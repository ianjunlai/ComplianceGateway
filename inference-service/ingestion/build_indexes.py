"""Offline multi-paradigm index builder.

One shared extraction pass feeds all three GraphRAG paradigms, and one database
holds the result: Neo4j stores the knowledge graph and, through native vector
indexes, the embeddings as well.

  1. chunking (semantic, structural)
  2. GPT-4o entity/relation extraction + embedding-based dedup   [SHARED]
  3. graph structure: Chunk / Entity nodes, MENTIONED_IN + RELATES edges [SHARED]
  4a. Hybrid + Vector RAG: chunk embeddings -> chunk vector index
  4b. LightRAG:            entity and relationship embeddings -> vector indexes
  4c. HippoRAG:            sparse adjacency + node-passage matrices -> artifacts/

Every phase is metered by CostTracker -> artifacts/indexing_cost_report.json.

Run:  python -m ingestion.build_indexes
"""
import json
import logging
from pathlib import Path

import numpy as np
from scipy import sparse

import config
from ingestion.chunking import Chunk, load_corpus
from ingestion.cost_tracker import CostTracker
from ingestion.dedup import CanonicalEntity, deduplicate_entities
from ingestion.extraction import extract_graph_elements
from pipeline.embeddings import embed
from pipeline.graph import get_driver

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_indexes")

CORPUS_DIR = Path(__file__).resolve().parents[2] / "dataset" / "corpus"
ARTIFACTS = Path(config.ARTIFACTS_DIR)

# Lookup indexes; without them every MATCH below is a label scan.
_PROPERTY_INDEXES = [
    "CREATE INDEX chunk_id_idx IF NOT EXISTS FOR (c:Chunk) ON (c.chunk_id)",
    "CREATE INDEX entity_id_idx IF NOT EXISTS FOR (e:Entity) ON (e.node_id)",
    "CREATE INDEX rel_id_idx IF NOT EXISTS FOR ()-[r:RELATES]-() ON (r.rel_id)",
]

_CREATE_CHUNKS = """
UNWIND $rows AS row
CREATE (:Chunk {chunk_id: row.chunk_id, text: row.text, source: row.source, title: row.title})
"""

_CREATE_ENTITIES = """
UNWIND $rows AS row
CREATE (:Entity {node_id: row.node_id, name: row.name, type: row.type, aliases: row.aliases})
"""

_CREATE_MENTIONS = """
UNWIND $rows AS row
MATCH (e:Entity {node_id: row.node_id}), (c:Chunk {chunk_id: row.chunk_id})
CREATE (e)-[:MENTIONED_IN]->(c)
"""

_CREATE_RELATIONS = """
UNWIND $rows AS row
MATCH (a:Entity {node_id: row.source_id}), (b:Entity {node_id: row.target_id})
CREATE (a)-[:RELATES {rel_id: row.rel_id, type: row.type,
                      description: row.description, chunk_id: row.chunk_id}]->(b)
"""

_SET_NODE_VECTORS = """
UNWIND $rows AS row
MATCH (n:%(label)s {%(key)s: row.id})
SET n.embedding = row.embedding
"""

_SET_EDGE_VECTORS = """
UNWIND $rows AS row
MATCH ()-[r:RELATES {rel_id: row.id}]->()
SET r.embedding = row.embedding
"""


def main() -> None:
    tracker = CostTracker()

    # ---- Stage 1: chunking -------------------------------------------------
    chunks = load_corpus(CORPUS_DIR)
    if not chunks:
        raise SystemExit(f"No corpus found under {CORPUS_DIR} — see dataset/corpus/README.md")
    log.info("Chunked corpus: %d chunks", len(chunks))

    # ---- Stage 2 (SHARED): extraction + dedup ------------------------------
    raw_entities: list[dict] = []
    raw_relations: list[dict] = []
    for chunk in chunks:
        data = extract_graph_elements(chunk, tracker)
        for e in data["entities"]:
            raw_entities.append({**e, "chunk_id": chunk.chunk_id})
        for r in data["relations"]:
            raw_relations.append({**r, "chunk_id": chunk.chunk_id})
    log.info("Extracted %d raw entities, %d relations", len(raw_entities), len(raw_relations))

    entities, name_to_node, merge_log = deduplicate_entities(raw_entities)
    relations = _rewire_relations(raw_relations, name_to_node)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "dedup_report.json").write_text(
        json.dumps(merge_log, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(
        "After dedup: %d canonical entities, %d relations. %d cross-name merges "
        "-> dedup_report.json — AUDIT IT before trusting the graph (legally "
        "distinct terms like controller/processor must not have merged)",
        len(entities), len(relations), len(merge_log),
    )

    driver = get_driver()

    # ---- Stage 3 (SHARED): graph structure ---------------------------------
    # Charged separately: hybrid traversal and LightRAG expansion both depend
    # on it, so attributing it to one paradigm would distort the comparison.
    with tracker.build_phase("shared_graph"):
        _create_graph(driver, entities, relations, chunks)

    # ---- Stage 4a: Hybrid + Vector RAG -------------------------------------
    with tracker.build_phase("hybrid"):
        _index_chunk_vectors(driver, chunks, tracker)

    # ---- Stage 4b: LightRAG ------------------------------------------------
    with tracker.build_phase("light_rag"):
        _index_entity_edge_vectors(driver, entities, relations, tracker)

    # ---- Stage 4c: HippoRAG ------------------------------------------------
    with tracker.build_phase("hippo_rag"):
        _build_hippo_artifacts(entities, relations, chunks, tracker)

    with driver.session() as s:
        s.run("CALL db.awaitIndexes()")  # vector indexes populate asynchronously
    log.info("Vector indexes populated")

    tracker.save(ARTIFACTS / "indexing_cost_report.json")
    log.info("Done. Cost report -> %s", ARTIFACTS / "indexing_cost_report.json")


def _rewire_relations(raw_relations: list[dict], name_to_node: dict[str, str]) -> list[dict]:
    """Relation endpoints follow their entities into canonical node IDs."""
    out = []
    for r in raw_relations:
        src, tgt = name_to_node.get(r["source"]), name_to_node.get(r["target"])
        if src and tgt and src != tgt:
            out.append({**r, "source_id": src, "target_id": tgt})
    return out


def _create_vector_index(session, name: str, pattern: str) -> None:
    """Index name and dimension come from config, so inlining them is safe;
    Cypher does not accept parameters in index options."""
    session.run(
        f"CREATE VECTOR INDEX {name} IF NOT EXISTS {pattern} "
        f"OPTIONS {{indexConfig: {{"
        f"`vector.dimensions`: {config.VECTOR_DIM}, "
        f"`vector.similarity_function`: 'cosine'}}}}"
    )


def _create_graph(driver, entities: list[CanonicalEntity], relations: list[dict],
                  chunks: list[Chunk]) -> None:
    with driver.session() as s:
        s.run("MATCH (n) DETACH DELETE n")  # rebuild from scratch (idempotent offline job)
        for stmt in _PROPERTY_INDEXES:
            s.run(stmt)

        s.run(_CREATE_CHUNKS, rows=[
            {"chunk_id": c.chunk_id, "text": c.text, "source": c.source, "title": c.title}
            for c in chunks
        ])
        s.run(_CREATE_ENTITIES, rows=[
            {"node_id": e.node_id, "name": e.name, "type": e.type, "aliases": sorted(e.aliases)}
            for e in entities
        ])
        s.run(_CREATE_MENTIONS, rows=[
            {"node_id": e.node_id, "chunk_id": cid}
            for e in entities for cid in sorted(e.chunk_ids)
        ])
        s.run(_CREATE_RELATIONS, rows=[
            {"rel_id": i, "source_id": r["source_id"], "target_id": r["target_id"],
             "type": r.get("type", "RELATES"), "description": r.get("description", ""),
             "chunk_id": r["chunk_id"]}
            for i, r in enumerate(relations)
        ])
    log.info("Neo4j graph: %d entities, %d relations, %d chunks",
             len(entities), len(relations), len(chunks))


def _index_chunk_vectors(driver, chunks: list[Chunk], tracker: CostTracker) -> None:
    vectors = embed([c.text for c in chunks])
    tracker.add_embeddings("hybrid", len(vectors))
    # Logical size estimate (float32 vectors + stored text); on-disk sizes
    # are measured separately from the docker volume
    tracker.add_storage("hybrid",
                        len(vectors) * config.VECTOR_DIM * 4
                        + sum(len(c.text.encode()) for c in chunks))
    with driver.session() as s:
        s.run(_SET_NODE_VECTORS % {"label": "Chunk", "key": "chunk_id"},
              rows=[{"id": c.chunk_id, "embedding": v} for c, v in zip(chunks, vectors)])
        _create_vector_index(s, config.INDEX_CHUNKS, "FOR (c:Chunk) ON (c.embedding)")
    log.info("Chunk vector index `%s`: %d vectors", config.INDEX_CHUNKS, len(vectors))


def _index_entity_edge_vectors(driver, entities: list[CanonicalEntity],
                               relations: list[dict], tracker: CostTracker) -> None:
    # Entity vectors (LightRAG low-level; also used by entity linking)
    ent_vectors = embed([e.name for e in entities])
    tracker.add_embeddings("light_rag", len(ent_vectors))
    tracker.add_storage("light_rag", len(ent_vectors) * config.VECTOR_DIM * 4)

    # Edge vectors (LightRAG high-level): embed the relation descriptions
    edge_texts = [
        r.get("description") or f'{r["source"]} {r.get("type", "")} {r["target"]}'
        for r in relations
    ]
    edge_vectors = embed(edge_texts) if edge_texts else []
    if edge_vectors:
        tracker.add_embeddings("light_rag", len(edge_vectors))
        tracker.add_storage("light_rag",
                            len(edge_vectors) * config.VECTOR_DIM * 4
                            + sum(len(t.encode()) for t in edge_texts))

    with driver.session() as s:
        s.run(_SET_NODE_VECTORS % {"label": "Entity", "key": "node_id"},
              rows=[{"id": e.node_id, "embedding": v} for e, v in zip(entities, ent_vectors)])
        _create_vector_index(s, config.INDEX_ENTITIES, "FOR (e:Entity) ON (e.embedding)")
        if edge_vectors:
            s.run(_SET_EDGE_VECTORS,
                  rows=[{"id": i, "embedding": v} for i, v in enumerate(edge_vectors)])
            _create_vector_index(s, config.INDEX_EDGES, "FOR ()-[r:RELATES]-() ON (r.embedding)")
    log.info("Entity/edge vector indexes: %d entities, %d edges",
             len(ent_vectors), len(edge_vectors))


def _build_hippo_artifacts(
    entities: list[CanonicalEntity], relations: list[dict], chunks: list[Chunk], tracker: CostTracker
) -> None:
    """Row-normalized adjacency matrix, node-passage count matrix, and provenance maps.

    The node-passage matrix holds how many times each entity occurs in each
    chunk; passage ranking multiplies the PPR node distribution by it, so a
    chunk backed by several activated entities outranks one backed by a single
    lucky match.
    """
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    node_index = {e.node_id: i for i, e in enumerate(entities)}
    n = len(entities)

    rows, cols = [], []
    for r in relations:
        i, j = node_index[r["source_id"]], node_index[r["target_id"]]
        rows.extend([i, j])  # treat as undirected for PPR connectivity
        cols.extend([j, i])
    adj = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))

    # Row-normalize -> column-stochastic transition on transpose (see hippo_rag.py)
    row_sums = np.asarray(adj.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    adj = sparse.diags(1.0 / row_sums) @ adj

    # Node-passage count matrix: rows = entities, cols = chunks
    chunk_index = {c.chunk_id: j for j, c in enumerate(chunks)}
    p_rows, p_cols, p_vals = [], [], []
    for e in entities:
        i = node_index[e.node_id]
        for cid, count in e.chunk_counts.items():
            if cid in chunk_index:
                p_rows.append(i)
                p_cols.append(chunk_index[cid])
                p_vals.append(float(count))
    passage_matrix = sparse.csr_matrix(
        (p_vals, (p_rows, p_cols)), shape=(n, len(chunks))
    )

    sparse.save_npz(ARTIFACTS / "hippo_adjacency.npz", adj.tocsr())
    sparse.save_npz(ARTIFACTS / "hippo_passage_matrix.npz", passage_matrix)
    (ARTIFACTS / "hippo_chunk_index.json").write_text(
        json.dumps(chunk_index), encoding="utf-8")
    (ARTIFACTS / "hippo_nodes.json").write_text(json.dumps(node_index), encoding="utf-8")
    (ARTIFACTS / "hippo_node_names.json").write_text(
        json.dumps({e.node_id: e.name for e in entities}), encoding="utf-8")
    (ARTIFACTS / "hippo_node_chunks.json").write_text(
        json.dumps({e.node_id: sorted(e.chunk_ids) for e in entities}), encoding="utf-8")
    (ARTIFACTS / "chunk_texts.json").write_text(
        json.dumps({c.chunk_id: c.text for c in chunks}), encoding="utf-8")

    total_bytes = sum((ARTIFACTS / f).stat().st_size for f in
                      ["hippo_adjacency.npz", "hippo_passage_matrix.npz",
                       "hippo_chunk_index.json", "hippo_nodes.json", "hippo_node_names.json",
                       "hippo_node_chunks.json", "chunk_texts.json"])
    tracker.add_storage("hippo_rag", total_bytes)
    log.info("HippoRAG artifacts: %d nodes, %d passages, %d bytes", n, len(chunks), total_bytes)


if __name__ == "__main__":
    main()
