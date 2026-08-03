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
import argparse
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy import sparse

import config
from ingestion.chunking import Chunk, load_corpus
from ingestion.cost_tracker import CostTracker
from ingestion.dedup import CanonicalEntity, deduplicate_entities
from ingestion.extraction import extract_graph_elements, normalise_elements
from pipeline.embeddings import embed
from pipeline.graph import get_driver

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_indexes")

CORPUS_DIR = Path(__file__).resolve().parents[2] / "dataset" / "corpus"
ARTIFACTS = Path(config.ARTIFACTS_DIR)
EXTRACTION_CACHE_FILE = ARTIFACTS / "extraction_cache.json"

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
                      description: row.description, chunk_ids: row.chunk_ids}]->(b)
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
    args = _parse_args()
    tracker = CostTracker()
    tracker.note("extraction_workers", args.workers)

    # ---- Stage 1: chunking -------------------------------------------------
    if args.corpus_json:
        chunks = _load_prechunked(Path(args.corpus_json))
        log.info("Pre-chunked corpus: %d passages from %s", len(chunks), args.corpus_json)
    else:
        chunks = load_corpus(CORPUS_DIR)
        if not chunks:
            raise SystemExit(f"No corpus found under {CORPUS_DIR} — see dataset/corpus/README.md")
        log.info("Chunked corpus: %d chunks", len(chunks))

    # ---- Stage 2 (SHARED): extraction + dedup ------------------------------
    # The profile is part of the key: a cache built by the legal extractor must
    # never be replayed for a general-domain corpus, and the failure would
    # otherwise be silent -- a full cache hit and a graph of the wrong shape.
    cache_key = f"{config.EXTRACTION_PROVIDER}:{config.EXTRACTION_MODEL}:{config.EXTRACTION_PROFILE}"
    cache = _load_extraction_cache(cache_key, args.reextract)
    cache_hits = _extract_all(chunks, cache, cache_key, tracker, args.workers)

    # Normalised on the way out of the cache, not only on the way in: a cache
    # written before a coercion rule existed still has to produce indexable
    # data, and re-extracting to pick the rule up is not affordable.
    raw_entities: list[dict] = []
    raw_relations: list[dict] = []
    salvaged = 0
    for chunk in chunks:
        entities, relations, dropped = normalise_elements(cache[chunk.chunk_id])
        salvaged += dropped
        for e in entities:
            raw_entities.append({**e, "chunk_id": chunk.chunk_id})
        for r in relations:
            raw_relations.append({**r, "chunk_id": chunk.chunk_id})
    if salvaged:
        log.info("Normalisation dropped %d malformed element(s) across the corpus", salvaged)
    log.info("Extracted %d raw entities, %d relations (%d/%d chunks served from cache)",
             len(raw_entities), len(raw_relations), cache_hits, len(chunks))

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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--corpus-json",
        help="pre-segmented corpus as a JSON list of {chunk_id, title, text}. "
             "Bypasses chunking.py, whose Article/Recital parsing and '###' "
             "markers only describe the GDPR corpus. Used by the public-benchmark run.")
    p.add_argument(
        "--reextract", action="store_true",
        help="discard an extraction cache whose provider/model/profile no longer "
             "matches, and pay for it again. Without this the mismatch is an error, "
             "because the usual cause is an edited .env rather than an intent to rebuild.")
    p.add_argument(
        "--workers", type=int, default=1,
        help="concurrent extraction calls. The default of 1 preserves the "
             "sequential behaviour every reported GDPR figure was produced "
             "with; the 6k-passage benchmark corpus needs 8 or so to finish in "
             "an evening rather than overnight.")
    return p.parse_args()


def _load_prechunked(path: Path) -> list[Chunk]:
    """Corpora that arrive already segmented — a retrieval benchmark ships its
    passages as the unit of retrieval, and re-splitting them would change the
    gold labels' meaning."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    chunks = [
        Chunk(chunk_id=r["chunk_id"], source=r.get("source", path.stem),
              title=r.get("title", ""), text=r["text"])
        for r in rows
    ]
    ids = [c.chunk_id for c in chunks]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{path}: chunk_id values are not unique — gold labels would be ambiguous")
    return chunks


def _extract_all(chunks: list[Chunk], cache: dict, cache_key: str,
                 tracker: CostTracker, workers: int) -> int:
    """Fill `cache` for every chunk, in parallel when asked, and return the
    number served from cache.

    Failures are collected rather than raised in flight: one bad chunk should
    not discard the hundreds already paid for. The cache is written before
    raising, so a re-run resumes instead of starting over.
    """
    hits = 0
    for chunk in chunks:
        cached = cache.get(chunk.chunk_id)
        if cached is not None:
            hits += 1
            # Replay the original token usage so the cost report stays complete
            # on cached runs (older caches predate this field).
            if cached.get("usage"):
                tracker.add_tokens("extraction", **cached["usage"])
    todo = [c for c in chunks if c.chunk_id not in cache]
    log.info("Extraction: %d/%d chunks served from cache, %d to fetch (%d worker(s))",
             hits, len(chunks), len(todo), workers)
    if not todo:
        return hits

    lock = threading.Lock()
    failures: list[tuple[str, BaseException]] = []
    done = 0
    started = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        pending = {pool.submit(extract_graph_elements, c, tracker): c for c in todo}
        for future in as_completed(pending):
            chunk = pending[future]
            try:
                data = future.result()
            except BaseException as exc:          # noqa: BLE001 - recorded, re-raised below
                log.warning("extraction failed for %s: %r", chunk.chunk_id, exc)
                with lock:
                    failures.append((chunk.chunk_id, exc))
                continue
            with lock:
                cache[chunk.chunk_id] = data
                done += 1
                # Periodic rather than per-chunk: at 8 workers a write per
                # result rewrites a growing multi-megabyte file often enough to
                # become the bottleneck.
                if done % 25 == 0:
                    _save_extraction_cache(cache_key, cache)
                    rate = done / (time.perf_counter() - started)
                    log.info("  extracted %d/%d (%.1f/s, ~%.0f min left)",
                             done, len(todo), rate, (len(todo) - done) / rate / 60)

    _save_extraction_cache(cache_key, cache)
    if failures:
        raise SystemExit(
            f"{len(failures)} chunk(s) failed extraction; {done} succeeded and are cached. "
            f"Re-run to retry only the failures. First: {failures[0][0]} {failures[0][1]!r}")
    return hits


def _load_extraction_cache(cache_key: str, reextract: bool = False) -> dict:
    """Per-chunk extraction results keyed by chunk_id, so re-running the
    pipeline to iterate on dedup/graph logic doesn't re-pay for extraction.
    Invalidated wholesale if EXTRACTION_PROVIDER/MODEL changed since the
    cache was written -- a cache built with one model must never be silently
    reused for another.

    Caches written before the profile existed carry a two-part
    "provider:model" key. There was only one prompt then, the legal one, so
    such a cache is exactly a "provider:model:legal" cache and is accepted as
    one. Without this the GDPR cache -- 345 chunks and a 430k-token extraction
    pass -- would be silently discarded by the introduction of the profile,
    re-spending the budget and rebuilding a graph the reported results no
    longer describe.
    """
    if not EXTRACTION_CACHE_FILE.exists():
        return {}
    payload = json.loads(EXTRACTION_CACHE_FILE.read_text(encoding="utf-8"))
    stored = payload.get("cache_key")
    if stored is not None and stored.count(":") == 1 and cache_key.endswith(":legal"):
        stored = f"{stored}:legal"
    if stored != cache_key and reextract:
        log.warning("Discarding %d cached chunks built with %r (--reextract)",
                    len(payload.get("chunks", {})), payload.get("cache_key"))
        return {}
    if stored != cache_key:
        # Stop rather than quietly re-extract. Discarding a cache is a decision
        # worth hundreds of thousands of tokens and a graph the existing
        # results no longer describe, and the usual cause is an edited .env
        # rather than an intent to rebuild.
        raise SystemExit(
            f"Extraction cache at {EXTRACTION_CACHE_FILE} was built with "
            f"{payload.get('cache_key')!r} but this run is configured for {cache_key!r}, "
            f"so all {len(payload.get('chunks', {}))} cached chunks would be re-extracted.\n"
            f"  - to reuse the cache: set EXTRACTION_PROVIDER/EXTRACTION_MODEL/"
            f"EXTRACTION_PROFILE back to match it\n"
            f"  - to build a different corpus: point ARTIFACTS_DIR somewhere else\n"
            f"  - to genuinely re-extract with the new model: --reextract")
    return payload.get("chunks", {})


def _save_extraction_cache(cache_key: str, chunks: dict) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    EXTRACTION_CACHE_FILE.write_text(
        json.dumps({"cache_key": cache_key, "chunks": chunks}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _rewire_relations(raw_relations: list[dict], name_to_node: dict[str, str]) -> list[dict]:
    """Relation endpoints follow their entities into canonical node IDs, then
    relations are deduplicated by (source, target, type): the same fact
    restated across several chunks must not become several near-identical
    edges. Undeduplicated, this would inflate Hybrid's traversal cost, crowd
    LightRAG's edge vector index with redundant near-duplicates that push out
    genuinely distinct relations, and -- since scipy sums duplicate
    coordinates when building a sparse matrix -- silently give HippoRAG's PPR
    an accidental frequency-weighting nobody decided on. Every contributing
    chunk is kept as provenance in chunk_ids.
    """
    dropped_self_loop = 0
    dropped_missing_entity = 0
    by_key: dict[tuple[str, str, str], dict] = {}
    for r in raw_relations:
        src, tgt = name_to_node.get(r["source"]), name_to_node.get(r["target"])
        if not (src and tgt):
            dropped_missing_entity += 1
            continue
        if src == tgt:
            # e.g. two ends of a relation merged into the same entity during
            # dedup -- a spike here often means DEDUP_THRESHOLD merged too
            # aggressively; cross-check against dedup_report.json
            dropped_self_loop += 1
            continue
        key = (src, tgt, r.get("type", "RELATES"))
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = {
                "source_id": src, "target_id": tgt, "type": r.get("type", "RELATES"),
                "description": r.get("description", ""),
                "chunk_ids": [r["chunk_id"]],
            }
        else:
            existing["chunk_ids"].append(r["chunk_id"])
    if dropped_self_loop or dropped_missing_entity:
        log.warning(
            "Relation rewiring dropped %d relation(s) collapsed to self-loops "
            "after entity dedup and %d with an unresolvable entity reference",
            dropped_self_loop, dropped_missing_entity,
        )
    deduped = len(raw_relations) - dropped_self_loop - dropped_missing_entity - len(by_key)
    if deduped:
        log.info("Merged %d duplicate relation restatement(s) into %d unique edges",
                 deduped, len(by_key))
    return list(by_key.values())


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
             "type": r["type"], "description": r["description"], "chunk_ids": r["chunk_ids"]}
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

    # Edge vectors (LightRAG high-level): embed the relation descriptions.
    # Fallback text uses canonical entity names (not raw extraction strings --
    # relations are deduplicated onto node IDs and no longer carry those).
    id_to_name = {e.node_id: e.name for e in entities}
    edge_texts = [
        r["description"] or f'{id_to_name[r["source_id"]]} {r["type"]} {id_to_name[r["target_id"]]}'
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
