"""LightRAG-style dual-level retrieval.

Three stages, following the published paradigm:
  1. Low-level retrieval — seed entity mentions are matched against entity
     vectors, covering specific, detail-oriented queries.
  2. High-level retrieval — the whole query is matched against relationship-edge
     vectors, covering abstract, thematic queries.
  3. High-order relatedness — the matched entities and edge endpoints are
     expanded to their one-hop graph neighbours, so the context includes
     structurally related clauses that neither vector search surfaced directly.

Expanded evidence is scored below direct hits by LIGHTRAG_NEIGHBOUR_DECAY, so
graph-derived clauses enrich the context without displacing exact matches.

This is a controlled re-implementation sharing the extraction pass, index and
SLM with the other strategies, not the reference library.
"""
import config
from pipeline.base import RetrievalStrategy, RetrievedChunk, RetrievedContext
from pipeline.embeddings import embed, embed_one
from pipeline.graph import get_driver, index_score_to_cosine

# Low level: every seed mention is matched in one round trip, carrying the
# provenance clauses of each matched entity.
#
# relation_count is what separates one clause of a matched entity from another.
# Without it every clause mentioning the entity inherits the entity's score and
# they all tie -- and this corpus's commonest entity appears in 53% of clauses,
# so a single hub match would flood the ranking with a 184-way tie broken by
# nothing. LightRAG orders a matched entity's clauses by how much of that
# entity's own neighbourhood each clause also contains, which is the signal
# reproduced here.
_ENTITY_QUERY = """
UNWIND $vectors AS vec
CALL db.index.vector.queryNodes($index, $limit, vec) YIELD node, score
WITH node, max(score) AS score
MATCH (node)-[:MENTIONED_IN]->(c:Chunk)
RETURN node.node_id AS node_id, node.name AS name, score,
       c.chunk_id AS chunk_id,
       COUNT {
           MATCH (node)-[:RELATES]-(nbr:Entity)-[:MENTIONED_IN]->(c)
           RETURN DISTINCT nbr
       } AS relation_count
ORDER BY score DESC, relation_count DESC
"""

# High level: relationship vector index, returning both endpoints so the
# expansion step can start from them.
_EDGE_QUERY = """
CALL db.index.vector.queryRelationships($index, $k, $vec) YIELD relationship AS r, score
RETURN r.description AS description, r.chunk_ids AS chunk_ids, score,
       startNode(r).node_id AS source_id, endNode(r).node_id AS target_id
"""

# One-hop expansion from matched graph elements to neighbouring entities and
# the clauses that mention them.
#
# The ORDER BY is load-bearing, not cosmetic: this LIMIT discards most of the
# expansion, and without an ordering Neo4j is free to return any rows it likes
# and to return different ones on an identical re-run -- which would make the
# strategy's retrieved set, and every metric computed from it, unreproducible.
# Rows are ranked by how many of the matched elements reach the neighbour, so
# the strongest structural evidence survives the cut; chunk_id breaks ties so
# the result is fully determined.
_NEIGHBOUR_QUERY = """
UNWIND $node_ids AS nid
MATCH (seed:Entity {node_id: nid})-[:RELATES]-(nbr:Entity)
MATCH (nbr)-[:MENTIONED_IN]->(c:Chunk)
WITH nbr, c, count(DISTINCT seed) AS support
RETURN nbr.name AS name, c.chunk_id AS chunk_id, c.text AS text, support
ORDER BY support DESC, c.chunk_id
LIMIT $limit
"""

_CHUNK_QUERY = """
UNWIND $chunk_ids AS cid
MATCH (c:Chunk {chunk_id: cid})
RETURN c.chunk_id AS chunk_id, c.text AS text
"""


class LightRagStrategy(RetrievalStrategy):
    name = "light_rag"

    # Low-level hits kept per seed mention before merging (disclosed constant).
    # No tau cutoff here: unlike entity linking, these hits are soft-ranked
    # evidence — a weak match ranks lower instead of poisoning a traversal.
    LOW_LEVEL_HITS_PER_MENTION = 3

    def retrieve(self, query: str, seed_entities: list[str], top_k: int) -> RetrievedContext:
        qvec = embed_one(query)
        mention_vecs = embed(seed_entities) if seed_entities else [qvec]

        with get_driver().session() as session:
            entity_hits = [
                dict(r) for r in session.run(
                    _ENTITY_QUERY,
                    vectors=mention_vecs,
                    index=config.INDEX_ENTITIES,
                    limit=self.LOW_LEVEL_HITS_PER_MENTION,
                )
            ]
            edge_hits = [
                dict(r) for r in session.run(
                    _EDGE_QUERY, index=config.INDEX_EDGES, k=top_k, vec=qvec
                )
            ]

            # A clause is ranked by (similarity of the element that surfaced it,
            # how much of that element's neighbourhood it contains). Keys are
            # negated so the natural sort order is best-first, and a clause
            # surfaced by several elements keeps its strongest claim.
            node_best: dict[str, float] = {}
            chunk_keys: dict[str, tuple[float, float]] = {}

            def claim(chunk_id: str, score: float, relation_count: int = 0) -> None:
                key = (-score, -relation_count)
                if key < chunk_keys.get(chunk_id, (1.0, 1.0)):
                    chunk_keys[chunk_id] = key

            for hit in entity_hits:
                score = index_score_to_cosine(hit["score"])
                node_best[hit["name"]] = max(node_best.get(hit["name"], 0.0), score)
                claim(hit["chunk_id"], score, hit["relation_count"])
            edges = []
            for hit in edge_hits:
                score = index_score_to_cosine(hit["score"])
                edges.append(hit["description"])
                for cid in hit["chunk_ids"] or []:
                    # An edge names its endpoints, not a neighbourhood within a
                    # clause, so it carries no relation_count of its own.
                    claim(cid, score)

            # High-order relatedness: one-hop neighbours of the matched entities
            # and of the entities the matched edges connect
            expansion_seeds = {h["node_id"] for h in entity_hits}
            for hit in edge_hits:
                expansion_seeds.update(
                    hit[key] for key in ("source_id", "target_id") if hit.get(key)
                )
            chunk_texts: dict[str, str] = {}
            if expansion_seeds:
                best_direct = -min(chunk_keys.values(), default=(-1.0, 0.0))[0]
                expanded_score = best_direct * config.LIGHTRAG_NEIGHBOUR_DECAY
                for r in session.run(
                    _NEIGHBOUR_QUERY, node_ids=sorted(expansion_seeds), limit=top_k * 4
                ):
                    node_best.setdefault(r["name"], expanded_score)
                    chunk_texts[r["chunk_id"]] = r["text"]
                    # Expanded evidence ranks below every direct hit by the
                    # decay, and among itself by how many matched elements
                    # reached it.
                    claim(r["chunk_id"], expanded_score, r["support"])

            nodes = [n for n, _ in sorted(node_best.items(), key=lambda kv: -kv[1])][:top_k]
            ranked = sorted(chunk_keys.items(), key=lambda kv: kv[1])[:top_k]
            chunks = self._fetch_chunks(session, ranked, chunk_texts)

        return RetrievedContext(chunks=chunks, graph_nodes=nodes, graph_edges=edges)

    def _fetch_chunks(self, session, ranked: list[tuple[str, tuple[float, float]]],
                      known_texts: dict[str, str]) -> list[RetrievedChunk]:
        """Chunk texts, reusing those already returned by the graph expansion."""
        missing = [cid for cid, _ in ranked if cid not in known_texts]
        by_id = dict(known_texts)
        if missing:
            by_id.update({
                r["chunk_id"]: r["text"]
                for r in session.run(_CHUNK_QUERY, chunk_ids=missing)
            })
        return [
            # key[0] is the negated similarity of the element that surfaced it
            RetrievedChunk(chunk_id=cid, text=by_id[cid], score=-key[0])
            for cid, key in ranked
            if cid in by_id
        ]
