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
_ENTITY_QUERY = """
UNWIND $vectors AS vec
CALL db.index.vector.queryNodes($index, $limit, vec) YIELD node, score
WITH node, max(score) AS score
MATCH (node)-[:MENTIONED_IN]->(c:Chunk)
WITH node, score, collect(c.chunk_id) AS chunk_ids
RETURN node.node_id AS node_id, node.name AS name, score, chunk_ids
"""

# High level: relationship vector index, returning both endpoints so the
# expansion step can start from them.
_EDGE_QUERY = """
CALL db.index.vector.queryRelationships($index, $k, $vec) YIELD relationship AS r, score
RETURN r.description AS description, r.chunk_id AS chunk_id, score,
       startNode(r).node_id AS source_id, endNode(r).node_id AS target_id
"""

# One-hop expansion from matched graph elements to neighbouring entities and
# the clauses that mention them.
_NEIGHBOUR_QUERY = """
UNWIND $node_ids AS nid
MATCH (seed:Entity {node_id: nid})-[:RELATES]-(nbr:Entity)
MATCH (nbr)-[:MENTIONED_IN]->(c:Chunk)
RETURN DISTINCT nbr.name AS name, c.chunk_id AS chunk_id, c.text AS text
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

            node_best: dict[str, float] = {}
            chunk_scores: dict[str, float] = {}
            for hit in entity_hits:
                score = index_score_to_cosine(hit["score"])
                node_best[hit["name"]] = max(node_best.get(hit["name"], 0.0), score)
                for cid in hit["chunk_ids"]:
                    chunk_scores[cid] = max(chunk_scores.get(cid, 0.0), score)
            edges = []
            for hit in edge_hits:
                score = index_score_to_cosine(hit["score"])
                edges.append(hit["description"])
                if hit["chunk_id"]:
                    chunk_scores[hit["chunk_id"]] = max(
                        chunk_scores.get(hit["chunk_id"], 0.0), score)

            # High-order relatedness: one-hop neighbours of the matched entities
            # and of the entities the matched edges connect
            expansion_seeds = {h["node_id"] for h in entity_hits}
            for hit in edge_hits:
                expansion_seeds.update(
                    hit[key] for key in ("source_id", "target_id") if hit.get(key)
                )
            chunk_texts: dict[str, str] = {}
            if expansion_seeds:
                expanded_score = (
                    max(chunk_scores.values(), default=1.0) * config.LIGHTRAG_NEIGHBOUR_DECAY
                )
                for r in session.run(
                    _NEIGHBOUR_QUERY, node_ids=sorted(expansion_seeds), limit=top_k * 4
                ):
                    node_best.setdefault(r["name"], expanded_score)
                    chunk_texts[r["chunk_id"]] = r["text"]
                    chunk_scores[r["chunk_id"]] = max(
                        chunk_scores.get(r["chunk_id"], 0.0), expanded_score)

            nodes = [n for n, _ in sorted(node_best.items(), key=lambda kv: -kv[1])][:top_k]
            top_chunk_ids = [
                cid for cid, _ in sorted(chunk_scores.items(), key=lambda kv: -kv[1])
            ][:top_k]
            chunks = self._fetch_chunks(session, top_chunk_ids, chunk_texts)

        return RetrievedContext(chunks=chunks, graph_nodes=nodes, graph_edges=edges)

    def _fetch_chunks(self, session, chunk_ids: list[str], known_texts: dict[str, str]) -> list[RetrievedChunk]:
        """Chunk texts, reusing those already returned by the graph expansion."""
        missing = [cid for cid in chunk_ids if cid not in known_texts]
        by_id = dict(known_texts)
        if missing:
            by_id.update({
                r["chunk_id"]: r["text"]
                for r in session.run(_CHUNK_QUERY, chunk_ids=missing)
            })
        return [RetrievedChunk(chunk_id=cid, text=by_id[cid]) for cid in chunk_ids if cid in by_id]
