"""Hybrid Vector-Graph RAG: link query entities to nodes, traverse, then rank the
admitted clauses by query similarity."""
import config
from pipeline.base import RetrievalStrategy, RetrievedChunk, RetrievedContext
from pipeline.embeddings import embed_one
from pipeline.entity_linking import link_entities
from pipeline.graph import get_driver, index_score_to_cosine

# 2-hop expansion from seed nodes to provenance chunks, ranked by query
# similarity.
# Graph schema (built by ingestion.build_indexes):
#   (:Entity {node_id, name}) -[:RELATES {type, description}]-> (:Entity)
#   (:Entity) -[:MENTIONED_IN]-> (:Chunk {chunk_id, text, embedding})
# vector.similarity.cosine scores the candidates in-database, so only the
# top-k rows cross the wire rather than the whole reachable set.
# %(hops)d is interpolated (an int from config): Cypher cannot parameterize
# the bounds of a variable-length pattern.
_TRAVERSAL_QUERY = """
UNWIND $seed_ids AS seed_id
MATCH (seed:Entity {node_id: seed_id})
CALL (seed) {
    MATCH (seed)-[:RELATES*1..%(hops)d]-(nbr:Entity)
    RETURN collect(DISTINCT nbr) AS nbrs
}
WITH seed, nbrs
UNWIND ([seed] + nbrs) AS ent
MATCH (ent)-[:MENTIONED_IN]->(c:Chunk)
WHERE ($allowed IS NULL OR c.jurisdiction IN $allowed)
WITH c, collect(DISTINCT ent.name) AS entities
RETURN c.chunk_id AS chunk_id, c.text AS text, entities,
       vector.similarity.cosine(c.embedding, $qvec) AS score
ORDER BY score DESC
LIMIT $limit
"""

# Same traversal, then one hop along the cross-tier citations.
_TRAVERSAL_WITH_IMPLEMENTS = """
UNWIND $seed_ids AS seed_id
MATCH (seed:Entity {node_id: seed_id})
CALL (seed) {
    MATCH (seed)-[:RELATES*1..%(hops)d]-(nbr:Entity)
    RETURN collect(DISTINCT nbr) AS nbrs
}
WITH seed, nbrs
UNWIND ([seed] + nbrs) AS ent
MATCH (ent)-[:MENTIONED_IN]->(c:Chunk)
WHERE ($allowed IS NULL OR c.jurisdiction IN $allowed)
WITH collect(DISTINCT c) AS reached
UNWIND reached AS c
OPTIONAL MATCH (c)-[:CITES|IMPLEMENTS]-(linked:Chunk)
WITH reached, collect(DISTINCT linked) AS crossed
UNWIND (reached + crossed) AS c
WITH DISTINCT c WHERE c IS NOT NULL
WITH c, vector.similarity.cosine(c.embedding, $qvec) AS score
ORDER BY score DESC
LIMIT $limit
// Entities are attached AFTER the cut, not before: collecting them over the
// whole admitted set returned 1,204 names for a ten-chunk answer, which would
// have gone straight into the generation prompt and made the two flag settings
// incomparable (78 names with it off).
OPTIONAL MATCH (e:Entity)-[:MENTIONED_IN]->(c)
RETURN c.chunk_id AS chunk_id, c.text AS text, score,
       collect(DISTINCT e.name) AS entities
ORDER BY score DESC
"""


class HybridGraphStrategy(RetrievalStrategy):
    name = "hybrid"

    def retrieve(self, query: str, seed_entities: list[str], top_k: int,
                 allowed_jurisdictions: list[str] | None = None) -> RetrievedContext:
        # Step 1: embedding-based entity linking
        seed_ids = link_entities(seed_entities)
        if not seed_ids:
            # Fallback: no linkable entity -> degrade to empty context (SLM must abstain)
            return RetrievedContext()

        # Step 2: 2-hop traversal to provenance chunks, ranked by query
        # similarity.
        cypher = (_TRAVERSAL_WITH_IMPLEMENTS if config.HYBRID_FOLLOW_IMPLEMENTS
                  else _TRAVERSAL_QUERY)
        with get_driver().session() as session:
            records = session.run(
                cypher % {"hops": config.GRAPH_HOPS},
                seed_ids=seed_ids,
                qvec=embed_one(query),
                limit=top_k,
                allowed=allowed_jurisdictions,
            )
            chunks, nodes = [], set()
            for rec in records:
                chunks.append(RetrievedChunk(
                    chunk_id=rec["chunk_id"],
                    text=rec["text"],
                    score=index_score_to_cosine(rec["score"]),
                ))
                nodes.update(rec["entities"])
        return RetrievedContext(chunks=chunks[:top_k], graph_nodes=sorted(nodes))
