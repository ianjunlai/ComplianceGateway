"""Hybrid Vector-Graph RAG — primary benchmark.

Entity linking locates seed graph nodes; a 2-hop Cypher traversal (GRAPH_HOPS
fixed at 2) expands to connected legal clauses; the query embedding ranks them.

The two stages have distinct jobs, and conflating them is what an earlier
version got wrong. The graph decides which clauses are *admissible* — reachable
from an entity the query mentions. The query vector decides which of those are
*relevant*. Ranking instead by graph-evidence density (how many of the expanded
entities mention a chunk) measures nothing but hub centrality: this corpus's
most common entity appears in 53% of clauses, so a 2-hop expansion reaches
essentially the whole corpus and the same well-connected clauses win every
query regardless of what was asked. Measured on 40 gold-bearing queries,
that ranking scored Recall@10 = 0.138 against 0.513 for the ranking below.
"""
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
WITH c, collect(DISTINCT ent.name) AS entities
RETURN c.chunk_id AS chunk_id, c.text AS text, entities,
       vector.similarity.cosine(c.embedding, $qvec) AS score
ORDER BY score DESC
LIMIT $limit
"""


class HybridGraphStrategy(RetrievalStrategy):
    name = "hybrid"

    def retrieve(self, query: str, seed_entities: list[str], top_k: int) -> RetrievedContext:
        # Step 1: embedding-based entity linking
        seed_ids = link_entities(seed_entities)
        if not seed_ids:
            # Fallback: no linkable entity -> degrade to empty context (SLM must abstain)
            return RetrievedContext()

        # Step 2: 2-hop traversal to provenance chunks, ranked by query similarity
        with get_driver().session() as session:
            records = session.run(
                _TRAVERSAL_QUERY % {"hops": config.GRAPH_HOPS},
                seed_ids=seed_ids,
                qvec=embed_one(query),
                limit=top_k,
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
