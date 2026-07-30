"""Hybrid Vector-Graph RAG — primary benchmark.

Dense vector search locates seed graph nodes; a 2-hop Cypher traversal
(GRAPH_HOPS fixed at 2) expands to connected legal clauses.
"""
import config
from pipeline.base import RetrievalStrategy, RetrievedChunk, RetrievedContext
from pipeline.entity_linking import link_entities
from pipeline.graph import get_driver

# 2-hop expansion from seed nodes to provenance chunks.
# Graph schema (built by ingestion.build_indexes):
#   (:Entity {node_id, name}) -[:RELATES {type, description}]-> (:Entity)
#   (:Entity) -[:MENTIONED_IN]-> (:Chunk {chunk_id, text})
# Chunks are ranked by graph-evidence density (how many retrieved entities
# mention them), so LIMIT keeps the best-supported clauses — comparable to the
# score-ranked output of the vector strategies.
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
RETURN c.chunk_id AS chunk_id, c.text AS text, entities
ORDER BY size(entities) DESC
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

        # Step 2: 2-hop traversal to provenance chunks
        with get_driver().session() as session:
            records = session.run(
                _TRAVERSAL_QUERY % {"hops": config.GRAPH_HOPS},
                seed_ids=seed_ids,
                limit=top_k,
            )
            chunks, nodes = [], set()
            for rec in records:
                chunks.append(RetrievedChunk(chunk_id=rec["chunk_id"], text=rec["text"]))
                nodes.update(rec["entities"])
        return RetrievedContext(chunks=chunks[:top_k], graph_nodes=sorted(nodes))
