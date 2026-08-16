"""Dense vector RAG over the chunk index. No graph steps."""
import config
from pipeline.base import RetrievalStrategy, RetrievedChunk, RetrievedContext
from pipeline.embeddings import embed_one
from pipeline.graph import get_driver, index_score_to_cosine

_SEARCH_QUERY = """
CALL db.index.vector.queryNodes($index, $k, $vec) YIELD node, score
WITH node, score WHERE ($allowed IS NULL OR node.jurisdiction IN $allowed)
RETURN node.chunk_id AS chunk_id, node.text AS text, score
"""

# Neo4j's vector index is approximate (HNSW), and an approximate search explores
# less of the graph when asked for fewer neighbours: on this corpus a k=10 query
# returned a different top 10 from an exact full-corpus ranking for 52% of
# queries. Over-fetching and slicing recovers the true top-k. This is not a
# change of method -- the strategy is still "the k nearest chunks by cosine" --
# it removes an index-tuning artefact that would otherwise be charged to dense
# retrieval, and would confound the comparison against the graph strategies,
# which score their candidates exactly with vector.similarity.cosine.
_OVERFETCH = 8
# Neo4j Community has no pre-filter on a vector index, so a jurisdiction scope
# is applied after the index returns.
_OVERFETCH_FILTERED = 20


class VectorRagStrategy(RetrievalStrategy):
    name = "vector_rag"

    def retrieve(self, query: str, seed_entities: list[str], top_k: int,
                 allowed_jurisdictions: list[str] | None = None) -> RetrievedContext:
        over = _OVERFETCH if allowed_jurisdictions is None else _OVERFETCH_FILTERED
        with get_driver().session() as session:
            records = session.run(
                _SEARCH_QUERY,
                index=config.INDEX_CHUNKS,
                k=top_k * over,
                vec=embed_one(query),
                allowed=allowed_jurisdictions,
            )
            chunks = [
                RetrievedChunk(
                    chunk_id=r["chunk_id"],
                    text=r["text"],
                    score=index_score_to_cosine(r["score"]),
                )
                for r in records
            ]
        # The over-fetch is an index-accuracy measure, not a wider result set.
        return RetrievedContext(chunks=chunks[:top_k])
