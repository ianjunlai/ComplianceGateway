"""Naive Vector RAG condition.

Plain dense retrieval over chunk embeddings — no graph traversal, even though
the vectors live in the graph database.
"""
import config
from pipeline.base import RetrievalStrategy, RetrievedChunk, RetrievedContext
from pipeline.embeddings import embed_one
from pipeline.graph import get_driver, index_score_to_cosine

_SEARCH_QUERY = """
CALL db.index.vector.queryNodes($index, $k, $vec) YIELD node, score
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


class VectorRagStrategy(RetrievalStrategy):
    name = "vector_rag"

    def retrieve(self, query: str, seed_entities: list[str], top_k: int) -> RetrievedContext:
        with get_driver().session() as session:
            records = session.run(
                _SEARCH_QUERY,
                index=config.INDEX_CHUNKS,
                k=top_k * _OVERFETCH,
                vec=embed_one(query),
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
