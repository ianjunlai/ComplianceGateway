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


class VectorRagStrategy(RetrievalStrategy):
    name = "vector_rag"

    def retrieve(self, query: str, seed_entities: list[str], top_k: int) -> RetrievedContext:
        with get_driver().session() as session:
            records = session.run(
                _SEARCH_QUERY,
                index=config.INDEX_CHUNKS,
                k=top_k,
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
        return RetrievedContext(chunks=chunks)
