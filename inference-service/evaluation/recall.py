"""Recall@K over chunk-ID sets.

Recall@K = |R ∩ GT| / |GT|, computed per query; K in {5, 10}.
All strategies emit chunk IDs as the normalized retrieval unit,
so the same function applies to vector RAG, hybrid, LightRAG and HippoRAG.
"""


def recall_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    if not gold_ids:
        raise ValueError("gold_ids must be non-empty (unanswerable queries are excluded from Recall@K)")
    top_k = set(retrieved_ids[:k])
    return len(top_k & set(gold_ids)) / len(gold_ids)


def mean_recall(results: list[dict], k: int) -> float:
    """results: [{"retrieved_chunk_ids": [...], "gold_chunk_ids": [...]}]
    Unanswerable queries (empty gold) are skipped by definition."""
    scored = [
        recall_at_k(r["retrieved_chunk_ids"], r["gold_chunk_ids"], k)
        for r in results
        if r.get("gold_chunk_ids")
    ]
    return sum(scored) / len(scored) if scored else 0.0
