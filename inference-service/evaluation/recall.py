"""Recall@K over chunk-id sets."""


def recall_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    if not gold_ids:
        raise ValueError("gold_ids must be non-empty (unanswerable queries are excluded from Recall@K)")
    top_k = set(retrieved_ids[:k])
    return len(top_k & set(gold_ids)) / len(gold_ids)


def recall_values(results: list[dict], k: int) -> list[float]:
    """Per-query Recall@K, for aggregation with a confidence interval."""
    return [
        recall_at_k(r["retrieved_chunk_ids"], r["gold_chunk_ids"], k)
        for r in results
        if r.get("gold_chunk_ids")
    ]


def mean_recall(results: list[dict], k: int) -> float | None:
    """Mean Recall@K; None when no query in the set has gold chunks."""
    scored = recall_values(results, k)
    return sum(scored) / len(scored) if scored else None
