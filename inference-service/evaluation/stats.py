"""Statistical procedures for the evaluation runs.

- Decision metrics with an abstention-aware scoring policy (abstention != dangerous failure)
- McNemar's test for paired decision outcomes between two systems
- Bootstrap 95% CIs for Recall@K
"""
import random

from scipy import stats as scipy_stats


def decision_metrics(predictions: list[str], golds: list[str]) -> dict:
    """Abstention-aware scoring policy:
    - accuracy over {APPROVE, DENY, UNKNOWN}
    - abstention rate: pred=UNKNOWN while gold in {APPROVE, DENY}
    - FPR (red line): P(pred=APPROVE | gold=DENY)
    - hallucination resistance: accuracy on the unanswerable subset (gold=UNKNOWN)
    """
    n = len(golds)
    correct = sum(p == g for p, g in zip(predictions, golds))
    decidable = [(p, g) for p, g in zip(predictions, golds) if g in ("APPROVE", "DENY")]
    abstained = sum(1 for p, g in decidable if p == "UNKNOWN")
    deny_gold = [(p, g) for p, g in decidable if g == "DENY"]
    false_approvals = sum(1 for p, _ in deny_gold if p == "APPROVE")
    unanswerable = [(p, g) for p, g in zip(predictions, golds) if g == "UNKNOWN"]

    # A rate with an empty denominator is undefined, and is reported as null
    # rather than 0.0. This matters for the per-stratum breakdown: the
    # unanswerable stratum contains no APPROVE/DENY gold at all, and a printed
    # "fpr: 0.0" there reads as "never wrongly approved" when the quantity
    # simply does not exist for that stratum.
    return {
        "n": n,
        "accuracy": correct / n if n else None,
        "abstention_rate": abstained / len(decidable) if decidable else None,
        "fpr": false_approvals / len(deny_gold) if deny_gold else None,
        "unanswerable_accuracy": (
            sum(1 for p, _ in unanswerable if p == "UNKNOWN") / len(unanswerable)
            if unanswerable else None
        ),
    }


def mcnemar_test(correct_a: list[bool], correct_b: list[bool]) -> dict:
    """Paired comparison of two systems on the same queries (exact binomial McNemar)."""
    b = sum(1 for ca, cb in zip(correct_a, correct_b) if ca and not cb)
    c = sum(1 for ca, cb in zip(correct_a, correct_b) if not ca and cb)
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p_value": 1.0}
    p_value = float(scipy_stats.binomtest(min(b, c), n, 0.5).pvalue)
    return {"b": b, "c": c, "p_value": p_value}


def bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 42) -> dict:
    """Percentile bootstrap 95% CI of the mean."""
    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choices(values, k=len(values))) / len(values) for _ in range(n_boot)
    )
    mean = sum(values) / len(values)
    return {
        "mean": mean,
        "ci_low": means[int(0.025 * n_boot)],
        "ci_high": means[int(0.975 * n_boot)],
    }
