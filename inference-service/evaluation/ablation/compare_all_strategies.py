# -*- coding: utf-8 -*-
"""Retrieval quality of all five strategies, as actually implemented.

Run before committing to the full evaluation: a strategy that cannot retrieve
its gold clauses will not produce a meaningful decision-accuracy number either,
and finding that out after 40 hours of local inference is expensive.

Only the retrieval step runs — no generation. NER seeds come from the cache
built by compare_hybrid_variants.py, so every strategy is scored on the same
seeds and the run costs no SLM time.
"""

import sys
from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[2]      # inference-service/
_REPO = _SERVICE.parent
sys.path.insert(0, str(_SERVICE))
import json
import random

from collections import defaultdict
from pathlib import Path

import config
from pipeline.strategies import build_strategy

HERE = Path(__file__).parent
NER_CACHE = HERE / "ner_seed_cache.json"
DATASET = (_REPO / "dataset" / "qa_dataset.json")
SEED = 42

STRATEGIES = ["vector_rag", "hybrid", "light_rag", "hippo_rag"]  # zero_shot retrieves nothing

def recall_at_k(retrieved, gold, k):
    return len(set(retrieved[:k]) & set(gold)) / len(gold)

def main() -> None:
    cache = json.loads(NER_CACHE.read_text(encoding="utf-8"))
    items = [q for q in json.loads(DATASET.read_text(encoding="utf-8")) if q["gold_chunk_ids"]]
    random.Random(SEED).shuffle(items)
    items = [q for q in items if q["query_id"] in cache]
    print(f"{len(items)} queries (gold-bearing, NER seeds cached)\n")

    hop_types = sorted({q["hop_type"] for q in items})
    r10 = {s: defaultdict(list) for s in STRATEGIES}
    r5 = {s: [] for s in STRATEGIES}
    empties = {s: 0 for s in STRATEGIES}

    for name in STRATEGIES:
        strategy = build_strategy(name)
        for q in items:
            ctx = strategy.retrieve(q["query_text"], cache[q["query_id"]], config.RETRIEVAL_K)
            ids = ctx.chunk_ids
            if not ids:
                empties[name] += 1
            r10[name][q["hop_type"]].append(recall_at_k(ids, q["gold_chunk_ids"], 10))
            r5[name].append(recall_at_k(ids, q["gold_chunk_ids"], 5))
        print(f"  {name} done", flush=True)

    header = f"\n{'strategy':<12}{'R@5':>8}{'R@10':>8}" + "".join(f"{h:>10}" for h in hop_types) + f"{'empty':>8}"
    print(header)
    print("-" * len(header.strip()))
    for name in STRATEGIES:
        allv = [v for h in hop_types for v in r10[name][h]]
        row = f"{name:<12}{sum(r5[name])/len(r5[name]):>8.3f}{sum(allv)/len(allv):>8.3f}"
        for h in hop_types:
            vals = r10[name][h]
            row += f"{sum(vals)/len(vals):>10.3f}" if vals else f"{'-':>10}"
        row += f"{empties[name]:>8}"
        print(row)
    print(f"\n(R@10 broken down by hop_type; 'empty' = queries that retrieved nothing at all)")

if __name__ == "__main__":
    main()
