# -*- coding: utf-8 -*-
"""Retrieval quality of every strategy over one question set: Recall@2/@5/@10 and the paired difference against vector_rag."""

import argparse
import sys
from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[2]      # inference-service/
_REPO = _SERVICE.parent
sys.path.insert(0, str(_SERVICE))
import json
import random

from collections import defaultdict
from pathlib import Path

import time

import config
from evaluation.stats import bootstrap_ci
from pipeline.attachment import attach_citations
from pipeline.jurisdiction import jurisdictions_for
from pipeline.strategies import build_strategy

HERE = Path(__file__).parent
NER_CACHE = HERE / "ner_seed_cache.json"
DATASET = (_REPO / "dataset" / "qa_dataset.json")
SEED = 42

# zero_shot retrieves nothing, so it has no place in a retrieval comparison.
DEFAULT_STRATEGIES = ["vector_rag", "hybrid", "light_rag", "hippo_rag"]
BASELINE = "vector_rag"   # the comparator every graph strategy has to beat

def recall_at_k(retrieved, gold, k):
    return len(set(retrieved[:k]) & set(gold)) / len(gold)

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=str(DATASET))
    ap.add_argument("--ner-cache", default=str(NER_CACHE))
    ap.add_argument("--limit", type=int, default=0, help="use only the first N queries")
    ap.add_argument("--paired-ci", action="store_true",
                    help="bootstrap CI for each strategy's R@5 difference against "
                         f"{BASELINE}, over the same queries")
    ap.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES),
                    help="comma-separated")
    args = ap.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    if BASELINE not in strategies:
        raise SystemExit(f"--strategies must include {BASELINE}")

    cache = json.loads(Path(args.ner_cache).read_text(encoding="utf-8"))
    items = [q for q in json.loads(Path(args.dataset).read_text(encoding="utf-8"))
             if q["gold_chunk_ids"]]
    random.Random(SEED).shuffle(items)
    items = [q for q in items if q["query_id"] in cache]
    if args.limit:
        items = items[:args.limit]
    print(f"{len(items)} queries (gold-bearing, NER seeds cached)")
    print(f"dataset: {args.dataset}\n")

    hop_types = sorted({q["hop_type"] for q in items})
    r10 = {s: defaultdict(list) for s in strategies}
    r5 = {s: [] for s in strategies}
    latency = {s: [] for s in strategies}
    ctx_size = {s: [] for s in strategies}
    empties = {s: 0 for s in strategies}

    for name in strategies:
        strategy = build_strategy(name)
        for q in items:
            # The same scope the online pipeline would apply, derived from the requesting institution.
            scope = jurisdictions_for(q.get("source_system"))
            t0 = time.perf_counter()
            ctx = strategy.retrieve(q["query_text"], cache[q["query_id"]],
                                    config.RETRIEVAL_K, allowed_jurisdictions=scope)
            latency[name].append((time.perf_counter() - t0) * 1000)
            ids = ctx.chunk_ids
            if not ids:
                empties[name] += 1
            # What the generator would actually read.
            ctx_size[name].append(
                len(attach_citations(ctx.truncated(config.GENERATION_CONTEXT_K),
                                     scope).chunks))
            r10[name][q["hop_type"]].append(recall_at_k(ids, q["gold_chunk_ids"], 10))
            r5[name].append(recall_at_k(ids, q["gold_chunk_ids"], 5))
        print(f"  {name} done", flush=True)

    mean = lambda v: sum(v) / len(v) if v else 0.0
    header = (f"\n{'strategy':<12}{'R@5':>8}{'R@10':>8}{'latency':>10}{'context':>9}"
              + "".join(f"{h:>12}" for h in hop_types) + f"{'empty':>7}")
    print(header)
    print("-" * len(header.strip()))
    for name in strategies:
        allv = [v for h in hop_types for v in r10[name][h]]
        row = (f"{name:<12}{mean(r5[name]):>8.3f}{mean(allv):>8.3f}"
               f"{mean(latency[name]):>9.0f}ms{mean(ctx_size[name]):>9.1f}")
        for h in hop_types:
            vals = r10[name][h]
            row += f"{mean(vals):>12.3f}" if vals else f"{'-':>12}"
        row += f"{empties[name]:>7}"
        print(row)
    print("\n(R@10 broken down by hop_type; 'context' = chunks the generator would")
    print(" read, attachments included; 'empty' = queries that retrieved nothing)")

    if args.paired_ci:
        print(f"\nR@5 difference against {BASELINE}, 95% bootstrap CI over {len(items)} queries")
        for name in strategies:
            if name == BASELINE:
                continue
            deltas = [a - b for a, b in zip(r5[name], r5[BASELINE])]
            ci = bootstrap_ci(deltas)
            lo, hi = ci["ci_low"], ci["ci_high"]
            verdict = "significant" if (lo > 0 or hi < 0) else "not significant"
            print(f"  {name:<12}{ci['mean']:+8.3f}   [{lo:+.3f}, {hi:+.3f}]   {verdict}")

if __name__ == "__main__":
    main()
