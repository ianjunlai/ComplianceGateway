# -*- coding: utf-8 -*-
"""Does the graph rescue the evidence dense retrieval misses?

That is the whole value proposition of graph retrieval, and averages hide it.
Most gold passages in a multi-hop question are easy — they name an entity the
question names, and any embedding finds them. The interesting one is the second
hop: the passage reachable only by following a relation, which the question does
not mention directly. If graph retrieval earns its indexing cost, that is where
it does so.

So this conditions on difficulty instead of averaging over it. Gold passages are
split by where dense retrieval put them, and the graph strategies are scored on
the hard half alone: passages `vector_rag` ranked outside its top 5.

A graph method that matches `vector_rag` overall while rescuing nothing here is
not a competitive retriever with a small deficit — it is re-deriving what dense
retrieval already had, and paying an LLM extraction pass for the privilege.

    python -m evaluation.benchmark.diagnose_rescue \
        --dataset ../dataset/benchmark/2wiki_qa.json \
        --ner-cache evaluation/benchmark/ner_seed_cache_2wiki.json
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SERVICE))

from pipeline.strategies import build_strategy       # noqa: E402

STRATEGIES = ["vector_rag", "hybrid", "light_rag", "hippo_rag"]
BASELINE = "vector_rag"
DEEP_K = 200
EASY_CUTOFF = 5      # what "dense retrieval already found it" means


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ner-cache", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cache = json.loads(Path(args.ner_cache).read_text(encoding="utf-8"))
    queries = [q for q in json.loads(Path(args.dataset).read_text(encoding="utf-8"))
               if q["query_id"] in cache and q.get("gold_chunk_ids")]
    if args.limit:
        queries = queries[:args.limit]

    # rank[strategy][(query_id, gold)] = rank or None
    rank: dict[str, dict] = {name: {} for name in STRATEGIES}
    for name in STRATEGIES:
        strategy = build_strategy(name)
        for q in queries:
            ids = strategy.retrieve(q["query_text"], cache[q["query_id"]], DEEP_K).chunk_ids
            pos = {cid: r for r, cid in enumerate(ids, 1)}
            for g in q["gold_chunk_ids"]:
                rank[name][(q["query_id"], g)] = pos.get(g)
        print(f"  {name} done", flush=True)

    keys = list(rank[BASELINE])
    hard = [k for k in keys if (rank[BASELINE][k] or 10 ** 6) > EASY_CUTOFF]
    easy = [k for k in keys if k not in set(hard)]
    hop_of = {q["query_id"]: q["hop_type"] for q in queries}

    print(f"\n{len(keys)} gold passages: {len(easy)} found by {BASELINE} in its top "
          f"{EASY_CUTOFF}, {len(hard)} not")

    def summarise(label, subset):
        print(f"\n{label}  (n={len(subset)})")
        print(f"  {'strategy':<12}{'median rank':>13}{'in top 5':>11}{'in top 10':>11}{'not in top ' + str(DEEP_K):>17}")
        for name in STRATEGIES:
            vals = [rank[name][k] for k in subset]
            found = [v for v in vals if v is not None]
            top5 = sum(1 for v in found if v <= 5)
            top10 = sum(1 for v in found if v <= 10)
            med = f"{statistics.median(found):.0f}" if found else "-"
            print(f"  {name:<12}{med:>13}{top5 / len(subset):>10.0%}{top10 / len(subset):>11.0%}"
                  f"{len(vals) - len(found):>17}")

    summarise(f"HARD — {BASELINE} ranked these outside its top {EASY_CUTOFF}", hard)
    summarise(f"EASY — {BASELINE} already had these in its top {EASY_CUTOFF}", easy)

    # Rescue: hard for the baseline, top-5 for the graph strategy.
    print(f"\nrescues (hard for {BASELINE}, in the graph strategy's top 5), by hop_type")
    by_type = defaultdict(list)
    for k in hard:
        by_type[hop_of[k[0]]].append(k)
    header = f"  {'hop_type':<20}{'hard':>6}" + "".join(f"{n:>12}" for n in STRATEGIES if n != BASELINE)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for ht in sorted(by_type):
        subset = by_type[ht]
        row = f"  {ht:<20}{len(subset):>6}"
        for name in STRATEGIES:
            if name == BASELINE:
                continue
            n = sum(1 for k in subset if (rank[name][k] or 10 ** 6) <= 5)
            row += f"{n:>12}"
        print(row)

    total_hard = len(hard)
    print(f"\n  {'TOTAL':<20}{total_hard:>6}", end="")
    for name in STRATEGIES:
        if name == BASELINE:
            continue
        n = sum(1 for k in hard if (rank[name][k] or 10 ** 6) <= 5)
        print(f"{n:>12}", end="")
    print(f"\n\nA graph strategy that rescues few of these is not adding retrieval "
          f"\nreach — it is re-ranking what dense retrieval already found.")


if __name__ == "__main__":
    main()
