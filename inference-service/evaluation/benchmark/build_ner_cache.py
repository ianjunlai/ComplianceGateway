# -*- coding: utf-8 -*-
"""Cache query seed entities for the benchmark questions.

Seeds are where every graph strategy enters the graph, so they have to be
identical across strategies or the comparison measures seed luck. Caching them
also keeps the comparison script free of Ollama, matching how the GDPR ablation
works (`evaluation/ablation/ner_seed_cache.json`).

Run with the general profile -- the legal NER prompt asked "Who is the mother of
the director of Polish-Russian War?" returns compliance vocabulary that links to
nothing:

    EXTRACTION_PROFILE=general python -m evaluation.benchmark.build_ner_cache

Resumable: an existing cache is extended, never rebuilt, so an interrupted run
costs only what it had not yet done.
"""
import argparse
import json
import sys
from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SERVICE))

import config                                    # noqa: E402
from pipeline.ner import extract_seed_entities   # noqa: E402

HERE = Path(__file__).parent
_REPO = _SERVICE.parent
DEFAULT_DATASET = _REPO / "dataset" / "benchmark" / "2wiki_qa.json"
DEFAULT_CACHE = HERE / "ner_seed_cache_2wiki.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--limit", type=int, default=0, help="only the first N queries")
    args = ap.parse_args()

    if config.EXTRACTION_PROFILE != "general":
        raise SystemExit(
            f"EXTRACTION_PROFILE is {config.EXTRACTION_PROFILE!r}; this benchmark needs "
            f"'general'. The legal NER prompt returns compliance vocabulary for "
            f"encyclopaedic questions, and the graph strategies would look broken "
            f"for a reason that has nothing to do with their implementation.")

    cache_path = Path(args.cache)
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    queries = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if args.limit:
        queries = queries[:args.limit]

    todo = [q for q in queries if q["query_id"] not in cache]
    print(f"{len(queries)} queries, {len(queries) - len(todo)} cached, {len(todo)} to extract "
          f"(model: {config.SLM_MODEL})")

    empty = 0
    for i, q in enumerate(todo, 1):
        seeds = extract_seed_entities(q["query_text"])
        cache[q["query_id"]] = seeds
        if not seeds:
            empty += 1
        if i % 10 == 0 or i == len(todo):
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"  {i}/{len(todo)}", flush=True)

    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    sizes = [len(v) for v in cache.values()]
    print(f"\nwrote {cache_path}  ({len(cache)} queries)")
    print(f"  seeds per query: min {min(sizes)}, median {sorted(sizes)[len(sizes) // 2]}, max {max(sizes)}")
    if empty:
        # Not fatal, but every empty-seed query is one the graph strategies
        # cannot answer at all, so it belongs in the write-up rather than in a
        # silently depressed average.
        print(f"  WARNING: {empty} query/queries produced no seeds at all")


if __name__ == "__main__":
    main()
