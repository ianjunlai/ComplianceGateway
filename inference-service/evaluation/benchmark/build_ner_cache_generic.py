# -*- coding: utf-8 -*-
"""Cache query seed entities for any QA set, under either prompt profile."""
import argparse
import json
import sys
from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SERVICE))

import config                                    # noqa: E402
from pipeline.ner import extract_seed_entities   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cache_path = Path(args.cache)
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    queries = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if args.limit:
        queries = queries[:args.limit]

    todo = [q for q in queries if q["query_id"] not in cache]
    print(f"{len(queries)} queries, {len(queries) - len(todo)} cached, {len(todo)} to extract")
    print(f"   model {config.SLM_MODEL}, prompt profile {config.EXTRACTION_PROFILE}")

    empty = 0
    for i, q in enumerate(todo, 1):
        seeds = extract_seed_entities(q["query_text"])
        cache[q["query_id"]] = seeds
        empty += not seeds
        if i % 10 == 0 or i == len(todo):
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
            print(f"   {i}/{len(todo)}", flush=True)

    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    sizes = [len(v) for v in cache.values()]
    print(f"\nwrote {cache_path}  ({len(cache)} queries)")
    print(f"   seeds per query: min {min(sizes)}, "
          f"median {sorted(sizes)[len(sizes) // 2]}, max {max(sizes)}")
    if empty:
        # A query with no seed is one no graph strategy can start from, so it belongs in the write-up rather than in a silently depressed average.
        print(f"   WARNING: {empty} query/queries produced no seeds at all")


if __name__ == "__main__":
    main()
