# -*- coding: utf-8 -*-
"""Where do the graph strategies lose — admissibility, or ranking?

Recall@k collapses two unrelated failures into one number. A gold passage the
traversal never reaches cannot be ranked at all, and no amount of scoring work
would recover it; a gold passage that is reached but placed 40th is a ranking
problem with a completely different fix. Reporting only R@5 makes the two
indistinguishable, and on the 2Wiki run the graph strategies lose most heavily
on `compositional` questions — the class that most needs a chain of hops, and
therefore the class where the distinction matters most.

For every query this measures, per hop_type:
  - whether each gold passage is reachable within GRAPH_HOPS of the linked seeds
    (the ceiling for hybrid: it can only rank what the traversal admits)
  - whether any linked seed entity is mentioned in the gold passage at all
    (a zero-hop connection, the easiest case there is)
  - where each strategy actually ranks the gold, retrieving far deeper than k=10
    so "missed" and "ranked 47th" stay distinguishable

    python -m evaluation.benchmark.diagnose_failure_modes \
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

import config                                        # noqa: E402
from pipeline.entity_linking import link_entities    # noqa: E402
from pipeline.graph import get_driver                # noqa: E402
from pipeline.strategies import build_strategy       # noqa: E402

STRATEGIES = ["vector_rag", "hybrid", "light_rag", "hippo_rag"]
DEEP_K = 200          # deep enough that a rank, not just a miss, is observable

_REACHABLE = """
UNWIND $seed_ids AS seed_id
MATCH (seed:Entity {node_id: seed_id})
CALL (seed) {
    MATCH (seed)-[:RELATES*1..%(hops)d]-(nbr:Entity)
    RETURN collect(DISTINCT nbr) AS nbrs
}
WITH seed, nbrs
UNWIND ([seed] + nbrs) AS ent
MATCH (ent)-[:MENTIONED_IN]->(c:Chunk)
RETURN collect(DISTINCT c.chunk_id) AS chunk_ids
"""

# Zero-hop: the seed entity is itself mentioned in the gold passage.
_DIRECT = """
UNWIND $seed_ids AS seed_id
MATCH (e:Entity {node_id: seed_id})-[:MENTIONED_IN]->(c:Chunk)
WHERE c.chunk_id IN $gold
RETURN collect(DISTINCT c.chunk_id) AS chunk_ids
"""


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

    strategies = {name: build_strategy(name) for name in STRATEGIES}
    driver = get_driver()

    # per hop_type -> accumulators
    reach = defaultdict(lambda: [0, 0])        # [gold reachable, gold total]
    direct = defaultdict(lambda: [0, 0])       # [gold sharing a seed entity, total]
    ranks = defaultdict(lambda: defaultdict(list))   # hop_type -> strategy -> ranks
    missed = defaultdict(lambda: defaultdict(int))   # gold not in top DEEP_K

    with driver.session() as s:
        for i, q in enumerate(queries, 1):
            ht, gold = q["hop_type"], q["gold_chunk_ids"]
            seeds = cache[q["query_id"]]
            seed_ids = link_entities(seeds)

            admitted = set()
            hit_direct = set()
            if seed_ids:
                rec = s.run(_REACHABLE % {"hops": config.GRAPH_HOPS},
                            seed_ids=seed_ids).single()
                admitted = set(rec["chunk_ids"] or [])
                rec = s.run(_DIRECT, seed_ids=seed_ids, gold=gold).single()
                hit_direct = set(rec["chunk_ids"] or [])

            for g in gold:
                reach[ht][1] += 1
                direct[ht][1] += 1
                if g in admitted:
                    reach[ht][0] += 1
                if g in hit_direct:
                    direct[ht][0] += 1

            for name, strategy in strategies.items():
                ids = strategy.retrieve(q["query_text"], seeds, DEEP_K).chunk_ids
                pos = {cid: r for r, cid in enumerate(ids, 1)}
                for g in gold:
                    if g in pos:
                        ranks[ht][name].append(pos[g])
                    else:
                        missed[ht][name] += 1
            if i % 25 == 0:
                print(f"  {i}/{len(queries)}", flush=True)

    hop_types = sorted(reach)
    print(f"\n{'hop_type':<20}{'n gold':>8}{'reachable':>11}{'shares seed':>13}")
    print("-" * 52)
    for ht in hop_types:
        r, tot = reach[ht]
        d, _ = direct[ht]
        print(f"{ht:<20}{tot:>8}{r/tot:>10.0%}{d/tot:>13.0%}")
    print("\n'reachable' = gold admitted by a %d-hop traversal, the ceiling for hybrid."
          % config.GRAPH_HOPS)
    print("'shares seed' = gold passage mentions a linked seed entity (zero hops).")

    print(f"\nmedian rank of gold within top {DEEP_K} (lower is better; "
          f"'miss' = outside top {DEEP_K})")
    header = f"{'hop_type':<20}" + "".join(f"{n:>22}" for n in STRATEGIES)
    print(header)
    print("-" * len(header))
    for ht in hop_types:
        row = f"{ht:<20}"
        for name in STRATEGIES:
            vals, m = ranks[ht][name], missed[ht][name]
            cell = f"{statistics.median(vals):.0f}" if vals else "-"
            row += f"{cell + f' ({m} miss)':>22}"
        print(row)


if __name__ == "__main__":
    main()
