# -*- coding: utf-8 -*-
"""Can PPR physically reach the second hop?

HippoRAG's ablation (paper Table 5) puts "query nodes only, no PPR" at R@5 61.4
on 2Wiki against 89.1 for the full method: almost thirty points of its
performance come from propagation. Our implementation scores 64.6 — barely
above the no-propagation baseline — which says the propagation is not
happening, whatever the ranking does afterwards.

Propagation needs a connected graph. The paper adds synonymy edges between
entities whose embeddings exceed tau=0.8, and on 2Wiki those outnumber the
extracted relation edges by more than ten to one; they are what stitches
separately-extracted mentions of the same thing into one component. This
project deduplicates entities at 0.90 instead, which merges the obvious cases
and leaves the rest as islands.

So this measures the thing that decides it: whether the entities of one gold
passage can reach the entities of the other at all.

    python -m evaluation.benchmark.audit_graph_connectivity \
        --dataset ../dataset/benchmark/2wiki_qa.json
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SERVICE))

import numpy as np                                   # noqa: E402
from scipy import sparse                             # noqa: E402

import config                                        # noqa: E402
from pipeline.graph import get_driver                # noqa: E402

_PAIR_HOPS = """
MATCH (a:Chunk {chunk_id: $a})<-[:MENTIONED_IN]-(ea:Entity)
MATCH (b:Chunk {chunk_id: $b})<-[:MENTIONED_IN]-(eb:Entity)
WITH collect(DISTINCT ea) AS eas, collect(DISTINCT eb) AS ebs
UNWIND eas AS ea
UNWIND ebs AS eb
WITH ea, eb WHERE ea <> eb
MATCH p = shortestPath((ea)-[:RELATES*1..6]-(eb))
RETURN min(length(p)) AS hops
"""

_SHARED = """
MATCH (a:Chunk {chunk_id: $a})<-[:MENTIONED_IN]-(e:Entity)-[:MENTIONED_IN]->(b:Chunk {chunk_id: $b})
RETURN count(DISTINCT e) AS n
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    art = Path(config.ARTIFACTS_DIR)
    adj = sparse.load_npz(art / "hippo_adjacency.npz")
    n_comp, labels = sparse.csgraph.connected_components(adj, directed=False)
    sizes = Counter(labels.tolist())
    largest = max(sizes.values())
    print(f"entity graph: {adj.shape[0]} nodes, {adj.nnz // 2} undirected edges, "
          f"mean degree {adj.nnz / adj.shape[0]:.2f}")
    print(f"connected components: {n_comp}")
    print(f"  largest holds {largest} nodes ({largest / adj.shape[0]:.0%} of the graph)")
    print(f"  singletons: {sum(1 for v in sizes.values() if v == 1)}")
    print(f"  median component size: {int(np.median(list(sizes.values())))}")

    queries = [q for q in json.loads(Path(args.dataset).read_text(encoding="utf-8"))
               if len(q.get("gold_chunk_ids", [])) > 1]
    if args.limit:
        queries = queries[:args.limit]

    # Only pairs matter: a multi-hop question is answered by connecting one gold
    # passage to another, and if the graph cannot join them no traversal will.
    buckets = Counter()
    per_type = {}
    with get_driver().session() as s:
        for q in queries:
            gold = q["gold_chunk_ids"]
            for i in range(len(gold)):
                for j in range(i + 1, len(gold)):
                    a, b = gold[i], gold[j]
                    shared = s.run(_SHARED, a=a, b=b).single()["n"]
                    if shared:
                        bucket = "shares an entity"
                    else:
                        rec = s.run(_PAIR_HOPS, a=a, b=b).single()
                        hops = rec["hops"] if rec else None
                        if hops is None:
                            bucket = "no path within 6 hops"
                        elif hops <= 2:
                            bucket = "2 hops or fewer"
                        elif hops <= 4:
                            bucket = "3-4 hops"
                        else:
                            bucket = "5-6 hops"
                    buckets[bucket] += 1
                    per_type.setdefault(q["hop_type"], Counter())[bucket] += 1

    total = sum(buckets.values())
    print(f"\n{total} gold pairs from {len(queries)} multi-gold questions")
    print("how one gold passage connects to the other:")
    order = ["shares an entity", "2 hops or fewer", "3-4 hops", "5-6 hops",
             "no path within 6 hops"]
    for k in order:
        if buckets[k]:
            print(f"   {k:<24}{buckets[k]:>5}  ({buckets[k] / total:.0%})")

    print("\nby hop_type:")
    for ht in sorted(per_type):
        c = per_type[ht]
        tot = sum(c.values())
        reachable = tot - c["no path within 6 hops"]
        print(f"   {ht:<20} {reachable}/{tot} connected ({reachable / tot:.0%})")

    print(f"\nGRAPH_HOPS is {config.GRAPH_HOPS}: anything beyond that is invisible to "
          f"hybrid,\nand anything unreachable is invisible to PPR no matter how many "
          f"iterations it runs.")


if __name__ == "__main__":
    main()
