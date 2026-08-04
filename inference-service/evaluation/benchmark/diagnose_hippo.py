# -*- coding: utf-8 -*-
"""Why does HippoRAG rank so much worse than dense retrieval here?

Three candidate explanations, and they call for different fixes, so they are
separated rather than argued about:

  1. Diffusion. If PPR spreads mass evenly over thousands of nodes, the passage
     projection is ranking noise. Measured as the participation ratio of the
     node distribution -- the effective number of nodes actually carrying mass.
  2. Reach. If the gold passage carries no activated entity at all, no scoring
     rule recovers it.
  3. Ranking. If the gold is reached and carries mass but still ranks low, the
     projection is the problem, not the walk.

The third case has a cheap test that is not a guess: HippoRAG is the only
strategy here that never looks at the query embedding, since it enters the
graph through NER seeds alone. Re-ranking its own admitted set by query
similarity is exactly the change that took hybrid from 0.138 to 0.513 and
light_rag from 0.196 to 0.567 on this corpus. If it helps here too, the deficit
is the projection; if it does not, the walk is landing in the wrong place.

    python -m evaluation.benchmark.diagnose_hippo \
        --dataset ../dataset/qa_dataset.json \
        --ner-cache evaluation/ablation/ner_seed_cache.json
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

_SERVICE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SERVICE))

import config                                          # noqa: E402
from pipeline.embeddings import embed_one              # noqa: E402
from pipeline.entity_linking import link_entities      # noqa: E402
from pipeline.graph import get_driver, index_score_to_cosine  # noqa: E402
from pipeline.strategies import build_strategy         # noqa: E402

DEEP_K = 200

_RERANK = """
UNWIND $chunk_ids AS cid
MATCH (c:Chunk {chunk_id: cid})
RETURN c.chunk_id AS chunk_id,
       vector.similarity.cosine(c.embedding, $qvec) AS score
ORDER BY score DESC
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

    hippo = build_strategy("hippo_rag")
    n_nodes = hippo.adj.shape[0]
    n_chunks = hippo.passage_matrix.shape[1]

    eff_nodes, eff_chunks = [], []
    gold_has_mass, gold_total = 0, 0
    rank_ppr, rank_reranked, missed_ppr = [], [], 0

    with get_driver().session() as s:
        for q in queries:
            seeds = cache[q["query_id"]]
            seed_ids = [n for n in link_entities(seeds) if n in hippo.node_index]
            if not seed_ids:
                continue
            node_scores = hippo._personalized_pagerank(seed_ids)
            passage_scores = hippo.passage_matrix.T @ node_scores

            # Participation ratio: (sum x)^2 / sum(x^2). For a distribution
            # concentrated on k nodes it is about k; for a uniform one it is n.
            for vec, store in ((node_scores, eff_nodes), (passage_scores, eff_chunks)):
                tot, sq = float(vec.sum()), float((vec ** 2).sum())
                store.append(tot * tot / sq if sq > 0 else 0.0)

            order = np.argsort(-passage_scores)
            pos = {hippo.index_chunk[int(j)]: r
                   for r, j in enumerate(order[:DEEP_K], 1)}
            for g in q["gold_chunk_ids"]:
                gold_total += 1
                col = hippo.index_chunk_inv.get(g) if hasattr(hippo, "index_chunk_inv") else None
                if col is None:
                    col = next((k for k, v in hippo.index_chunk.items() if v == g), None)
                if col is not None and passage_scores[col] > 0:
                    gold_has_mass += 1
                if g in pos:
                    rank_ppr.append(pos[g])
                else:
                    missed_ppr += 1

            # Re-rank HippoRAG's own admitted set by query similarity
            admitted = [hippo.index_chunk[int(j)] for j in order[:DEEP_K]
                        if passage_scores[j] > 0]
            if admitted:
                reranked = [r["chunk_id"] for r in
                            s.run(_RERANK, chunk_ids=admitted, qvec=embed_one(q["query_text"]))]
                rpos = {cid: r for r, cid in enumerate(reranked, 1)}
                for g in q["gold_chunk_ids"]:
                    if g in rpos:
                        rank_reranked.append(rpos[g])

    print(f"graph: {n_nodes} entity nodes, {n_chunks} passages")
    print(f"queries scored: {len(eff_nodes)}\n")

    print("1. DIFFUSION — effective number of items carrying mass")
    print(f"   nodes    : median {statistics.median(eff_nodes):8.1f} of {n_nodes}"
          f"   ({statistics.median(eff_nodes) / n_nodes:.1%} of the graph)")
    print(f"   passages : median {statistics.median(eff_chunks):8.1f} of {n_chunks}"
          f"   ({statistics.median(eff_chunks) / n_chunks:.1%} of the corpus)")
    print("   A passage figure approaching the corpus size means the projection is\n"
          "   ranking noise: everything scores about the same.\n")

    print("2. REACH")
    print(f"   gold passages carrying any activated entity: {gold_has_mass}/{gold_total}"
          f"  ({gold_has_mass / gold_total:.0%})\n")

    print("3. RANKING")
    if rank_ppr:
        print(f"   gold rank under PPR projection : median {statistics.median(rank_ppr):.0f}"
              f"   ({missed_ppr} outside top {DEEP_K})")
    if rank_reranked:
        print(f"   same set re-ranked by query    : median {statistics.median(rank_reranked):.0f}")
        top5_ppr = sum(1 for r in rank_ppr if r <= 5)
        top5_re = sum(1 for r in rank_reranked if r <= 5)
        print(f"   gold in top 5: {top5_ppr} under PPR -> {top5_re} after re-ranking")


if __name__ == "__main__":
    main()
