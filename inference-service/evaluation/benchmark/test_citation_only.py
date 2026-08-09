# -*- coding: utf-8 -*-
"""Traverse only explicit legal citations, instead of extracted entity relations.

The entity graph admits 99.3% of the corpus within two hops, so it selects
nothing and whatever ranks the candidates afterwards does all the work. The
cause is what RELATES means: two entities co-occurring in a clause with an
LLM-asserted relation between them. "Personal data" relates to almost
everything in a data-protection statute, so the hub connects the corpus to
itself.

A citation is a different kind of edge. "This provision gives effect to Article
21" is a claim the drafter made, not one an extractor inferred, and it is rare
by construction. Traversing only those should admit a small, targeted set.

Design, and why it is not the same as the earlier hybrid run:
  * entry by vector search over chunks, because a citation graph has no entity
    layer to link query mentions into
  * expand one hop along IMPLEMENTS, in both directions
  * rank the union by query similarity, so a cited provision competes rather
    than jumps the queue

    python -m evaluation.benchmark.test_citation_only \
        --dataset ../dataset/crosstier_qa.json
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SERVICE))

import config                                              # noqa: E402
from evaluation.stats import bootstrap_ci                  # noqa: E402
from pipeline.embeddings import embed_one                  # noqa: E402
from pipeline.graph import get_driver                      # noqa: E402

# Entry points: nearest chunks by embedding.
_ENTRY = """
CALL db.index.vector.queryNodes($index, $k, $qvec) YIELD node, score
RETURN node.chunk_id AS chunk_id, score
"""

# One hop along citations from those entry points, then rank everything
# admitted by similarity to the question.
_EXPAND = """
UNWIND $entry AS cid
MATCH (c:Chunk {chunk_id: cid})
OPTIONAL MATCH (c)-[:IMPLEMENTS]-(linked:Chunk)
WITH collect(DISTINCT c) + collect(DISTINCT linked) AS all_nodes
UNWIND all_nodes AS n
WITH DISTINCT n WHERE n IS NOT NULL
RETURN n.chunk_id AS chunk_id,
       vector.similarity.cosine(n.embedding, $qvec) AS score
ORDER BY score DESC
"""


def recall(retrieved, gold, k):
    return len(set(retrieved[:k]) & set(gold)) / len(gold)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--entry-k", default="3,5,10",
                    help="comma-separated entry-point counts to sweep")
    args = ap.parse_args()

    ks = [int(x) for x in args.entry_k.split(",")]
    queries = [q for q in json.loads(Path(args.dataset).read_text(encoding="utf-8"))
               if q.get("gold_chunk_ids")]
    driver = get_driver()

    # Embed once per query, not once per configuration: the sweep would
    # otherwise spend most of its time re-encoding the same 38 questions.
    with driver.session() as s:
        total = s.run("MATCH (c:Chunk) RETURN count(c) AS n").single()["n"]
        cited = s.run("MATCH (c:Chunk)-[:IMPLEMENTS]-() "
                      "RETURN count(DISTINCT c) AS n").single()["n"]
        qvecs = [embed_one(q["query_text"]) for q in queries]

        baseline = [[r["chunk_id"] for r in
                     s.run(_ENTRY, index=config.INDEX_CHUNKS, k=10, qvec=v)]
                    for v in qvecs]

        variants, stats = {}, {}
        for k in ks:
            got, sizes, grew = [], [], 0
            for q, v in zip(queries, qvecs):
                entry = [r["chunk_id"] for r in
                         s.run(_ENTRY, index=config.INDEX_CHUNKS, k=k, qvec=v)]
                ids = [r["chunk_id"] for r in s.run(_EXPAND, entry=entry, qvec=v)]
                got.append(ids)
                sizes.append(len(ids))
                grew += len(ids) > len(entry)
            variants[k] = got
            stats[k] = (statistics.median(sizes), grew)

    print(f"{len(queries)} queries, corpus {total} chunks, "
          f"{cited} of them ({cited / total:.0%}) touched by a citation\n")

    def scores(retrieved, k):
        return [recall(g, q["gold_chunk_ids"], k) for g, q in zip(retrieved, queries)]

    head = (f"{'method':<30}{'R@2':>8}{'R@5':>8}{'R@10':>8}"
            f"{'admitted':>11}{'expanded':>10}")
    print(head); print("-" * len(head))
    b2, b5, b10 = (scores(baseline, n) for n in (2, 5, 10))
    print(f"{'vector_rag (baseline)':<30}{statistics.fmean(b2):>8.3f}"
          f"{statistics.fmean(b5):>8.3f}{statistics.fmean(b10):>8.3f}"
          f"{10:>11}{'-':>10}")
    for k in ks:
        r = variants[k]
        med, grew = stats[k]
        print(f"{'citation-only, entry ' + str(k):<30}"
              f"{statistics.fmean(scores(r, 2)):>8.3f}"
              f"{statistics.fmean(scores(r, 5)):>8.3f}"
              f"{statistics.fmean(scores(r, 10)):>8.3f}"
              f"{med:>11.0f}{f'{grew}/{len(queries)}':>10}")

    # Paired: the same questions under both methods, so the interval is over
    # per-question differences and is not inflated by the variation in question
    # difficulty that both methods share.
    print(f"\nR@10 difference against the baseline, 95% paired bootstrap "
          f"({len(queries)} queries)")
    for k in ks:
        deltas = [a - b for a, b in zip(scores(variants[k], 10), b10)]
        ci = bootstrap_ci(deltas)
        sig = "significant" if (ci["ci_low"] > 0 or ci["ci_high"] < 0) else "not significant"
        print(f"   entry {k:<3}{ci['mean']:+8.3f}   "
              f"[{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]   {sig}")

    print("\nThe entity traversal admits 99.3% of this corpus. Selectivity is the point:\n"
          "an admitted set the size of the corpus cannot rank anything.")


if __name__ == "__main__":
    main()
