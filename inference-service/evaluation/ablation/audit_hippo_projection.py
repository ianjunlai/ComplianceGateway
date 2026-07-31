# -*- coding: utf-8 -*-
"""Where does HippoRAG lose the gold passage after activating the right nodes?

Node activation was shown to work: PPR surfaces specific, on-topic entities and
resists the corpus hubs. Recall is nonetheless low, which puts the loss in the
projection step, where passage score = sum over entities of (mention count x
node score). That sum is unnormalized, so a passage accumulates score simply by
mentioning many entities -- and legal clauses vary in length by an order of
magnitude, unlike the uniform Wikipedia passages the method was published on.

Reports the gold passage's rank under the published projection and under two
normalizations, and the correlation between a passage's score and its size.
"""

import sys
from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[2]      # inference-service/
_REPO = _SERVICE.parent
sys.path.insert(0, str(_SERVICE))
import json

from pathlib import Path

import numpy as np

from pipeline.entity_linking import link_entities
from pipeline.strategies.hippo_rag import HippoRagStrategy

HERE = Path(__file__).parent
cache = json.loads((HERE / "ner_seed_cache.json").read_text(encoding="utf-8"))
items = {q["query_id"]: q for q in json.loads(
    (_REPO / "dataset" / "qa_dataset.json").read_text(encoding="utf-8"))}

s = HippoRagStrategy()
P = s.passage_matrix                      # |entities| x |chunks| mention counts
ents_per_chunk = np.asarray(P.sum(axis=0)).ravel()      # total mentions per chunk
distinct_per_chunk = np.asarray((P > 0).sum(axis=0)).ravel()

print(f"mentions per chunk: min {ents_per_chunk.min():.0f}, "
      f"median {np.median(ents_per_chunk):.0f}, max {ents_per_chunk.max():.0f}")
print(f"  -> the busiest clause carries {ents_per_chunk.max()/max(np.median(ents_per_chunk),1):.0f}x "
      f"the median clause's mention mass\n")

variants = {
    "published (raw sum)": lambda v: v,
    "/ mentions": lambda v: v / np.maximum(ents_per_chunk, 1),
    "/ sqrt(mentions)": lambda v: v / np.sqrt(np.maximum(ents_per_chunk, 1)),
}
ranks = {k: [] for k in variants}
hits10 = {k: 0 for k in variants}
n = 0

for qid, seeds in cache.items():
    gold = items[qid]["gold_chunk_ids"]
    seed_ids = [x for x in link_entities(seeds) if x in s.node_index]
    if not seed_ids:
        continue
    node_scores = s._personalized_pagerank(seed_ids)
    base = P.T @ node_scores
    gold_cols = [s.index_chunk and None]  # placeholder, resolved below
    col_of = {cid: j for cid, j in json.loads(
        (Path(str(_SERVICE / "artifacts"))
         / "hippo_chunk_index.json").read_text(encoding="utf-8")).items()}
    cols = [col_of[g] for g in gold if g in col_of]
    if not cols:
        continue
    n += 1
    for label, fn in variants.items():
        scored = fn(base)
        order = np.argsort(-scored)
        pos = {int(c): int(np.where(order == c)[0][0]) + 1 for c in cols}
        best = min(pos.values())
        ranks[label].append(best)
        if best <= 10:
            hits10[label] += 1

print(f"{'projection':<22}{'median gold rank':>18}{'gold in top-10':>16}")
print("-" * 56)
for label in variants:
    r = ranks[label]
    print(f"{label:<22}{np.median(r):>18.0f}{hits10[label] / n:>15.0%}")
print(f"\n{n} queries with a linkable seed and an indexed gold chunk "
      f"(of {len(cache)}); rank is the best-placed gold chunk out of {P.shape[1]}")
