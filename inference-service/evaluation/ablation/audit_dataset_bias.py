# -*- coding: utf-8 -*-
"""Is the graph's failure a property of the corpus, or of how the questions were made?

Every question was written by an LLM that had the gold clauses in front of it,
so the question inherits their wording. That hands dense retrieval a paraphrase-
matching task and can leave nothing for graph traversal to contribute -- in
which case the negative result would be an artefact of the dataset rather than
a finding about the corpus.

The decisive check is on multi-hop items: if each gold clause is independently
findable by vector search, the question never required a second hop, and no
graph method could have shown an advantage on it.
"""

import sys
from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[2]      # inference-service/
_REPO = _SERVICE.parent
sys.path.insert(0, str(_SERVICE))
import json

from pathlib import Path

import numpy as np

import config
from pipeline.embeddings import embed_one
from pipeline.graph import get_driver

DATASET = (_REPO / "dataset" / "qa_dataset.json")
items = [q for q in json.loads(DATASET.read_text(encoding="utf-8")) if q["gold_chunk_ids"]]

_RANK = """
CALL db.index.vector.queryNodes($index, $k, $vec) YIELD node, score
RETURN node.chunk_id AS chunk_id, score
"""

with get_driver().session() as s:
    n_chunks = s.run("MATCH (c:Chunk) RETURN count(c) AS n").single()["n"]

    per_gold_rank = {"single": [], "multi": [], "trap": []}
    all_reachable = {"multi": [], "trap": []}
    top1_sim, gold_sim = [], []

    for q in items:
        qvec = embed_one(q["query_text"])
        rows = [(r["chunk_id"], r["score"]) for r in s.run(
            _RANK, index=config.INDEX_CHUNKS, k=n_chunks, vec=qvec)]
        rank = {cid: i + 1 for i, (cid, _) in enumerate(rows)}
        score = dict(rows)

        ranks = [rank[g] for g in q["gold_chunk_ids"] if g in rank]
        if not ranks:
            continue
        per_gold_rank[q["hop_type"]].extend(ranks)
        top1_sim.append(2 * rows[0][1] - 1)
        gold_sim.append(max(2 * score[g] - 1 for g in q["gold_chunk_ids"] if g in score))

        if q["hop_type"] in ("multi", "trap") and len(q["gold_chunk_ids"]) > 1:
            # Did the question actually need a hop? If every gold clause is
            # already in the vector top-10, one dense query answered it.
            all_reachable[q["hop_type"]].append(all(r <= 10 for r in ranks))

print(f"{len(items)} gold-bearing queries, corpus of {n_chunks} clauses\n")

print("rank of the gold clause under pure vector search (1 = best):")
for ht, rs in per_gold_rank.items():
    if not rs:
        continue
    rs = np.array(rs)
    print(f"   {ht:<8} n={len(rs):<4} median {np.median(rs):>5.0f}   "
          f"in top-10: {(rs <= 10).mean():>5.0%}   in top-50: {(rs <= 50).mean():>5.0%}")

print("\nmulti-gold questions where EVERY gold clause is already in the vector top-10")
print("(i.e. one dense query sufficed and no second hop was required):")
for ht, flags in all_reachable.items():
    if flags:
        print(f"   {ht:<8} {sum(flags)}/{len(flags)} = {sum(flags)/len(flags):.0%}")

print(f"\nsimilarity of the query to its gold clause vs to the corpus's best match:")
print(f"   query <-> gold          mean {np.mean(gold_sim):.3f}")
print(f"   query <-> top-1 overall mean {np.mean(top1_sim):.3f}")
print(f"   gold is the top-1 match for {np.mean([g >= t for g, t in zip(gold_sim, top1_sim)]):.0%} of queries")
