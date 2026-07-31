# -*- coding: utf-8 -*-
"""Is the proximity gain on multi-hop questions real, or sampling noise?

The sweep showed graph proximity helping multi-hop and trap questions and
hurting single-hop ones -- the pattern theory predicts, but the margins are
small enough that reporting them as findings requires a test. Paired bootstrap
over the same queries, since both rankings are scored on identical items and
the pairing removes between-query variance.
"""

import sys
from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[2]      # inference-service/
_REPO = _SERVICE.parent
sys.path.insert(0, str(_SERVICE))
import json

from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse

import config
from pipeline.embeddings import embed
from pipeline.graph import get_driver

ART = Path(config.ARTIFACTS_DIR)
DATASET = (_REPO / "dataset" / "qa_dataset.json")
BETA, ANCHORS, TOP_K, N_BOOT = 0.1, 5, 10, 10000

items = [q for q in json.loads(DATASET.read_text(encoding="utf-8")) if q["gold_chunk_ids"]]
chunk_index = json.loads((ART / "hippo_chunk_index.json").read_text(encoding="utf-8"))
col_of = dict(chunk_index)
id_of = {j: cid for cid, j in col_of.items()}

P = sparse.load_npz(ART / "hippo_passage_matrix.npz")
B = (P > 0).astype(float)
occ = np.asarray(B.sum(axis=1)).ravel()
S = (B.T @ sparse.diags(1.0 / np.maximum(occ, 1)) @ B).toarray()
np.fill_diagonal(S, 0.0)

print(f"embedding {len(items)} queries...", flush=True)
qvecs = embed([q["query_text"] for q in items])

_RANK = """
CALL db.index.vector.queryNodes($index, $k, $vec) YIELD node, score
RETURN node.chunk_id AS chunk_id, score
"""
base, prox_scores = defaultdict(list), defaultdict(list)
with get_driver().session() as s:
    n_chunks = s.run("MATCH (c:Chunk) RETURN count(c) AS n").single()["n"]
    for q, v in zip(items, qvecs):
        c = np.zeros(len(col_of))
        for r in s.run(_RANK, index=config.INDEX_CHUNKS, k=n_chunks, vec=v):
            if r["chunk_id"] in col_of:
                c[col_of[r["chunk_id"]]] = 2 * r["score"] - 1
        gold = set(q["gold_chunk_ids"])

        def rec(scores):
            top = [id_of[int(j)] for j in np.argsort(-scores)[:TOP_K]]
            return len(set(top) & gold) / len(gold)

        anchors = np.argsort(-c)[:ANCHORS]
        p = S[:, anchors].sum(axis=1)
        if p.max() > 0:
            p = p / p.max()
        base[q["hop_type"]].append(rec(c))
        prox_scores[q["hop_type"]].append(rec(c + BETA * p))

rng = np.random.default_rng(42)
print(f"\nspecificity proximity, beta={BETA}, anchors={ANCHORS}, {N_BOOT} bootstrap resamples")
print(f"{'hop_type':<10}{'vector':>9}{'proximity':>11}{'diff':>9}{'95% CI of diff':>22}{'':>4}")
print("-" * 66)
for ht in ["single", "multi", "trap", "ALL"]:
    if ht == "ALL":
        a = np.array([v for k in base for v in base[k]])
        b = np.array([v for k in prox_scores for v in prox_scores[k]])
    else:
        a, b = np.array(base[ht]), np.array(prox_scores[ht])
    d = b - a
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    sig = "significant" if lo > 0 or hi < 0 else "not significant"
    print(f"{ht:<10}{a.mean():>9.3f}{b.mean():>11.3f}{d.mean():>+9.3f}"
          f"   [{lo:>+6.3f}, {hi:>+6.3f}]  {sig}")
print(f"\nn per stratum: " + ", ".join(f"{k}={len(v)}" for k, v in base.items()))
