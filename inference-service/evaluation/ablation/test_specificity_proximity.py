# -*- coding: utf-8 -*-
"""Can specificity-weighted graph proximity rank what vector search alone cannot?

86% of the gold clauses vector search misses share an entity with a gold clause
it found, so the connection needed is present in the graph -- what has been
missing is a way to rank on it. Sharing "personal data" is worthless because
half the corpus shares it; sharing "article 42(1)" is decisive. Weighting each
shared entity by 1/(clauses mentioning it) is the difference between the two.

    score(c) = cos(query, c) + beta * proximity(c, anchors)
    proximity(c, A) = sum over anchors a, over entities e in both c and a, of 1/occ(e)

Anchors are the clauses vector search is already confident about, so the graph
is used to reach outward from confirmed relevance rather than to replace it.

Two controls decide whether the weighting is what matters:
  * beta = 0 reduces exactly to vector_rag
  * an unweighted variant keeps the proximity and drops the specificity
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

BETAS = [0.0, 0.1, 0.25, 0.5, 1.0]
ANCHOR_KS = [3, 5]
TOP_K = 10

def build_proximity(weighted: bool) -> np.ndarray:
    """Clause-by-clause proximity via shared entities."""
    P = sparse.load_npz(ART / "hippo_passage_matrix.npz")     # |entities| x |clauses|
    B = (P > 0).astype(float)
    occ = np.asarray(B.sum(axis=1)).ravel()                   # clauses per entity
    w = 1.0 / np.maximum(occ, 1) if weighted else np.ones_like(occ)
    S = (B.T @ sparse.diags(w) @ B).toarray()
    np.fill_diagonal(S, 0.0)      # a clause is not its own neighbour
    return S

def main() -> None:
    chunk_index = json.loads((ART / "hippo_chunk_index.json").read_text(encoding="utf-8"))
    col_of = {cid: j for cid, j in chunk_index.items()}
    id_of = {j: cid for cid, j in col_of.items()}

    items = [q for q in json.loads(DATASET.read_text(encoding="utf-8")) if q["gold_chunk_ids"]]
    print(f"{len(items)} gold-bearing queries; embedding them...", flush=True)
    qvecs = embed([q["query_text"] for q in items])

    _RANK = """
    CALL db.index.vector.queryNodes($index, $k, $vec) YIELD node, score
    RETURN node.chunk_id AS chunk_id, score
    """
    with get_driver().session() as s:
        n_chunks = s.run("MATCH (c:Chunk) RETURN count(c) AS n").single()["n"]
        cos = []
        for v in qvecs:
            row = np.zeros(len(col_of))
            for r in s.run(_RANK, index=config.INDEX_CHUNKS, k=n_chunks, vec=v):
                if r["chunk_id"] in col_of:
                    row[col_of[r["chunk_id"]]] = 2 * r["score"] - 1   # index score -> cosine
            cos.append(row)
    cos = np.vstack(cos)
    print("vector scores collected\n", flush=True)

    results = {}
    for weighted in (True, False):
        S = build_proximity(weighted)
        for k_anchor in ANCHOR_KS:
            for beta in BETAS:
                if beta == 0.0 and (not weighted or k_anchor != ANCHOR_KS[0]):
                    continue      # beta=0 is the same baseline in every arm
                per_hop = defaultdict(list)
                for i, q in enumerate(items):
                    c = cos[i]
                    anchors = np.argsort(-c)[:k_anchor]
                    prox = S[:, anchors].sum(axis=1)
                    if prox.max() > 0:
                        prox = prox / prox.max()
                    final = c + beta * prox
                    top = [id_of[int(j)] for j in np.argsort(-final)[:TOP_K]]
                    gold = set(q["gold_chunk_ids"])
                    per_hop[q["hop_type"]].append(len(set(top) & gold) / len(gold))
                label = ("vector_rag (beta=0)" if beta == 0.0
                         else f"{'specificity' if weighted else 'unweighted '} "
                              f"beta={beta:<4} anchors={k_anchor}")
                results[label] = per_hop

    hops = ["single", "multi", "trap"]
    print(f"{'ranking':<34}" + "".join(f"{h:>9}" for h in hops) + f"{'overall':>10}")
    print("-" * 71)
    for label, per_hop in results.items():
        allv = [v for h in hops for v in per_hop[h]]
        row = f"{label:<34}"
        for h in hops:
            row += f"{np.mean(per_hop[h]):>9.3f}" if per_hop[h] else f"{'-':>9}"
        print(row + f"{np.mean(allv):>10.3f}")
    print(f"\nRecall@{TOP_K}, {len(items)} queries")

if __name__ == "__main__":
    main()
