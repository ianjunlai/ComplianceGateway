# -*- coding: utf-8 -*-
"""Does excluding hub entities from graph expansion restore selectivity?

Two hops currently reach 99% of the corpus because a handful of entities appear
in a third to a half of all clauses and act as bridges between everything else.
This sweeps a cutoff on entity ubiquity and measures what each graph strategy
recovers, against the vector baseline it has to beat.

The filter removes hubs from the *path*, not just from the endpoint: a hub in
the middle of a two-hop walk is exactly what connects two unrelated clauses.
Seeds are exempt — a query genuinely about "personal data" should still link.

hippo_rag applies the same idea to its adjacency matrix, so PPR mass cannot
flow through a hub either.
"""

import sys
from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[2]      # inference-service/
_REPO = _SERVICE.parent
sys.path.insert(0, str(_SERVICE))
import json

from pathlib import Path

import numpy as np
from scipy import sparse

import config
from pipeline.embeddings import embed_one
from pipeline.entity_linking import link_entities
from pipeline.graph import get_driver
from pipeline.strategies.hippo_rag import HippoRagStrategy

HERE = Path(__file__).parent
cache = json.loads((HERE / "ner_seed_cache.json").read_text(encoding="utf-8"))
items = {q["query_id"]: q for q in json.loads(
    (_REPO / "dataset" / "qa_dataset.json").read_text(encoding="utf-8"))}

CUTOFFS = [None, 0.20, 0.10, 0.05]   # exclude entities in > this share of clauses

_HYBRID = """
UNWIND $seed_ids AS seed_id
MATCH (seed:Entity {node_id: seed_id})
CALL (seed) {
    MATCH p = (seed)-[:RELATES*1..%(hops)d]-(nbr:Entity)
    WHERE none(n IN nodes(p)[1..] WHERE n.node_id IN $hubs)
    RETURN collect(DISTINCT nbr) AS nbrs
}
WITH seed, nbrs
UNWIND ([seed] + nbrs) AS ent
MATCH (ent)-[:MENTIONED_IN]->(c:Chunk)
WITH c, count(DISTINCT ent) AS n
RETURN c.chunk_id AS chunk_id,
       vector.similarity.cosine(c.embedding, $qvec) AS score
ORDER BY score DESC
LIMIT $limit
"""

_LIGHT_NBR = """
UNWIND $node_ids AS nid
MATCH (seed:Entity {node_id: nid})-[:RELATES]-(nbr:Entity)
WHERE NOT nbr.node_id IN $hubs
MATCH (nbr)-[:MENTIONED_IN]->(c:Chunk)
WITH nbr, c, count(DISTINCT seed) AS support
RETURN c.chunk_id AS chunk_id, support
ORDER BY support DESC, c.chunk_id
LIMIT $limit
"""

_LIGHT_ENT = """
UNWIND $vectors AS vec
CALL db.index.vector.queryNodes($index, $limit, vec) YIELD node, score
WITH node, max(score) AS score
MATCH (node)-[:MENTIONED_IN]->(c:Chunk)
RETURN node.node_id AS node_id, score, c.chunk_id AS chunk_id,
       COUNT { MATCH (node)-[:RELATES]-(nbr:Entity)-[:MENTIONED_IN]->(c)
               RETURN DISTINCT nbr } AS relation_count
ORDER BY score DESC, relation_count DESC
"""

def recall(retrieved, gold, k=10):
    return len(set(retrieved[:k]) & set(gold)) / len(gold)

def main() -> None:
    hippo = HippoRagStrategy()
    from pipeline.embeddings import embed

    with get_driver().session() as s:
        n_chunks = s.run("MATCH (c:Chunk) RETURN count(c) AS n").single()["n"]
        occ = {r["nid"]: r["n"] for r in s.run("""
            MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)
            RETURN e.node_id AS nid, count(DISTINCT c) AS n
        """)}

        rows = []
        for cutoff in CUTOFFS:
            hubs = [nid for nid, n in occ.items() if cutoff and n > cutoff * n_chunks]
            hub_set = set(hubs)
            res = {"hybrid": [], "light_rag": [], "hippo_rag": []}

            # hippo: mask hub rows/cols so PPR cannot route through them
            if hubs:
                keep = np.ones(hippo.adj.shape[0], dtype=bool)
                for nid in hubs:
                    if nid in hippo.node_index:
                        keep[hippo.node_index[nid]] = False
                mask = sparse.diags(keep.astype(float))
                adj = (mask @ hippo.adj @ mask).tocsr()
            else:
                adj = hippo.adj
            original_adj = hippo.adj
            hippo.adj = adj

            for qid, seeds in cache.items():
                gold = items[qid]["gold_chunk_ids"]
                qvec = embed_one(items[qid]["query_text"])
                seed_ids = link_entities(seeds)

                # --- hybrid ---
                ids = []
                if seed_ids:
                    ids = [r["chunk_id"] for r in s.run(
                        _HYBRID % {"hops": config.GRAPH_HOPS},
                        seed_ids=seed_ids, hubs=hubs, qvec=qvec, limit=config.RETRIEVAL_K)]
                res["hybrid"].append(recall(ids, gold))

                # --- light_rag (low level + filtered expansion) ---
                keys = {}
                mention_vecs = embed(seeds) if seeds else [qvec]
                for r in s.run(_LIGHT_ENT, vectors=mention_vecs,
                               index=config.INDEX_ENTITIES, limit=3):
                    k = (-(2 * r["score"] - 1), -r["relation_count"])
                    if k < keys.get(r["chunk_id"], (1.0, 1.0)):
                        keys[r["chunk_id"]] = k
                if seed_ids:
                    decay = (-min(keys.values(), default=(-1.0, 0.0))[0]
                             * config.LIGHTRAG_NEIGHBOUR_DECAY)
                    for r in s.run(_LIGHT_NBR, node_ids=sorted(set(seed_ids)),
                                   hubs=hubs, limit=config.RETRIEVAL_K * 4):
                        k = (-decay, -r["support"])
                        if k < keys.get(r["chunk_id"], (1.0, 1.0)):
                            keys[r["chunk_id"]] = k
                ranked = [c for c, _ in sorted(keys.items(), key=lambda kv: kv[1])]
                res["light_rag"].append(recall(ranked, gold))

                # --- hippo_rag ---
                hids = [n for n in seed_ids if n in hippo.node_index]
                if hids:
                    ns = hippo._personalized_pagerank(hids)
                    ps = hippo.passage_matrix.T @ ns
                    top = np.argsort(-ps)[:config.RETRIEVAL_K]
                    hp = [hippo.index_chunk[int(j)] for j in top if ps[j] > 0]
                else:
                    hp = []
                res["hippo_rag"].append(recall(hp, gold))

            hippo.adj = original_adj
            label = "no filter" if cutoff is None else f">{cutoff:.0%} of clauses"
            rows.append((label, len(hub_set),
                         *(sum(v) / len(v) for v in (res["hybrid"], res["light_rag"], res["hippo_rag"]))))

    print(f"Recall@{config.RETRIEVAL_K} on {len(cache)} queries "
          f"(vector_rag baseline = 0.525)\n")
    print(f"{'hub cutoff':<20}{'hubs excl.':>11}{'hybrid':>9}{'light_rag':>11}{'hippo_rag':>11}")
    print("-" * 62)
    for label, n_hub, h, l, p in rows:
        print(f"{label:<20}{n_hub:>11}{h:>9.3f}{l:>11.3f}{p:>11.3f}")

if __name__ == "__main__":
    main()
