# -*- coding: utf-8 -*-
"""Recall comparison of four ways to build the Hybrid Vector-Graph candidate set.

The current strategy reaches 99% of the corpus in two hops and then ranks by
hub popularity, so it returns the same clauses for every query. Each variant
below changes exactly one thing, so the contribution of graph *seeding*,
graph *depth*, and *ranking* can be read off separately:

  vector_rag  no graph at all — the reference line the graph has to beat
  A  current  NER seeds -> 2 hops -> rank by mention count (hub popularity)
  B  rerank   same candidates as A -> rank by cosine(query, chunk)
  C  vseed    query-vector-seeded entities -> 2 hops -> cosine ranking
  D  vseed-1  query-vector-seeded entities -> 1 hop  -> cosine ranking

NER is the only SLM call; it is computed once per query and cached, so all
variants score the identical seed set and re-runs are free.
"""

import sys
from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[2]      # inference-service/
_REPO = _SERVICE.parent
sys.path.insert(0, str(_SERVICE))
import json
import random

import time
from pathlib import Path

import config
from pipeline.embeddings import embed_one
from pipeline.entity_linking import link_entities
from pipeline.graph import get_driver, index_score_to_cosine

HERE = Path(__file__).parent
NER_CACHE = HERE / "ner_seed_cache.json"
DATASET = (_REPO / "dataset" / "qa_dataset.json")
N_QUERIES = 40
SEED = 42

# Whole ranked chunk list for the query; intersecting it with a graph candidate
# set gives "graph filters, vector ranks" without refetching any embeddings.
_ALL_CHUNKS_BY_VECTOR = """
CALL db.index.vector.queryNodes($index, $k, $vec) YIELD node, score
RETURN node.chunk_id AS chunk_id, score
"""

_ENTITIES_BY_VECTOR = """
CALL db.index.vector.queryNodes($index, $k, $vec) YIELD node, score
RETURN node.node_id AS node_id, score
"""

_TRAVERSE = """
UNWIND $seed_ids AS seed_id
MATCH (seed:Entity {node_id: seed_id})
CALL (seed) {
    MATCH (seed)-[:RELATES*1..%(hops)d]-(nbr:Entity)
    RETURN collect(DISTINCT nbr) AS nbrs
}
WITH seed, nbrs
UNWIND ([seed] + nbrs) AS ent
MATCH (ent)-[:MENTIONED_IN]->(c:Chunk)
WITH c, collect(DISTINCT ent.name) AS entities
RETURN c.chunk_id AS chunk_id, size(entities) AS n_ent
ORDER BY n_ent DESC
"""

def recall_at_k(retrieved, gold, k):
    return len(set(retrieved[:k]) & set(gold)) / len(gold)

def main() -> None:
    items = [q for q in json.loads(DATASET.read_text(encoding="utf-8")) if q["gold_chunk_ids"]]
    random.Random(SEED).shuffle(items)
    items = items[:N_QUERIES]
    print(f"{len(items)} queries with gold chunks (excludes unanswerable)\n")

    # --- NER: the only SLM cost, paid once ---
    cache = json.loads(NER_CACHE.read_text(encoding="utf-8")) if NER_CACHE.exists() else {}
    missing = [q for q in items if q["query_id"] not in cache]
    if missing:
        from pipeline import ner
        print(f"Extracting NER seeds for {len(missing)} queries (~8s each)...")
        t0 = time.perf_counter()
        for i, q in enumerate(missing, 1):
            cache[q["query_id"]] = ner.extract_seed_entities(q["query_text"])
            if i % 5 == 0:
                print(f"  {i}/{len(missing)}  ({time.perf_counter() - t0:.0f}s elapsed)", flush=True)
        NER_CACHE.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"NER seeds ready ({len(cache)} cached)\n")

    with get_driver().session() as session:
        n_chunks = session.run("MATCH (c:Chunk) RETURN count(c) AS n").single()["n"]

        variants = ["vector_rag", "A_current", "B_rerank", "C_vseed_2hop", "D_vseed_1hop"]
        hits = {v: {"r5": [], "r10": [], "cand": []} for v in variants}
        degenerate_seeds = {}

        for q in items:
            gold = q["gold_chunk_ids"]
            qvec = embed_one(q["query_text"])
            seeds = cache[q["query_id"]]
            degenerate_seeds[tuple(sorted(seeds))] = degenerate_seeds.get(tuple(sorted(seeds)), 0) + 1

            # Full corpus ranked by query similarity, reused by every variant.
            ranked = [r["chunk_id"] for r in session.run(
                _ALL_CHUNKS_BY_VECTOR, index=config.INDEX_CHUNKS, k=n_chunks, vec=qvec)]

            def by_vector(candidates):
                """Rank a candidate set by the query-similarity order."""
                cand = set(candidates)
                return [c for c in ranked if c in cand]

            def traverse(seed_ids, hops):
                if not seed_ids:
                    return []
                return [(r["chunk_id"], r["n_ent"]) for r in session.run(
                    _TRAVERSE % {"hops": hops}, seed_ids=seed_ids)]

            # --- reference: no graph ---
            record("vector_rag", hits, ranked, gold, len(ranked))

            # --- A: current implementation ---
            ner_ids = link_entities(seeds)
            cand_a = traverse(ner_ids, config.GRAPH_HOPS)
            record("A_current", hits, [c for c, _ in cand_a], gold, len(cand_a))

            # --- B: same candidates, vector-ranked ---
            record("B_rerank", hits, by_vector([c for c, _ in cand_a]), gold, len(cand_a))

            # --- C/D: entities seeded by the query vector, not by NER ---
            vseed_ids = [r["node_id"] for r in session.run(
                _ENTITIES_BY_VECTOR, index=config.INDEX_ENTITIES, k=10, vec=qvec)
                if index_score_to_cosine(r["score"]) >= config.ENTITY_LINK_THRESHOLD]
            for label, hops in (("C_vseed_2hop", 2), ("D_vseed_1hop", 1)):
                cand = traverse(vseed_ids, hops)
                record(label, hits, by_vector([c for c, _ in cand]), gold, len(cand))

    print(f"{'variant':<14} {'Recall@5':>9} {'Recall@10':>10} {'候选集均值':>12}  (语料库 {n_chunks} chunks)")
    print("-" * 60)
    for v in variants:
        h = hits[v]
        n = len(h["r5"])
        print(f"{v:<14} {sum(h['r5'])/n:>9.3f} {sum(h['r10'])/n:>10.3f} "
              f"{sum(h['cand'])/n:>12.1f}")

    repeats = sum(c for c in degenerate_seeds.values() if c > 1)
    print(f"\nNER 种子重复: {len(items) - len(degenerate_seeds)} / {len(items)} 条查询与另一条共享完全相同的种子集")

def record(label, hits, retrieved, gold, n_cand):
    hits[label]["r5"].append(recall_at_k(retrieved, gold, 5))
    hits[label]["r10"].append(recall_at_k(retrieved, gold, 10))
    hits[label]["cand"].append(n_cand)

if __name__ == "__main__":
    main()
