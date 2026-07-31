# -*- coding: utf-8 -*-
"""Does graph expansion recover multi-hop evidence that vector search misses?

The earlier comparison showed the graph only ever subtracts: its best variant
still trailed plain vector search. But every variant there used the graph to
*restrict* the candidate set. This tests the opposite arrangement — vector
search locates entry chunks, the graph expands outward from them — which is
where a knowledge graph should earn its place: a gold clause that is not
worded like the query but is connected to one that is.

Results are broken down by hop_type, because expansion should pay off on
multi-hop questions and do nothing on single-hop ones. An aggregate number
would hide exactly the effect being tested.
"""

import sys
from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[2]      # inference-service/
_REPO = _SERVICE.parent
sys.path.insert(0, str(_SERVICE))
import json
import random

from collections import defaultdict
from pathlib import Path

import config
from pipeline.embeddings import embed_one
from pipeline.graph import get_driver

DATASET = (_REPO / "dataset" / "qa_dataset.json")
N_QUERIES = 60
SEED = 42
ENTRY_K = 5          # entry chunks located by vector search
TOP_K = 10

_RANK_ALL = """
CALL db.index.vector.queryNodes($index, $k, $vec) YIELD node, score
RETURN node.chunk_id AS chunk_id
"""

# Chunks sharing an entity with the entry chunks (1 hop), and chunks two hops
# out. Ordered by how many entry-linked entities reach them, so the strongest
# graph evidence is appended first.
_EXPAND = """
MATCH (c:Chunk) WHERE c.chunk_id IN $entry_ids
MATCH (e:Entity)-[:MENTIONED_IN]->(c)
WITH collect(DISTINCT e) AS entry_ents
UNWIND entry_ents AS e
CALL (e) {
    MATCH (e)-[:RELATES*0..%(hops)d]-(nbr:Entity)
    RETURN collect(DISTINCT nbr) AS nbrs
}
WITH entry_ents, collect(nbrs) AS all_nbrs
WITH entry_ents, reduce(acc = [], n IN all_nbrs | acc + n) AS flat
UNWIND flat AS ent
MATCH (ent)-[:MENTIONED_IN]->(c2:Chunk)
WHERE NOT c2.chunk_id IN $entry_ids
WITH c2, count(DISTINCT ent) AS support
RETURN c2.chunk_id AS chunk_id, support
ORDER BY support DESC
"""

def recall_at_k(retrieved, gold, k):
    return len(set(retrieved[:k]) & set(gold)) / len(gold)

def main() -> None:
    items = [q for q in json.loads(DATASET.read_text(encoding="utf-8")) if q["gold_chunk_ids"]]
    random.Random(SEED).shuffle(items)
    items = items[:N_QUERIES]

    variants = ["vector_rag", "E_expand_1hop", "F_expand_2hop"]
    scores = {v: defaultdict(list) for v in variants}

    with get_driver().session() as session:
        n_chunks = session.run("MATCH (c:Chunk) RETURN count(c) AS n").single()["n"]

        for q in items:
            gold, ht = q["gold_chunk_ids"], q["hop_type"]
            qvec = embed_one(q["query_text"])
            ranked = [r["chunk_id"] for r in session.run(
                _RANK_ALL, index=config.INDEX_CHUNKS, k=n_chunks, vec=qvec)]

            scores["vector_rag"][ht].append(recall_at_k(ranked, gold, TOP_K))

            entry = ranked[:ENTRY_K]
            for label, hops in (("E_expand_1hop", 1), ("F_expand_2hop", 2)):
                expanded = [r["chunk_id"] for r in session.run(
                    _EXPAND % {"hops": hops}, entry_ids=entry)]
                # Entry chunks keep their vector order; graph-found chunks are
                # appended by graph support. Gold reached only by expansion can
                # therefore enter the top-10 without displacing a vector hit.
                merged = entry + [c for c in expanded if c not in set(entry)]
                scores[label][ht].append(recall_at_k(merged, gold, TOP_K))

    hop_types = sorted({q["hop_type"] for q in items})
    header = f"{'variant':<16}" + "".join(f"{ht:>12}" for ht in hop_types) + f"{'overall':>10}"
    print(f"Recall@{TOP_K}, {len(items)} queries, entry_k={ENTRY_K}\n")
    print(header)
    print("-" * len(header))
    for v in variants:
        row = f"{v:<16}"
        allv = []
        for ht in hop_types:
            vals = scores[v][ht]
            allv += vals
            row += f"{sum(vals)/len(vals):>12.3f}" if vals else f"{'-':>12}"
        row += f"{sum(allv)/len(allv):>10.3f}"
        print(row)

    counts = {ht: len(scores['vector_rag'][ht]) for ht in hop_types}
    print(f"\nqueries per hop_type: {counts}")

if __name__ == "__main__":
    main()
