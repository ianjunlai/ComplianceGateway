# -*- coding: utf-8 -*-
"""Does asking the vector index for 10 results give the same 10 as asking for all 345?

Two harnesses scored vector_rag on the identical 40 queries and disagreed (0.525 vs
0.537). One issued a k=10 index query, the other ranked the whole corpus and sliced the
top 10. Neo4j's vector index is approximate (HNSW), and an approximate search explores
more of the graph when asked for more neighbours -- so the two need not agree, and if
they do not, every strategy that issues a small-k query carries the same quiet recall
loss. Worth knowing before any of these numbers go into a results chapter.
"""

import sys
from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[2]      # inference-service/
_REPO = _SERVICE.parent
sys.path.insert(0, str(_SERVICE))
import json

from pathlib import Path

import config
from pipeline.embeddings import embed_one
from pipeline.graph import get_driver

HERE = Path(__file__).parent
cache = json.loads((HERE / "ner_seed_cache.json").read_text(encoding="utf-8"))
items = {q["query_id"]: q for q in json.loads(
    (_REPO / "dataset" / "qa_dataset.json").read_text(encoding="utf-8"))}

Q = """
CALL db.index.vector.queryNodes($index, $k, $vec) YIELD node, score
RETURN node.chunk_id AS chunk_id
"""

differing = 0
rank_moves = []
with get_driver().session() as s:
    n_chunks = s.run("MATCH (c:Chunk) RETURN count(c) AS n").single()["n"]
    for qid in cache:
        v = embed_one(items[qid]["query_text"])
        small = [r["chunk_id"] for r in s.run(Q, index=config.INDEX_CHUNKS, k=10, vec=v)]
        full = [r["chunk_id"] for r in s.run(Q, index=config.INDEX_CHUNKS, k=n_chunks, vec=v)]
        if small != full[:10]:
            differing += 1
            missing = [c for c in full[:10] if c not in small]
            rank_moves.append((qid, missing))

print(f"{len(cache)} queries")
print(f"k=10 result differs from the top-10 of a full ranking: {differing} "
      f"({differing / len(cache):.0%})")
if rank_moves:
    print("\nclauses the full ranking places in the top 10 that the k=10 query misses:")
    for qid, missing in rank_moves[:8]:
        print(f"   {qid}: {missing}")
