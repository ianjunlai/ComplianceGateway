# -*- coding: utf-8 -*-
"""Does the knowledge graph connect the clauses the multi-hop questions pair up?

The questions were built from *citation* links: generate_qa.py pairs clauses
that cross-reference each other by article number. The graph's RELATES edges
come from something else entirely -- semantic relations an LLM extracted per
chunk. Nothing guarantees the two structures coincide.

That distinction decides how to read the negative result. If a gold pair is
connected in the graph and the strategies still missed the second clause, the
methods failed at something reachable. If the pair is not connected at all, no
traversal could have bridged it, and the gap is in graph construction rather
than in graph retrieval.

Measured only on the pairs that matter: those where vector search already found
one gold clause but missed the other, so a hop was the only way through.
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

DATASET = (_REPO / "dataset" / "qa_dataset.json")
items = [q for q in json.loads(DATASET.read_text(encoding="utf-8"))
         if len(q["gold_chunk_ids"]) > 1]

_RANK = """
CALL db.index.vector.queryNodes($index, $k, $vec) YIELD node, score
RETURN node.chunk_id AS chunk_id
"""

# Shortest path between any entity of clause A and any entity of clause B.
_PATH = """
MATCH (a:Chunk {chunk_id: $a})<-[:MENTIONED_IN]-(ea:Entity)
MATCH (b:Chunk {chunk_id: $b})<-[:MENTIONED_IN]-(eb:Entity)
WITH collect(DISTINCT ea) AS eas, collect(DISTINCT eb) AS ebs
UNWIND eas AS ea
UNWIND ebs AS eb
WITH ea, eb WHERE ea <> eb
MATCH p = shortestPath((ea)-[:RELATES*1..4]-(eb))
RETURN min(length(p)) AS hops
"""

# Do the two clauses simply share an entity? That is a zero-hop connection and
# the easiest thing any graph method could exploit.
_SHARED = """
MATCH (a:Chunk {chunk_id: $a})<-[:MENTIONED_IN]-(e:Entity)-[:MENTIONED_IN]->(b:Chunk {chunk_id: $b})
RETURN count(DISTINCT e) AS n
"""

with get_driver().session() as s:
    n_chunks = s.run("MATCH (c:Chunk) RETURN count(c) AS n").single()["n"]

    considered = shared = within2 = within4 = unreachable = 0
    examples = []

    for q in items:
        qvec = embed_one(q["query_text"])
        top10 = [r["chunk_id"] for r in s.run(
            _RANK, index=config.INDEX_CHUNKS, k=10, vec=qvec)]
        found = [g for g in q["gold_chunk_ids"] if g in top10]
        missed = [g for g in q["gold_chunk_ids"] if g not in top10]
        if not found or not missed:
            continue  # vector search got all or none; no hop to make

        for a in found:
            for b in missed:
                considered += 1
                n_shared = s.run(_SHARED, a=a, b=b).single()["n"]
                rec = s.run(_PATH, a=a, b=b).single()
                hops = rec["hops"] if rec and rec["hops"] is not None else None
                if n_shared:
                    shared += 1
                elif hops is not None and hops <= 2:
                    within2 += 1
                elif hops is not None:
                    within4 += 1
                else:
                    unreachable += 1
                    if len(examples) < 6:
                        examples.append((q["query_id"], a, b))

print(f"{considered} (found, missed) gold pairs where vector search got one clause "
      f"but not the other\n")
print("how the missed clause connects to the one that was found:")
print(f"   shares an entity outright   : {shared:>4}  ({shared/considered:.0%})")
print(f"   reachable within 2 RELATES  : {within2:>4}  ({within2/considered:.0%})")
print(f"   reachable within 3-4 RELATES: {within4:>4}  ({within4/considered:.0%})")
print(f"   no path within 4 hops       : {unreachable:>4}  ({unreachable/considered:.0%})")

if examples:
    print("\nunreachable examples (question, clause found, clause missed):")
    for qid, a, b in examples:
        print(f"   {qid}: {a}  -/->  {b}")
