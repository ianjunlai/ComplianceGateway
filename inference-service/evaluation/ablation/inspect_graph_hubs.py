# -*- coding: utf-8 -*-
"""Is the entity layer discriminative enough to retrieve with?

All three graph strategies underperform plain vector search, which points past
any single ranking bug to the graph itself. An entity mentioned in most of the
corpus carries almost no information about which clause answers a query, so
this reports how concentrated the mention distribution is.
"""

import sys
from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[2]      # inference-service/
_REPO = _SERVICE.parent
sys.path.insert(0, str(_SERVICE))

from pipeline.graph import get_driver

with get_driver().session() as s:
    n_chunks = s.run("MATCH (c:Chunk) RETURN count(c) AS n").single()["n"]
    n_ents = s.run("MATCH (e:Entity) RETURN count(e) AS n").single()["n"]

    rows = [(r["name"], r["n"]) for r in s.run("""
        MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)
        WITH e, count(DISTINCT c) AS n
        RETURN e.name AS name, n ORDER BY n DESC
    """)]

    degree = [r["d"] for r in s.run("""
        MATCH (e:Entity) RETURN size([(e)-[:RELATES]-() | 1]) AS d ORDER BY d DESC
    """)]

print(f"corpus: {n_chunks} chunks, {n_ents} entities\n")

print("most-mentioned entities (entity -> how many chunks mention it):")
for name, n in rows[:15]:
    print(f"   {n:>4} chunks ({n / n_chunks:>5.1%})  {name}")

singletons = sum(1 for _, n in rows if n == 1)
broad = sum(1 for _, n in rows if n >= 0.10 * n_chunks)
print(f"\nentities mentioned in exactly 1 chunk : {singletons} ({singletons / len(rows):.0%})")
print(f"entities mentioned in >=10% of corpus : {broad}")
print(f"top-1 entity covers                   : {rows[0][1] / n_chunks:.0%} of the corpus")

print(f"\nRELATES degree — max {degree[0]}, median {degree[len(degree)//2]}, "
      f"mean {sum(degree)/len(degree):.1f}")
print(f"entities with degree >= 50            : {sum(1 for d in degree if d >= 50)}")
