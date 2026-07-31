# -*- coding: utf-8 -*-
"""Is the graph's type vocabulary and deduplication the reason graph retrieval lags?

Deduplication runs once at ingestion and touches only the entity layer, so it
can degrade all three graph strategies while leaving vector_rag -- which reads
chunk embeddings and never looks at an entity -- untouched. That is exactly the
pattern the measurements show, which makes it worth testing rather than
assuming.

Reports what the type vocabulary actually looks like, how much the dedup pass
collapsed, and whether the types survive into anything retrieval can use.
"""

import sys
from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[2]      # inference-service/
_REPO = _SERVICE.parent
sys.path.insert(0, str(_SERVICE))
import json

from collections import Counter
from pathlib import Path

from pipeline.graph import get_driver

ART = Path(str(_SERVICE / "artifacts"))

with get_driver().session() as s:
    props = s.run("""
        MATCH (e:Entity) WITH keys(e) AS k LIMIT 1 RETURN k
    """).single()
    print(f"Entity properties stored in Neo4j: {props['k']}\n")

    ent_types = [(r["t"], r["n"]) for r in s.run("""
        MATCH (e:Entity) RETURN e.type AS t, count(*) AS n ORDER BY n DESC
    """)]
    rel_types = [(r["t"], r["n"]) for r in s.run("""
        MATCH ()-[r:RELATES]->() RETURN r.type AS t, count(*) AS n ORDER BY n DESC
    """)]

print(f"entity types: {len(ent_types)}")
for t, n in ent_types:
    print(f"   {str(t):<14} {n}")

print(f"\nrelation types: {len(rel_types)} distinct")
for t, n in rel_types[:12]:
    print(f"   {str(t):<28} {n}")
tail = sum(n for _, n in rel_types[12:])
singles = sum(1 for _, n in rel_types if n == 1)
print(f"   ... {len(rel_types) - 12} more types covering {tail} edges")
print(f"   relation types used exactly once: {singles} ({singles / len(rel_types):.0%})")

# --- what the dedup pass actually collapsed ---
report = json.loads((ART / "dedup_report.json").read_text(encoding="utf-8"))
print(f"\ndedup merges recorded: {len(report)}")
print(f"record shape: {list(report[0].keys())}")

cache = json.loads((ART / "extraction_cache.json").read_text(encoding="utf-8"))
raw = sum(len(v["entities"]) for v in cache.values() if isinstance(v, dict) and "entities" in v)
with get_driver().session() as s:
    final = s.run("MATCH (e:Entity) RETURN count(e) AS n").single()["n"]
print(f"raw extracted entity mentions : {raw}")
print(f"entities after dedup          : {final}   ({1 - final / raw:.0%} collapsed)")
print(f"  of which by embedding similarity across DIFFERENT names: {len(report)}")
print(f"  the rest collapsed by exact repeated name across chunks: {raw - final - len(report)}")

# Are merged entities of the same declared type?
if "type" in report[0] or "canonical_type" in report[0]:
    pass
print("\nsample merges (rarest first is how they were reviewed before):")
for m in report[:8]:
    print("   " + json.dumps(m, ensure_ascii=False)[:150])
