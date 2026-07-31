# -*- coding: utf-8 -*-
"""Did deduplication collapse distinctions the graph strategies needed?

Dedup runs once at ingestion and rewrites only the entity layer, so it can
degrade all three graph strategies while leaving vector_rag untouched — the
exact asymmetry the recall numbers show. The declared entity type gives a cheap
correctness probe the merge itself never consulted: a merge that folds an ACTOR
into a CONCEPT changed what the node means, whatever its embedding said.
"""

import sys
from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[2]      # inference-service/
_REPO = _SERVICE.parent
sys.path.insert(0, str(_SERVICE))
import json

from collections import Counter
from pathlib import Path

import config
from pipeline.graph import get_driver

ART = Path(str(_SERVICE / "artifacts"))
report = json.loads((ART / "dedup_report.json").read_text(encoding="utf-8"))

cross = [m for m in report if m["alias_type"] != m["canonical_type"]]
print(f"dedup threshold (config)      : {config.DEDUP_THRESHOLD}")
print(f"merges across different names : {len(report)}")
print(f"  of which cross-TYPE         : {len(cross)} ({len(cross) / len(report):.0%})\n")

print("cross-type merges (the declared types disagree, yet they became one node):")
for m in sorted(cross, key=lambda m: -m["similarity"])[:12]:
    print(f"   {m['alias']!r} [{m['alias_type']}] -> {m['canonical']!r} "
          f"[{m['canonical_type']}]  sim={m['similarity']:.3f}")

buckets = Counter(f"{m['similarity']:.2f}"[:4] for m in report)
print(f"\nsimilarity distribution of merges: "
      f"{sorted(buckets.items())}")

# How much of the entity layer was touched at all?
with get_driver().session() as s:
    total = s.run("MATCH (e:Entity) RETURN count(e) AS n").single()["n"]
    merged = s.run("""
        MATCH (e:Entity) WHERE e.aliases IS NOT NULL AND size(e.aliases) > 0
        RETURN count(e) AS n
    """).single()["n"]
    widest = [(r["name"], r["k"]) for r in s.run("""
        MATCH (e:Entity) WHERE e.aliases IS NOT NULL
        RETURN e.name AS name, size(e.aliases) AS k ORDER BY k DESC LIMIT 8
    """)]

print(f"\nentities in graph            : {total}")
print(f"entities carrying an alias   : {merged} ({merged / total:.0%})")
print(f"\nnodes that absorbed the most names:")
for name, k in widest:
    print(f"   {k:>3} aliases   {name}")

# The counterfactual: if dedup were the cause, undoing it should matter. The
# cheapest proxy is how many gold-bearing queries even touch a merged node.
print("\n--- reach of the merged nodes ---")
with get_driver().session() as s:
    chunks_via_merged = s.run("""
        MATCH (e:Entity) WHERE e.aliases IS NOT NULL AND size(e.aliases) > 0
        MATCH (e)-[:MENTIONED_IN]->(c:Chunk)
        RETURN count(DISTINCT c) AS n
    """).single()["n"]
    all_chunks = s.run("MATCH (c:Chunk) RETURN count(c) AS n").single()["n"]
print(f"clauses reachable through a merged entity: {chunks_via_merged} / {all_chunks} "
      f"({chunks_via_merged / all_chunks:.0%})")
