# -*- coding: utf-8 -*-
"""Did deduplication collapse distinctions the graph strategies needed?

Dedup runs once at ingestion and rewrites only the entity layer, so it can
degrade all three graph strategies while leaving vector_rag untouched — the
exact asymmetry the recall numbers show. The declared entity type gives a cheap
correctness probe the merge itself never consulted: a merge that folds an ACTOR
into a CONCEPT changed what the node means, whatever its embedding said.

SUPERSEDED. This measures similarity-based merging, which has since been
removed: entities are now merged only when their names match exactly, and
near-identity is expressed as a SYNONYM edge instead, as both source papers
specify. The script is kept because it is the provenance for
docs/retrieval_ablation.md §4.3, and that section describes the configuration
the reported GDPR figures were produced under. It refuses to run against a
report written by the current pipeline rather than silently reinterpreting
fields that no longer mean the same thing.
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
report = json.loads((ART / "dedup_report.json").read_text(encoding="utf-8"))

if report and "alias_type" not in report[0]:
    raise SystemExit(
        f"{ART / 'dedup_report.json'} was written by the exact-match dedup, which "
        f"performs no similarity merging — there is no merge damage to audit.\n"
        f"This script describes the superseded 0.90-cosine configuration; see the "
        f"module docstring.")

cross = [m for m in report if m["alias_type"] != m["canonical_type"]]
print("configuration audited        : similarity merge at cosine 0.90 (SUPERSEDED)")
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
