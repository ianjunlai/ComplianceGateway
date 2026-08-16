# -*- coding: utf-8 -*-
"""How much of the corpus a two-hop traversal reaches. A median near 100% means
the graph selects nothing."""
import argparse
import json
import statistics
import sys
from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SERVICE))

import config                                      # noqa: E402
from pipeline.entity_linking import link_entities  # noqa: E402
from pipeline.graph import get_driver              # noqa: E402

# Same traversal as hybrid_graph.py, counting what it admits instead of ranking
# it. Measuring a different expansion would describe a strategy nobody runs.
_REACH_QUERY = """
UNWIND $seed_ids AS seed_id
MATCH (seed:Entity {node_id: seed_id})
CALL (seed) {
    MATCH (seed)-[:RELATES*1..%(hops)d]-(nbr:Entity)
    RETURN collect(DISTINCT nbr) AS nbrs
}
WITH seed, nbrs
UNWIND ([seed] + nbrs) AS ent
MATCH (ent)-[:MENTIONED_IN]->(c:Chunk)
RETURN count(DISTINCT c) AS reached
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ner-cache", required=True)
    ap.add_argument("--limit", type=int, default=200,
                    help="queries to sample; the distribution stabilises well before this")
    # See the note in selectivity.py: stdout-only output has already cost this
    # project one set of measurements.
    ap.add_argument("--out", default="../results/hop_coverage.json",
                    help="where to write the result; --out '' to skip")
    ap.add_argument("--label", default=None, help="name for this corpus")
    args = ap.parse_args()

    cache = json.loads(Path(args.ner_cache).read_text(encoding="utf-8"))
    queries = [q for q in json.loads(Path(args.dataset).read_text(encoding="utf-8"))
               if q["query_id"] in cache and q.get("gold_chunk_ids")]
    queries = queries[:args.limit]

    driver = get_driver()
    with driver.session() as s:
        total = s.run("MATCH (c:Chunk) RETURN count(c) AS n").single()["n"]
        entities = s.run("MATCH (e:Entity) RETURN count(e) AS n").single()["n"]

    fractions, unlinked = [], 0
    with driver.session() as s:
        for q in queries:
            seed_ids = link_entities(cache[q["query_id"]])
            if not seed_ids:
                # A query the graph strategies cannot start from at all.
                unlinked += 1
                continue
            reached = s.run(_REACH_QUERY % {"hops": config.GRAPH_HOPS},
                            seed_ids=seed_ids).single()["reached"]
            fractions.append(reached / total)

    print(f"corpus: {total} chunks, {entities} entities, {config.GRAPH_HOPS} hops")
    print(f"queries: {len(queries)} sampled, {unlinked} with no linkable seed\n")
    if not fractions:
        raise SystemExit("no query linked to the graph — is the right corpus loaded?")

    pct = sorted(f * 100 for f in fractions)
    print(f"  chunks reachable within {config.GRAPH_HOPS} hops, as % of corpus")
    print(f"    median : {statistics.median(pct):6.1f}%")
    print(f"    mean   : {statistics.fmean(pct):6.1f}%")
    print(f"    p10-p90: {pct[len(pct) // 10]:6.1f}% - {pct[-max(1, len(pct) // 10)]:6.1f}%")
    print(f"    max    : {pct[-1]:6.1f}%  ({pct[-1] / 100 * total:.0f} chunks)")
    print("\nA median near 100% means the traversal admits the whole corpus and the "
          "\nranking step, not the graph, decides the result.")

    if args.out:
        from datetime import datetime, timezone
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        existing = {}
        if p.exists():
            try:
                existing = json.loads(p.read_text(encoding="utf-8"))
            except ValueError:
                pass
        existing[args.label or args.dataset] = {
            "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset": args.dataset,
            "chunks": total, "entities": entities, "hops": config.GRAPH_HOPS,
            "queries_sampled": len(queries), "queries_unlinked": unlinked,
            "reachable_pct": {
                "median": round(statistics.median(pct), 2),
                "mean": round(statistics.fmean(pct), 2),
                "p10": round(pct[len(pct) // 10], 2),
                "p90": round(pct[-max(1, len(pct) // 10)], 2),
                "max": round(pct[-1], 2),
            },
        }
        p.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
