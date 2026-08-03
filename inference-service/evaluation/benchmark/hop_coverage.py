# -*- coding: utf-8 -*-
"""How much of the corpus a 2-hop traversal can reach.

This is the mechanism behind the GDPR result, and it is measured rather than
argued. On that corpus the traversal in hybrid_graph.py reaches 343 of 345
clauses from typical query seeds -- 99% -- so the graph admits nearly everything
and contributes no discrimination; whatever ranks the candidates afterwards is
doing all the work, which is exactly why hybrid converges on plain dense
retrieval there. A corpus where graph retrieval helps must reach a small
fraction instead.

Runs against whichever graph is currently loaded in Neo4j, so it produces the
GDPR figure and the benchmark figure with the same code:

    python -m evaluation.benchmark.hop_coverage \
        --dataset ../dataset/qa_dataset.json \
        --ner-cache evaluation/ablation/ner_seed_cache.json

    python -m evaluation.benchmark.hop_coverage \
        --dataset ../dataset/benchmark/2wiki_qa.json \
        --ner-cache evaluation/benchmark/ner_seed_cache_2wiki.json
"""
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
                # A query the graph strategies cannot start from at all. Folding
                # it in as 0% coverage would flatter the graph; it is reported
                # separately instead.
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


if __name__ == "__main__":
    main()
