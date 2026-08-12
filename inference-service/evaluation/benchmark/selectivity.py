# -*- coding: utf-8 -*-
"""How much of a corpus one hop through the entity layer admits.

This is the diagnostic that explains why an entity graph helps on one corpus
and not on another, and it can be measured before any retrieval experiment is
run. If a single hop from a handful of entry chunks admits nearly the whole
corpus, the graph is performing no selection: ranking the admitted set by
similarity to the query reproduces dense retrieval, and no traversal policy
built on those edges can do better.

The cause is the register of the source text. Legislation is written in a
shared vocabulary -- "personal data" appears in roughly half the clauses of the
GDPR corpus -- so co-mention edges extracted from it connect nearly everything
to nearly everything. A benchmark built on proper nouns does not have this
property, because a person or a film is discriminative by nature.

Run it against each graph in turn and compare:

    NEO4J_URI=bolt://localhost:7687 python -m evaluation.benchmark.selectivity
    NEO4J_URI=bolt://localhost:7688 ARTIFACTS_DIR=... python -m evaluation.benchmark.selectivity
"""
import argparse
import statistics
import sys
from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SERVICE))

import config                                    # noqa: E402
from pipeline.graph import get_driver            # noqa: E402

# chunk -> entity it mentions -> related entity -> the chunks mentioning that.
# One hop of RELATES; the MENTIONED_IN steps are the projection back to the
# chunk layer and are not a second hop of the relation.
_ADMIT = """
MATCH (c:Chunk) WITH c, rand() AS r ORDER BY r LIMIT $entry
WITH collect(c) AS seeds
UNWIND seeds AS c
OPTIONAL MATCH (c)<-[:MENTIONED_IN]-(:Entity)-[:RELATES]-(:Entity)-[:MENTIONED_IN]->(l:Chunk)
WITH collect(DISTINCT c) + collect(DISTINCT l) AS pool
UNWIND pool AS n
RETURN count(DISTINCT n) AS admitted
"""

# The same measurement along the citations the drafters wrote, where the corpus
# has them. Included because the contrast between the two is the point: at the
# same depth the two kinds of edge admit very different fractions.
_ADMIT_CITES = """
MATCH (c:Chunk) WITH c, rand() AS r ORDER BY r LIMIT $entry
WITH collect(c) AS seeds
UNWIND seeds AS c
OPTIONAL MATCH (c)-[:CITES|IMPLEMENTS]-(l:Chunk)
WITH collect(DISTINCT c) + collect(DISTINCT l) AS pool
UNWIND pool AS n
RETURN count(DISTINCT n) AS admitted
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entry", type=int, nargs="+", default=[1, 5],
                    help="entry-set sizes to report")
    ap.add_argument("--repeats", type=int, default=20,
                    help="random entry sets per size; the entry set is random "
                         "because selectivity is a property of the graph, not "
                         "of any one query")
    args = ap.parse_args()

    with get_driver().session() as s:
        total = s.run("MATCH (c:Chunk) RETURN count(c) AS n").single()["n"]
        ents = s.run("MATCH (e:Entity) RETURN count(e) AS n").single()["n"]
        rels = s.run("MATCH ()-[r:RELATES]->() RETURN count(r) AS n").single()["n"]
        cites = s.run("MATCH ()-[r:CITES|IMPLEMENTS]->() "
                      "RETURN count(r) AS n").single()["n"]
        print(f"{config.NEO4J_URI}")
        print(f"  {total} chunks, {ents} entities, {rels} RELATES, "
              f"{cites} citation edges\n")

        print(f"{'edges followed':<18}{'entry':>7}{'admitted':>11}{'% of corpus':>14}")
        print("-" * 50)
        for label, query, available in (("entity co-mention", _ADMIT, rels),
                                        ("citations", _ADMIT_CITES, cites)):
            if not available:
                print(f"{label:<18}{'--':>7}{'n/a':>11}{'(no such edges)':>14}")
                continue
            for entry in args.entry:
                runs = [s.run(query, entry=entry).single()["admitted"]
                        for _ in range(args.repeats)]
                mean = statistics.mean(runs)
                print(f"{label:<18}{entry:>7}{mean:>11.1f}{100*mean/total:>13.1f}%")

    print("\n  Near 100% means the hop selects nothing: the admitted set is the")
    print("  corpus, so ranking it by query similarity is dense retrieval.")


if __name__ == "__main__":
    main()
