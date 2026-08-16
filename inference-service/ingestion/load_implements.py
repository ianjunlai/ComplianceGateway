# -*- coding: utf-8 -*-
"""Load the Chunk-to-Chunk citation edges into an existing graph. Separate from
build_indexes so they can be added or removed without a rebuild."""
import argparse
import json
import sys
from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SERVICE))

from pipeline.graph import get_driver   # noqa: E402

EDGES = _SERVICE.parent / "dataset" / "corpus" / "nations" / "implements_edges.json"

# One statement per relationship type: Cypher cannot parameterise a type, and
# the two mean different things.
_CREATE = """
UNWIND $rows AS row
MATCH (a:Chunk {chunk_id: row.source}), (b:Chunk {chunk_id: row.target})
CREATE (a)-[:%(type)s {citation: row.citation, resolution: row.resolution}]->(b)
"""
_TYPES = ("IMPLEMENTS", "CITES")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--remove", action="store_true")
    ap.add_argument("--edges", default=str(EDGES))
    ap.add_argument("--skip-spread", action="store_true",
                    help="drop edges whose citation named a paragraph with no chunk of "
                         "its own; those point at the article's other paragraphs")
    args = ap.parse_args()

    driver = get_driver()
    with driver.session() as s:
        if args.remove:
            for t in _TYPES:
                n = s.run(f"MATCH ()-[r:{t}]->() DELETE r "
                          f"RETURN count(r) AS n").single()["n"]
                print(f"removed {n} {t} edges")
            return

        rows = json.loads(Path(args.edges).read_text(encoding="utf-8"))
        if args.skip_spread:
            rows = [r for r in rows if r.get("resolution") != "spread"]
        unknown = {r.get("type") for r in rows} - set(_TYPES)
        if unknown:
            raise SystemExit(f"edge file contains unsupported type(s) {unknown}; "
                             f"expected one of {_TYPES}")

        # Only edges whose BOTH ends are in this graph can be created.
        ids = {r["chunk_id"] for r in
               s.run("MATCH (c:Chunk) RETURN c.chunk_id AS chunk_id")}
        usable = [r for r in rows if r["source"] in ids and r["target"] in ids]
        print(f"{len(rows)} edges in file, {len(usable)} with both ends in this graph")

        for t in _TYPES:
            batch = [r for r in usable if r.get("type") == t]
            if not batch:
                continue
            existing = s.run(f"MATCH ()-[r:{t}]->() RETURN count(r) AS n").single()["n"]
            if existing:
                raise SystemExit(f"{existing} {t} edges already present; run with "
                                 f"--remove first so the count stays meaningful")
            for start in range(0, len(batch), 5000):
                s.run(_CREATE % {"type": t}, rows=batch[start:start + 5000])
            print(f"   created {len(batch)} {t} edges")

        total = s.run("MATCH (c:Chunk) RETURN count(c) AS n").single()["n"]
        linked = s.run("MATCH (c:Chunk)-[:CITES|IMPLEMENTS]-() "
                       "RETURN count(DISTINCT c) AS n").single()["n"]
        print(f"chunks touched by any citation edge: {linked} of {total} "
              f"({linked / total:.0%})")


if __name__ == "__main__":
    main()
