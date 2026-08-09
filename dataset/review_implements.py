# -*- coding: utf-8 -*-
"""Build a spot-check sheet for the IMPLEMENTS edges.

The edges are only worth building a graph on if they are right, and 194 is too
many to read. This samples 20, stratified so the sample is informative rather
than flattering: 'spread' edges are over-sampled because they are the weak kind
(the cited paragraph has no chunk of its own, so the edge points at the
article's other paragraphs), and every jurisdiction appears because the three
Acts cite in visibly different styles.

Each entry puts both sides on the page -- the citing provision with the
sentence that carries the citation, and the GDPR text it resolves to -- so the
judgement is "does this national provision give effect to that article?" and
needs no lookup.

    python dataset/review_implements.py            # writes the sheet
    python dataset/review_implements.py --n 30
"""
import argparse
import json
import random
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
EDGES = HERE / "corpus" / "nations" / "implements_edges.json"
NATIONS = HERE / "corpus" / "nations" / "nations.chunks.json"
GDPR = REPO / "inference-service" / "artifacts" / "chunk_texts.json"
OUT = HERE / "corpus" / "nations" / "implements_review.md"
SEED = 42


def wrap(s, width=94, indent="   "):
    return textwrap.fill(" ".join(s.split()), width=width,
                         initial_indent=indent, subsequent_indent=indent)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    edges = json.loads(EDGES.read_text(encoding="utf-8"))
    nat = {c["chunk_id"]: c for c in json.loads(NATIONS.read_text(encoding="utf-8"))}
    gdpr = json.loads(GDPR.read_text(encoding="utf-8"))

    by_kind = defaultdict(list)
    for e in edges:
        by_kind[e.get("resolution", "?")].append(e)
    print("edges by resolution:", {k: len(v) for k, v in by_kind.items()})

    # Over-sample the weak kind: it is where an error is most likely, and a
    # sample drawn proportionally would mostly show the easy cases.
    quota = {"spread": args.n // 2, "exact": args.n // 4, "whole": args.n - args.n // 2 - args.n // 4}
    rng = random.Random(SEED)
    picked = []
    for kind, want in quota.items():
        pool = by_kind.get(kind, [])
        picked.extend(rng.sample(pool, min(want, len(pool))))
    # Top up from anywhere if a kind was short.
    if len(picked) < args.n:
        rest = [e for e in edges if e not in picked]
        picked.extend(rng.sample(rest, min(args.n - len(picked), len(rest))))

    lines = ["# IMPLEMENTS edge spot-check", "",
             f"{len(picked)} of {len(edges)} edges, stratified by how the citation resolved.",
             "",
             "For each: does the national provision give effect to the GDPR article it points at?",
             "Mark each **yes / no / unclear**. A 'no' on a `spread` edge is expected and tells us",
             "whether that resolution kind should be dropped rather than kept.", "",
             "| # | verdict |", "|---|---|"]
    lines += [f"| {i} |  |" for i in range(1, len(picked) + 1)]
    lines.append("")

    for i, e in enumerate(picked, 1):
        src = nat.get(e["source"], {})
        lines.append("---")
        lines.append(f"\n## {i}. `{e['source']}` → `{e['target']}`  "
                     f"[{e['jurisdiction']}, {e.get('resolution')}]")
        lines.append(f"\ncites **{e['citation']}**\n")
        lines.append(f"**{src.get('title', '?')}** — the citing sentence:\n")
        lines.append("```")
        lines.append(wrap(e["context"], 92, ""))
        lines.append("```\n")
        lines.append(f"**{e['target']}** — what it resolves to:\n")
        lines.append("```")
        lines.append(wrap(gdpr.get(e["target"], "(chunk not found)")[:900], 92, ""))
        lines.append("```\n")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nsampled {len(picked)}: "
          f"{dict(Counter(e.get('resolution') for e in picked))}, "
          f"jurisdictions {dict(Counter(e['jurisdiction'] for e in picked))}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
