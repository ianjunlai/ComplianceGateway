# -*- coding: utf-8 -*-
"""Extract GDPR's internal cross-references as CITES edges.

The pilot suggested that traversing citations rather than extracted entity
relations is what makes a graph selective, but 38 questions cannot settle it.
The GDPR corpus has 144 gold-bearing questions and a full set of baselines, so
the claim can be tested there properly -- if the same citation structure exists
inside a single instrument, which it does: the Regulation refers to its own
articles constantly ("the conditions referred to in Article 6(1)").

Different pattern from extract_implements.py. Inside the Regulation a bare
"Article 6(1)" means this Regulation, so the "of the Data Protection
Regulation" anchor that disambiguated national citations is absent. What has to
be excluded instead is references to OTHER instruments, which GDPR names
explicitly ("Article 6 of Directive 2002/58/EC").

    python dataset/extract_gdpr_citations.py
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CHUNKS = REPO / "inference-service" / "artifacts" / "chunk_texts.json"
OUT = HERE / "corpus" / "gdpr_citations.json"

# "Article 6(1)(a)" / "Articles 13 and 14" / "Article 9 (2)"
CITE = re.compile(r"Articles?\s+(\d{1,3})((?:\s*\(\d+[a-z]?\))*)"
                  r"((?:\s*(?:,|and|or|to)\s*\d{1,3}(?:\s*\(\d+[a-z]?\))*)*)", re.I)
# A reference that names another instrument is not an internal one.
FOREIGN = re.compile(r"\bof\s+(?:Directive|Regulation|Decision|Treaty|Charter)\b", re.I)
INNER = re.compile(r"(?<![(\w])(\d{1,3})((?:\s*\(\d+[a-z]?\))*)")


def resolve(article: str, paras: str, ids: set[str]) -> tuple[list[str], str]:
    first = re.match(r"\s*\((\d+)", paras)
    if first:
        sub = f"gdpr-art-{article}-{first.group(1)}"
        if sub in ids:
            return [sub], "exact"
    whole = f"gdpr-art-{article}"
    if whole in ids:
        return [whole], "whole"
    parts = sorted(i for i in ids if re.fullmatch(rf"gdpr-art-{article}-\d+", i))
    return parts, "spread"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()

    texts = json.loads(CHUNKS.read_text(encoding="utf-8"))
    gdpr = {k: v for k, v in texts.items() if k.startswith("gdpr-")}
    article_ids = {k for k in gdpr if k.startswith("gdpr-art-")}
    print(f"{len(gdpr)} GDPR chunks, {len(article_ids)} of them articles")

    edges, unresolved, foreign_skipped = [], Counter(), 0
    for cid, text in gdpr.items():
        seen = set()
        for m in CITE.finditer(text):
            # Look just past the match: "Article 6 of Directive 2002/58/EC"
            tail = text[m.end():m.end() + 40]
            if FOREIGN.match(tail.lstrip()):
                foreign_skipped += 1
                continue
            span = m.group(0)
            for art, paras in INNER.findall(span):
                targets, kind = resolve(art, paras, article_ids)
                if not targets:
                    unresolved[f"Article {art}"] += 1
                    continue
                for t in targets:
                    if t == cid or (cid, t) in seen:
                        continue        # a provision citing itself is not an edge
                    seen.add((cid, t))
                    edges.append({"source": cid, "target": t, "type": "CITES",
                                  "citation": f"Article {art}{paras}".strip(),
                                  "resolution": kind})

    OUT.write_text(json.dumps(edges, ensure_ascii=False, indent=1), encoding="utf-8")

    srcs = {e["source"] for e in edges}
    tgts = {e["target"] for e in edges}
    touched = srcs | tgts
    print(f"\n{len(edges)} CITES edges")
    print(f"   from {len(srcs)} chunks to {len(tgts)} articles")
    print(f"   chunks touched: {len(touched)} of {len(gdpr)} ({len(touched) / len(gdpr):.0%})")
    print(f"   references to other instruments skipped: {foreign_skipped}")
    print(f"   resolution: {dict(Counter(e['resolution'] for e in edges))}")
    if unresolved:
        print(f"   citations to an article with no chunk: {sum(unresolved.values())}")

    deg = Counter(e["target"] for e in edges)
    print(f"\nmost-cited: {dict(deg.most_common(8))}")
    print(f"mean out-degree over citing chunks: {len(edges) / max(1, len(srcs)):.1f}")

    print(f"\nsamples:")
    for e in edges[:args.show]:
        print(f"   {e['source']} --CITES--> {e['target']}  ({e['citation']})")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
