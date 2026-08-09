# -*- coding: utf-8 -*-
"""Extract IMPLEMENTS edges: national provision -> the GDPR article it gives effect to.

This is the one thing the three-tier corpus gives graph retrieval that a
metadata filter cannot. A jurisdiction filter is a flat WHERE clause every
vector database supports, so handing it only to the graph strategies would
manufacture an advantage that has nothing to do with graph structure. A typed
cross-tier link is different: "Irish DPA section 148 gives effect to GDPR
Article 80(1)" connects two provisions whose texts share almost no vocabulary --
one is about complaint procedure, the other about mandated representation -- so
embedding similarity finds the pair only by accident.

Regex rather than an LLM. The citations are formulaic, the mapping is
checkable, and an extraction pass over 614 provisions would cost API budget to
produce something less reliable than a pattern that either matches or does not.

Two precision traps are handled explicitly:
  * The Irish and German Acts transpose BOTH Regulation 2016/679 and Directive
    2016/680, and cite them in the same sentence. "Article 22(3) of the
    Directive" must not resolve to GDPR Article 22.
  * The GDPR corpus splits long articles into sub-chunks (gdpr-art-9-2), so a
    citation to Article 9(2) resolves to that sub-chunk while a citation to
    Article 9 alone resolves to whichever chunks exist for it.

    python dataset/extract_implements.py
    python dataset/extract_implements.py --show 20
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
NATIONS = HERE / "corpus" / "nations" / "nations.chunks.json"
GDPR_IDS = REPO / "inference-service" / "artifacts" / "chunk_texts.json"
OUT = HERE / "corpus" / "nations" / "implements_edges.json"

# The instrument named after the article number decides whether this is a GDPR
# citation at all. Directive 2016/680 is transposed by the same Acts and its
# article numbering overlaps completely.
REGULATION = (r"(?:the\s+)?(?:Data Protection Regulation|GDPR|UK GDPR"
              r"|Regulation\s*\(EU\)\s*(?:No\.?\s*)?2016/679)")
DIRECTIVE = r"(?:the\s+)?(?:Directive|Directive\s*\(EU\)\s*2016/680)"

# Real citation shapes, counted across the three Acts before writing this:
#   "Article 9(2)(b) of the Regulation"          the simple case
#   "Article 58 (2) of Regulation ..."           German translation spaces the bracket
#   "Article 15(1) to (3) of the UK GDPR"        a range
#   "Article 6(3), 8A(3)(e), 9(2)(g) or 10(1) of the UK GDPR"   several at once
#   "Article 15 (Right of access) of the Regulation"  Irish drafting inserts the title
# The first pattern alone reached 107 of the 212 citations that name an article.
_SUBS = r"(?:\s*\(\d+[a-z]?\))*"                        # (2)(b), with optional spaces
_TITLE = r"(?:\s*\([A-Z][^)]{2,40}\))?"                 # "(Right of access)"
_TAIL = r"(?:\s*(?:,|and|or|to)\s*\(?\d{0,3}[a-z]?\)?" + _SUBS + r")*"
_ONE = r"(\d{1,3}[A-Z]?)" + _SUBS + _TITLE + _TAIL

CITE = re.compile(r"Articles?\s+" + _ONE + r"(?:\s*(?:,|and|or)\s*" + _ONE + r")*"
                  r"\s+of\s+" + REGULATION, re.I)
CITE_DIR = re.compile(r"Articles?\s+" + _ONE + r"(?:\s*(?:,|and|or)\s*" + _ONE + r")*"
                      r"\s+of\s+" + DIRECTIVE, re.I)
# Pulls each article out of a matched span, so "Article 6(3), 8A(3)(e) or 10(1)
# of ..." yields three edges rather than one.
#
# (?<![(\w]) is what keeps it honest. Without it, "Article 35(4) and (5)" also
# yields an Article 5, because the paragraph number inside the brackets looks
# exactly like an article number -- a bare digit. Requiring that the digit not
# follow an opening bracket distinguishes "article 5" from "paragraph (5)".
INNER = re.compile(r"(?<![(\w])(\d{1,3}[A-Z]?)((?:\s*\(\d+[a-z]?\))*)")


def load_gdpr_ids() -> set[str]:
    return {k for k in json.loads(GDPR_IDS.read_text(encoding="utf-8"))
            if k.startswith("gdpr-art-")}


def resolve(article: str, paras: str, ids: set[str]) -> tuple[list[str], str]:
    """Citation -> the chunk ids that actually exist for it, and how it got there.

    The three outcomes differ in how much they can be trusted, so the kind is
    recorded rather than flattened away:
      exact      the citation named a paragraph and that sub-chunk exists
      whole      the article is one chunk, so the citation lands on it
      spread     the citation named a paragraph with no chunk of its own, so it
                 falls back to every sub-chunk of that article. This is the
                 weakest kind: a citation to Article 28(3) becomes edges to
                 28-1 and 28-2, neither of which is the cited paragraph.
    """
    first_para = re.match(r"\((\d+)", paras)
    if first_para:
        specific = f"gdpr-art-{article}-{first_para.group(1)}"
        if specific in ids:
            return [specific], "exact"
    whole = f"gdpr-art-{article}"
    if whole in ids:
        return [whole], "whole"
    parts = sorted(i for i in ids if re.fullmatch(rf"gdpr-art-{article}-\d+", i))
    return parts, "spread"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--show", type=int, default=8)
    args = ap.parse_args()

    ids = load_gdpr_ids()
    chunks = json.loads(NATIONS.read_text(encoding="utf-8"))
    print(f"{len(chunks)} national provisions, {len(ids)} GDPR article chunks\n")

    edges, unresolved = [], Counter()
    directive_only = 0
    per_j = defaultdict(lambda: {"cites": 0, "edges": 0, "provisions": set()})

    for c in chunks:
        text = c["text"]
        # Blank out Directive citations first so the Regulation pattern cannot
        # pick up an article number that belongs to the other instrument.
        masked = CITE_DIR.sub(lambda m: " " * len(m.group(0)), text)
        directive_only += len(CITE_DIR.findall(text))

        seen = set()
        for m in CITE.finditer(masked):
            span = m.group(0)
            # The span may name several articles ("Article 6(3), 8A(3)(e) or
            # 10(1) of ..."); take each in turn. Stop at " of ", after which
            # the digits belong to the instrument's own name (2016/679).
            head = span[:span.lower().rfind(" of ")] if " of " in span.lower() else span
            for art, paras in INNER.findall(head):
                per_j[c["jurisdiction"]]["cites"] += 1
                targets, kind = resolve(art, paras.replace(" ", ""), ids)
                if not targets:
                    unresolved[f"Article {art}{paras}".strip()] += 1
                    continue
                for t in targets:
                    key = (c["chunk_id"], t)
                    if key in seen:
                        continue
                    seen.add(key)
                    edges.append({
                        "source": c["chunk_id"], "target": t, "type": "IMPLEMENTS",
                        "jurisdiction": c["jurisdiction"],
                        "citation": f"Article {art}{paras}".strip(),
                        "resolution": kind,
                        "context": re.sub(r"\s+", " ",
                                          text[max(0, m.start() - 70):m.end() + 40]).strip(),
                    })
                    per_j[c["jurisdiction"]]["edges"] += 1
                    per_j[c["jurisdiction"]]["provisions"].add(c["chunk_id"])

    print("coverage by jurisdiction")
    head = f"{'':<6}{'provisions':>12}{'linked':>8}{'%':>6}{'citations':>11}{'edges':>7}"
    print(head); print("-" * len(head))
    for j in sorted(per_j):
        total = sum(1 for c in chunks if c["jurisdiction"] == j)
        linked = len(per_j[j]["provisions"])
        print(f"{j:<6}{total:>12}{linked:>8}{linked / total:>5.0%}"
              f"{per_j[j]['cites']:>11}{per_j[j]['edges']:>7}")

    print(f"\ntotal IMPLEMENTS edges: {len(edges)}")
    print(f"distinct GDPR articles targeted: {len({e['target'] for e in edges})}")
    print(f"Directive citations excluded: {directive_only}"
          "   <- would have been wrong GDPR links")
    if unresolved:
        print(f"\ncitations naming an article with no chunk: {sum(unresolved.values())}")
        print("   ", dict(unresolved.most_common(8)))

    kinds = Counter(e["resolution"] for e in edges)
    print(f"\nhow each edge resolved: {dict(kinds)}")
    print("   'spread' is the weak kind — the cited paragraph has no chunk of its own,")
    print("   so the edge points at the article's other paragraphs instead.")

    tgt = Counter(e["target"] for e in edges)
    print(f"\nmost-linked GDPR provisions: {dict(tgt.most_common(10))}")

    print(f"\nsample edges:")
    for e in edges[:args.show]:
        print(f"\n   {e['source']} --IMPLEMENTS--> {e['target']}   ({e['citation']})")
        print(f"      ...{e['context'][:150]}...")

    OUT.write_text(json.dumps(edges, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
