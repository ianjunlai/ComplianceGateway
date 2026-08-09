# -*- coding: utf-8 -*-
"""Extract every citation edge the corpus contains, of every kind.

Supersedes extract_implements.py (national -> GDPR) and
extract_gdpr_citations.py (inside GDPR), which between them covered two of the
five kinds present. A survey of the corpus found the missing three are the
larger part: statutes cite their own sections far more often than they cite
anything external -- 204 references inside the UK Act alone.

Two relationship types come out, and the distinction is the experiment:

    IMPLEMENTS   crosses a tier. A national provision giving effect to a GDPR
                 article, or a university policy invoking either. This is the
                 structure a jurisdiction-blind retriever cannot see.
    CITES        stays inside one instrument. Dense, and the reason a citation
                 graph is connected at all.

Regex, not an LLM: the forms are formulaic, the output is checkable against the
chunk ids that exist, and a pattern either matches or does not. Every citation
naming a different instrument (Directive 2016/680, Directive 2002/58/EC) is
excluded explicitly -- their article numbering overlaps GDPR's completely, so a
missed exclusion silently produces a confident wrong edge.

    python dataset/extract_citations.py --corpus corpus/pilot_corpus.json
    python dataset/extract_citations.py --corpus corpus/full_corpus.json
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent

# "the Data Protection Regulation" (IE), "the UK GDPR" (UK), "Regulation (EU)
# 2016/679" (DE), "the EU General Data Protection Regulation" (Göttingen) --
# four instruments-by-name for one instrument.
REGULATION = (r"(?:the\s+)?(?:EU\s+)?(?:General\s+)?"
              r"(?:Data Protection Regulation|UK GDPR|GDPR"
              r"|Regulation\s*\(EU\)\s*(?:No\.?\s*)?2016/679)")
# Named instruments that are NOT the GDPR. Their articles are numbered the same.
#
# The separator belongs INSIDE the lookahead. Written as `\(EU\)\s*(?!2016/679)`
# the `\s*` backtracks to zero width, the lookahead then tests against a leading
# space, does not match, and the negative succeeds -- so "Regulation (EU)
# 2016/679" is excluded as foreign, which is the opposite of the intent. Only
# the German Act writes GDPR citations in that formal style (the UK Act says
# "the UK GDPR", the Irish "the Data Protection Regulation"), so the hole
# removed one jurisdiction's cross-tier edges entirely and left the other two
# looking correct.
FOREIGN = (r"(?:Directive|Regulation\s*\(EU\)(?!\s*(?:No\.?\s*)?2016/679)|Decision"
           r"|Treaty|Charter|Convention)")

_SUBS = r"(?:\s*\(\d+[a-z]?\))*"
_TITLE = r"(?:\s*\([A-Z][^)]{2,40}\))?"
_TAIL = r"(?:\s*(?:,|and|or|to)\s*\(?\d{0,3}[a-z]?\)?" + _SUBS + r")*"
_ART = r"(\d{1,3}[A-Z]?)" + _SUBS + _TITLE + _TAIL

# "Article 9(2)(b) of the Regulation" — an explicit external reference.
ART_EXPLICIT = re.compile(r"Articles?\s+" + _ART + r"(?:\s*(?:,|and|or)\s*" + _ART + r")*"
                          r"\s+of\s+" + REGULATION, re.I)
# "Article 6(1)" with no instrument named. Inside GDPR that means GDPR.
ART_BARE = re.compile(r"Articles?\s+" + _ART + r"(?:\s*(?:,|and|or)\s*" + _ART + r")*", re.I)
ART_FOREIGN = re.compile(r"Articles?\s+" + _ART + r"[^.]{0,30}?\s+of\s+(?:the\s+)?" + FOREIGN, re.I)
# "Article 6(1)(1)(a) GDPR", "Article 6 Paragraph 1 Subparagraph 1 lit. b) GDPR"
# -- the instrument trails the article instead of following "of". This is how
# the university privacy notices cite, and only how they cite; the statutes all
# use the "of the ..." form. Applied to the institutional tier alone, because
# the bounded gap would otherwise let an unrelated later "GDPR" pull in a
# statute's self-reference ("under Article 15 or under section 91 ... GDPR").
ART_TRAILING = re.compile(r"Articles?\s+" + _ART + r"[^.;]{0,60}?\s*" + REGULATION, re.I)

# Each statute names its own provisions its own way.
SELF_REF = {
    "ie_dpa_2018": (re.compile(r"\bsections?\s+(\d{1,3})\b", re.I), "ie-dpa-s"),
    "uk_dpa_2018": (re.compile(r"\bsections?\s+(\d{1,3}[A-Z]?)\b", re.I), "uk-dpa-s"),
    "de_bdsg": (re.compile(r"\bSections?\s+(\d{1,3}[a-z]?)\b", re.I), "de-bdsg-s"),
}
INNER = re.compile(r"(?<![(\w])(\d{1,3}[A-Z]?)((?:\s*\(\d+[a-z]?\))*)")


def resolve_article(article: str, paras: str, ids: set[str]) -> tuple[list[str], str]:
    """A citation to a GDPR article -> the chunk ids that exist for it."""
    first = re.match(r"\s*\((\d+)", paras)
    if first:
        sub = f"gdpr-art-{article}-{first.group(1)}"
        if sub in ids:
            return [sub], "exact"
    whole = f"gdpr-art-{article}"
    if whole in ids:
        return [whole], "whole"
    parts = sorted(i for i in ids if re.fullmatch(rf"gdpr-art-{re.escape(article)}-\d+", i))
    return parts, "spread"


# Words that introduce a subdivision spelled out rather than bracketed. Göttingen
# writes "Article 6 Paragraph 1 Subparagraph 1 lit. b) GDPR", where 1 and 1 are
# subdivisions of Article 6 -- read as article numbers they yield a confident
# edge to Article 1.
_QUALIFIER = re.compile(r"\b(?:paragraph|subparagraph|sentence|point|lit\.?|no\.?)\b", re.I)


def articles_in(span: str) -> list[tuple[str, str]]:
    """Pull each article out of a citation span, and only articles.

    The negative lookbehind is what stops "Article 35(4) and (5)" also yielding
    an Article 5: a paragraph number inside brackets is a bare digit and looks
    identical to an article number.
    """
    head = span[:span.lower().rfind(" of ")] if " of " in span.lower() else span
    qualifier = _QUALIFIER.search(head)
    if qualifier:
        head = head[:qualifier.start()]
    return INNER.findall(head)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="corpus/pilot_corpus.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--drop-spread", action="store_true",
                    help="omit edges whose cited paragraph has no chunk of its own")
    args = ap.parse_args()

    corpus_path = HERE / args.corpus if not Path(args.corpus).is_absolute() else Path(args.corpus)
    chunks = json.loads(corpus_path.read_text(encoding="utf-8"))
    by_id = {c["chunk_id"]: c for c in chunks}
    ids = set(by_id)
    out_path = Path(args.out) if args.out else corpus_path.with_name(
        corpus_path.stem.replace("_corpus", "") + "_citations.json")

    edges, dropped_foreign, unresolved = [], 0, Counter()

    def add(src, tgt, kind, citation, resolution):
        if tgt not in ids or tgt == src:
            return False
        edges.append({"source": src, "target": tgt, "type": kind,
                      "citation": citation, "resolution": resolution})
        return True

    for c in chunks:
        cid, text, tier = c["chunk_id"], c["text"], c.get("tier", "")
        seen = set()

        # Mask references to other instruments before anything else reads the
        # text: "Article 22(3) of the Directive" must not become a GDPR edge.
        masked = ART_FOREIGN.sub(lambda m: " " * len(m.group(0)), text)
        dropped_foreign += len(ART_FOREIGN.findall(text))

        # --- references to a GDPR article -----------------------------------
        # Inside GDPR a bare "Article 6" means GDPR; elsewhere the instrument
        # must be named, or the reference is to the citing statute's own text.
        if tier == "regional":
            pattern = ART_BARE
        elif tier == "institutional":
            pattern = ART_TRAILING
        else:
            pattern = ART_EXPLICIT
        for m in pattern.finditer(masked):
            for art, paras in articles_in(m.group(0)):
                targets, resolution = resolve_article(art, paras, ids)
                if not targets:
                    unresolved[f"Article {art}"] += 1
                    continue
                if args.drop_spread and resolution == "spread":
                    continue
                kind = "CITES" if tier == "regional" else "IMPLEMENTS"
                for t in targets:
                    if (cid, t) not in seen and add(cid, t, kind,
                                                    f"Article {art}{paras}".strip(),
                                                    resolution):
                        seen.add((cid, t))

        # --- a statute citing its own sections -------------------------------
        rule = SELF_REF.get(c.get("source", ""))
        if rule:
            rx, prefix = rule
            for num in rx.findall(text):
                t = f"{prefix}{num}"
                if (cid, t) not in seen and add(cid, t, "CITES", f"section {num}", "whole"):
                    seen.add((cid, t))

    kinds = Counter(e["type"] for e in edges)
    per_pair = Counter((by_id[e["source"]].get("tier"), by_id[e["target"]].get("tier"))
                       for e in edges)
    touched = {e["source"] for e in edges} | {e["target"] for e in edges}

    out_path.write_text(json.dumps(edges, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"corpus {len(chunks)} chunks -> {len(edges)} citation edges")
    print(f"   by type      : {dict(kinds)}")
    print(f"   by tier pair : {dict(per_pair)}")
    print(f"   resolution   : {dict(Counter(e['resolution'] for e in edges))}")
    print(f"   chunks touched: {len(touched)} ({len(touched) / len(chunks):.0%})")
    print(f"   foreign-instrument references excluded: {dropped_foreign}")
    if unresolved:
        print(f"   citations with no chunk to point at: {sum(unresolved.values())}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
