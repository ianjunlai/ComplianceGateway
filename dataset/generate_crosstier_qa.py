# -*- coding: utf-8 -*-
"""Generate audit questions whose evidence deliberately spans two tiers.

The existing QA set never tests what graph retrieval is for: 126 of its 144
gold-bearing questions need only GDPR, and just 3 need two documents. So the
comparison has been measuring single-document retrieval all along.

Here the gold is fixed by construction, not by the generator. Each question is
built from one IMPLEMENTS edge -- a national provision and the GDPR article it
gives effect to -- so `gold_chunk_ids` is exactly those two chunks and the model
is asked only to write a question that needs both. That inverts the failure
found in the original set, where the generator chose the label as well as the
question and marked items UNKNOWN relative to the single clause it happened to
sample, never checking whether the corpus could answer them.

The institution is named, and it is the one actually bound by that national
law: a question about Cambridge cites the UK Act, not the German one. That is
both realistic and the condition under which a jurisdiction-aware system can
differ from a jurisdiction-blind one.

    python dataset/generate_crosstier_qa.py --n 40
    python dataset/generate_crosstier_qa.py --n 4 --dry-run   # prompts only, no API
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "inference-service"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config                                    # noqa: E402
from common.llm_clients import complete_json     # noqa: E402

CORPUS = HERE / "corpus" / "pilot_corpus.json"
EDGES = HERE / "corpus" / "nations" / "implements_edges.json"
OUT = HERE / "crosstier_qa.json"
SEED = 42

# Which jurisdiction a chunk belongs to, read off its id. extract_citations.py
# emits no jurisdiction field -- it works over any corpus and has no concept of
# one -- so the prefix is the only carrier.
JURIS_PREFIX = {"ie": "IE", "uk": "UK", "de": "DE"}

# Which institution each national law binds. Only these four appear in the
# corpus, so a question naming any other would have no institutional tier.
INSTITUTIONS = {
    "IE": [("Trinity College Dublin", "tcd"), ("University of Limerick", "ul")],
    "UK": [("the University of Cambridge", "cambridge")],
    "DE": [("Georg-August-Universität Göttingen", "goettingen")],
}
LAW_NAME = {"IE": "the Irish Data Protection Act 2018",
            "UK": "the UK Data Protection Act 2018",
            "DE": "the German Federal Data Protection Act (BDSG)"}

PROMPT = """You are writing audit requests for a GDPR compliance system used by universities.

Write ONE realistic compliance question that a data protection officer at {institution}
would submit, and that CANNOT be answered without BOTH provisions below. The national
provision gives effect to the GDPR article; a correct answer needs the general rule and
the national detail together.

Requirements:
- name {institution} explicitly
- concern a concrete operation on student or staff personal data (transcripts, grades,
  application records, research data, attendance, references)
- do not quote either provision, and do not mention article or section numbers
- answerable as APPROVE (the operation is permitted) or DENY (it is not)
- AT MOST 60 WORDS, two or three sentences. State the situation and ask. Do not
  narrate background, motive or mitigating detail.

Then describe the scenario your question encodes, as separate fields. These are NOT
added to the question text -- the question must read as a natural request from a
colleague. They record what a correct answer would have to recognise.

GDPR {gdpr_id}:
{gdpr_text}

{law} {nat_id} — {nat_title}:
{nat_text}

Return STRICT JSON:
{{"question": "...",
  "decision": "APPROVE|DENY",
  "rationale": "one sentence naming what each provision contributes",
  "purpose": "why the data is being processed, a short noun phrase",
  "data_category": "what kind of personal data, e.g. academic records, health data",
  "data_subject": "STUDENT|STAFF|APPLICANT|ALUMNUS|OTHER",
  "recipient": "who receives the data; 'internal' if it stays within the institution",
  "cross_border": true or false,
  "legal_basis": "the lawful basis the operation would rely on, in the Regulation's own
                  vocabulary (consent, contract, legal obligation, vital interests,
                  public task, legitimate interests), or 'none' if there is none"}}
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--edges", default=str(EDGES))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    out_path = Path(args.out)

    corpus = {c["chunk_id"]: c
              for c in json.loads(Path(args.corpus).read_text(encoding="utf-8"))}
    edges = [e for e in json.loads(Path(args.edges).read_text(encoding="utf-8"))
             if e["source"] in corpus and e["target"] in corpus]

    # A citation file may carry every edge kind; only a national provision
    # giving effect to a GDPR article yields the two-tier pair this question
    # form is built on. Institutional sources are excluded too -- the prompt
    # needs a national law to name, and a university policy has none.
    for e in edges:
        e.setdefault("jurisdiction",
                     JURIS_PREFIX.get(e["source"].split("-")[0], ""))
    edges = [e for e in edges
             if e.get("type", "IMPLEMENTS") == "IMPLEMENTS" and e["jurisdiction"]]

    # 'spread' edges point at a paragraph other than the one cited, so the pair
    # may not actually belong together. Excluded: a question built on a wrong
    # pair would produce a gold label the corpus does not support.
    edges = [e for e in edges if e.get("resolution") in ("exact", "whole")]
    print(f"{len(edges)} usable edges (national IMPLEMENTS GDPR, exact/whole, "
          f"both ends in {Path(args.corpus).name})")

    rng = random.Random(SEED)
    rng.shuffle(edges)
    # One question per national provision keeps the set from concentrating on
    # whichever provision cites the most articles.
    seen_src, picked = set(), []
    for e in edges:
        if e["source"] in seen_src:
            continue
        seen_src.add(e["source"])
        picked.append(e)
        if len(picked) >= args.n:
            break
    print(f"selected {len(picked)}: {dict(Counter(e['jurisdiction'] for e in picked))}\n")

    out, failures = [], 0
    for i, e in enumerate(picked, 1):
        juris = e["jurisdiction"]
        inst_name, inst_slug = rng.choice(INSTITUTIONS[juris])
        nat, gd = corpus[e["source"]], corpus[e["target"]]
        prompt = PROMPT.format(
            institution=inst_name, law=LAW_NAME[juris],
            gdpr_id=e["target"], gdpr_text=gd["text"][:2500],
            nat_id=e["source"], nat_title=nat.get("title", ""),
            nat_text=nat["text"][:2500])

        if args.dry_run:
            print("=" * 70)
            print(prompt[:1100])
            continue

        try:
            data, _ = complete_json(config.QA_GENERATION_PROVIDER,
                                    config.QA_GENERATION_MODEL, prompt, max_tokens=1200)
        except Exception as exc:            # noqa: BLE001
            print(f"   {i}/{len(picked)} FAILED {e['source']}: {exc!r}")
            failures += 1
            continue
        q = (data.get("question") or "").strip()
        dec = (data.get("decision") or "").strip().upper()
        if not q or dec not in ("APPROVE", "DENY"):
            print(f"   {i}/{len(picked)} malformed for {e['source']}: {str(data)[:90]}")
            failures += 1
            continue

        out.append({
            "query_id": f"ct-{i:03d}",
            "query_text": q,
            "source_system": inst_slug,
            "hop_type": "cross_tier",
            "gold_decision": dec,
            # Fixed by construction: the two provisions the question was built
            # from. The generator never chooses these.
            "gold_chunk_ids": sorted([e["source"], e["target"]]),
            "jurisdiction": juris,
            "generator_rationale": (data.get("rationale") or "").strip(),
            "citation": e["citation"],
            # The scenario behind the question, as fields rather than prose.
            # Deliberately NOT concatenated into query_text: spelling the data
            # category and legal basis out in the question would hand keywords
            # to dense retrieval and narrow the gap being measured, so these
            # feed the judge and the analysis only.
            "scenario": {
                "purpose": (data.get("purpose") or "").strip(),
                "data_category": (data.get("data_category") or "").strip(),
                "data_subject": (data.get("data_subject") or "").strip().upper(),
                "recipient": (data.get("recipient") or "").strip(),
                "cross_border": bool(data.get("cross_border")),
                "legal_basis": (data.get("legal_basis") or "").strip().lower(),
            },
        })
        if i % 10 == 0:
            print(f"   {i}/{len(picked)}")

    if args.dry_run:
        return
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {len(out)} questions ({failures} failed) -> {out_path}")
    print(f"   decisions : {dict(Counter(q['gold_decision'] for q in out))}")
    print(f"   by juris  : {dict(Counter(q['jurisdiction'] for q in out))}")
    print(f"   every question has exactly 2 gold chunks spanning national+GDPR: "
          f"{all(len(q['gold_chunk_ids']) == 2 for q in out)}")
    print(f"   subjects  : {dict(Counter(q['scenario']['data_subject'] for q in out))}")
    print(f"   bases     : {dict(Counter(q['scenario']['legal_basis'] for q in out))}")
    print(f"   cross-border: {sum(q['scenario']['cross_border'] for q in out)}")
    # A field the model left blank is not a failed question, but it is one the
    # rubric cannot score, so the count belongs in the log rather than buried.
    thin = [q["query_id"] for q in out
            if not all(str(v) for v in q["scenario"].values())]
    if thin:
        print(f"   INCOMPLETE scenario on {len(thin)}: {thin[:8]}")
    # Question length is an experimental condition, not a style preference. The
    # scenario fields tempt the generator into writing a case study, and a
    # longer question gives dense retrieval more surface to match on -- which
    # would narrow the very gap this compares. The pre-scenario set ran a median
    # of 36 words; anything far above that is not comparable to it.
    words = sorted(len(q["query_text"].split()) for q in out)
    median = words[len(words) // 2]
    print(f"   length    : median {median} words, max {words[-1]} (target <= 60)")
    if median > 60:
        print("   WARNING questions are much longer than the pre-scenario set "
              "(median 36 words); dense retrieval is advantaged and the "
              "comparison to earlier runs is weakened")


if __name__ == "__main__":
    main()
