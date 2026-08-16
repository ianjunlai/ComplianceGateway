# -*- coding: utf-8 -*-
"""Generate the evaluation questions in three strata: cross-tier, single-tier and
unanswerable. Gold is the seed provision alone."""
import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "inference-service"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config                                    # noqa: E402
from common.llm_clients import complete_json     # noqa: E402
from pipeline.attachment import _RESOLUTION_ORDER  # noqa: E402

CORPUS = HERE / "corpus" / "full_corpus.json"
EDGES = HERE / "corpus" / "full_citations.json"
OUT = HERE / "qa_v2.json"
SEED = 42

# Which institutions sit in which state.
INSTITUTIONS = {
    "IE": ["tcd", "ul"],
    "UK": ["cambridge"],
    "DE": ["goettingen"],
}
JURIS_PREFIX = {"ie": "IE", "uk": "UK", "de": "DE"}
LAW_NAME = {"IE": "the Irish Data Protection Act 2018",
            "UK": "the UK Data Protection Act 2018",
            "DE": "the German Federal Data Protection Act (BDSG)"}

# Provisions that exist but leave a detail to secondary legislation.
DEFERRAL = re.compile(
    r"(may|shall) by regulations|as (may be )?prescribed|regulations under|"
    r"the (Minister|Secretary of State) may|specified in regulations|by order|"
    r"(Federal )?Ministry may|legal ordinance|statutory instrument|"
    r"by ordinance|further details? (are|shall be)", re.I)

_COMMON_RULES = """
Rules for the question text:
- Do NOT name the university, and do not write "our university" or similar.
  Write it as the request itself: what is being done, to whose data, for what.
- Do NOT name the recipient organisation either; describe it generically
  ("a partner university", "a research funder", "a public authority").
- Do NOT mention article, section, schedule or paragraph numbers.
- Do NOT quote the provision.
- At most 60 words, two or three sentences.
"""

# Recorded as fields, never written into the question.
_TARGET_FIELDS = """  "target_kind": "internal|another_institution|public_authority|third_country|none",
  "target_jurisdiction": "IE|UK|DE|EEA|non-EEA|none",
"""


def _attachment_preview(seed_id: str, nbrs: dict, corpus: dict,
                        cap: int) -> list[tuple[str, str, str]]:
    """What the retrieval layer would attach to this provision."""
    seen, out = {seed_id}, []
    for tgt, typ, res in sorted(nbrs.get(seed_id, []),
                                key=lambda r: (_RESOLUTION_ORDER.get(r[2], 3), r[0])):
        if tgt in seen or tgt not in corpus:
            continue
        seen.add(tgt)
        out.append((tgt, typ, res))
        if len(out) >= cap:
            break
    return out


def _render_attachments(picks: list[tuple[str, str, str]], corpus: dict) -> str:
    if not picks:
        return "(this provision cites nothing else in the corpus)"
    return "\n\n".join(
        f"[{cid}] — {'gives effect to / implemented by' if typ == 'IMPLEMENTS' else 'cross-reference within the same Act'}\n"
        f"{corpus[cid]['text'][:1800]}"
        for cid, typ, _ in picks)

PROMPT_CROSSTIER = """You are writing audit requests for a GDPR compliance system used by universities.

Write ONE realistic compliance question that CANNOT be answered from the main
provision alone: it must also turn on the GDPR article that provision gives
effect to. A correct answer needs the general rule and the national detail
together.
{rules}
The operation must concern student or staff personal data (transcripts, grades,
application records, research data, attendance, references).
Answerable as APPROVE (permitted) or DENY (not permitted).

MAIN PROVISION — {law} {nat_id} — {nat_title}:
{nat_text}

PROVISIONS THE SYSTEM WILL SUPPLY ALONGSIDE IT (these are the ones this
provision cites; the retrieval layer attaches them automatically, so you may
rely on them):
{attachments}

Return STRICT JSON:
{{"question": "...", "decision": "APPROVE|DENY",
{target}  "rationale": "one sentence naming what each provision contributes",
  "uses_attached": ["chunk ids from the list above that a correct answer needs"]}}
"""

PROMPT_SINGLE = """You are writing audit requests for a GDPR compliance system used by universities.

Write ONE realistic compliance question that is answerable from the main
provision below, together with the provisions listed after it if they are
relevant. Do not require anything beyond those.
{rules}
The operation must concern student or staff personal data.
Answerable as APPROVE (permitted) or DENY (not permitted).

MAIN PROVISION — {law} {nat_id} — {nat_title}:
{nat_text}

PROVISIONS THE SYSTEM WILL SUPPLY ALONGSIDE IT:
{attachments}

Return STRICT JSON:
{{"question": "...", "decision": "APPROVE|DENY",
{target}  "rationale": "one sentence on how these provisions settle it",
  "uses_attached": ["chunk ids from the list above that a correct answer needs, [] if none"]}}
"""

PROMPT_UNANSWERABLE = """You are writing audit requests for a GDPR compliance system used by universities.

The main provision below leaves a specific detail to secondary legislation, or
does not specify it at all. Write ONE realistic compliance question that turns
on exactly that missing detail, so it CANNOT be answered even with the
provisions listed after it.
{rules}
The operation must concern student or staff personal data. A correct system
should decline to decide rather than guess.

MAIN PROVISION — {law} {nat_id} — {nat_title}:
{nat_text}

PROVISIONS THE SYSTEM WILL SUPPLY ALONGSIDE IT (check that none of them
supplies the missing detail):
{attachments}

Return STRICT JSON:
{{"question": "...",
{target}  "missing": "the specific detail no provision above supplies",
  "why_unanswerable": "one sentence explaining why the text cannot settle it"}}
"""


def _ask(prompt: str) -> dict | None:
    try:
        data, _ = complete_json(config.QA_GENERATION_PROVIDER,
                                config.QA_GENERATION_MODEL, prompt, max_tokens=1200)
        return data
    except Exception as exc:                                    # noqa: BLE001
        print(f"      FAILED: {exc!r}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-crosstier", type=int, default=50)
    ap.add_argument("--n-single", type=int, default=25)
    ap.add_argument("--n-unanswerable", type=int, default=25)
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--edges", default=str(EDGES))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--dry-run", action="store_true",
                    help="report the candidate pools and print one prompt per stratum")
    args = ap.parse_args()

    corpus = {c["chunk_id"]: c
              for c in json.loads(Path(args.corpus).read_text(encoding="utf-8"))}
    edges = json.loads(Path(args.edges).read_text(encoding="utf-8"))
    rng = random.Random(SEED)

    national = [c for c in corpus.values() if c["tier"] == "national"]

    # Every citation out of a chunk, both directions, matching the undirected
    # pattern the retrieval layer uses.
    nbrs: dict[str, list[tuple[str, str, str]]] = {}
    for e in edges:
        if e["source"] in corpus and e["target"] in corpus:
            nbrs.setdefault(e["source"], []).append(
                (e["target"], e["type"], e["resolution"]))
            nbrs.setdefault(e["target"], []).append(
                (e["source"], e["type"], e["resolution"]))

    def preview(cid: str):
        return _attachment_preview(cid, nbrs, corpus, config.ATTACH_PER_CHUNK)

    # Cross-tier: one usable IMPLEMENTS edge per seed provision.
    ct_edges = [e for e in edges
                if e["type"] == "IMPLEMENTS"
                and e["resolution"] in ("exact", "whole")
                and e["source"] in corpus and e["target"] in corpus
                and corpus[e["source"]]["tier"] == "national"]
    rng.shuffle(ct_edges)
    by_source: dict[str, list[dict]] = {}
    for e in ct_edges:
        by_source.setdefault(e["source"], []).append(e)
    ct_pool, crowded_out = [], 0
    for source, candidates in by_source.items():
        deliverable = {c for c, _, _ in preview(source)}
        # A provision may give effect to several articles.
        survivors = [e for e in candidates if e["target"] in deliverable]
        if survivors:
            ct_pool.append(survivors[0])
        else:
            crowded_out += 1
    rng.shuffle(ct_pool)

    # Single: a national provision with no usable cross-tier citation, so the
    # question does not need a GDPR article.
    cross_sources = {e["source"] for e in ct_edges}
    single_pool = [c for c in national if c["chunk_id"] not in cross_sources
                   and len(c["text"].split()) >= 60]
    rng.shuffle(single_pool)

    unans_pool = [c for c in national if DEFERRAL.search(c["text"])]
    rng.shuffle(unans_pool)

    print(f"candidate pools: cross-tier {len(ct_pool)}, single {len(single_pool)}, "
          f"unanswerable {len(unans_pool)}")
    print(f"  cross-tier seeds dropped because the cap would hide the partner: "
          f"{crowded_out}")
    print(f"  unanswerable by jurisdiction: "
          f"{dict(Counter(c['jurisdiction'] for c in unans_pool))}")
    for need, have, label in ((args.n_crosstier, len(ct_pool), "cross-tier"),
                              (args.n_single, len(single_pool), "single"),
                              (args.n_unanswerable, len(unans_pool), "unanswerable")):
        if need > have:
            raise SystemExit(f"need {need} {label} questions but only {have} candidates")

    out: list[dict] = []
    failures = 0

    def emit(stratum: str, seed_chunk: dict, data: dict, decision: str,
             extra: dict) -> None:
        juris = seed_chunk["jurisdiction"]
        row = {
            "query_id": f"q2-{len(out) + 1:03d}",
            "query_text": (data.get("question") or "").strip(),
            # The gateway knows who sent the request; the question text does not
            # say. Retrieval scope is derived from this field.
            "source_system": rng.choice(INSTITUTIONS[juris]),
            "jurisdiction": juris,
            "hop_type": stratum,
            "gold_decision": decision,
            # Seed only. The retrieval layer attaches what the seed cites, so
            # reaching the seed is what the metric needs to measure.
            "gold_chunk_ids": [seed_chunk["chunk_id"]] if stratum != "unanswerable" else [],
        }
        # The recipient, recorded but never used to widen the retrieval
        # scope: the obligations under audit belong to the sender.
        row["target"] = {
            "kind": (data.get("target_kind") or "none").strip().lower(),
            "jurisdiction": (data.get("target_jurisdiction") or "none").strip(),
        }
        row.update(extra)
        out.append(row)

    # ---- cross-tier -----------------------------------------------------
    print(f"\ncross-tier ({args.n_crosstier})")
    for e in ct_pool:
        if sum(1 for r in out if r["hop_type"] == "cross_tier") >= args.n_crosstier:
            break
        nat = corpus[e["source"]]
        picks = preview(e["source"])
        prompt = PROMPT_CROSSTIER.format(
            rules=_COMMON_RULES, target=_TARGET_FIELDS,
            law=LAW_NAME[nat["jurisdiction"]],
            nat_id=e["source"], nat_title=nat.get("title", ""),
            nat_text=nat["text"][:2500],
            attachments=_render_attachments(picks, corpus))
        if args.dry_run:
            print(prompt[:900]); break
        data = _ask(prompt)
        dec = (data or {}).get("decision", "").strip().upper()
        if not data or dec not in ("APPROVE", "DENY") or not data.get("question"):
            failures += 1
            continue
        emit("cross_tier", nat, data, dec,
             {"partner_chunk_id": e["target"], "citation": e["citation"],
              "attached": [c for c, _, _ in picks],
              "uses_attached": data.get("uses_attached") or [],
              "generator_rationale": (data.get("rationale") or "").strip()})
        if len(out) % 10 == 0:
            print(f"   {len(out)} done")

    # ---- single ---------------------------------------------------------
    print(f"\nsingle ({args.n_single})")
    for c in single_pool:
        if sum(1 for r in out if r["hop_type"] == "single") >= args.n_single:
            break
        picks = preview(c["chunk_id"])
        prompt = PROMPT_SINGLE.format(
            rules=_COMMON_RULES, target=_TARGET_FIELDS,
            law=LAW_NAME[c["jurisdiction"]],
            nat_id=c["chunk_id"], nat_title=c.get("title", ""),
            nat_text=c["text"][:2500],
            attachments=_render_attachments(picks, corpus))
        if args.dry_run:
            print(prompt[:900]); break
        data = _ask(prompt)
        dec = (data or {}).get("decision", "").strip().upper()
        if not data or dec not in ("APPROVE", "DENY") or not data.get("question"):
            failures += 1
            continue
        emit("single", c, data, dec,
             {"attached": [x for x, _, _ in picks],
              "uses_attached": data.get("uses_attached") or [],
              "generator_rationale": (data.get("rationale") or "").strip()})

    # ---- unanswerable ---------------------------------------------------
    print(f"\nunanswerable ({args.n_unanswerable})")
    for c in unans_pool:
        if sum(1 for r in out if r["hop_type"] == "unanswerable") >= args.n_unanswerable:
            break
        picks = preview(c["chunk_id"])
        prompt = PROMPT_UNANSWERABLE.format(
            rules=_COMMON_RULES, target=_TARGET_FIELDS,
            law=LAW_NAME[c["jurisdiction"]],
            nat_id=c["chunk_id"], nat_title=c.get("title", ""),
            nat_text=c["text"][:2500],
            attachments=_render_attachments(picks, corpus))
        if args.dry_run:
            print(prompt[:900]); break
        data = _ask(prompt)
        if not data or not data.get("question") or not data.get("missing"):
            failures += 1
            continue
        emit("unanswerable", c, data, "UNKNOWN",
             {"missing_detail": (data.get("missing") or "").strip(),
              "why_unanswerable": (data.get("why_unanswerable") or "").strip(),
              "attached": [x for x, _, _ in picks],
              "provision_asked_about": c["chunk_id"]})

    if args.dry_run:
        return

    # ---- validation -----------------------------------------------------
    # "the university" is fine and is left alone: it names no institution, and
    # as a query mention it is high-degree and therefore harmless. What must not
    # appear is a specific institution, which would reintroduce the seeding
    # problem the redesign exists to remove.
    inst_names = re.compile(
        r"trinity|cambridge|limerick|göttingen|goettingen|georg-august|"
        r"\bour university\b|\bthis university\b", re.I)
    # Schedules and paragraphs are clause references too.
    clause_ref = re.compile(
        r"\b(article|articles|section|sections|schedule|schedules|sch\.?|"
        r"paragraph|paragraphs|para\.?|regulation)\s+\d", re.I)
    dropped = []
    for r in list(out):
        why = None
        if inst_names.search(r["query_text"]):
            why = "names an institution"
        elif clause_ref.search(r["query_text"]):
            why = "cites a clause number"
        elif r["hop_type"] != "unanswerable" and \
                r["gold_chunk_ids"][0] not in corpus:
            why = "gold does not resolve"
        if why:
            dropped.append((r["query_id"], why))
            out.remove(r)

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    words = sorted(len(r["query_text"].split()) for r in out)
    print(f"\nwrote {len(out)} questions ({failures} generation failures, "
          f"{len(dropped)} dropped in validation) -> {args.out}")
    if dropped:
        for qid, why in dropped[:8]:
            print(f"   dropped {qid}: {why}")
    print(f"   by stratum   : {dict(Counter(r['hop_type'] for r in out))}")
    print(f"   by decision  : {dict(Counter(r['gold_decision'] for r in out))}")
    print(f"   by requester : {dict(Counter(r['source_system'] for r in out))}")
    print(f"   median length: {words[len(words) // 2]} words")
    print(f"   gold size    : "
          f"{dict(Counter(len(r['gold_chunk_ids']) for r in out))}")
    print(f"   target kind  : "
          f"{dict(Counter(r['target']['kind'] for r in out))}")
    uses = [len(r.get('uses_attached', [])) for r in out]
    print(f"   questions relying on an attached provision: "
          f"{sum(1 for n in uses if n)} of {len(out)}")


if __name__ == "__main__":
    main()
