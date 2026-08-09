# -*- coding: utf-8 -*-
"""Assemble a small three-tier corpus for the pilot.

The question the pilot answers is narrow: on questions whose evidence
deliberately spans two tiers, does graph retrieval beat dense retrieval? If it
does not here -- on material built to favour it -- expanding the full corpus to
three tiers has no expected payoff and the budget is better spent elsewhere.

So the corpus is built around the IMPLEMENTS edges, plus enough unlinked
material that retrieval is not trivial. Distractors are the point: with only
the linked provisions present, every method finds the gold and the comparison
measures nothing.

    python dataset/build_pilot_corpus.py
    python dataset/build_pilot_corpus.py --distractor-ratio 2.0
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
NATIONS = HERE / "corpus" / "nations" / "nations.chunks.json"
EDGES = HERE / "corpus" / "nations" / "implements_edges.json"
GDPR_TEXTS = REPO / "inference-service" / "artifacts" / "chunk_texts.json"
OUT = HERE / "corpus" / "pilot_corpus.json"
SEED = 42

# Which national law binds which institution. The corpus already carries four
# universities; source_system in the QA set was uncorrelated with any of them,
# which is one reason no question ever needed two documents.
INSTITUTIONS = {
    "tcd": "IE", "ul": "IE", "cambridge": "UK", "goettingen": "DE",
}


def tier_of(chunk_id: str) -> tuple[str, str]:
    if chunk_id.startswith("gdpr-"):
        return "regional", "EU"
    for inst, juris in INSTITUTIONS.items():
        if chunk_id.startswith(inst + "-"):
            return "institutional", juris
    return "national", "?"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distractor-ratio", type=float, default=1.3,
                    help="unlinked chunks per linked chunk, per tier")
    args = ap.parse_args()

    nations = {c["chunk_id"]: c for c in json.loads(NATIONS.read_text(encoding="utf-8"))}
    edges = json.loads(EDGES.read_text(encoding="utf-8"))
    gdpr_all = json.loads(GDPR_TEXTS.read_text(encoding="utf-8"))
    rng = random.Random(SEED)

    linked_nat = {e["source"] for e in edges}
    linked_gdpr = {e["target"] for e in edges}
    print(f"IMPLEMENTS edges: {len(edges)}  "
          f"({len(linked_nat)} national provisions -> {len(linked_gdpr)} GDPR chunks)")

    # Institutional tier: everything. It is small and it is the tier the audit
    # questions are actually about.
    inst = {k: v for k, v in gdpr_all.items() if tier_of(k)[0] == "institutional"}

    # Regional distractors: GDPR chunks nothing cites.
    gdpr_pool = [k for k in gdpr_all if k.startswith("gdpr-") and k not in linked_gdpr]
    n_gd = int(len(linked_gdpr) * args.distractor_ratio)
    gdpr_dist = rng.sample(gdpr_pool, min(n_gd, len(gdpr_pool)))

    # National distractors, drawn per jurisdiction so no country disappears.
    nat_dist = []
    by_j = {}
    for cid, c in nations.items():
        if cid not in linked_nat:
            by_j.setdefault(c["jurisdiction"], []).append(cid)
    per_j_linked = Counter(nations[c]["jurisdiction"] for c in linked_nat)
    for j, pool in by_j.items():
        want = int(per_j_linked.get(j, 0) * args.distractor_ratio)
        nat_dist.extend(rng.sample(pool, min(want, len(pool))))

    rows = []
    for cid in sorted(linked_gdpr) + sorted(gdpr_dist):
        rows.append({"chunk_id": cid, "source": "gdpr", "title": "",
                     "text": gdpr_all[cid], "tier": "regional", "jurisdiction": "EU",
                     "linked": cid in linked_gdpr})
    for cid in sorted(linked_nat) + sorted(nat_dist):
        c = nations[cid]
        rows.append({"chunk_id": cid, "source": c["source"], "title": c["title"],
                     "text": c["text"], "tier": "national",
                     "jurisdiction": c["jurisdiction"], "linked": cid in linked_nat})
    for cid, text in sorted(inst.items()):
        _, juris = tier_of(cid)
        rows.append({"chunk_id": cid, "source": cid.split("-")[0], "title": "",
                     "text": text, "tier": "institutional", "jurisdiction": juris,
                     "linked": False})

    ids = [r["chunk_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate chunk_id in the pilot corpus"

    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n{'tier':<16}{'jurisdiction':>14}{'linked':>9}{'distractor':>12}{'total':>8}")
    print("-" * 59)
    for tier in ("regional", "national", "institutional"):
        sub = [r for r in rows if r["tier"] == tier]
        for j in sorted({r["jurisdiction"] for r in sub}):
            s = [r for r in sub if r["jurisdiction"] == j]
            print(f"{tier:<16}{j:>14}{sum(1 for r in s if r['linked']):>9}"
                  f"{sum(1 for r in s if not r['linked']):>12}{len(s):>8}")
    chars = sum(len(r["text"]) for r in rows)
    print(f"\n{len(rows)} chunks, {chars:,} characters")
    print(f"estimated extraction cost: {len(rows) * 1025 / 1000:.0f}k tokens"
          "   (1,025/chunk measured on the 2Wiki run)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
