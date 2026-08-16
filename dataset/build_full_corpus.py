# -*- coding: utf-8 -*-
"""Assemble the complete three-tier corpus from the GDPR, the national Acts and
the university policies. No sampling."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference-service"))

import config   # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
NATIONS = HERE / "corpus" / "nations" / "nations.chunks.json"
GDPR_TEXTS = REPO / "inference-service" / "artifacts" / "chunk_texts.json"
OUT = HERE / "corpus" / "full_corpus.json"

# Which national law binds which institution.
INSTITUTIONS = {"tcd": "IE", "ul": "IE", "cambridge": "UK", "goettingen": "DE"}


def classify(chunk_id: str) -> tuple[str, str]:
    if chunk_id.startswith("gdpr-"):
        return "regional", "EU"
    for inst, juris in INSTITUTIONS.items():
        if chunk_id.startswith(inst + "-"):
            return "institutional", juris
    return "national", "?"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    nations = json.loads(NATIONS.read_text(encoding="utf-8"))
    # chunk_texts.json is what the GDPR build wrote: GDPR articles and recitals
    # plus the four university policies, already chunked and already extracted.
    existing = json.loads(GDPR_TEXTS.read_text(encoding="utf-8"))

    rows = []
    for cid, text in sorted(existing.items()):
        tier, juris = classify(cid)
        rows.append({"chunk_id": cid, "source": cid.split("-")[0],
                     "title": "", "text": text, "tier": tier, "jurisdiction": juris})
    for c in nations:
        rows.append({"chunk_id": c["chunk_id"], "source": c["source"],
                     "title": c.get("title", ""), "text": c["text"],
                     "tier": "national", "jurisdiction": c["jurisdiction"]})

    ids = [r["chunk_id"] for r in rows]
    dupes = [i for i, n in Counter(ids).items() if n > 1]
    if dupes:
        raise SystemExit(f"duplicate chunk_id across sources: {dupes[:8]}")
    empty = [r["chunk_id"] for r in rows if len(r["text"].strip()) < 40]
    if empty:
        raise SystemExit(f"{len(empty)} chunks are effectively empty, e.g. {empty[:5]}")

    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                              encoding="utf-8")

    print(f"{len(rows)} chunks\n")
    print(f"{'tier':<16}{'jurisdiction':>14}{'chunks':>9}{'median chars':>14}")
    print("-" * 53)
    for tier in ("regional", "national", "institutional"):
        sub = [r for r in rows if r["tier"] == tier]
        for j in sorted({r["jurisdiction"] for r in sub}):
            s = [r for r in sub if r["jurisdiction"] == j]
            lens = sorted(len(r["text"]) for r in s)
            print(f"{tier:<16}{j:>14}{len(s):>9}{lens[len(lens) // 2]:>14}")

    # Read the cache that will actually be used: it is keyed by
    # provider/model/profile, so a changed model makes every chunk billable.
    cache_file = Path(config.ARTIFACTS_DIR) / "extraction_cache.json"
    cached: set[str] = set()
    if cache_file.exists():
        blob = json.loads(cache_file.read_text(encoding="utf-8"))
        want = (f"{config.EXTRACTION_PROVIDER}:{config.EXTRACTION_MODEL}"
                f":{config.EXTRACTION_PROFILE}")
        stored = blob.get("cache_key") or ""
        if stored.count(":") == 1 and want.endswith(":legal"):
            stored += ":legal"          # pre-profile cache; see build_indexes
        if stored == want:
            cached = set(blob.get("chunks", {}))
    todo = [r for r in rows if r["chunk_id"] not in cached]
    print(f"\nextraction with {config.EXTRACTION_PROVIDER}/{config.EXTRACTION_MODEL} "
          f"(profile={config.EXTRACTION_PROFILE}, cache={cache_file})")
    print(f"   {len(cached & set(ids))} already cached, {len(todo)} to fetch")
    print(f"   estimated {len(todo) * 1025 / 1000:.0f}k tokens "
          f"(1,025/chunk measured on the 2Wiki run)")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
