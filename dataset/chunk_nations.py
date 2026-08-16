# -*- coding: utf-8 -*-
"""Cut the three national Acts into provision-level chunks, one parser each
because the source formats differ."""
import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
NATIONS = HERE / "corpus" / "nations"
OUT = NATIONS / "nations.chunks.json"

# A provision shorter than this is a heading the parser mistook for a section,
# or a repealed one left as a stub.
MIN_CHARS = 80


def _clean(s: str) -> str:
    return re.sub(r"[ \t]+", " ", re.sub(r"\s*\n\s*", " ", s)).strip()


# --------------------------------------------------------------- Ireland
def parse_ireland(path: Path) -> list[dict]:
    """Irish Statute Book web copy."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    # The Schedules restart at 1.
    body_end = next((i for i, l in enumerate(lines)
                     if re.match(r"(?i)^\s*schedule\s+1\b", l)), len(lines))

    def sections(lo, hi, prefix, sched=None):
        starts = []
        last = 0
        for i in range(lo, hi):
            m = re.match(r"^\s*(\d{1,3})\.\s+\S", lines[i])
            if not m:
                continue
            n = int(m.group(1))
            # A section number that goes backwards belongs to another statute
            # quoted inside this one, not to a second section 1.
            if n <= last:
                continue
            last = n
            title = ""
            for j in range(i - 1, max(lo - 1, i - 4), -1):
                if lines[j].strip():
                    title = lines[j].strip()
                    break
            starts.append((i, m.group(1), title))
        out = []
        for k, (i, num, title) in enumerate(starts):
            end = starts[k + 1][0] - 1 if k + 1 < len(starts) else hi
            out.append({
                "chunk_id": f"{prefix}{num}", "source": "ie_dpa_2018",
                "title": title, "text": _clean("\n".join(lines[i:end])),
                "jurisdiction": "IE", "tier": "national",
                **({"schedule": sched} if sched else {}),
            })
        return out

    out = sections(0, body_end, "ie-dpa-s")
    sched_starts = [i for i in range(body_end, len(lines))
                    if re.match(r"(?i)^\s*schedule\s+(\d+)", lines[i])]
    for n, start in enumerate(sched_starts, 1):
        stop = sched_starts[n] if n < len(sched_starts) else len(lines)
        out.extend(sections(start, stop, f"ie-dpa-sch{n}-p", sched=n))
    return out


# --------------------------------------------------------------- Germany
def parse_germany(path: Path) -> list[dict]:
    """gesetze-im-internet official English translation."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    starts = []
    for i, line in enumerate(lines):
        m = re.match(r"^\s*Section\s+(\d{1,3}[a-z]?)\s*$", line)
        if m:
            title = next((lines[j].strip() for j in range(i + 1, min(len(lines), i + 4))
                          if lines[j].strip()), "")
            starts.append((i, m.group(1), title))

    out = []
    for k, (i, num, title) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
        body = _clean("\n".join(lines[i:end]))
        out.append({
            "chunk_id": f"de-bdsg-s{num}", "source": "de_bdsg",
            "title": title, "text": body,
            "jurisdiction": "DE", "tier": "national",
        })
    return out


# --------------------------------------------------------------- UK
def parse_uk(path: Path) -> list[dict]:
    """legislation.gov.uk XML, point-in-time snapshot."""
    root = ET.parse(path).getroot()
    nsuri = root.tag.split("}")[0].strip("{") if "}" in root.tag else ""

    def q(name):
        return f"{{{nsuri}}}{name}" if nsuri else name

    def identify(group):
        """The stable identifier is P1's id, not its Pnumber."""
        for p1 in group.findall(q("P1")):
            ident = p1.get("id") or ""
            m = re.match(r"section-(\d+[A-Za-z]?)$", ident)
            if m:
                return f"uk-dpa-s{m.group(1)}", None
            m = re.match(r"schedule-(\w+)-paragraph-(\d+[A-Za-z]?)$", ident)
            if m:
                return f"uk-dpa-sch{m.group(1)}-p{m.group(2)}", m.group(1)
            if ident:
                return "uk-dpa-" + re.sub(r"[^A-Za-z0-9]+", "-", ident), None
        # Provisions this Act inserts into OTHER statutes carry no id; keying
        # them by Pnumber collided them with the real sections 2 and 61.
        return None, None

    out = []
    for group in root.iter(q("P1group")):
        cid, sched = identify(group)
        if not cid:
            continue
        title = ""
        for t in group.findall(q("Title")):
            title = _clean("".join(t.itertext()))
            break
        body = " ".join("".join(t.itertext()) for t in group.iter(q("Text")))
        # Repealed text is a run of dot leaders. Strip the leaders only: the
        # rest of the section is still law.
        body = _clean(re.sub(r"(?:\.\s){4,}\.?", "", body))
        if not re.search(r"[A-Za-z]{3}", body):
            continue
        out.append({
            "chunk_id": cid, "source": "uk_dpa_2018",
            "title": title, "text": body,
            "jurisdiction": "UK", "tier": "national",
            **({"schedule": sched} if sched else {}),
        })
    return out


SOURCES = [
    ("Ireland", parse_ireland, "Ireland_DATA PROTECTION ACT 2018.txt"),
    ("Germany", parse_germany, "Germany_Federal Data Protection Act.txt"),
    ("UK", parse_uk, "UK_Data Protection Act 2018_2026.xml"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    everything, problems = [], []
    for label, parser, fname in SOURCES:
        path = NATIONS / fname
        if not path.exists():
            problems.append(f"{label}: {fname} not found")
            continue
        chunks = parser(path)
        lengths = sorted(len(c["text"]) for c in chunks)
        short = [c for c in chunks if len(c["text"]) < MIN_CHARS]
        untitled = [c for c in chunks if not c["title"]]
        print(f"\n{label:<9} {len(chunks):>4} provisions from {fname}")
        if lengths:
            print(f"          chars  min {lengths[0]:>5}  median {lengths[len(lengths) // 2]:>6}"
                  f"  max {lengths[-1]:>7}")
        print(f"          under {MIN_CHARS} chars: {len(short)}   without a title: {len(untitled)}")
        if short[:3]:
            for c in short[:3]:
                print(f"             {c['chunk_id']}: {c['text'][:70]!r}")
        everything.extend(chunks)

    ids = [c["chunk_id"] for c in everything]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        problems.append(f"duplicate chunk_id: {sorted(dupes)[:8]}")

    # Apparatus none of the three parsers should have let through. Match the
    # bracketed F-number form only; a bare \bF\d+\b flags clean prose.
    residue = [c["chunk_id"] for c in everything
               if re.search(r"\[F\d+|\.\s\.\s\.\s\.\s\.|Textual Amendments"
                            r"|Commencement Information", c["text"])]
    if residue:
        problems.append(f"{len(residue)} chunks still carry editorial apparatus, "
                        f"e.g. {residue[:5]}")

    print(f"\ntotal {len(everything)} provisions, {len(set(ids))} distinct ids")
    if problems:
        print("\nPROBLEMS")
        for p in problems:
            print("   -", p)
    else:
        print("no duplicate ids, no apparatus residue")

    if args.check_only:
        print("\n--check-only: nothing written")
        return
    if problems:
        raise SystemExit("\nrefusing to write a corpus with the problems above")
    OUT.write_text(json.dumps(everything, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
