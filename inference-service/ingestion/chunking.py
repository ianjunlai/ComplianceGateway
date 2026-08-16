"""Structural chunking: one chunk is one complete legal directive, never a token window."""
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

MAX_CHUNK_TOKENS = 800  # fallback split threshold (~ tokens, approximated by words/0.75)

_ARTICLE_RE = re.compile(r"^Article\s+(\d+[a-z]?)\b", re.MULTILINE)
_RECITAL_RE = re.compile(r"^\((\d+)\)\s", re.MULTILINE)
_POLICY_MARKER_RE = re.compile(r"^###\s*(\S+)\s*\n", re.MULTILINE)

_GERMAN_TRANSLITERATION = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})


def slugify_university_name(name: str) -> str:
    """ASCII, lowercase identifier for a university name (e.g. "Göttingen" ->
    "goettingen"). Chunk IDs, logs and CSVs all handle plain ASCII more safely
    across platforms and consoles than a table of every accented character."""
    name = name.lower().translate(_GERMAN_TRANSLITERATION)
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_")


@dataclass
class Chunk:
    chunk_id: str
    source: str        # "gdpr" | national act, e.g. "uk_dpa_2018" | university id
    title: str
    text: str
    # Where the provision sits in the hierarchy of law, and which state it
    # binds.
    tier: str = ""          # "regional" | "national" | "institutional"
    jurisdiction: str = ""  # "EU" | "UK" | "IE" | "DE"

    @property
    def approx_tokens(self) -> int:
        return int(len(self.text.split()) / 0.75)


def chunk_gdpr(path: Path) -> list[Chunk]:
    """Split the GDPR plain text on Article boundaries."""
    text = path.read_text(encoding="utf-8")
    chunks: list[Chunk] = []
    matches = list(_ARTICLE_RE.finditer(text))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.start():end].strip()
        art_no = m.group(1)
        chunks.extend(_maybe_split(Chunk(
            chunk_id=f"gdpr-art-{art_no}",
            source="gdpr",
            title=f"GDPR Article {art_no}",
            text=body,
        )))
    return chunks


def chunk_gdpr_recitals(path: Path) -> list[Chunk]:
    """Split the GDPR recitals text on "(N) " paragraph boundaries."""
    text = path.read_text(encoding="utf-8")
    chunks: list[Chunk] = []
    matches = list(_RECITAL_RE.finditer(text))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.start():end].strip()
        rec_no = m.group(1)
        chunks.extend(_maybe_split(Chunk(
            chunk_id=f"gdpr-rec-{rec_no}",
            source="gdpr",
            title=f"GDPR Recital {rec_no}",
            text=body,
        )))
    return chunks


def chunk_university_policy(path: Path, uni_id: str) -> list[Chunk]:
    """Split a university policy on its hand-placed "### <id>" markers."""
    text = path.read_text(encoding="utf-8")
    matches = list(_POLICY_MARKER_RE.finditer(text))
    chunks: list[Chunk] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end].strip()
        suffix = m.group(1)
        chunks.extend(_maybe_split(Chunk(
            chunk_id=f"{uni_id}-policy-{suffix}",
            source=uni_id,
            title=body.splitlines()[0] if body else f"{uni_id} policy {suffix}",
            text=body,
        )))
    return chunks


def _maybe_split(chunk: Chunk) -> list[Chunk]:
    """Fallback split for oversized chunks, preserving paragraph boundaries."""
    if chunk.approx_tokens <= MAX_CHUNK_TOKENS:
        return [chunk]
    paras = chunk.text.split("\n\n")
    parts: list[Chunk] = []
    buf: list[str] = []
    part_no = 1
    for p in paras:
        buf.append(p)
        if int(len(" ".join(buf).split()) / 0.75) >= MAX_CHUNK_TOKENS:
            parts.append(Chunk(f"{chunk.chunk_id}-{part_no}", chunk.source,
                               f"{chunk.title} (part {part_no})", "\n\n".join(buf)))
            buf, part_no = [], part_no + 1
    if buf:
        parts.append(Chunk(f"{chunk.chunk_id}-{part_no}", chunk.source,
                           f"{chunk.title} (part {part_no})", "\n\n".join(buf)))
    return parts


def load_corpus(corpus_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    gdpr_file = corpus_dir / "gdpr" / "gdpr_articles.txt"
    if gdpr_file.exists():
        chunks.extend(chunk_gdpr(gdpr_file))
    recitals_file = corpus_dir / "gdpr" / "gdpr_recitals.txt"
    if recitals_file.exists():
        chunks.extend(chunk_gdpr_recitals(recitals_file))

    uni_root = corpus_dir / "universities"
    if uni_root.exists():
        for doc in sorted(uni_root.glob("data_protection_policy_*.txt")):
            name = doc.stem.removeprefix("data_protection_policy_")
            chunks.extend(chunk_university_policy(doc, slugify_university_name(name)))
    return chunks
