"""Semantic chunking of legal documents.

Structural chunking: one chunk = one complete legal directive (Article /
Clause / Recital), never token-window chunking. Oversized articles fall back
to a paragraph split at MAX_CHUNK_TOKENS (disclosed parameter).

GDPR is chunked automatically from its Article/Recital numbering (regular and
predictable enough to parse reliably). University policies vary too much in
numbering style to parse safely, so their chunk boundaries are marked by hand:
a line "### <chunk_id_suffix>" placed before every clause meant to become its
own chunk. Everything between one marker and the next is that chunk's text,
including any sub-item numbering that should stay together (e.g. Cambridge's
1.6.1-1.6.4 under 1.6) -- simply don't mark those sub-items.

Input corpus layout (dataset/corpus/):
    gdpr/gdpr_articles.txt                           plain-text GDPR, "Article N ..." headed
    gdpr/gdpr_recitals.txt                           optional: recitals as "(N) ..." paragraphs
    universities/data_protection_policy_<Name>.txt   one manually marked-up file per institution

Chunk ID convention (stable, used as ground-truth keys in the QA set):
    gdpr-art-46        gdpr-rec-108        cambridge-policy-3.3        tcd-policy-8
"""
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
    source: str        # "gdpr" | university id, e.g. "cambridge"
    title: str
    text: str

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
    """Split a university policy on its hand-placed "### <id>" markers.

    Each marker's own line is removed; everything from there to the next
    marker (or end of file) becomes that chunk's text, verbatim. Text before
    the first marker, if any, is discarded — mark a "### preamble" line there
    if it should be kept.
    """
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
