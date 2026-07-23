"""Semantic chunking of legal documents.

Structural chunking: one chunk = one complete legal directive (Article /
Clause / Recital), never token-window chunking. Oversized articles fall back
to a paragraph split at MAX_CHUNK_TOKENS (disclosed parameter).

Input corpus layout (dataset/corpus/):
    gdpr/gdpr_articles.txt        plain-text GDPR, "Article N ..." headed
    gdpr/gdpr_recitals.txt        optional: recitals as "(N) ..." paragraphs
    universities/<uni_id>/*.txt   one policy document per file

Chunk ID convention (stable, used as ground-truth keys in the QA set):
    gdpr-art-46        gdpr-rec-108        uni_a-policy-3
"""
import re
from dataclasses import dataclass
from pathlib import Path

MAX_CHUNK_TOKENS = 800  # fallback split threshold (~ tokens, approximated by words/0.75)

_ARTICLE_RE = re.compile(r"^Article\s+(\d+[a-z]?)\b", re.MULTILINE)
_RECITAL_RE = re.compile(r"^\((\d+)\)\s", re.MULTILINE)


@dataclass
class Chunk:
    chunk_id: str
    source: str        # "gdpr" | university id, e.g. "uni_a"
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
    """Split a university policy document on numbered-section boundaries.

    TODO: adapt the section regex per downloaded policy format once
    the 3-5 target universities are chosen.
    """
    text = path.read_text(encoding="utf-8")
    sections = re.split(r"\n(?=\d+\.\s)", text)
    chunks = []
    for i, sec in enumerate(s for s in sections if s.strip()):
        chunks.extend(_maybe_split(Chunk(
            chunk_id=f"{uni_id}-policy-{i + 1}",
            source=uni_id,
            title=f"{uni_id} policy section {i + 1}",
            text=sec.strip(),
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
        for uni_dir in sorted(p for p in uni_root.iterdir() if p.is_dir()):
            for doc in sorted(uni_dir.glob("*.txt")):
                chunks.extend(chunk_university_policy(doc, uni_dir.name))
    return chunks
