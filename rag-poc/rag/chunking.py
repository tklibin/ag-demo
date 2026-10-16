"""Section-aware chunking.

Policy documents are organised by numbered clauses, and a question is usually answered by one
clause. So chunks follow the document's own structure: one chunk per section, split at paragraph
boundaries only when a section is long. Each chunk remembers its clause number, its parent
headings and its pages, which is what a citation needs ("§4.2 Non-emergency services, p.5").
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import config
from .models import Chunk, Page

MARKED = re.compile(r"^(#{1,6})\s+(.+)$")
# "Article 7", "Section 3.2: Scope", "Annex A – Prior authorisation list", "المادة 5"
KEYWORD = re.compile(
    r"^(Part|Chapter|Annex|Appendix|Schedule|Article|Section|الباب|الفصل|المادة|الملحق)\s+"
    r"([0-9]+(?:\.[0-9]+)*|[IVXLC]+|[A-Z])\b\s*[:.\-–—]?\s*(.*)$")
# "4.2 Non-emergency services", "12. Appeals". The title must start with a capital letter (or an
# Arabic letter) and stay short, so numbered list items and sentences are not mistaken for headings.
NUMBERED = re.compile(r"^(\d{1,2}(?:\.\d{1,2})*)\.?\s+([A-Z؀-ۿ].*)$")
TOP_LEVEL_KEYWORDS = {"Part", "Chapter", "Annex", "Appendix", "Schedule", "الباب", "الفصل", "الملحق"}
MAX_HEADING_WORDS = 12


@dataclass
class Heading:
    level: int
    number: str  # "4.2", "Annex A", "" when unnumbered
    line: str    # the heading as written, without '#' markers


def detect_heading(line: str) -> Heading | None:
    line = line.strip()
    marked = MARKED.match(line)
    text = marked.group(2).strip() if marked else line
    if not text or len(text.split()) > MAX_HEADING_WORDS:
        return None

    m = KEYWORD.match(text)
    if m:
        keyword, number = m.group(1), m.group(2)
        if keyword in TOP_LEVEL_KEYWORDS:  # Annex A, Chapter 2: top level, numbered by name
            return Heading(1, f"{keyword} {number}", text)
        return Heading(2 + number.count("."), number, text)  # Article 7 sits below any chapter

    m = NUMBERED.match(text)
    if m and not text.endswith((".", ";", ",")):
        return Heading(m.group(1).count(".") + 1, m.group(1), text)

    if marked:
        return Heading(len(marked.group(1)), "", text)
    return None


@dataclass
class _Paragraph:
    page: int | None
    text: str


def _paragraphs(pages: list[Page]) -> list[_Paragraph]:
    """Split pages into paragraphs. A heading on the first line of a paragraph becomes its own."""
    out = []
    for page in pages:
        for block in re.split(r"\n\s*\n", page.text):
            lines = [ln for ln in block.strip().splitlines() if ln.strip()]
            if not lines:
                continue
            if len(lines) > 1 and detect_heading(lines[0]):
                out.append(_Paragraph(page.number, lines[0].strip()))
                lines = lines[1:]
            out.append(_Paragraph(page.number, "\n".join(ln.strip() for ln in lines)))
    return out


def chunk(pages: list[Page], doc_id: str, doc_title: str, max_chars: int = config.CHUNK_MAX_CHARS) -> list[Chunk]:
    chunks: list[Chunk] = []
    stack: list[Heading] = []
    body: list[_Paragraph] = []

    def flush() -> None:
        if not body:
            return
        current = stack[-1] if stack else None
        for part in _split(body, max_chars):
            pages_ = [p.page for p in part if p.page is not None]
            chunks.append(Chunk(
                chunk_id=f"{doc_id}:{len(chunks):04d}",
                doc_id=doc_id,
                doc_title=doc_title,
                section=current.number if current else "",
                heading=current.line if current else "",
                path=[h.line for h in stack],
                page_start=min(pages_) if pages_ else None,
                page_end=max(pages_) if pages_ else None,
                text="\n\n".join(p.text for p in part),
            ))
        body.clear()

    for para in _paragraphs(pages):
        heading = detect_heading(para.text) if "\n" not in para.text else None
        if heading:
            flush()
            while stack and stack[-1].level >= heading.level:
                stack.pop()
            stack.append(heading)
        else:
            body.append(para)
    flush()
    return chunks


def _split(paragraphs: list[_Paragraph], max_chars: int) -> list[list[_Paragraph]]:
    """Group paragraphs into parts of at most max_chars. Overly long paragraphs are split by
    sentence. Consecutive parts share their boundary paragraph when it is short, so a sentence
    that depends on the previous one keeps its context."""
    units: list[_Paragraph] = []
    for p in paragraphs:
        if len(p.text) <= max_chars:
            units.append(p)
            continue
        sentence_group = ""
        for sentence in re.split(r"(?<=[.!?؟])\s+", p.text):
            if sentence_group and len(sentence_group) + len(sentence) + 1 > max_chars:
                units.append(_Paragraph(p.page, sentence_group))
                sentence_group = ""
            sentence_group = f"{sentence_group} {sentence}".strip()
        if sentence_group:
            units.append(_Paragraph(p.page, sentence_group))

    parts: list[list[_Paragraph]] = [[]]
    size = 0
    for u in units:
        if parts[-1] and size + len(u.text) > max_chars:
            last = parts[-1][-1]
            carry = [last] if len(last.text) < max_chars // 3 and len(parts[-1]) > 1 else []
            parts.append(carry)
            size = sum(len(c.text) for c in carry)
        parts[-1].append(u)
        size += len(u.text)
    return parts


def embedding_text(c: Chunk) -> str:
    """What gets embedded: the chunk prefixed with its document and heading path. A clause such as
    'Requests must be submitted within 24 hours' is only findable if the vector knows it sits under
    'Prior authorisation > Retrospective authorisation'."""
    context = " > ".join([c.doc_title, *c.path])
    return f"{context}\n\n{c.text}"
