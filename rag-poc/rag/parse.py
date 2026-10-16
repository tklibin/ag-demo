"""Turn an uploaded file into pages of plain text that the chunker understands.

Conventions in the returned text:
  * paragraphs are separated by a blank line;
  * a line starting with '#' marks a heading the parser is sure about (a DOCX "Heading n" style,
    or a short bold line in a PDF). Numbered headings ("4.2 Prior authorisation", "Article 7")
    are left as they are: the chunker recognises them itself.

Scanned PDF pages have no text layer. With ocr=True those pages are rendered to an image and
transcribed by the local vision model (gemma4 can read images).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf

from . import config
from .models import Page

pymupdf.no_recommend_layout()  # silence the "consider pymupdf_layout" notice on table search

SUPPORTED = {".pdf", ".docx", ".md", ".txt"}
PAGE_NUMBER = re.compile(r"^(page\s+)?\d+(\s*(/|of)\s*\d+)?$", re.IGNORECASE)
BOLD = 1 << 4  # pymupdf span flag
MAX_HEADING_WORDS = 12


def parse(path: Path, ocr: bool = False) -> list[Page]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path, ocr=ocr)
    if suffix == ".docx":
        return parse_docx(path)
    if suffix in (".md", ".txt"):
        return [Page(None, path.read_text(encoding="utf-8"))]
    raise ValueError(f"Unsupported file type {suffix}; expected one of {', '.join(sorted(SUPPORTED))}")


# --- PDF ----------------------------------------------------------------------------------------

def parse_pdf(path: Path, ocr: bool = False) -> list[Page]:
    with pymupdf.open(path) as doc:
        raw = [_pdf_page_paragraphs(page) for page in doc]
        if ocr:
            for i, page in enumerate(doc):
                if sum(len(p) for p in raw[i]) < 20:  # no usable text layer: a scan
                    raw[i] = _ocr_page(page)
    repeated = _repeated_lines(raw)
    pages = []
    for number, paragraphs in enumerate(raw, 1):
        kept = [p for p in paragraphs if p.lstrip("# ") not in repeated and not PAGE_NUMBER.match(p)]
        # NFKC turns PDF ligatures ("ﬁ", "ﬀ") back into plain letters, which keyword search and
        # quote matching need.
        pages.append(Page(number, unicodedata.normalize("NFKC", "\n\n".join(kept))))
    return pages


def _pdf_page_paragraphs(page: pymupdf.Page) -> list[str]:
    """One paragraph per text block; a short, fully bold line becomes its own '# ' heading.
    Tables are read row by row ("cell | cell | cell") and placed where they sit on the page."""
    tables = [(t.bbox, _table_rows(t)) for t in page.find_tables().tables]
    items: list[tuple[float, list[str]]] = [(bbox[1], rows) for bbox, rows in tables]
    for block in page.get_text("dict", sort=True)["blocks"]:
        if block.get("type") != 0:  # 0 = text, 1 = image
            continue
        if any(pymupdf.Rect(block["bbox"]).intersects(bbox) for bbox, _ in tables):
            continue  # already read as part of a table
        items.append((block["bbox"][1], _block_paragraphs(block)))
    items.sort(key=lambda item: item[0])
    return [p for _, paragraphs in items for p in paragraphs]


def _table_rows(table) -> list[str]:
    rows = []
    for row in table.extract():
        cells = [re.sub(r"\s+", " ", c or "").strip() for c in row]
        if any(cells):
            rows.append(" | ".join(cells))
    return rows


def _block_paragraphs(block: dict) -> list[str]:
    paragraphs: list[str] = []
    body: list[str] = []
    for line in block["lines"]:
        spans = [s for s in line["spans"] if s["text"].strip()]
        text = "".join(s["text"] for s in line["spans"]).strip()
        if not text:
            continue
        if spans and all(s["flags"] & BOLD for s in spans) and _looks_like_heading(text):
            if body:
                paragraphs.append(_join_lines(body))
                body = []
            paragraphs.append(f"# {text}")
        else:
            body.append(text)
    if body:
        paragraphs.append(_join_lines(body))
    return paragraphs


def _looks_like_heading(text: str) -> bool:
    return len(text.split()) <= MAX_HEADING_WORDS and not text.endswith((".", ":", ";", ","))


def _join_lines(lines: list[str]) -> str:
    """Join wrapped lines of one paragraph, undoing end-of-line hyphenation ("authori-\nsation")."""
    out = lines[0]
    for line in lines[1:]:
        if out.endswith("-") and line[:1].islower():
            out = out[:-1] + line
        else:
            out = f"{out} {line}"
    return out


def _repeated_lines(pages: list[list[str]]) -> set[str]:
    """Running headers and footers: short lines that appear on most pages."""
    if len(pages) < 3:
        return set()
    counts = Counter(p.lstrip("# ") for paragraphs in pages for p in set(paragraphs) if len(p) < 120)
    return {text for text, n in counts.items() if n >= 0.6 * len(pages)}


def _ocr_page(page: pymupdf.Page) -> list[str]:
    import ollama  # only needed for OCR

    png = page.get_pixmap(dpi=200).tobytes("png")
    response = ollama.Client(host=config.OLLAMA_HOST).chat(
        model=config.CHAT_MODEL,
        messages=[{"role": "user", "images": [png], "content": (
            "Transcribe all text on this page exactly as written, in reading order. "
            "Put a blank line between paragraphs and start each heading line with '# '. "
            "Output only the transcription.")}],
        think=False,
        options={"temperature": 0},
    )
    return [p.strip() for p in response.message.content.split("\n\n") if p.strip()]


# --- DOCX ---------------------------------------------------------------------------------------

def parse_docx(path: Path) -> list[Page]:
    import docx
    from docx.table import Table

    document = docx.Document(str(path))
    paragraphs = []
    for item in document.iter_inner_content():
        if isinstance(item, Table):
            for row in item.rows:
                cells = [c.text.strip() for c in row.cells]
                # Merged cells repeat their text in python-docx; drop consecutive duplicates.
                cells = [c for i, c in enumerate(cells) if c and (i == 0 or c != cells[i - 1])]
                if cells:
                    paragraphs.append(" | ".join(cells))
            continue
        text = item.text.strip()
        if not text:
            continue
        style = (item.style.name if item.style is not None else "") or ""
        level = _heading_level(style)
        paragraphs.append(f"{'#' * level} {text}" if level else text)
    return [Page(None, "\n\n".join(paragraphs))]


def _heading_level(style: str) -> int:
    if style == "Title":
        return 1
    m = re.match(r"Heading (\d)", style)
    return int(m.group(1)) if m else 0
