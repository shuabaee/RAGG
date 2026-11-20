"""PDF reading for RAGG: page-by-page text with metadata.

Uses PyMuPDF (fitz) for extraction and page rendering.
Every page keeps its ORIGINAL 1-based page number so citations are real.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date

import pymupdf


_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "‘": "'", "’": "'", "“": '"', "”": '"', "­": "",
    "–": "-", "—": "-", "−": "-", "\u00a0": " ", "\u2009": " ",
}


def normalize_text(text: str) -> str:
    """Light normalization used for BOTH display and verification."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    for k, v in _LIGATURES.items():
        text = text.replace(k, v)
    # join words broken by end-of-line hyphenation: "classifi-\ncation"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def squash(text: str) -> str:
    """Whitespace-insensitive form used to verify verbatim quotes."""
    text = normalize_text(text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class Paper:
    paper_id: str
    filename: str
    data: bytes
    pages: list[str] = field(default_factory=list)  # index 0 -> page 1
    title: str = ""
    authors: str = ""
    year: str = ""
    notebook: str = "General"
    status: str = "ready"  # ready | empty | error
    message: str = ""

    @property
    def n_pages(self) -> int:
        return len(self.pages)

    @property
    def n_chars(self) -> int:
        return sum(len(p) for p in self.pages)

    @property
    def short_label(self) -> str:
        who = self.authors or ""
        if who:
            first = re.split(r",| and |;|&", who)[0].strip()
            multi = bool(re.search(r",| and |;|&", who))
            who = first + (" et al." if multi else "")
        bits = [b for b in [who, self.year] if b]
        return " ".join(bits) if bits else self.filename

    @property
    def display_title(self) -> str:
        return self.title or re.sub(r"\.pdf$", "", self.filename, flags=re.I)


def paper_id_for(filename: str, data: bytes) -> str:
    return hashlib.sha1(filename.encode() + data[:200000]).hexdigest()[:12]


def _pages_fitz(data: bytes) -> tuple[list[str], dict, list[tuple[float, str]]]:
    doc = pymupdf.open(stream=data, filetype="pdf")
    try:
        pages = [normalize_text(p.get_text("text")) for p in doc]
        meta = doc.metadata or {}
        # big-font lines on page 1 -> title candidates
        spans: list[tuple[float, str]] = []
        if len(doc):
            try:
                d = doc[0].get_text("dict")
                for b in d.get("blocks", []):
                    for line in b.get("lines", []):
                        txt = "".join(s.get("text", "") for s in line.get("spans", [])).strip()
                        if txt:
                            size = max(s.get("size", 0) for s in line.get("spans", []))
                            spans.append((size, txt))
            except Exception:
                pass
        return pages, meta, spans
    finally:
        doc.close()


_BAD_TITLES = re.compile(r"^(microsoft word|untitled|document\d*|slide \d+|\W*$)|\.(docx?|pdf|tex)$", re.I)


def _guess_title(meta: dict, spans: list[tuple[float, str]], first_page: str) -> str:
    t = (meta.get("title") or "").strip()
    if t and len(t) > 8 and not _BAD_TITLES.search(t):
        return t[:200]
    if spans:
        top = max(s for s, _ in spans)
        lines = [txt for s, txt in spans if s >= top - 0.6 and len(txt) > 2]
        cand = " ".join(lines[:4]).strip()
        if 8 < len(cand) < 250:
            return cand
    for line in first_page.splitlines()[:12]:
        line = line.strip()
        if 15 < len(line) < 200 and not re.search(r"journal|vol\.|issn|doi|http|©|copyright|arxiv", line, re.I):
            return line
    return ""


def _guess_authors(meta: dict, first_page: str, title: str) -> str:
    a = (meta.get("author") or "").strip()
    if a and len(a) < 150 and not re.search(r"user|admin|owner|microsoft", a, re.I):
        return a
    lines = [l.strip() for l in first_page.splitlines()[:25] if l.strip()]
    start = 0
    if title:
        for i, l in enumerate(lines):
            if l and l[:20] in title:
                start = i + 1
    for l in lines[start:start + 6]:
        # a line of 2-6 capitalised name-like words, maybe separated by commas/and
        clean = re.sub(r"[\d\*†‡§¹²³,]+", " ", l)
        words = [w for w in clean.split() if w.lower() not in {"and", "&"}]
        if 2 <= len(words) <= 12 and all(re.match(r"^[A-Z][a-zA-Z.\-']*$", w) for w in words):
            if not re.search(r"abstract|university|department|institute|introduction", l, re.I):
                return l.strip(" ,")
    return ""


def _guess_year(first_page: str, meta: dict) -> str:
    this_year = date.today().year
    for src in [meta.get("creationDate", "") or "", first_page[:3000]]:
        for m in re.finditer(r"(?<!\d)(19[5-9]\d|20[0-4]\d)(?!\d)", src):
            y = int(m.group(1))
            if 1950 <= y <= this_year:
                return str(y)
    return ""


def load_pdf(filename: str, data: bytes, notebook: str = "General") -> Paper:
    pid = paper_id_for(filename, data)
    paper = Paper(paper_id=pid, filename=filename, data=data, notebook=notebook)
    try:
        pages, meta, spans = _pages_fitz(data)
    except Exception as e:
        paper.status = "error"
        paper.message = f"Could not open this PDF ({type(e).__name__}). It may be encrypted or damaged."
        return paper

    paper.pages = pages
    first = pages[0] if pages else ""
    paper.title = _guess_title(meta, spans, first)
    paper.authors = _guess_authors(meta, first, paper.title)
    paper.year = _guess_year(first, meta)

    if paper.n_chars < 200:
        paper.status = "empty"
        paper.message = (
            "No readable text found. This looks like a scanned or image-only PDF. "
            "RAGG v1 does not run OCR, so it cannot read these pages."
        )
    else:
        blank = sum(1 for p in pages if len(p.strip()) < 30)
        if blank:
            paper.message = f"{blank} page(s) had no extractable text and will not be searched."
    return paper


def render_page_png(data: bytes, page_number: int, highlight: str | None = None, dpi: int = 110) -> bytes | None:
    """Render one page as PNG, highlighting the quote if it can be found. PyMuPDF only."""
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
        try:
            page = doc[page_number - 1]
            if highlight:
                words = highlight.split()
                probes = [highlight]
                if len(words) > 8:
                    probes = [" ".join(words[i:i + 6]) for i in range(0, len(words) - 5, 6)]
                for probe in probes:
                    try:
                        rects = page.search_for(probe)
                        for r in rects:
                            page.add_highlight_annot(r)
                    except Exception:
                        continue
            pix = page.get_pixmap(dpi=dpi)
            return pix.tobytes("png")
        finally:
            doc.close()
    except Exception:
        return None