"""Citation verification and grounding validation for RAGG."""
from __future__ import annotations

import difflib
import html

from .chunker import split_sentences
from .models import Chunk
from .pdf_utils import squash


def _cite_html(filename: str, page: int) -> str:
    return f'<span class="cite">{html.escape(filename)}, p. {page}</span>'


def _cite_plain(filename: str, page: int) -> str:
    return f"[{filename}, p. {page}]"


def verify_quote(quote: str, chunk: Chunk, page_text: str) -> tuple[str, str]:
    """Return (text_to_show, status). Only text that really exists in the PDF page is shown."""
    q = squash(quote).strip(" \"'“”‘’")
    if len(q) < 12:
        return "", "rejected"
    page_sq = squash(page_text)
    if q in page_sq or q.lower() in page_sq.lower():
        # return the original-case text from the page
        i = page_sq.lower().find(q.lower())
        return page_sq[i : i + len(q)], "verified"
    # closest real sentence on that chunk
    best, best_r = "", 0.0
    for s in split_sentences(chunk.text):
        r = difflib.SequenceMatcher(None, q.lower(), s.lower()).ratio()
        if r > best_r:
            best, best_r = s, r
    if best_r >= 0.6:
        return best, "closest"
    return "", "rejected"
