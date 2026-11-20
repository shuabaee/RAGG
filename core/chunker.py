"""Document chunking and sentence boundary detection for RAGG."""
from __future__ import annotations

import re

from .models import Chunk
from .pdf_utils import Paper
from .retrieval import tokenize

SECTION_PATTERNS = [
    ("abstract", r"abstract|summary"),
    ("intro", r"introduction|background|rationale|statement of the problem|problem statement"),
    ("related", r"related works?|literature review|review of (the )?related literature|related literature|theoretical framework|conceptual framework"),
    ("method", r"methodology|methods?|materials and methods|research design|research methodology|proposed (method|approach|system|model)|approach|experimental (setup|design)|implementation|system design|procedure"),
    ("data", r"datasets?|data( collection| sources?| and materials)?|participants|respondents|sample|corpus"),
    ("results", r"results?( and discussions?)?|experiments?|findings|evaluation|performance|analysis"),
    ("discussion", r"discussions?"),
    ("conclusion", r"conclusions?( and (future work|recommendations?))?|concluding remarks|summary and conclusions?"),
    ("limitations", r"limitations?( and future (work|research|directions))?|threats to validity"),
    ("future", r"future (work|research|directions)|recommendations?"),
    ("references", r"references|bibliography|works cited|literature cited|acknowledge?ments?|appendix|appendices"),
]
_HEADING_RE = [
    (name, re.compile(r"^\s*((\d+(\.\d+)*\.?)|([IVX]+\.)|([A-H]\.))?\s*(" + pat + r")\s*[:.]?\s*$", re.I))
    for name, pat in SECTION_PATTERNS
]


def detect_heading(line: str) -> str | None:
    line = line.strip()
    if not line or len(line) > 70:
        return None
    for name, rx in _HEADING_RE:
        if rx.match(line):
            return name
    return None


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"(\[])", text)
    out = []
    for p in parts:
        p = p.strip()
        if len(p) > 600:  # very long run-on (tables etc.) -> cut on semicolons
            out.extend(s.strip() for s in re.split(r"(?<=;)\s+", p) if s.strip())
        elif p:
            out.append(p)
    return out


def chunk_paper(paper: Paper, target_chars: int = 900, overlap_sents: int = 1) -> list[Chunk]:
    chunks: list[Chunk] = []
    section = "front"
    n = 0
    for pi, page_text in enumerate(paper.pages, start=1):
        if len(page_text.strip()) < 30:
            continue
        # split page into (section, text) runs by heading lines
        runs: list[tuple[str, list[str]]] = [(section, [])]
        for line in page_text.splitlines():
            h = detect_heading(line)
            if h:
                section = h
                runs.append((section, []))
            else:
                runs[-1][1].append(line)
        for sec, lines in runs:
            sents = split_sentences(" ".join(lines))
            if not sents:
                continue
            buf: list[str] = []
            for s in sents:
                buf.append(s)
                if sum(len(x) + 1 for x in buf) >= target_chars:
                    txt = " ".join(buf)
                    n += 1
                    chunks.append(Chunk(f"{paper.paper_id}-{n}", paper.paper_id, paper.filename, pi, sec, txt, tokenize(txt)))
                    buf = buf[-overlap_sents:] if overlap_sents else []
            if buf and (not chunks or chunks[-1].page != pi or " ".join(buf) not in chunks[-1].text):
                txt = " ".join(buf)
                if len(txt) > 40:
                    n += 1
                    chunks.append(Chunk(f"{paper.paper_id}-{n}", paper.paper_id, paper.filename, pi, sec, txt, tokenize(txt)))
    return chunks


def best_sentences(text: str, query: str, n: int = 2, min_len: int = 40) -> list[str]:
    """Pick the most query-relevant sentences from a passage (verbatim)."""
    sents = [s for s in split_sentences(text) if len(s) >= min_len]
    if not sents:
        return [text[:400]] if text else []
    q = set(tokenize(query))
    scored = []
    for i, s in enumerate(sents):
        toks = tokenize(s)
        ov = len(q & set(toks))
        scored.append((ov + 0.001 * (len(sents) - i), i, s))
    scored.sort(reverse=True)
    picked = sorted(scored[:n], key=lambda x: x[1])
    return [s for _, _, s in picked]
