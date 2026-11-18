"""Chunking + hybrid retrieval (BM25 keyword search + optional dense embeddings).

Chunks never cross a page boundary, so each chunk has one exact page number.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .pdf_utils import Paper

STOPWORDS = set(
    """a about above after again against all am an and any are as at be because been before being below
    between both but by can could did do does doing down during each few for from further had has have having
    he her here hers herself him himself his how i if in into is it its itself just me more most my myself no
    nor not now of off on once only or other our ours ourselves out over own same she should so some such than
    that the their theirs them themselves then there these they this those through to too under until up very
    was we were what when where which while who whom why will with would you your yours yourself yourselves
    also may might must shall within without upon via per et al eg ie use used using show shows shown give tell
    me please kindly explain describe discuss list find get make s t""".split()
)

# words that say "this is about a paper" but carry no topic
GENERIC = STOPWORDS | set(
    """paper papers study studies research article articles document documents pdf pdfs author authors work
    works researcher researchers journal uploaded upload file files source sources text page pages section
    evidence proof quote quotes excerpt excerpts cite citation citations main major key overall summary summarize
    summarise summarizes tldr overview point points thing things something anything everything ragg""".split()
)

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


def stem(w: str) -> str:
    for suf in ("ational", "ations", "ation", "ings", "ing", "ness", "ments", "ment", "ies", "ied", "ly", "ed", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            if suf in ("ies", "ied"):
                return w[: -len(suf)] + "y"
            return w[: -len(suf)]
    return w


def tokenize(text: str, drop: set[str] = STOPWORDS) -> list[str]:
    toks = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]+|\d+(?:\.\d+)?%?", text.lower())
    return [stem(t) for t in toks if t not in drop and len(t) > 1]


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


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    filename: str
    page: int
    section: str
    text: str
    tokens: list[str]


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


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.N = len(docs)
        self.avgdl = (sum(len(d) for d in docs) / self.N) if self.N else 0
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.N - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tfs = [Counter(d) for d in docs]

    def scores(self, q: list[str]) -> np.ndarray:
        out = np.zeros(self.N)
        if not self.N:
            return out
        for i, tf in enumerate(self.tfs):
            dl = len(self.docs[i]) or 1
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf.get(t, 0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
            out[i] = s
        return out


@dataclass
class Hit:
    chunk: Chunk
    score: float
    lexical: float
    dense: float | None
    overlap: int  # how many topical query terms appear in the chunk


def search(
    chunks: list[Chunk],
    query: str,
    k: int = 6,
    embeddings: np.ndarray | None = None,
    query_vec: np.ndarray | None = None,
    section_boost: dict[str, float] | None = None,
) -> list[Hit]:
    if not chunks:
        return []
    topical = set(tokenize(query, GENERIC))
    q = list(topical) if topical else tokenize(query)
    bm = BM25([c.tokens for c in chunks]).scores(q)
    order_lex = np.argsort(-bm)
    rank_lex = np.empty(len(chunks), dtype=int)
    rank_lex[order_lex] = np.arange(len(chunks))

    dense = None
    if embeddings is not None and query_vec is not None and len(embeddings) == len(chunks):
        dense = embeddings @ query_vec
        order_d = np.argsort(-dense)
        rank_d = np.empty(len(chunks), dtype=int)
        rank_d[order_d] = np.arange(len(chunks))

    fused = np.zeros(len(chunks))
    for i, c in enumerate(chunks):
        s = (1 / (60 + rank_lex[i])) if bm[i] > 0 else 0.0
        if dense is not None:
            s += 1 / (60 + rank_d[i])
        if c.section == "references":
            s *= 0.35
        if section_boost and c.section in section_boost:
            s *= section_boost[c.section]
        fused[i] = s

    best = np.argsort(-fused)[: max(k * 3, k)]
    hits = []
    for i in best:
        if fused[i] <= 0:
            continue
        c = chunks[i]
        ov = len(topical & set(c.tokens))
        hits.append(Hit(c, float(fused[i]), float(bm[i]), None if dense is None else float(dense[i]), ov))
    # de-duplicate near-identical overlapping chunks from the same page
    out: list[Hit] = []
    for h in hits:
        if any(o.chunk.page == h.chunk.page and o.chunk.paper_id == h.chunk.paper_id and _jacc(o.chunk.tokens, h.chunk.tokens) > 0.7 for o in out):
            continue
        out.append(h)
        if len(out) >= k:
            break
    return out


def _jacc(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / (len(sa | sb) or 1)


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
