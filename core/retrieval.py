"""Hybrid retrieval (BM25 keyword search + optional dense embeddings) for RAGG."""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from .models import Chunk, Hit

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


def _jacc(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / (len(sa | sb) or 1)


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


def __getattr__(name: str):
    if name in ("best_sentences", "chunk_paper", "detect_heading", "split_sentences"):
        from . import chunker
        return getattr(chunker, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "STOPWORDS",
    "GENERIC",
    "stem",
    "tokenize",
    "BM25",
    "Chunk",
    "Hit",
    "search",
    "split_sentences",
    "chunk_paper",
    "detect_heading",
    "best_sentences",
]
