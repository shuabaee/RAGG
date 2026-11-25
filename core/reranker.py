"""Cross-encoder reranking with FlashRank (ONNX, no PyTorch).

Search finds ~20 candidates quickly; the reranker reads the question and each candidate
together and keeps the best few. If FlashRank or its model cannot load, the original order is
kept and a one-line notice is returned instead of an error.
"""
from __future__ import annotations

import os

from .models import Hit

MODEL_NAME = "ms-marco-MiniLM-L-12-v2"

_ranker = None
_error: str | None = None


def _cache_dir() -> str:
    return os.environ.get("RAGG_RERANK_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "ragg_flashrank")


def _load_ranker():
    from flashrank import Ranker

    os.makedirs(_cache_dir(), exist_ok=True)
    return Ranker(model_name=MODEL_NAME, cache_dir=_cache_dir(), max_length=512)


def reset_for_tests() -> None:
    global _ranker, _error
    _ranker, _error = None, None


def _get():
    global _ranker, _error
    if _ranker is None and _error is None:
        try:
            _ranker = _load_ranker()
        except Exception as e:
            _error = f"{type(e).__name__}: {e}"[:160]
    return _ranker


def rerank(query: str, hits: list[Hit], k: int = 6) -> tuple[list[Hit], str]:
    """Return the k most relevant hits by cross-encoder score, plus a notice ('' when it worked)."""
    if not hits:
        return [], ""
    ranker = _get()
    if ranker is None:
        return hits[:k], f"Reranker unavailable ({_error}). Using standard ranking."
    try:
        from flashrank import RerankRequest
    except Exception:  # tests may inject a fake ranker without flashrank installed
        RerankRequest = _Request
    try:
        req = RerankRequest(query=query, passages=[{"id": i, "text": h.chunk.text} for i, h in enumerate(hits)])
        ranked = ranker.rerank(req)
    except Exception as e:
        return hits[:k], f"Reranker failed ({type(e).__name__}). Using standard ranking."
    out: list[Hit] = []
    for r in ranked[:k]:
        h = hits[int(r["id"])]
        h.rerank = float(r["score"])
        out.append(h)
    return out, ""


class _Request:
    def __init__(self, query: str, passages: list[dict]):
        self.query, self.passages = query, passages
