"""Cross-encoder reranking with FlashRank (ONNX, no PyTorch).

Search finds ~20 candidates quickly; the reranker reads the question and each candidate
together and keeps the best few. If FlashRank or its model cannot load, the original order is
kept and a one-line notice is returned instead of an error.
"""
from __future__ import annotations

import os
import shutil
import threading
import time

from .models import Hit

MODEL_NAME = "ms-marco-MiniLM-L-12-v2"
REFERENCES_PENALTY = 0.35  # same factor retrieval.search applies to reference-list chunks
LOAD_TIMEOUT = 20.0  # seconds a question waits for the first model download/load
RETRY_AFTER = 300.0  # seconds before a failed load is tried again

_lock = threading.Lock()
_ranker = None
_error: str | None = None
_error_at = 0.0
_loading: threading.Thread | None = None


def _cache_dir() -> str:
    return os.environ.get("RAGG_RERANK_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "ragg_flashrank")


def _load_ranker():
    from flashrank import Ranker

    os.makedirs(_cache_dir(), exist_ok=True)
    return Ranker(model_name=MODEL_NAME, cache_dir=_cache_dir(), max_length=512)


def reset_for_tests() -> None:
    global _ranker, _error, _error_at, _loading
    _ranker, _error, _error_at, _loading = None, None, 0.0, None


def _load_in_background() -> None:
    """Runs in a worker thread: a slow first download never blocks a question for long."""
    global _ranker, _error, _error_at
    try:
        r = _load_ranker()
        with _lock:
            _ranker, _error = r, None
    except Exception as e:
        # a half-downloaded model dir makes FlashRank skip the download forever; remove it
        shutil.rmtree(os.path.join(_cache_dir(), MODEL_NAME), ignore_errors=True)
        with _lock:
            _error, _error_at = f"{type(e).__name__}: {e}"[:160], time.time()


def _get():
    global _loading, _error
    with _lock:
        if _ranker is not None:
            return _ranker
        if _error is not None and time.time() - _error_at < RETRY_AFTER:
            return None
        if _loading is None or not _loading.is_alive():
            _error = None
            _loading = threading.Thread(target=_load_in_background, daemon=True)
            _loading.start()
        worker = _loading
    worker.join(LOAD_TIMEOUT)
    with _lock:
        if _ranker is None and _error is None:
            return _TimedOut("model still loading, timed out")
        return _ranker


class _TimedOut:
    def __init__(self, msg: str):
        self.msg = msg


def rerank(query: str, hits: list[Hit], k: int = 6) -> tuple[list[Hit], str]:
    """Return the k most relevant hits by cross-encoder score, plus a notice ('' when it worked)."""
    if not hits:
        return [], ""
    ranker = _get()
    if isinstance(ranker, _TimedOut):
        return hits[:k], f"Reranker {ranker.msg}; using standard ranking for now."
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
    scored: list[Hit] = []
    for r in ranked:
        h = hits[int(r["id"])]
        # the cross-encoder ignores sections; keep search's down-weighting of bibliography text
        h.rerank = float(r["score"]) * (REFERENCES_PENALTY if h.chunk.section == "references" else 1.0)
        scored.append(h)
    scored.sort(key=lambda h: -h.rerank)
    return scored[:k], ""


class _Request:
    def __init__(self, query: str, passages: list[dict]):
        self.query, self.passages = query, passages
