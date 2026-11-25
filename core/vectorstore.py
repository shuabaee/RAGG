"""Persistent vector store for chunk embeddings (ChromaDB).

Embeddings are kept on disk so a PDF that was already embedded is never embedded again.
One collection per embedding key ("provider:model") so vectors of different sizes never mix.
Every function fails safe: if ChromaDB is missing or errors, callers get "not stored" / no results
and RAGG falls back to its in-memory search.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

from .models import Chunk

_client = None
_path_override: str | None = None
_failed = False


def _path() -> str:
    return _path_override or os.environ.get("RAGG_CHROMA_DIR") or "./chroma_db"


def _get_client():
    global _client, _failed
    if _client is not None or _failed:
        return _client
    try:
        import chromadb
        from chromadb.config import Settings

        _client = chromadb.PersistentClient(path=_path(), settings=Settings(anonymized_telemetry=False))
    except Exception:
        _failed = True
        _client = None
    return _client


def reset_client_for_tests(path: str | None) -> None:
    global _client, _path_override, _failed
    _client, _path_override, _failed = None, path, False


def available() -> bool:
    return _get_client() is not None


def location() -> str:
    return _path()


def _name(key: str) -> str:
    return "ragg_" + hashlib.sha1(key.encode()).hexdigest()[:16]


def _text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()[:16]


def _collection(key: str, create: bool):
    client = _get_client()
    if client is None:
        return None
    try:
        if create:
            return client.get_or_create_collection(_name(key), metadata={"hnsw:space": "cosine"})
        return client.get_collection(_name(key))
    except Exception:
        return None


def _where(paper_ids: list[str]) -> dict:
    return {"paper_id": paper_ids[0]} if len(paper_ids) == 1 else {"paper_id": {"$in": list(paper_ids)}}


def has_paper(key: str, paper_id: str, chunks: list[Chunk]) -> bool:
    """True when this paper's current chunks are all stored unchanged. Stale records are deleted."""
    col = _collection(key, create=False)
    if col is None:
        return False
    try:
        got = col.get(where=_where([paper_id]), include=["metadatas"])
        stored = {i: (m or {}).get("text_hash") for i, m in zip(got["ids"], got["metadatas"])}
        want = {c.chunk_id: _text_hash(c.text) for c in chunks}
        if stored == want:
            return bool(want)
        if stored:
            col.delete(ids=list(stored))
        return False
    except Exception:
        return False


def _upsert(col, chunks: list[Chunk], vectors: np.ndarray) -> None:
    for i in range(0, len(chunks), 1000):
        part = chunks[i : i + 1000]
        col.upsert(
            ids=[c.chunk_id for c in part],
            embeddings=np.asarray(vectors[i : i + len(part)], dtype=np.float32).tolist(),
            documents=[c.text for c in part],
            metadatas=[
                {"paper_id": c.paper_id, "page": c.page, "section": c.section, "text_hash": _text_hash(c.text)}
                for c in part
            ],
        )


def _stored_dim(col) -> int | None:
    try:
        got = col.get(limit=1, include=["embeddings"])
        embs = got.get("embeddings")
        return len(embs[0]) if embs is not None and len(embs) else None
    except Exception:
        return None


def add(key: str, chunks: list[Chunk], vectors: np.ndarray) -> bool:
    """Store vectors; True on success. If the model behind this key changed vector size,
    the old (now unusable) collection is replaced."""
    col = _collection(key, create=True)
    if col is None or not chunks:
        return False
    try:
        _upsert(col, chunks, vectors)
        return True
    except Exception:
        pass
    dim = _stored_dim(col)
    if dim is None or dim == np.asarray(vectors).shape[1]:
        return False
    try:
        _get_client().delete_collection(_name(key))
        _upsert(_collection(key, create=True), chunks, vectors)
        return True
    except Exception:
        return False


def query(key: str, qvec: np.ndarray, paper_ids: list[str], n: int = 40) -> dict[str, float]:
    """chunk_id -> cosine similarity for the n nearest chunks among paper_ids; {} on any problem."""
    col = _collection(key, create=False)
    if col is None or not paper_ids:
        return {}
    try:
        total = col.count()
        if not total:
            return {}
        res = col.query(
            query_embeddings=[np.asarray(qvec, dtype=np.float32).tolist()],
            n_results=min(n, total),
            where=_where(paper_ids),
            include=["distances"],
        )
        return {cid: 1.0 - float(d) for cid, d in zip(res["ids"][0], res["distances"][0])}
    except Exception:
        return {}
