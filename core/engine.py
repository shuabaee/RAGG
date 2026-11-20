"""Backward-compatibility facade for RAGG engine.

Re-exports data structures, chunking, retrieval, verifier, context builders,
and the central RAGPipeline orchestrator so existing imports remain fully functional.
"""
from __future__ import annotations

from .chunker import (
    best_sentences,
    chunk_paper,
    detect_heading,
    split_sentences,
)
from .context_builder import (
    INTENT_QUERY,
    INTENT_WORDS,
    SCOPE_CAN,
    SECTION_BOOST,
    _SEC_NAMES,
    _ev_from_hit,
    _overview_chunks,
    _paper_list,
    no_papers_reply,
    scope_reply,
    suggested_prompts,
)
from .models import (
    Chunk,
    Evidence,
    Hit,
    PaperIndex,
    Reply,
)
from .pipeline import (
    RAGPipeline,
    answer,
    build_index,
    ensure_embeddings,
)
from .retrieval import (
    BM25,
    GENERIC,
    STOPWORDS,
    search,
    stem,
    tokenize,
)
from .verifier import (
    _cite_html,
    _cite_plain,
    verify_quote,
)

__all__ = [
    "Chunk",
    "Hit",
    "PaperIndex",
    "Evidence",
    "Reply",
    "detect_heading",
    "split_sentences",
    "chunk_paper",
    "best_sentences",
    "STOPWORDS",
    "GENERIC",
    "stem",
    "tokenize",
    "BM25",
    "search",
    "_cite_html",
    "_cite_plain",
    "verify_quote",
    "INTENT_WORDS",
    "INTENT_QUERY",
    "SECTION_BOOST",
    "_SEC_NAMES",
    "_paper_list",
    "suggested_prompts",
    "SCOPE_CAN",
    "scope_reply",
    "no_papers_reply",
    "_ev_from_hit",
    "_overview_chunks",
    "RAGPipeline",
    "build_index",
    "ensure_embeddings",
    "answer",
]
