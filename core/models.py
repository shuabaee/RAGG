"""Data contracts for RAGG: chunks, search hits, paper indexes, evidence, and replies."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    filename: str
    page: int
    section: str
    text: str
    tokens: list[str]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    lexical: float
    dense: float | None
    overlap: int  # how many topical query terms appear in the chunk
    rerank: float | None = None  # cross-encoder relevance score, when reranked


@dataclass
class PaperIndex:
    chunks: list[Chunk]
    emb: np.ndarray | None = None
    emb_key: str = ""  # embed_key() the vectors were made with
    in_store: bool = False  # vectors confirmed in the persistent vector store


@dataclass
class Evidence:
    paper_id: str
    filename: str
    page: int
    section: str
    quote: str
    status: str  # verified | closest | retrieved


@dataclass
class Reply:
    content: str  # markdown/html for display
    plain: str  # text kept in chat history for the model
    evidence: list[Evidence] = field(default_factory=list)
    passages: list[Evidence] = field(default_factory=list)
    focus_ids: list[str] = field(default_factory=list)
    mode: str = ""  # scope | ai | extractive | refuse
    notice: str = ""
    reasoning: str = ""  # the model's private reasoning, shown collapsed; never sent back as history
