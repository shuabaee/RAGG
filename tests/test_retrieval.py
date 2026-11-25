import pytest

from conftest import make_chunks
from core.retrieval import search

TEXTS = [
    "Hybrid retrieval combines lexical matching with dense vectors for student papers.",
    "Participants reported the interface was easy to learn within one week.",
    "Our results show citation grounding reduced hallucinated quotes considerably.",
]


def test_dense_scores_promote_chunk():
    chunks = make_chunks("p1", TEXTS)
    hits = search(chunks, "hybrid retrieval", k=6, dense_scores={"p1-2": 0.9})
    ids = {h.chunk.chunk_id: h for h in hits}
    assert "p1-2" in ids  # no keyword overlap, found through the vector store
    assert ids["p1-2"].dense == pytest.approx(0.9)
    assert ids["p1-1"].dense is None


def test_dense_scores_unknown_ids_ignored():
    chunks = make_chunks("p1", TEXTS)
    base = search(chunks, "citation grounding", k=6)
    hits = search(chunks, "citation grounding", k=6, dense_scores={"gone-99": 0.99})
    assert [h.chunk.chunk_id for h in hits] == [h.chunk.chunk_id for h in base]


def test_search_without_dense_unchanged():
    chunks = make_chunks("p1", TEXTS)
    hits = search(chunks, "citation grounding hallucinated", k=6)
    assert hits[0].chunk.chunk_id == "p1-3"
    assert all(h.dense is None for h in hits)
