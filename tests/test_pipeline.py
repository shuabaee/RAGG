from pathlib import Path

import numpy as np
import pytest

from core import llm as llm_mod
from core import pipeline, reranker
from core.llm import LLMConfig
from core.pdf_utils import load_pdf

SAMPLE = sorted((Path(__file__).resolve().parents[1] / "sample_papers").glob("*.pdf"))
QUESTION = "What were the limitations of the sample size?"


@pytest.fixture(scope="module")
def corpus():
    papers = [load_pdf(p.name, p.read_bytes()) for p in SAMPLE]
    return papers


def _indexes(papers):
    return {p.paper_id: pipeline.build_index(p) for p in papers}


class PickLast:
    """Fake cross-encoder: puts the lowest-ranked candidate first."""

    def __init__(self):
        self.calls, self.chosen, self.n = 0, None, 0

    def rerank(self, req):
        self.calls += 1
        self.n = len(req.passages)
        last = req.passages[-1]
        self.chosen = last["text"]
        return [{"id": last["id"], "text": last["text"], "score": 0.99}] + [
            {"id": p["id"], "text": p["text"], "score": 0.01} for p in req.passages[:-1]
        ]


@pytest.fixture(autouse=True)
def _reset():
    reranker.reset_for_tests()
    yield
    reranker.reset_for_tests()


def _spy_search(monkeypatch):
    ks = []
    real = pipeline.search

    def spy(*a, **kw):
        ks.append(kw.get("k"))
        return real(*a, **kw)

    monkeypatch.setattr(pipeline, "search", spy)
    return ks


def test_answer_uses_reranker_in_extractive_mode(corpus, monkeypatch):
    fake = PickLast()
    monkeypatch.setattr(reranker, "_load_ranker", lambda: fake)
    ks = _spy_search(monkeypatch)
    rep = pipeline.answer(QUESTION, corpus, _indexes(corpus), [], [], LLMConfig(provider="none", rerank=True))
    assert fake.calls == 1
    assert ks == [20]  # fetch extra candidates for the reranker
    assert fake.n >= 1
    assert rep.mode == "extractive"
    assert rep.evidence and set(rep.evidence[0].quote.split()) <= set(fake.chosen.split())


def test_answer_rerank_off(corpus, monkeypatch):
    fake = PickLast()
    monkeypatch.setattr(reranker, "_load_ranker", lambda: fake)
    ks = _spy_search(monkeypatch)
    rep = pipeline.answer(QUESTION, corpus, _indexes(corpus), [], [], LLMConfig(provider="none", rerank=False))
    assert fake.calls == 0
    assert ks == [6]
    assert rep.mode == "extractive"


def test_ensure_embeddings_skips_stored_paper(corpus, store, monkeypatch):
    calls = {"n": 0}

    def fake_embed(cfg, texts, is_query=False):
        calls["n"] += 1
        rng = np.random.default_rng(len(texts))
        v = rng.normal(size=(len(texts), 8)).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    monkeypatch.setattr(llm_mod, "embed", fake_embed)
    cfg = LLMConfig(provider="ollama", model="fake-chat", embed_model="fake-embed")
    ids = [p.paper_id for p in corpus]
    assert pipeline.ensure_embeddings(cfg, _indexes(corpus), ids) is None
    first = calls["n"]
    assert first == len(corpus)
    # simulate an app restart: fresh in-memory indexes, same on-disk store
    assert pipeline.ensure_embeddings(cfg, _indexes(corpus), ids) is None
    assert calls["n"] == first


def test_answer_queries_chroma(corpus, store, monkeypatch):
    def fake_embed(cfg, texts, is_query=False):
        v = np.ones((len(texts), 8), dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    seen = {}
    real_query = pipeline.vectorstore.query

    def spy(key, qvec, paper_ids, n=40):
        seen["ids"] = list(paper_ids)
        return real_query(key, qvec, paper_ids, n)

    monkeypatch.setattr(llm_mod, "embed", fake_embed)
    monkeypatch.setattr(pipeline.vectorstore, "query", spy)
    monkeypatch.setattr(llm_mod, "chat", lambda *a, **k: (_ for _ in ()).throw(llm_mod.LLMError("offline")))
    monkeypatch.setattr(reranker, "_load_ranker", lambda: PickLast())
    cfg = LLMConfig(provider="ollama", model="fake-chat", embed_model="fake-embed")
    ix = _indexes(corpus)
    pipeline.answer(QUESTION, corpus, ix, [], [], cfg)
    assert set(seen["ids"]) <= {p.paper_id for p in corpus} and seen["ids"]
