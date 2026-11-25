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
    monkeypatch.setattr(llm_mod, "chat_full", lambda *a, **k: (_ for _ in ()).throw(llm_mod.LLMError("offline")))
    monkeypatch.setattr(reranker, "_load_ranker", lambda: PickLast())
    cfg = LLMConfig(provider="ollama", model="fake-chat", embed_model="fake-embed")
    ix = _indexes(corpus)
    pipeline.answer(QUESTION, corpus, ix, [], [], cfg)
    assert set(seen["ids"]) <= {p.paper_id for p in corpus} and seen["ids"]


def _unit_embed(cfg, texts, is_query=False):
    rng = np.random.default_rng(abs(hash(tuple(texts))) % 2**32)
    v = rng.normal(size=(len(texts), 8)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _spy_dense(monkeypatch):
    seen = {}
    real = pipeline.search

    def spy(*a, **kw):
        seen["dense"] = kw.get("dense_scores")
        return real(*a, **kw)

    monkeypatch.setattr(pipeline, "search", spy)
    return seen


def _cfg():
    return LLMConfig(provider="ollama", model="fake-chat", embed_model="fake-embed", rerank=False)


def _no_chat(monkeypatch):
    monkeypatch.setattr(llm_mod, "chat_full", lambda *a, **k: (_ for _ in ()).throw(llm_mod.LLMError("offline")))


def test_embed_key_includes_base_url():
    a = LLMConfig(provider="openai", api_key="k", embed_model="text-embedding-3-small")
    b = LLMConfig(provider="openai", api_key="k", embed_model="text-embedding-3-small", base_url="https://api.groq.com/openai/v1")
    assert pipeline.embed_key(a) != pipeline.embed_key(b)


def test_chroma_query_failure_gives_notice(corpus, store, monkeypatch):
    monkeypatch.setattr(llm_mod, "embed", _unit_embed)
    _no_chat(monkeypatch)
    ids = [p.paper_id for p in corpus]
    pipeline.ensure_embeddings(_cfg(), _indexes(corpus), ids)
    ix = _indexes(corpus)  # restart: vectors only on disk
    pipeline.ensure_embeddings(_cfg(), ix, ids)
    monkeypatch.setattr(pipeline.vectorstore, "query", lambda *a, **k: {})
    rep = pipeline.answer(QUESTION, corpus, ix, [], [], _cfg())
    assert "Vector store unavailable" in rep.notice


def test_paper_not_stored_still_gets_dense_scores(corpus, store, monkeypatch):
    monkeypatch.setattr(llm_mod, "embed", _unit_embed)
    _no_chat(monkeypatch)
    a, b = corpus[0], corpus[1]
    real_add = pipeline.vectorstore.add
    monkeypatch.setattr(
        pipeline.vectorstore, "add",
        lambda key, chunks, vecs: False if chunks and chunks[0].paper_id == b.paper_id else real_add(key, chunks, vecs),
    )
    ix = _indexes(corpus)
    pipeline.ensure_embeddings(_cfg(), ix, [a.paper_id, b.paper_id])
    assert ix[a.paper_id].in_store and not ix[b.paper_id].in_store and ix[b.paper_id].emb is not None
    seen = _spy_dense(monkeypatch)
    pipeline.answer(QUESTION, corpus, ix, [], [], _cfg())
    pids = {cid.rsplit("-", 1)[0] for cid in seen["dense"]}
    assert pids == {a.paper_id, b.paper_id}


def test_source_attributes_are_escaped(corpus, monkeypatch):
    seen = {}

    def capture(cfg, system, history, user, **k):
        seen["user"] = user
        return "The hybrid method worked best [S1].", ""

    monkeypatch.setattr(llm_mod, "chat_full", capture)
    monkeypatch.setattr(reranker, "_load_ranker", lambda: PickLast())
    papers = [load_pdf(p.name, p.read_bytes()) for p in SAMPLE]
    papers[0].title = 'Say "hi" <b>loud</b>'
    cfg = LLMConfig(provider="openai", api_key="k", model="m", embed_model="")
    pipeline.answer("Which retrieval method worked best?", papers, _indexes(papers), [], [], cfg)
    assert 'paper_title="Say "hi"' not in seen["user"]
    assert "Say &quot;hi&quot; &lt;b&gt;loud&lt;/b&gt;" in seen["user"]


def test_reply_carries_reasoning(corpus, monkeypatch):
    monkeypatch.setattr(llm_mod, "chat_full", lambda *a, **k: ("The hybrid method worked best [S1].", "Looked at S1."))
    monkeypatch.setattr(reranker, "_load_ranker", lambda: PickLast())
    cfg = LLMConfig(provider="openai", api_key="k", model="m")
    rep = pipeline.answer("Which retrieval method worked best?", corpus, _indexes(corpus), [], [], cfg)
    assert rep.mode == "ai" and rep.reasoning == "Looked at S1."
    assert "Looked at S1" not in rep.plain  # reasoning never goes back into chat history
