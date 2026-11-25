import pytest

from conftest import make_chunks
from core import reranker
from core.models import Hit

TEXTS = [
    "The classroom used tablets for every lesson in the semester.",
    "Teachers reported higher engagement during group activities.",
    "Only 42 students participated, so the small sample size limits generalizability.",
]


def _hits():
    return [Hit(c, 1.0 / (i + 1), 1.0, None, 1) for i, c in enumerate(make_chunks("p1", TEXTS))]


class FakeRanker:
    calls = 0

    def rerank(self, req):
        FakeRanker.calls += 1
        return sorted(
            ({"id": p["id"], "text": p["text"], "score": 0.9 if "sample size" in p["text"] else 0.1} for p in req.passages),
            key=lambda d: -d["score"],
        )


@pytest.fixture(autouse=True)
def _reset():
    reranker.reset_for_tests()
    yield
    reranker.reset_for_tests()


def test_rerank_reorders(monkeypatch):
    monkeypatch.setattr(reranker, "_load_ranker", lambda: FakeRanker())
    out, notice = reranker.rerank("limitations of the sample size", _hits(), k=2)
    assert notice == ""
    assert len(out) == 2
    assert out[0].chunk.text == TEXTS[2]
    assert out[0].rerank == pytest.approx(0.9)
    assert out[1].rerank == pytest.approx(0.1)


def test_rerank_fallback_when_unavailable(monkeypatch):
    def boom():
        raise ImportError("No module named 'flashrank'")

    monkeypatch.setattr(reranker, "_load_ranker", boom)
    hits = _hits()
    out, notice = reranker.rerank("anything", hits, k=2)
    assert [h.chunk.chunk_id for h in out] == [h.chunk.chunk_id for h in hits[:2]]
    assert notice.startswith("Reranker unavailable")
    # a failed load is remembered, not retried on every question
    out2, notice2 = reranker.rerank("anything", _hits(), k=2)
    assert notice2.startswith("Reranker unavailable")


def test_rerank_empty_hits(monkeypatch):
    monkeypatch.setattr(reranker, "_load_ranker", lambda: FakeRanker())
    assert reranker.rerank("q", [], k=6) == ([], "")


@pytest.mark.slow
def test_rerank_real_model():
    pytest.importorskip("flashrank")
    try:
        out, notice = reranker.rerank("What were the limitations of the sample size?", _hits(), k=3)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"model unavailable: {e}")
    if notice:
        pytest.skip(notice)
    assert out[0].chunk.text == TEXTS[2]


def test_references_section_penalised(monkeypatch):
    class RefFirst:
        def rerank(self, req):
            return [
                {"id": p["id"], "text": p["text"], "score": 0.8 if "Sample, A." in p["text"] else 0.5}
                for p in req.passages
            ]

    monkeypatch.setattr(reranker, "_load_ranker", lambda: RefFirst())
    body = make_chunks("p1", ["Only 42 students took part, which limits generalizability of the findings."])
    refs = make_chunks("p1", ["Sample, A. (2020). An example reference used for demonstration only."], page=3, section="references")
    refs[0].chunk_id = "p1-ref"
    hits = [Hit(refs[0], 1.0, 1.0, None, 1), Hit(body[0], 0.5, 1.0, None, 1)]
    out, _ = reranker.rerank("sample size limitations", hits, k=2)
    assert out[0].chunk.section != "references"
