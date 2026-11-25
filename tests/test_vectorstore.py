import numpy as np
import pytest

from conftest import make_chunks

TEXTS = ["alpha beta gamma words here", "delta epsilon zeta words here", "eta theta iota words here"]


def _unit(dim, i):
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1.0
    return v


def test_roundtrip_and_reuse(store):
    chunks = make_chunks("p1", TEXTS)
    vecs = np.stack([_unit(3, i) for i in range(3)])
    assert not store.has_paper("ollama:m", "p1", chunks)
    store.add("ollama:m", chunks, vecs)
    assert store.has_paper("ollama:m", "p1", chunks)
    res = store.query("ollama:m", _unit(3, 1), ["p1"], n=40)
    best = max(res, key=res.get)
    assert best == "p1-2"
    assert res["p1-2"] == pytest.approx(1.0, abs=1e-4)


def test_stale_text_hash_readds(store):
    chunks = make_chunks("p1", TEXTS)
    store.add("ollama:m", chunks, np.stack([_unit(3, i) for i in range(3)]))
    changed = make_chunks("p1", [TEXTS[0], "totally different text now", TEXTS[2]])
    assert not store.has_paper("ollama:m", "p1", changed)
    assert store.query("ollama:m", _unit(3, 0), ["p1"], n=40) == {}


def test_query_filters_by_paper(store):
    a = make_chunks("pa", TEXTS)
    b = make_chunks("pb", TEXTS)
    vecs = np.stack([_unit(3, i) for i in range(3)])
    store.add("ollama:m", a, vecs)
    store.add("ollama:m", b, vecs)
    res = store.query("ollama:m", _unit(3, 0), ["pb"], n=40)
    assert res and all(cid.startswith("pb-") for cid in res)
    res2 = store.query("ollama:m", _unit(3, 0), ["pa", "pb"], n=40)
    assert {c[:2] for c in res2} == {"pa", "pb"}


def test_separate_collections_per_key(store):
    chunks = make_chunks("p1", TEXTS)
    store.add("ollama:A", chunks, np.stack([_unit(3, i) for i in range(3)]))
    store.add("gemini:B", chunks, np.stack([_unit(5, i) for i in range(3)]))
    assert store.query("ollama:A", _unit(3, 0), ["p1"])
    assert store.query("gemini:B", _unit(5, 0), ["p1"])
    assert store.has_paper("ollama:A", "p1", chunks) and store.has_paper("gemini:B", "p1", chunks)


def test_query_unknown_key_is_empty(store):
    assert store.query("none:x", _unit(3, 0), ["p1"]) == {}


def test_add_reports_success(store):
    chunks = make_chunks("p1", TEXTS)
    assert store.add("ollama:m", chunks, np.stack([_unit(3, i) for i in range(3)])) is True


def test_dimension_change_same_key_recovers(store):
    chunks = make_chunks("p1", TEXTS)
    store.add("ollama:m", chunks, np.stack([_unit(3, i) for i in range(3)]))
    # the model behind the same name now returns 5-d vectors
    assert store.add("ollama:m", chunks, np.stack([_unit(5, i) for i in range(3)])) is True
    res = store.query("ollama:m", _unit(5, 2), ["p1"])
    assert max(res, key=res.get) == "p1-3"
