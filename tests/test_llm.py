import pytest

from core import llm
from core.llm import LLMConfig, LLMError


class Resp:
    def __init__(self, status, data):
        self.status_code, self._data, self.text = status, data, str(data)

    def json(self):
        return self._data


@pytest.fixture(autouse=True)
def _clear_cache():
    llm.clear_model_cache()
    yield
    llm.clear_model_cache()


def _tags(monkeypatch, names, counter=None):
    def get(url, timeout=None):
        if counter is not None:
            counter["n"] += 1
        return Resp(200, {"models": [{"name": n} for n in names]})

    monkeypatch.setattr(llm.requests, "get", get)


def test_default_chat_model_skips_embedding_models(monkeypatch):
    _tags(monkeypatch, ["nomic-embed-text:latest", "llama3:latest"])
    assert LLMConfig(provider="ollama").resolved_model() == "llama3:latest"


def test_only_embedding_models_means_no_chat_model(monkeypatch):
    _tags(monkeypatch, ["nomic-embed-text:latest"])
    cfg = LLMConfig(provider="ollama")
    assert cfg.resolved_model() == "" and not cfg.enabled


def test_embed_model_uses_installed_tag(monkeypatch):
    _tags(monkeypatch, ["llama3:latest", "nomic-embed-text:v1.5"])
    assert LLMConfig(provider="ollama").resolved_embed() == "nomic-embed-text:v1.5"


def test_chat_models_list_excludes_embedders(monkeypatch):
    _tags(monkeypatch, ["qwen3:4b", "bge-m3:latest", "llama3:latest"])
    assert llm.ollama_chat_models("http://localhost:11434") == ["qwen3:4b", "llama3:latest"]


def test_tags_are_cached_between_calls(monkeypatch):
    counter = {"n": 0}
    _tags(monkeypatch, ["llama3:latest", "nomic-embed-text:latest"], counter)
    cfg = LLMConfig(provider="ollama")
    for _ in range(5):
        cfg.enabled, cfg.label(), cfg.resolved_embed()
    assert counter["n"] == 1


def test_unreachable_ollama_is_cached_too(monkeypatch):
    counter = {"n": 0}

    def down(url, timeout=None):
        counter["n"] += 1
        raise llm.requests.ConnectionError("refused")

    monkeypatch.setattr(llm.requests, "get", down)
    cfg = LLMConfig(provider="ollama")
    for _ in range(4):
        assert not cfg.enabled
    assert counter["n"] == 1


def _chat_reply(monkeypatch, message):
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: Resp(200, {"message": message}))


def test_chat_never_returns_hidden_reasoning(monkeypatch):
    _chat_reply(monkeypatch, {"content": "", "thinking": "Okay, the user wants me to..."})
    with pytest.raises(LLMError):
        llm.chat(LLMConfig(provider="ollama", model="qwen3:4b"), "sys", [], "q")


def test_chat_strips_think_tags(monkeypatch):
    _chat_reply(monkeypatch, {"content": "<think>secret plan</think>\nThe hybrid method worked best [S1]."})
    out = llm.chat(LLMConfig(provider="ollama", model="qwen3:4b"), "sys", [], "q")
    assert out == "The hybrid method worked best [S1]."


def test_chat_full_returns_reasoning_separately(monkeypatch):
    _chat_reply(monkeypatch, {"content": "Hybrid worked best [S1].", "thinking": "Compare S1 and S2 first."})
    content, reasoning = llm.chat_full(LLMConfig(provider="ollama", model="qwen3:4b"), "sys", [], "q")
    assert content == "Hybrid worked best [S1]." and reasoning == "Compare S1 and S2 first."


def test_think_tags_become_reasoning(monkeypatch):
    _chat_reply(monkeypatch, {"content": "<think>check S2</think>Answer [S2]."})
    content, reasoning = llm.chat_full(LLMConfig(provider="ollama", model="m"), "sys", [], "q")
    assert content == "Answer [S2]." and reasoning == "check S2"


def test_think_flag_retries_for_models_without_thinking(monkeypatch):
    bodies = []

    def post(url, json=None, timeout=None, **k):
        bodies.append(dict(json))
        if json.get("think"):
            return Resp(400, {"error": '"llama3:latest" does not support thinking'})
        return Resp(200, {"message": {"content": "Plain answer."}})

    monkeypatch.setattr(llm.requests, "post", post)
    cfg = LLMConfig(provider="ollama", model="llama3:latest", think=True)
    assert llm.chat_full(cfg, "sys", [], "q") == ("Plain answer.", "")
    assert [b["think"] for b in bodies] == [True, False]
    llm.chat_full(cfg, "sys", [], "q")  # remembered: no second failed attempt
    assert [b["think"] for b in bodies] == [True, False, False]


def _record_posts(monkeypatch):
    bodies = []

    def post(url, json=None, timeout=None, **k):
        bodies.append(json)
        return Resp(200, {"message": {"content": "ok"}})

    monkeypatch.setattr(llm.requests, "post", post)
    return bodies


def test_think_on_by_default_with_bigger_budget(monkeypatch):
    bodies = _record_posts(monkeypatch)
    llm.chat_full(LLMConfig(provider="ollama", model="qwen3:4b"), "sys", [], "q", max_tokens=2048)
    assert bodies[0]["think"] is True and bodies[0]["options"]["num_predict"] == 6144


def test_think_can_be_turned_off(monkeypatch):
    bodies = _record_posts(monkeypatch)
    llm.chat_full(LLMConfig(provider="ollama", model="qwen3:4b", think=False), "sys", [], "q", max_tokens=2048)
    assert bodies[0]["think"] is False and bodies[0]["options"]["num_predict"] == 2048


def test_reasoning_with_only_closing_tag(monkeypatch):
    # qwen3 with thinking off: reasoning, then a bare </think>, then the answer
    _chat_reply(monkeypatch, {"content": "Okay, the user wants the finding. S1 says 6%.\n</think>\n\nGrounding cut errors to 6% [S1]."})
    content, reasoning = llm.chat_full(LLMConfig(provider="ollama", model="qwen3:4b"), "sys", [], "q")
    assert content == "Grounding cut errors to 6% [S1]."
    assert reasoning == "Okay, the user wants the finding. S1 says 6%."


def test_unclosed_think_is_not_an_answer(monkeypatch):
    # ran out of tokens mid-reasoning: nothing usable to show as the answer
    _chat_reply(monkeypatch, {"content": "<think>Let me look at S1, S2 and"})
    with pytest.raises(LLMError):
        llm.chat_full(LLMConfig(provider="ollama", model="qwen3:4b"), "sys", [], "q")
