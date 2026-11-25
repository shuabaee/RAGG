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
