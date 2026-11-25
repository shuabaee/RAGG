"""Language-model + embedding providers (plain HTTPS, no heavy SDKs).

Supported:
  - "ollama" / "llama": local Ollama server (e.g. http://localhost:11434)
  - "openai"          : any OpenAI-compatible API (OpenAI, Groq, OpenRouter, ...) via base_url
  - "none"            : internal fallback when nothing is configured -> extractive mode
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

import numpy as np
import requests

TIMEOUT = 90
TAGS_TTL = 15.0  # seconds to reuse Ollama's model list (Streamlit re-runs the script on every click)
EMBED_FAMILIES = ("nomic-embed-text", "mxbai-embed-large", "bge-m3", "snowflake-arctic-embed", "all-minilm", "embed")
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S | re.I)
_tags_cache: dict[str, tuple[float, list[str]]] = {}


def clear_model_cache() -> None:
    _tags_cache.clear()


def ollama_models(base: str) -> list[str]:
    """Installed Ollama model names, cached for TAGS_TTL seconds ([] when Ollama is unreachable)."""
    base = base.rstrip("/")
    hit = _tags_cache.get(base)
    if hit and time.time() - hit[0] < TAGS_TTL:
        return hit[1]
    names: list[str] = []
    try:
        r = requests.get(f"{base}/api/tags", timeout=1.0)
        if r.status_code == 200:
            names = [m.get("name") for m in r.json().get("models", []) if m.get("name")]
    except Exception:
        pass
    _tags_cache[base] = (time.time(), names)
    return names


def is_embedding_model(name: str) -> bool:
    return any(f in name.lower() for f in EMBED_FAMILIES)


def ollama_chat_models(base: str) -> list[str]:
    return [n for n in ollama_models(base) if not is_embedding_model(n)]


class LLMError(Exception):
    pass


@dataclass
class LLMConfig:
    provider: str = "ollama"  # ollama | llama | openai | none (internal fallback)
    api_key: str = ""
    model: str = ""
    base_url: str = ""
    embed_model: str = ""
    rerank: bool = True  # rerank retrieved passages with FlashRank
    think: bool = True  # Ollama: let reasoning models think first with a bigger budget; reasoning shown separately

    @property
    def enabled(self) -> bool:
        if self.provider in ("ollama", "llama"):
            return bool(self.resolved_model())
        return self.provider == "openai" and bool(self.api_key)

    def resolved_base_url(self) -> str:
        if self.base_url:
            return self.base_url.rstrip("/")
        if self.provider in ("ollama", "llama"):
            return "http://localhost:11434"
        if self.provider == "openai":
            return "https://api.openai.com/v1"
        return ""

    def resolved_model(self) -> str:
        if self.model:
            return self.model
        if self.provider in ("ollama", "llama"):
            chat_models = ollama_chat_models(self.resolved_base_url())
            return chat_models[0] if chat_models else ""
        return {"openai": "gpt-4o-mini"}.get(self.provider, "")

    def resolved_embed(self) -> str:
        if self.embed_model:
            return self.embed_model
        if self.provider in ("ollama", "llama"):
            names = ollama_models(self.resolved_base_url())
            for family in EMBED_FAMILIES:  # in order of preference
                for n in names:
                    if family in n:
                        return n  # the installed tag, e.g. "nomic-embed-text:v1.5"
            return ""  # no embedding model installed -> keyword search only
        if self.provider == "openai" and "api.openai.com" in (self.base_url or "https://api.openai.com/v1"):
            return "text-embedding-3-small"
        return ""

    def label(self) -> str:
        if not self.enabled or not self.resolved_model():
            if self.provider in ("ollama", "llama"):
                return "Ollama · No model detected"
            return "Extractive mode (no AI model)"
        prov_name = "Ollama" if self.provider in ("ollama", "llama") else "ChatGPT"
        return f"{prov_name} · {self.resolved_model()}"


def _raise(resp: requests.Response, who: str):
    try:
        detail = resp.json()
        msg = detail.get("error", {}).get("message") if isinstance(detail.get("error"), dict) else str(detail)[:300]
    except Exception:
        msg = resp.text[:300]
    raise LLMError(f"{who} error {resp.status_code}: {msg}")


_NO_THINK: set[str] = set()  # Ollama models that rejected "think": true


def _split_reasoning(content: str, thinking: str = "") -> tuple[str, str]:
    """Separate a reasoning model's private reasoning from its answer."""
    answer = _THINK_RE.sub("", content or "")
    parts = [(thinking or "").strip(), *(t.strip() for t in _THINK_RE.findall(content or ""))]
    low = answer.lower()
    if "</think>" in low:  # opening tag missing (qwen3 with thinking off): all before it is reasoning
        cut = low.rindex("</think>")
        parts.append(answer[:cut].replace("<think>", "").strip())
        answer = answer[cut + len("</think>"):]
    elif "<think>" in low:  # never closed: it ran out of tokens while reasoning
        cut = low.index("<think>")
        parts.append(answer[cut + len("<think>"):].strip())
        answer = answer[:cut]
    answer = answer.strip()
    reasoning = "\n\n".join(x for x in parts if x)
    return answer, reasoning


def chat_full(cfg: LLMConfig, system: str, history: list[dict], user: str, temperature: float = 0.2,
              max_tokens: int = 2048) -> tuple[str, str]:
    """Return (answer, reasoning). Reasoning is "" for models that do not expose it.

    history: list of {"role": "user"|"assistant", "content": str}
    """
    if cfg.provider in ("ollama", "llama"):
        base = cfg.resolved_base_url()
        model = cfg.resolved_model()
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": m["role"], "content": m["content"]} for m in history
        ] + [{"role": "user", "content": user}]

        def _post(think: bool):
            body = {
                "model": model,
                "messages": msgs,
                "stream": False,
                "think": think,
                # reasoning eats tokens before the answer starts, so give it room
                "options": {"temperature": temperature, "num_predict": max_tokens * (3 if think else 1), "num_ctx": 8192},
            }
            return requests.post(f"{base}/api/chat", json=body, timeout=TIMEOUT * (3 if think else 1))

        think = cfg.think and model not in _NO_THINK
        r = _post(think)
        if think and r.status_code == 400 and "think" in (r.text or "").lower():
            _NO_THINK.add(model)  # this model cannot reason separately; don't ask again
            r = _post(False)
        if r.status_code != 200:
            _raise(r, "Ollama")
        msg = r.json().get("message") or {}
        answer, reasoning = _split_reasoning(msg.get("content") or "", msg.get("thinking") or "")
        if not answer:
            raise LLMError("the model returned no answer (it may have used its whole token budget on reasoning)")
        return answer, reasoning

    if cfg.provider == "openai":
        base = (cfg.base_url or "https://api.openai.com/v1").rstrip("/")
        msgs = [{"role": "system", "content": system}] + [
            {"role": m["role"], "content": m["content"]} for m in history
        ] + [{"role": "user", "content": user}]
        r = requests.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {cfg.api_key}"},
            json={"model": cfg.resolved_model(), "messages": msgs, "temperature": temperature, "max_tokens": max_tokens},
            timeout=TIMEOUT,
        )
        if r.status_code != 200:
            _raise(r, "Model API")
        msg = r.json()["choices"][0]["message"]
        answer, reasoning = _split_reasoning(msg.get("content") or "", msg.get("reasoning_content") or "")
        if not answer:
            raise LLMError("the model returned no answer")
        return answer, reasoning

    raise LLMError("No language model configured.")


def chat(cfg: LLMConfig, system: str, history: list[dict], user: str, temperature: float = 0.2, max_tokens: int = 2048) -> str:
    """Answer only (reasoning, if any, is dropped)."""
    return chat_full(cfg, system, history, user, temperature, max_tokens)[0]


def embed(cfg: LLMConfig, texts: list[str], is_query: bool = False) -> np.ndarray | None:
    """Return L2-normalised embeddings, or None if the provider has no embedding model."""
    model = cfg.resolved_embed()
    if not cfg.enabled or not model or not texts:
        return None
    vecs: list[list[float]] = []
    if cfg.provider in ("ollama", "llama"):
        base = cfg.resolved_base_url()
        for i in range(0, len(texts), 64):
            batch = [t[:4000] for t in texts[i : i + 64]]
            r = requests.post(
                f"{base}/api/embed",
                json={"model": model, "input": batch},
                timeout=TIMEOUT,
            )
            if r.status_code != 200:
                _raise(r, "Ollama embeddings")
            vecs.extend(r.json().get("embeddings", []))
    else:
        base = (cfg.base_url or "https://api.openai.com/v1").rstrip("/")
        for i in range(0, len(texts), 256):
            batch = [t[:8000] for t in texts[i : i + 256]]
            r = requests.post(
                f"{base}/embeddings",
                headers={"Authorization": f"Bearer {cfg.api_key}"},
                json={"model": model, "input": batch},
                timeout=TIMEOUT,
            )
            if r.status_code != 200:
                _raise(r, "Embeddings")
            vecs.extend(d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"]))
    arr = np.array(vecs, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    return arr
