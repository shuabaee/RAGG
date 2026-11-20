"""Language-model + embedding providers (plain HTTPS, no heavy SDKs).

Supported:
- "ollama" / "llama": local Ollama server (e.g. http://localhost:11434)
- "gemini"          : Google Gemini API (free tier available at aistudio.google.com)
- "openai"          : any OpenAI-compatible API (OpenAI, Groq, OpenRouter, ...) via base_url
- "none"            : no model -> RAGG runs in extractive mode (retrieval + verbatim excerpts only)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import requests

TIMEOUT = 90


class LLMError(Exception):
    pass


@dataclass
class LLMConfig:
    provider: str = "none"  # ollama | llama | gemini | openai | none
    api_key: str = ""
    model: str = ""
    base_url: str = ""

    @property
    def enabled(self) -> bool:
        if self.provider in ("ollama", "llama"):
            return True
        return self.provider in ("gemini", "openai") and bool(self.api_key)

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
        return {"gemini": "gemini-2.5-flash", "openai": "gpt-4o-mini", "ollama":"llama3.2:1b"}.get(self.provider, "")

    def label(self) -> str:
        if not self.enabled:
            return "Extractive mode (no AI model)"
        prov_name = "Ollama" if self.provider in ("ollama", "llama") else self.provider.title()
        return f"{prov_name} · {self.resolved_model()}"


@dataclass
class EmbedConfig:
    provider: str = "none"  # ollama | llama | gemini | openai | none
    api_key: str = ""
    model: str = ""
    base_url: str = ""

    @property
    def enabled(self) -> bool:
        if self.provider in ("ollama", "llama"):
            True
        return self.provider in ("gemini", "openai") and bool(self.api_key)

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
        return {
            "gemini": "gemini-embedding-001",
            "openai": "text-embedding-3-small",
            "ollama": "nomic-embed-text",
        }.get(self.provider, "")

    def label(self) -> str:
        if not self.enabled or not self.resolved_model():
            return "BM25 only (no embedding model)"
        prov_name = "Ollama" if self.provider in ("ollama", "llama") else self.provider.title()
        return f"{prov_name} · {self.resolved_model()}"

    def label(self) -> str:
        if not self.enabled or not self.resolved_model():
            return "BM25 only (no embedding model)"
        prov_name = "Ollama" if self.provider in ("ollama", "llama") else self.provider.title()
        return f"{prov_name} · {self.resolved_model()}"


def _raise(resp: requests.Response, who: str):
    try:
        detail = resp.json()
        msg = detail.get("error", {}).get("message") if isinstance(detail.get("error"), dict) else str(detail)[:300]
    except Exception:
        msg = resp.text[:300]
    raise LLMError(f"{who} error {resp.status_code}: {msg}")


def chat(cfg: LLMConfig, system: str, history: list[dict], user: str, temperature: float = 0.2, max_tokens: int = 2048) -> str:
    """history: list of {"role": "user"|"assistant", "content": str}"""
    if cfg.provider in ("ollama", "llama"):
        base = cfg.resolved_base_url()
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": m["role"], "content": m["content"]} for m in history
        ] + [{"role": "user", "content": user}]
        body = {
            "model": cfg.resolved_model(),
            "messages": msgs,
            "stream": False,
            "think": False,
            "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": 8192},
        }
        r = requests.post(f"{base}/api/chat", json=body, timeout=TIMEOUT*2)
        if r.status_code != 200:
            _raise(r, "Ollama")
        data = r.json()
        msg = data.get("message") or {}
        content = (msg.get("content") or "").strip()
        if not content:
            content = (msg.get("thinking") or "").strip()
        return content

    if cfg.provider == "gemini":
        contents = []
        for m in history:
            contents.append({"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]})
        contents.append({"role": "user", "parts": [{"text": user}]})
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{cfg.resolved_model()}:generateContent"
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": contents,
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        r = requests.post(url, params={"key": cfg.api_key}, json=body, timeout=TIMEOUT)
        if r.status_code != 200:
            _raise(r, "Gemini")
        data = r.json()
        cands = data.get("candidates") or []
        if not cands:
            reason = (data.get("promptFeedback") or {}).get("blockReason", "no answer returned")
            raise LLMError(f"Gemini returned no answer ({reason}).")
        parts = (cands[0].get("content") or {}).get("parts") or []
        return "".join(p.get("text", "") for p in parts).strip()

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
        return (r.json()["choices"][0]["message"]["content"] or "").strip()

    raise LLMError("No language model configured.")


def embed(cfg: EmbedConfig, texts: list[str], is_query: bool = False) -> np.ndarray | None:
    """Return L2-normalised embeddings, or None if the provider has no embedding model."""
    model = cfg.resolved_model()
    if not cfg.enabled or not model or not texts:
        return None
    vecs: list[list[float]] = []
    if cfg.provider == "gemini":
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents"
        task = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
        for i in range(0, len(texts), 90):
            batch = texts[i : i + 90]
            body = {
                "requests": [
                    {"model": f"models/{model}", "content": {"parts": [{"text": t[:6000]}]}, "taskType": task}
                    for t in batch
                ]
            }
            r = requests.post(url, params={"key": cfg.api_key}, json=body, timeout=TIMEOUT)
            if r.status_code != 200:
                _raise(r, "Gemini embeddings")
            vecs.extend(e["values"] for e in r.json()["embeddings"])
    elif cfg.provider in ("ollama", "llama"):
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