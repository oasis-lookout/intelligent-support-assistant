"""OpenAI-compatible HTTP clients (vLLM, TGI, llama.cpp server...) using only the stdlib."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

import numpy as np

from .base import LLM, Embedder, alternate_roles, normalize


def _post(url: str, payload: dict, api_key: str = "", timeout: float = 120.0) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{url} -> HTTP {e.code}: {e.read()[:300]!r}") from e


class OpenAICompatEmbedder(Embedder):
    def __init__(self, base_url: str, model: str, dim: int, api_key: str = "", batch: int = 32) -> None:
        self.base_url, self.model, self.dim, self.api_key, self._batch = base_url, model, dim, api_key, batch
        self.name = f"openai:{model}:{dim}"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            data = _post(f"{self.base_url}/embeddings",
                         {"model": self.model, "input": texts[i:i + self._batch]}, self.api_key)["data"]
            out.extend(d["embedding"] for d in sorted(data, key=lambda d: d["index"]))
        arr = np.array(out, dtype=np.float32)
        if arr.shape[1] != self.dim:
            raise ValueError(f"embedding model returned dim={arr.shape[1]}, EMBEDDING_DIM={self.dim}")
        return normalize(arr)


class OpenAICompatLLM(LLM):
    def __init__(self, base_url: str, model: str, api_key: str = "") -> None:
        self.base_url, self.model, self.api_key = base_url, model, api_key
        self.name = f"openai:{model}"

    def generate(self, system: str, messages: list[dict], max_tokens: int, temperature: float) -> str:
        payload = {"model": self.model, "max_tokens": max_tokens, "temperature": temperature,
                   "messages": [{"role": "system", "content": system}] + alternate_roles(messages)}
        data = _post(f"{self.base_url}/chat/completions", payload, self.api_key)
        return data["choices"][0]["message"]["content"].strip()
