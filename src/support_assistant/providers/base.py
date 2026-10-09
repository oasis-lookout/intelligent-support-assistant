"""Provider contracts. The RAG pipeline only ever talks to these interfaces, which is what lets
the same code run on Bedrock (AWS-managed) or vLLM + local vectors (self-hosted)."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    name: str
    dim: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Returns an (n, dim) float32 array of L2-normalised vectors."""

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class LLM(ABC):
    name: str

    @abstractmethod
    def generate(self, system: str, messages: list[dict], max_tokens: int, temperature: float) -> str:
        """messages: [{"role": "user"|"assistant", "content": str}, ...] ending with a user turn."""


def normalize(vecs: np.ndarray) -> np.ndarray:
    vecs = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def alternate_roles(messages: list[dict]) -> list[dict]:
    """Chat APIs (Bedrock Converse, some vLLM templates) require strictly alternating roles that
    start with 'user'. Merge consecutive same-role turns and drop a leading assistant turn."""
    out: list[dict] = []
    for m in messages:
        if not out and m["role"] != "user":
            continue
        if out and out[-1]["role"] == m["role"]:
            out[-1] = {"role": m["role"], "content": out[-1]["content"] + "\n" + m["content"]}
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out
