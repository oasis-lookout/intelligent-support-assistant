"""Offline providers: no network, no GPU, no AWS account. Used for local dev, CI and unit tests.

HashEmbedder is a feature-hashing bag-of-words(+bigrams) model - lexical, not semantic. It exists to
exercise the plumbing and to give the offline evaluation something deterministic to run against.
ExtractiveLLM "answers" by quoting the best-matching past agent replies; it is NOT a language model.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

import numpy as np

from .base import LLM, Embedder, normalize

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset("""a an the and or but if then so of to in on at for from by with about as is are was were be been am i me my
mine we our you your he she it its they them their this that these those there here do does did done have has had will would
can could should may might not no yes please thanks thank hi hello ok okay just also very really what which who whom when where
why how into out up down over under again more most some any all each other than too s t get got""".split())


def _features(text: str) -> Counter:
    toks = [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]
    feats: Counter = Counter(toks)
    for a, b in zip(toks, toks[1:]):
        feats[f"{a}_{b}"] += 0.7
    return feats


class HashEmbedder(Embedder):
    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hash-{dim}"

    def _one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for f, c in _features(text).items():
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
            sign = 1.0 if (h >> 40) & 1 else -1.0
            v[h % self.dim] += sign * (1.0 + math.log(max(c, 1.0)))
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return normalize(np.stack([self._one(t) for t in texts]))


class ExtractiveLLM(LLM):
    name = "extractive-stub"

    def generate(self, system: str, messages: list[dict], max_tokens: int, temperature: float) -> str:
        prompt = messages[-1]["content"]
        blocks = re.split(r"(?m)^\[S\d+\]\s*", prompt)[1:]
        public = [b for b in blocks if b.startswith("(public")]
        internal = [b for b in blocks if b.startswith("(internal")]
        if not public and not internal:
            return "I could not find relevant information."
        out: list[str] = []
        if public:
            agent_lines = [ln[len("Agent:"):].strip() for ln in public[0].splitlines() if ln.startswith("Agent:")]
            out.append(" ".join(agent_lines[:3]) or "Our team will review your request.")
        if internal:
            note = "\n".join(internal[0].splitlines()[2:]).strip() or internal[0].strip()
            out.append(f"Internal guidance: {note}")
        return "\n\n".join(out)
