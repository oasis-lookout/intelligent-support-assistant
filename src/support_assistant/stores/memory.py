"""NumPy-backed vector store with per-tenant matrices. Fine up to ~10^5 chunks; persists to disk."""
from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict
from typing import Collection

import numpy as np

from ..models import Chunk, Hit
from .base import VectorStore


class InMemoryVectorStore(VectorStore):
    def __init__(self, path: str | None = None, dim: int | None = None) -> None:
        self.path, self.dim = path, dim
        self._chunks: dict[str, Chunk] = {}
        self._vecs: dict[str, np.ndarray] = {}
        self._by_tenant: dict[str, set[str]] = {}
        self._cache: dict[str, tuple[list[str], np.ndarray]] = {}
        self._lock = threading.RLock()
        if path and os.path.exists(os.path.join(path, "manifest.json")):
            self._load()

    # -- write path ---------------------------------------------------------------
    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks/vectors length mismatch")
        with self._lock:
            for c, v in zip(chunks, vectors):
                if self.dim is None:
                    self.dim = int(v.shape[0])
                if v.shape[0] != self.dim:
                    raise ValueError(f"vector dim {v.shape[0]} != store dim {self.dim}")
                self._chunks[c.chunk_id] = c
                self._vecs[c.chunk_id] = np.asarray(v, dtype=np.float32)
                self._by_tenant.setdefault(c.tenant_id, set()).add(c.chunk_id)
                self._cache.pop(c.tenant_id, None)

    def delete_tenant(self, tenant_id: str) -> None:
        with self._lock:
            for cid in self._by_tenant.pop(tenant_id, set()):
                self._chunks.pop(cid, None)
                self._vecs.pop(cid, None)
            self._cache.pop(tenant_id, None)

    # -- read path ----------------------------------------------------------------
    def _matrix(self, tenant_id: str) -> tuple[list[str], np.ndarray] | None:
        with self._lock:
            if tenant_id not in self._by_tenant or not self._by_tenant[tenant_id]:
                return None
            if tenant_id not in self._cache:
                ids = sorted(self._by_tenant[tenant_id])
                self._cache[tenant_id] = (ids, np.stack([self._vecs[i] for i in ids]))
            return self._cache[tenant_id]

    def search(self, tenant_id: str, query: np.ndarray, top_k: int, visibilities: Collection[str]) -> list[Hit]:
        m = self._matrix(tenant_id)
        if m is None:
            return []
        ids, mat = m
        scores = mat @ np.asarray(query, dtype=np.float32)
        allowed = np.array([self._chunks[i].visibility in visibilities for i in ids])
        scores = np.where(allowed, scores, -np.inf)
        k = min(top_k, int(allowed.sum()))
        if k <= 0:
            return []
        order = np.argsort(-scores)[:k]
        return [Hit(self._chunks[ids[i]], float(scores[i])) for i in order]

    def count(self, tenant_id: str | None = None) -> int:
        with self._lock:
            return len(self._chunks) if tenant_id is None else len(self._by_tenant.get(tenant_id, ()))

    def tenants(self) -> list[str]:
        return sorted(t for t, ids in self._by_tenant.items() if ids)

    # -- persistence --------------------------------------------------------------
    def save(self) -> None:
        if not self.path:
            return
        os.makedirs(self.path, exist_ok=True)
        with self._lock:
            ids = sorted(self._chunks)
            mat = np.stack([self._vecs[i] for i in ids]) if ids else np.zeros((0, self.dim or 0), np.float32)
            np.save(os.path.join(self.path, "vectors.npy"), mat)
            with open(os.path.join(self.path, "chunks.jsonl"), "w", encoding="utf-8") as fh:
                for i in ids:
                    fh.write(json.dumps(asdict(self._chunks[i])) + "\n")
            with open(os.path.join(self.path, "manifest.json"), "w") as fh:
                json.dump({"dim": self.dim, "count": len(ids)}, fh)

    def _load(self) -> None:
        with open(os.path.join(self.path, "manifest.json")) as fh:
            manifest = json.load(fh)
        if self.dim is not None and manifest["dim"] != self.dim:
            raise ValueError(f"stored vectors have dim={manifest['dim']} but EMBEDDING_DIM={self.dim}; re-ingest")
        self.dim = manifest["dim"]
        mat = np.load(os.path.join(self.path, "vectors.npy"))
        with open(os.path.join(self.path, "chunks.jsonl"), encoding="utf-8") as fh:
            chunks = [Chunk(**json.loads(line)) for line in fh if line.strip()]
        for c, v in zip(chunks, mat):
            self._chunks[c.chunk_id] = c
            self._vecs[c.chunk_id] = v
            self._by_tenant.setdefault(c.tenant_id, set()).add(c.chunk_id)
