"""Vector store contract.

Tenant isolation is enforced by the *signature*: `search` requires a tenant_id and a set of
visibilities, so there is no code path that can query across tenants or ignore visibility.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Collection

import numpy as np

from ..models import Chunk, Hit


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...

    @abstractmethod
    def search(self, tenant_id: str, query: np.ndarray, top_k: int, visibilities: Collection[str]) -> list[Hit]: ...

    @abstractmethod
    def count(self, tenant_id: str | None = None) -> int: ...

    @abstractmethod
    def delete_tenant(self, tenant_id: str) -> None: ...

    def save(self) -> None:  # optional persistence hook
        return None
