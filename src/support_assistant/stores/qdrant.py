"""Qdrant adapter (runs on EKS for both deployment modes). Single collection; tenant_id and
visibility are indexed payload fields and are *always* part of the query filter."""
from __future__ import annotations

import uuid
from typing import Collection

import numpy as np

from ..models import Chunk, Hit
from .base import VectorStore

_NS = uuid.UUID("6f1c2f0e-3b5a-4c8e-9a47-5d0c1f5a9b11")


def _pid(chunk_id: str) -> str:
    return str(uuid.uuid5(_NS, chunk_id))


class QdrantVectorStore(VectorStore):
    def __init__(self, url: str, collection: str, dim: int, api_key: str = "") -> None:
        from qdrant_client import QdrantClient

        self.client = QdrantClient(url=url, api_key=api_key or None, timeout=30)
        self.collection, self.dim = collection, dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        from qdrant_client import models as qm

        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection not in existing:
            self.client.create_collection(self.collection,
                                          vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE))
            for field in ("tenant_id", "visibility", "ticket_id"):
                self.client.create_payload_index(self.collection, field_name=field,
                                                 field_schema=qm.PayloadSchemaType.KEYWORD)
        else:
            size = self.client.get_collection(self.collection).config.params.vectors.size
            if size != self.dim:
                raise ValueError(f"collection {self.collection!r} has dim={size}, expected {self.dim}")

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        from qdrant_client import models as qm

        points = [qm.PointStruct(id=_pid(c.chunk_id), vector=v.tolist(),
                                 payload={"chunk_id": c.chunk_id, "tenant_id": c.tenant_id, "ticket_id": c.ticket_id,
                                          "visibility": c.visibility, "text": c.text,
                                          "ticket_status": c.ticket_status})
                  for c, v in zip(chunks, vectors)]
        self.client.upsert(self.collection, points=points, wait=True)

    def search(self, tenant_id: str, query: np.ndarray, top_k: int, visibilities: Collection[str]) -> list[Hit]:
        from qdrant_client import models as qm

        flt = qm.Filter(must=[
            qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id)),
            qm.FieldCondition(key="visibility", match=qm.MatchAny(any=list(visibilities))),
        ])
        res = self.client.query_points(self.collection, query=query.tolist(), query_filter=flt,
                                       limit=top_k, with_payload=True).points
        return [Hit(Chunk(chunk_id=p.payload["chunk_id"], tenant_id=p.payload["tenant_id"],
                          ticket_id=p.payload["ticket_id"], visibility=p.payload["visibility"],
                          text=p.payload["text"], ticket_status=p.payload.get("ticket_status", "")),
                    float(p.score)) for p in res]

    def count(self, tenant_id: str | None = None) -> int:
        from qdrant_client import models as qm

        flt = None if tenant_id is None else qm.Filter(
            must=[qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id))])
        return self.client.count(self.collection, count_filter=flt, exact=True).count

    def delete_tenant(self, tenant_id: str) -> None:
        from qdrant_client import models as qm

        self.client.delete(self.collection, points_selector=qm.FilterSelector(filter=qm.Filter(
            must=[qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id))])))
