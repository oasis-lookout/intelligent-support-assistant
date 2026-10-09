"""Offline ingestion path: tickets -> redact -> chunk -> embed -> upsert."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Iterable

from ..models import Chunk, Ticket
from ..observability import log
from ..providers.base import Embedder
from ..stores.base import VectorStore
from .chunker import chunk_ticket

logger = logging.getLogger(__name__)


@dataclass
class IngestStats:
    tickets: int = 0
    chunks: int = 0
    seconds: float = 0.0


def ingest(tickets: Iterable[Ticket], embedder: Embedder, store: VectorStore, *,
           batch_size: int = 64, tenant: str | None = None, max_words: int = 160) -> IngestStats:
    stats = IngestStats()
    t0 = time.perf_counter()
    batch: list[Chunk] = []

    def flush() -> None:
        if not batch:
            return
        vecs = embedder.embed_documents([c.text for c in batch])
        store.upsert(batch, vecs)
        stats.chunks += len(batch)
        batch.clear()

    for t in tickets:
        if tenant and t.company_id != tenant:
            continue
        stats.tickets += 1
        batch.extend(chunk_ticket(t, max_words=max_words))
        if len(batch) >= batch_size:
            flush()
    flush()
    stats.seconds = round(time.perf_counter() - t0, 2)
    log(logger, logging.INFO, "ingest complete", tickets=stats.tickets, chunks=stats.chunks, seconds=stats.seconds)
    return stats
