"""Plain data types shared across the pipeline (no third-party dependencies)."""
from __future__ import annotations

from dataclasses import dataclass, field

PUBLIC = "public"      # customer-safe knowledge (past conversations)
INTERNAL = "internal"  # agent-only knowledge (internal notes)


@dataclass(frozen=True)
class Ticket:
    ticket_id: str
    company_id: str
    conversation: str
    status: str
    internal_notes: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    tenant_id: str          # == company_id; the unit of data isolation
    ticket_id: str
    visibility: str         # PUBLIC | INTERNAL
    text: str
    ticket_status: str = ""


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float            # cosine similarity in [-1, 1]


@dataclass(frozen=True)
class SessionContext:
    """Identity attached to a request. Derived from a signed token, never from the request body."""
    session_id: str
    tenant_id: str
    role: str               # "customer" | "agent"


@dataclass
class Answer:
    text: str
    confidence: float
    handoff_recommended: bool
    fallback_reason: str | None = None
    sources: list[dict] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
