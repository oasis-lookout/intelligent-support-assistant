"""Turn a ticket into retrievable chunks.

* Conversation -> PUBLIC chunks (turn windows, usually one chunk per ticket).
* Internal notes -> INTERNAL chunks, prefixed with the customer's opening question so that
  an agent searching "how do we handle X" can find them.

Visibility is decided here, once, and enforced at retrieval time.
"""
from __future__ import annotations

import re

from ..models import INTERNAL, PUBLIC, Chunk, Ticket
from .redact import redact

_TURN = re.compile(r"^\s*(Customer|Agent)\s*:\s*(.*)$", re.IGNORECASE)
_SENT = re.compile(r"(?<=[.!?])\s+")


def parse_turns(conversation: str) -> list[str]:
    turns: list[str] = []
    for line in conversation.replace("\r\n", "\n").split("\n"):
        line = line.strip()
        if not line:
            continue
        m = _TURN.match(line)
        if m:
            turns.append(f"{m.group(1).capitalize()}: {m.group(2).strip()}")
        elif turns:                      # continuation of the previous turn
            turns[-1] += " " + line
        else:
            turns.append(line)
    return turns


def _words(s: str) -> int:
    return len(s.split())


def window_turns(turns: list[str], max_words: int, overlap: int = 1) -> list[list[str]]:
    windows: list[list[str]] = []
    cur: list[str] = []
    cur_words = 0
    for t in turns:
        w = _words(t)
        if cur and cur_words + w > max_words:
            windows.append(cur)
            cur = cur[-overlap:] if overlap else []
            cur_words = sum(_words(x) for x in cur)
        cur.append(t)
        cur_words += w
    if cur:
        windows.append(cur)
    return windows


def _split_sentences(text: str, max_words: int) -> list[str]:
    parts, cur, n = [], [], 0
    for s in _SENT.split(text.strip()):
        w = _words(s)
        if cur and n + w > max_words:
            parts.append(" ".join(cur))
            cur, n = [], 0
        cur.append(s)
        n += w
    if cur:
        parts.append(" ".join(cur))
    return parts


def chunk_ticket(t: Ticket, max_words: int = 160) -> list[Chunk]:
    conv = redact(t.conversation)
    notes = redact(t.internal_notes or "")
    turns = parse_turns(conv)
    chunks: list[Chunk] = []

    for i, win in enumerate(window_turns(turns, max_words)):
        chunks.append(Chunk(chunk_id=f"{t.ticket_id}:c{i}", tenant_id=t.company_id, ticket_id=t.ticket_id,
                            visibility=PUBLIC, text="\n".join(win), ticket_status=t.status))

    if notes.strip():
        first_q = next((x for x in turns if x.lower().startswith("customer:")), "")
        topic = " ".join(first_q.split()[:40])
        for j, part in enumerate(_split_sentences(notes, max_words)):
            text = f"Internal notes for a ticket about: {topic}\n{part}" if topic else part
            chunks.append(Chunk(chunk_id=f"{t.ticket_id}:n{j}", tenant_id=t.company_id, ticket_id=t.ticket_id,
                                visibility=INTERNAL, text=text, ticket_status=t.status))
    return chunks
