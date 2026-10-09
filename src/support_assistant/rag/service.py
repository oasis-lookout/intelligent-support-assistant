"""The online query path: guard -> retrieve -> confidence gate -> generate -> output guard -> persist."""
from __future__ import annotations

import logging

from ..config import Settings
from ..ingestion.redact import redact
from ..memory.conversations import ConversationStore
from ..models import INTERNAL, PUBLIC, Answer, Hit, SessionContext
from ..observability import Timings, log
from ..providers.base import LLM, Embedder
from ..stores.base import VectorStore
from . import guardrails as g
from .prompt import system_prompt, user_message

logger = logging.getLogger(__name__)

VISIBILITY_BY_ROLE = {"customer": (PUBLIC,), "agent": (PUBLIC, INTERNAL)}


class AssistantService:
    def __init__(self, settings: Settings, embedder: Embedder, llm: LLM, store: VectorStore,
                 conversations: ConversationStore) -> None:
        self.s, self.embedder, self.llm, self.store, self.conv = settings, embedder, llm, store, conversations

    # ------------------------------------------------------------------ retrieval
    def _retrieval_query(self, question: str, history: list[dict]) -> str:
        """Cheap, deterministic follow-up handling (no extra LLM call): very short questions are
        prefixed with the previous user message so 'and how long will it take?' still retrieves."""
        if len(question.split()) >= 6:
            return question
        prev = next((m["content"] for m in reversed(history) if m["author"] == "user"), "")
        return f"{prev} {question}".strip()

    def retrieve(self, ctx: SessionContext, question: str, history: list[dict] | None = None) -> list[Hit]:
        q = self._retrieval_query(redact(question), history or [])
        vec = self.embedder.embed_query(q)
        hits = self.store.search(ctx.tenant_id, vec, self.s.top_k, VISIBILITY_BY_ROLE[ctx.role])
        # Defence in depth: never trust the backend filter alone.
        assert all(h.chunk.tenant_id == ctx.tenant_id for h in hits), "cross-tenant hit returned by store"
        seen, unique = set(), []
        for h in hits:                       # templated tickets produce near-identical chunks; dedupe by text
            key = h.chunk.text
            if key not in seen:
                seen.add(key)
                unique.append(h)
        return unique

    # ------------------------------------------------------------------ main entry
    def answer(self, ctx: SessionContext, question: str) -> Answer:
        tm = Timings()
        question = question.strip()
        safe_q = redact(question)
        history = self.conv.history(ctx.session_id, self.s.history_turns * 2)

        def finish(text: str, conf: float, reason: str | None, handoff: bool, hits: list[Hit] | None = None) -> Answer:
            sources = []
            if ctx.role == "agent" and hits:      # customers never see ticket ids
                sources = [{"ticket_id": h.chunk.ticket_id, "chunk_id": h.chunk.chunk_id,
                            "visibility": h.chunk.visibility, "status": h.chunk.ticket_status,
                            "score": round(h.score, 4)} for h in hits]
            ans = Answer(text=text, confidence=round(conf, 4), handoff_recommended=handoff or reason is not None
                         and reason != "prompt_injection", fallback_reason=reason, sources=sources,
                         timings_ms=tm.finish())
            self.conv.append(ctx.session_id, "user", safe_q)
            self.conv.append(ctx.session_id, "assistant", text,
                             {"confidence": ans.confidence, "fallback": reason})
            log(logger, logging.INFO, "answered", tenant=ctx.tenant_id, role=ctx.role, session=ctx.session_id,
                confidence=ans.confidence, fallback=reason, timings_ms=ans.timings_ms)
            return ans

        verdict = g.check_input(question, self.s.max_question_chars)
        if not verdict.ok:
            if verdict.reason in ("empty", "too_long"):
                raise ValueError(verdict.reason)
            return finish(g.FALLBACK_REFUSAL, 0.0, verdict.reason, False)

        with tm.stage("retrieve"):
            hits = self.retrieve(ctx, question, history)
        confident = [h for h in hits if h.score >= self.s.min_confidence]
        top = hits[0].score if hits else 0.0
        if len(confident) < self.s.min_hits:
            return finish(g.FALLBACK_LOW_CONFIDENCE, top, "low_confidence", True)

        context_hits = confident[: self.s.context_k]
        messages = [{"role": "user" if m["author"] == "user" else "assistant", "content": m["content"]}
                    for m in history]
        messages.append({"role": "user", "content": user_message(safe_q, context_hits)})
        try:
            with tm.stage("generate"):
                text = self.llm.generate(system_prompt(ctx.role), messages, self.s.max_answer_tokens,
                                         self.s.temperature)
        except Exception:  # noqa: BLE001 - provider outage must degrade gracefully
            logger.exception("llm generation failed")
            return finish(g.FALLBACK_ERROR, top, "provider_error", True)

        violations = g.output_violations(text, ctx.role)
        if violations:
            log(logger, logging.WARNING, "output blocked by guard", tenant=ctx.tenant_id,
                session=ctx.session_id, violations=violations)
            return finish(g.FALLBACK_OUTPUT_BLOCKED, top, "output_guard", True)

        return finish(text, top, None, g.needs_human(question), context_hits)
