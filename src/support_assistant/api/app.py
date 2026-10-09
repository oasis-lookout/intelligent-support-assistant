"""FastAPI surface. Thin by design: authentication -> SessionContext -> AssistantService."""
from __future__ import annotations

import logging
import time
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from .. import __version__
from ..config import Settings
from ..factory import Container, build_container
from ..models import SessionContext
from ..observability import log, setup_logging
from ..security import tokens

logger = logging.getLogger("support_assistant.api")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    setup_logging(settings.log_level)
    c = container or build_container(settings)
    app = FastAPI(title="Intelligent Assistant for Customer Support", version=__version__)

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        rid, t0 = request.headers.get("x-request-id", uuid.uuid4().hex[:12]), time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        if request.url.path not in ("/healthz", "/readyz"):
            log(logger, logging.INFO, "request", request_id=rid, method=request.method, path=request.url.path,
                status=response.status_code, ms=round((time.perf_counter() - t0) * 1000, 1))
        return response

    def current_session(authorization: str | None = Header(default=None)) -> SessionContext:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(401, "missing bearer token")
        try:
            claims = tokens.verify(c.settings.session_secret, authorization[7:].strip())
        except tokens.InvalidToken as e:
            raise HTTPException(401, f"invalid token: {e}") from e
        row = c.conversations.get_session(claims["sid"])
        if not row or row["tenant_id"] != claims["tid"] or row["role"] != claims["role"]:
            raise HTTPException(401, "session expired or revoked")
        return SessionContext(row["session_id"], row["tenant_id"], row["role"])

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "version": __version__}

    @app.get("/readyz")
    def readyz():
        try:
            c.conversations.ping()
            chunks = c.service.store.count()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(503, f"not ready: {e}") from e
        return {"status": "ready", "chunks_indexed": chunks, "mode": c.settings.deployment_mode}

    @app.post("/v1/sessions", status_code=201)
    def create_session(x_api_key: str | None = Header(default=None)):
        principal = c.registry.authenticate(x_api_key)
        if principal is None:
            raise HTTPException(401, "invalid API key")
        sid, exp = c.conversations.create_session(principal.tenant_id, principal.role, c.settings.session_ttl_seconds)
        token = tokens.issue(c.settings.session_secret, sid, principal.tenant_id, principal.role, exp)
        return {"session_token": token, "session_id": sid, "tenant_id": principal.tenant_id,
                "role": principal.role, "expires_at": exp}

    @app.post("/v1/chat")
    def chat(req: ChatRequest, ctx: SessionContext = Depends(current_session)):
        try:
            a = c.service.answer(ctx, req.message)
        except ValueError as e:
            raise HTTPException(422, f"invalid message: {e}") from e
        return {"answer": a.text, "confidence": a.confidence, "handoff_recommended": a.handoff_recommended,
                "fallback_reason": a.fallback_reason, "sources": a.sources, "timings_ms": a.timings_ms}

    @app.get("/v1/sessions/me/history")
    def history(ctx: SessionContext = Depends(current_session)):
        return {"messages": c.conversations.history(ctx.session_id, 100)}

    @app.delete("/v1/sessions/me", status_code=204)
    def end_session(ctx: SessionContext = Depends(current_session)):
        c.conversations.delete_session(ctx.session_id)

    return app


def app_factory() -> FastAPI:
    """uvicorn entrypoint: `uvicorn support_assistant.api.app:app_factory --factory`"""
    return create_app()
