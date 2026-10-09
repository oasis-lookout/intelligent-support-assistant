"""Stateless HMAC-signed session tokens: v1.<payload>.<signature>.

The token carries tenant_id and role, so those are never taken from the request body.
The session row in the DB is still checked on every call (revocation + expiry).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time


class InvalidToken(Exception):
    pass


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _sig(secret: str, payload: str) -> str:
    return _b64(hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest())


def issue(secret: str, session_id: str, tenant_id: str, role: str, expires_at: int) -> str:
    payload = _b64(json.dumps({"sid": session_id, "tid": tenant_id, "role": role, "exp": expires_at},
                              separators=(",", ":")).encode())
    body = f"v1.{payload}"
    return f"{body}.{_sig(secret, body)}"


def verify(secret: str, token: str, now: float | None = None) -> dict:
    try:
        version, payload, sig = token.split(".")
    except ValueError as e:
        raise InvalidToken("malformed token") from e
    if version != "v1" or not hmac.compare_digest(sig, _sig(secret, f"{version}.{payload}")):
        raise InvalidToken("bad signature")
    try:
        data = json.loads(_unb64(payload))
    except Exception as e:  # noqa: BLE001
        raise InvalidToken("bad payload") from e
    if data.get("exp", 0) < (time.time() if now is None else now):
        raise InvalidToken("expired")
    return data
