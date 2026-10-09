"""Tenant API keys. Only SHA-256 hashes are stored; each key is bound to (tenant_id, role).

In AWS, mount the JSON from Secrets Manager (External Secrets / CSI driver) instead of a file in the image.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
from dataclasses import dataclass

ROLES = ("customer", "agent")


@dataclass(frozen=True)
class Principal:
    tenant_id: str
    role: str


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class ApiKeyRegistry:
    def __init__(self, entries: dict[str, dict] | None = None) -> None:
        self._entries = entries or {}

    @classmethod
    def load(cls, path: str) -> "ApiKeyRegistry":
        if not os.path.exists(path):
            return cls({})
        with open(path) as fh:
            return cls(json.load(fh))

    def authenticate(self, key: str | None) -> Principal | None:
        if not key:
            return None
        e = self._entries.get(hash_key(key))
        return Principal(e["tenant_id"], e["role"]) if e else None

    @staticmethod
    def create(path: str, tenant_id: str, role: str) -> str:
        """Generate a key, append its hash to `path`, return the plaintext key (shown once)."""
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        key = f"sk_{role[:1]}_{secrets.token_urlsafe(24)}"
        entries: dict = {}
        if os.path.exists(path):
            with open(path) as fh:
                entries = json.load(fh)
        entries[hash_key(key)] = {"tenant_id": tenant_id, "role": role}
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(entries, fh, indent=2)
        return key
