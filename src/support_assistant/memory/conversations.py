"""Conversation / session state. SQLite for dev and tests, MySQL (Amazon RDS) in the cloud.

Only portable SQL is used so the same statements run on both. Message content is stored
*after* PII redaction.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from urllib.parse import parse_qs, unquote, urlparse

_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS sessions (
         session_id VARCHAR(64) PRIMARY KEY,
         tenant_id  VARCHAR(64) NOT NULL,
         role       VARCHAR(16) NOT NULL,
         created_at BIGINT NOT NULL,
         expires_at BIGINT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS messages (
         session_id VARCHAR(64) NOT NULL,
         seq        INT NOT NULL,
         author     VARCHAR(16) NOT NULL,
         content    TEXT NOT NULL,
         meta       TEXT,
         created_at BIGINT NOT NULL,
         PRIMARY KEY (session_id, seq))""",
]


class Database:
    """Minimal DB-API wrapper. `?` placeholders are translated to `%s` for MySQL."""

    def __init__(self, url: str) -> None:
        p = urlparse(url)
        self._lock = threading.RLock()
        self._local = threading.local()
        if p.scheme == "sqlite":
            self.kind = "sqlite"
            path = url[len("sqlite:///"):] or ":memory:"
            if path != ":memory:":
                import os
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self._conn = sqlite3.connect(path, check_same_thread=False)
        elif p.scheme in ("mysql", "mysql+pymysql"):
            self.kind = "mysql"
            q = parse_qs(p.query)
            self._mysql = dict(host=p.hostname, port=p.port or 3306, user=unquote(p.username or ""),
                               password=unquote(p.password or ""), database=p.path.lstrip("/"),
                               charset="utf8mb4", autocommit=True)
            if "ssl_ca" in q:                       # e.g. the RDS global CA bundle
                self._mysql["ssl"] = {"ca": q["ssl_ca"][0]}
        else:
            raise ValueError(f"unsupported DATABASE_URL scheme {p.scheme!r} (use sqlite:/// or mysql://)")

    def _mysql_conn(self):
        import pymysql

        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._local.conn = pymysql.connect(**self._mysql)
        else:
            conn.ping(reconnect=True)
        return conn

    def execute(self, sql: str, params: tuple = (), fetch: bool = False):
        if self.kind == "mysql":
            cur = self._mysql_conn().cursor()
            cur.execute(sql.replace("?", "%s"), params)
            rows = cur.fetchall() if fetch else None
            cur.close()
            return rows
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(sql, params)
            rows = cur.fetchall() if fetch else None
            self._conn.commit()
            cur.close()
            return rows


class ConversationStore:
    def __init__(self, db: Database) -> None:
        self.db = db
        for stmt in _SCHEMA:
            db.execute(stmt)

    def ping(self) -> bool:
        self.db.execute("SELECT 1", fetch=True)
        return True

    def create_session(self, tenant_id: str, role: str, ttl_seconds: int) -> tuple[str, int]:
        sid, now = uuid.uuid4().hex, int(time.time())
        self.db.execute("INSERT INTO sessions (session_id, tenant_id, role, created_at, expires_at) VALUES (?,?,?,?,?)",
                        (sid, tenant_id, role, now, now + ttl_seconds))
        return sid, now + ttl_seconds

    def get_session(self, session_id: str) -> dict | None:
        rows = self.db.execute("SELECT session_id, tenant_id, role, expires_at FROM sessions WHERE session_id = ?",
                               (session_id,), fetch=True)
        if not rows:
            return None
        sid, tenant, role, exp = rows[0]
        if exp < time.time():
            return None
        return {"session_id": sid, "tenant_id": tenant, "role": role, "expires_at": int(exp)}

    def append(self, session_id: str, author: str, content: str, meta: dict | None = None) -> None:
        rows = self.db.execute("SELECT COALESCE(MAX(seq), 0) FROM messages WHERE session_id = ?",
                               (session_id,), fetch=True)
        seq = int(rows[0][0]) + 1
        self.db.execute("INSERT INTO messages (session_id, seq, author, content, meta, created_at) VALUES (?,?,?,?,?,?)",
                        (session_id, seq, author, content, json.dumps(meta) if meta else None, int(time.time())))

    def history(self, session_id: str, limit: int = 8) -> list[dict]:
        rows = self.db.execute("SELECT author, content FROM messages WHERE session_id = ? ORDER BY seq DESC LIMIT ?",
                               (session_id, int(limit)), fetch=True)
        return [{"author": a, "content": c} for a, c in reversed(rows)]

    def delete_session(self, session_id: str) -> None:
        self.db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        self.db.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    def purge_expired(self) -> int:
        now = int(time.time())
        rows = self.db.execute("SELECT session_id FROM sessions WHERE expires_at < ?", (now,), fetch=True)
        for (sid,) in rows:
            self.delete_session(sid)
        return len(rows)
