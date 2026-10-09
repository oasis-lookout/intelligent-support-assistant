"""Structured JSON logging + a tiny stage timer. On EKS, stdout is shipped to CloudWatch Logs."""
from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": round(record.created, 3), "level": record.levelname, "logger": record.name,
                   "msg": record.getMessage()}
        extra = getattr(record, "fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    root.addHandler(h)
    root.setLevel(level)


def log(logger: logging.Logger, level: int, msg: str, **fields) -> None:
    logger.log(level, msg, extra={"fields": fields})


class Timings:
    """Collects per-stage latencies in milliseconds."""

    def __init__(self) -> None:
        self.ms: dict[str, float] = {}
        self._t0 = time.perf_counter()

    @contextmanager
    def stage(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.ms[name] = round((time.perf_counter() - t) * 1000, 2)

    def finish(self) -> dict[str, float]:
        self.ms["total"] = round((time.perf_counter() - self._t0) * 1000, 2)
        return self.ms
