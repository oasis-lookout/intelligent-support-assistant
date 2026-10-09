"""Concurrent load test for the deployment comparison (throughput, latency percentiles, errors).

    sa-loadtest --base-url http://localhost:8080 --api-key sk_c_... --concurrency 1,5,10,25 --requests 200

Run once per deployment (AWS-managed vs self-hosted) with the same questions and compare the CSV output.
Uses only the stdlib so it can run from any bastion / CI runner.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from ..ingestion.loader import read_tickets
from .evaluate import first_customer_question


def _call(url: str, payload: dict | None, headers: dict, method: str = "POST") -> tuple[int, dict]:
    req = urllib.request.Request(url, data=json.dumps(payload).encode() if payload is not None else None,
                                 headers={"Content-Type": "application/json", **headers}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, {}
    except Exception:  # noqa: BLE001
        return 0, {}


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))] if xs else float("nan")


def run_level(base: str, api_key: str, questions: list[str], concurrency: int, total: int) -> dict:
    lat: list[float] = []
    errors = fallbacks = 0
    lock = threading.Lock()
    local = threading.local()

    def one(i: int) -> None:
        nonlocal errors, fallbacks
        if not hasattr(local, "token"):            # one session per worker thread
            st, body = _call(f"{base}/v1/sessions", {}, {"X-API-Key": api_key})
            local.token = body.get("session_token", "")
        t0 = time.perf_counter()
        st, body = _call(f"{base}/v1/chat", {"message": questions[i % len(questions)]},
                         {"Authorization": f"Bearer {local.token}"})
        dt = (time.perf_counter() - t0) * 1000
        with lock:
            if st == 200:
                lat.append(dt)
                fallbacks += bool(body.get("fallback_reason"))
            else:
                errors += 1

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        list(ex.map(one, range(total)))
    wall = time.perf_counter() - t0
    return {"concurrency": concurrency, "requests": total, "errors": errors, "fallbacks": fallbacks,
            "throughput_rps": round(len(lat) / wall, 2),
            "p50_ms": round(pct(lat, 50), 1), "p95_ms": round(pct(lat, 95), 1), "p99_ms": round(pct(lat, 99), 1),
            "mean_ms": round(statistics.fmean(lat), 1) if lat else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key", required=True, help="customer-role key for ONE tenant")
    ap.add_argument("--tenant", required=True, help="that tenant's company_id (used to pick realistic questions)")
    ap.add_argument("--source", default="data/support_tickets_1000.csv")
    ap.add_argument("--concurrency", default="1,5,10,25")
    ap.add_argument("--requests", type=int, default=200, help="requests per concurrency level")
    ap.add_argument("--out", default="reports/loadtest.csv")
    args = ap.parse_args()

    qs = [q for t in read_tickets(args.source) if t.company_id == args.tenant
          and (q := first_customer_question(t.conversation))]
    if not qs:
        raise SystemExit(f"no questions found for tenant {args.tenant}")
    rows = []
    for c in (int(x) for x in args.concurrency.split(",")):
        r = run_level(args.base_url.rstrip("/"), args.api_key, qs, c, args.requests)
        rows.append(r)
        print(r)
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
