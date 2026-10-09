"""Offline ingestion: `sa-ingest --source data/support_tickets_1000.csv [--tenant CMP-101] [--reset]`.

Source may be a local file or s3://bucket/key. In EKS this runs as a Kubernetes Job (deploy/k8s/jobs).
"""
from __future__ import annotations

import argparse

from ..config import Settings
from ..factory import build_embedder, build_store
from ..ingestion.loader import read_tickets
from ..ingestion.pipeline import ingest
from ..observability import setup_logging


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="CSV path or s3://bucket/key")
    ap.add_argument("--tenant", help="ingest only this company_id")
    ap.add_argument("--reset", action="store_true", help="delete the tenant's existing vectors first (needs --tenant)")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    s = Settings.from_env()
    setup_logging(s.log_level)
    embedder, store = build_embedder(s), build_store(s)
    if args.reset:
        if not args.tenant:
            ap.error("--reset requires --tenant")
        store.delete_tenant(args.tenant)
    stats = ingest(read_tickets(args.source), embedder, store, batch_size=args.batch_size, tenant=args.tenant)
    store.save()
    print(f"ingested {stats.tickets} tickets -> {stats.chunks} chunks in {stats.seconds}s "
          f"[embedder={embedder.name}, store={s.vector_store}]")


if __name__ == "__main__":
    main()
