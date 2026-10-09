"""Read tickets from a local path or s3://bucket/key."""
from __future__ import annotations

import csv
import io
from typing import Iterator

from ..models import Ticket

REQUIRED = ("ticket_id", "company_id", "conversation", "ticket_status", "internal_notes")


def read_text(uri: str) -> str:
    if uri.startswith("s3://"):
        import boto3  # lazy: only needed for S3 sources

        bucket, _, key = uri[5:].partition("/")
        body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
        return body.decode("utf-8-sig")
    with open(uri, "r", encoding="utf-8-sig", newline="") as fh:
        return fh.read()


def read_tickets(uri: str) -> Iterator[Ticket]:
    reader = csv.DictReader(io.StringIO(read_text(uri), newline=""))
    missing = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
    if missing:
        raise ValueError(f"{uri}: missing required columns {missing}")
    for row in reader:
        tid, cid = (row.get("ticket_id") or "").strip(), (row.get("company_id") or "").strip()
        if not tid or not cid:
            continue
        yield Ticket(ticket_id=tid, company_id=cid,
                     conversation=(row.get("conversation") or "").replace("\r\n", "\n"),
                     status=(row.get("ticket_status") or "").strip(),
                     internal_notes=(row.get("internal_notes") or "").replace("\r\n", "\n"))
