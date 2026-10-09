"""Interactive terminal chat against the service layer (no HTTP): `sa-chat --tenant CMP-101 --role customer`."""
from __future__ import annotations

import argparse

from ..config import Settings
from ..factory import build_container
from ..models import SessionContext


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--role", choices=("customer", "agent"), default="customer")
    args = ap.parse_args()

    c = build_container(Settings.from_env())
    sid, _ = c.conversations.create_session(args.tenant, args.role, 3600)
    ctx = SessionContext(sid, args.tenant, args.role)
    print(f"tenant={args.tenant} role={args.role} chunks={c.service.store.count(args.tenant)}  (Ctrl-D to quit)")
    while True:
        try:
            q = input("\nyou> ").strip()
        except EOFError:
            break
        if not q:
            continue
        try:
            a = c.service.answer(ctx, q)
        except ValueError as e:
            print(f"[invalid: {e}]")
            continue
        print(f"\nbot> {a.text}")
        print(f"     [confidence={a.confidence} handoff={a.handoff_recommended} fallback={a.fallback_reason} "
              f"ms={a.timings_ms.get('total')}]")
        for s in a.sources:
            print(f"     source {s['ticket_id']} {s['visibility']} score={s['score']}")


if __name__ == "__main__":
    main()
