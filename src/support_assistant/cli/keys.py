"""Create a tenant API key: `sa-keys --tenant CMP-101 --role customer`. The plaintext key is printed once."""
from __future__ import annotations

import argparse

from ..config import Settings
from ..security.apikeys import ROLES, ApiKeyRegistry


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--role", choices=ROLES, required=True)
    args = ap.parse_args()
    path = Settings.from_env().api_keys_file
    print(ApiKeyRegistry.create(path, args.tenant, args.role))
    print(f"(hash stored in {path})")


if __name__ == "__main__":
    main()
