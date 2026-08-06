#!/usr/bin/env python3
"""Phase-0 spike: prove a store -> query round trip through the adapter.

Builds a real local ALMA (SQLite + sentence-transformers), stores two facts via
`handle_memory_store`, then queries via `handle_memory_query` and prints results.

Prerequisites:
    pip install 'alma-memory[local]'

Run (from the repo root):
    python integrations/maia/spike_roundtrip.py
"""

from __future__ import annotations

import sys

from integrations.maia.alma_gateway_adapter import (
    build_alma,
    handle_memory_query,
    handle_memory_store,
)


def main() -> int:
    print("Building local ALMA (quickstart: SQLite + local embeddings)...")
    try:
        alma = build_alma()  # no config_path -> ALMA.quickstart()
    except ImportError as exc:
        print(f"\n[spike] ALMA local backend unavailable: {exc}")
        print("[spike] Install it with: pip install 'alma-memory[local]'")
        return 1

    facts = [
        "The Maia gateway runs on 127.0.0.1:3600 behind Caddy on :443.",
        "ALMA backs Maia memory via memory.query and memory.store RPC methods.",
    ]

    print("\nStoring 2 facts via handle_memory_store...")
    for fact in facts:
        result = handle_memory_store(
            {"content": fact, "metadata": {"persona": "maia"}}, alma=alma
        )
        print(f"  store -> {result}")
        if not result.get("ok"):
            print("[spike] store failed; aborting.")
            return 1

    print("\nQuerying via handle_memory_query (query='Maia gateway', limit=5)...")
    query_result = handle_memory_query(
        {"query": "Maia gateway", "limit": 5}, alma=alma
    )
    print(f"  ok={query_result.get('ok')}")
    results = query_result.get("payload", {}).get("results", [])
    print(f"  {len(results)} result(s):")
    for item in results:
        print(f"    - {item['content']}")

    print("\n[spike] Round trip complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
