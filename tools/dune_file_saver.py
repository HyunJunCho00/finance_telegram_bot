"""Run Dune collection from CLI using the shared collector implementation.

This keeps manual runs consistent with scheduler behavior:
- Same cadence map and query definitions
- Same stale/404 re-execution logic
- Same row sanitization and DB persistence model
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running as: python tools/dune_file_saver.py
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

DEFAULT_QUERY_IDS = [6638261, 5924114, 21689, 4319, 3383110]


def load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Dune collector and save query snapshots")
    parser.add_argument("--query-ids", nargs="+", type=int, default=DEFAULT_QUERY_IDS, help="Dune Query IDs")
    parser.add_argument("--limit", type=int, default=200, help="Rows to fetch per query")
    parser.add_argument("--offset", type=int, default=0, help="Pagination offset")
    parser.add_argument("--save-dir", type=str, default=".", help="Directory to save JSON files")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds")
    parser.add_argument("--force", action="store_true", help="Run selected queries even if not yet due")
    return parser.parse_args()


def main() -> int:
    load_env_file()
    args = parse_args()

    api_key = os.getenv("DUNE_API_KEY", "").strip()
    if not api_key:
        print("ERROR: DUNE_API_KEY not found. Add it to .env or environment variables.")
        return 1

    save_dir = Path(args.save_dir).resolve()
    from collectors.dune_collector import DuneCollector

    collector = DuneCollector(api_key=api_key, timeout=args.timeout, save_dir=str(save_dir))

    print(f"Dune save directory: {save_dir}")
    print(f"Selected query IDs: {args.query_ids}")
    print(f"Mode: {'force' if args.force else 'due-only'}")

    stats = collector.run_due_queries(
        limit=args.limit,
        offset=args.offset,
        query_ids=args.query_ids,
        force=args.force,
    )
    print(f"Completed. selected={stats['selected']} ran={stats['ran']} skipped={stats['skipped']} failed={stats['failed']}")

    return 1 if stats["failed"] > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
