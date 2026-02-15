"""Fetch and summarize Dune query results using query IDs.

Usage examples:
  python tools/dune_results_runner.py
  python tools/dune_results_runner.py --query-ids 4319 21689 --limit 5
  python tools/dune_results_runner.py --query-ids 3383110 --limit 20 --save-dir ./tmp/dune
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE_URL = "https://api.dune.com/api/v1"
DEFAULT_QUERY_IDS = [3383110, 4319, 21689,
    1551,     # ETH Netflow (Unified)
    6638261,  # ETH Netflow (Top 3)
    5924114,  # BTC Netflow
    21689,    # DEX Volume
    137977,   # NFT Market
    2280      # Stablecoin Supply
]


def load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


class DuneClient:
    def __init__(self, api_key: str, timeout: int = 30) -> None:
        self.api_key = api_key
        self.timeout = timeout

    def fetch_results(self, query_id: int, limit: int = 5, offset: int = 0) -> dict[str, Any]:
        params = urlencode({"limit": limit, "offset": offset})
        url = f"{BASE_URL}/query/{query_id}/results?{params}"
        req = Request(url, headers={"X-Dune-API-Key": self.api_key}, method="GET")
        with urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Dune results-only fetch for one or more query IDs")
    parser.add_argument("--query-ids", nargs="+", type=int, default=DEFAULT_QUERY_IDS, help="Dune Query IDs (default: 3383110 4319 21689)")
    parser.add_argument("--limit", type=int, default=5, help="Rows to fetch per query (default: 5)")
    parser.add_argument("--offset", type=int, default=0, help="Pagination offset (default: 0)")
    parser.add_argument("--save-dir", type=str, default="", help="Optional directory to save raw JSON responses")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds (default: 30)")
    return parser.parse_args()


def summarize_payload(query_id: int, payload: dict[str, Any]) -> str:
    result = payload.get("result", {})
    rows = result.get("rows", [])
    metadata = result.get("metadata", {})
    row_count = result.get("row_count", len(rows))

    lines = [
        f"[query_id={query_id}] rows_returned={len(rows)} total_row_count={row_count}",
        f"  columns={metadata.get('column_names', [])}",
    ]
    if rows:
        lines.append(f"  sample_row={json.dumps(rows[0], ensure_ascii=False)}")
    return "\n".join(lines)


def main() -> int:
    load_env_file()
    args = parse_args()

    api_key = os.getenv("DUNE_API_KEY", "").strip()
    if not api_key:
        print("ERROR: DUNE_API_KEY not found. Put it in .env or env vars.")
        return 1

    client = DuneClient(api_key=api_key, timeout=args.timeout)

    save_dir = Path(args.save_dir).resolve() if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    failed = False

    print(f"Running query_ids={args.query_ids} limit={args.limit} offset={args.offset}")
    for qid in args.query_ids:
        try:
            payload = client.fetch_results(query_id=qid, limit=args.limit, offset=args.offset)
            print(summarize_payload(qid, payload))

            if save_dir:
                out = save_dir / f"query_{qid}_limit{args.limit}_offset{args.offset}.json"
                out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"  saved={out}")
        except HTTPError as exc:
            failed = True
            body = exc.read().decode("utf-8", errors="ignore")[:500]
            print(f"[query_id={qid}] HTTP error: {exc}. body={body}")
        except URLError as exc:
            failed = True
            print(f"[query_id={qid}] URL error: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed = True
            print(f"[query_id={qid}] error: {exc}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
