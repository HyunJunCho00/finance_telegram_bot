"""Fetch Dune query results and SAVE AS JSON FILES.

Features:
- Fetches results from Dune API
- Automatically re-executes queries if results are missing (404) or stale (>24h)
- Saves each query result as a separate JSON file: 'dune_data_{QUERY_ID}.json'

Recommended Query IDs:
- 6638261: ETH Exchange Netflow (Hourly)
- 5924114: BTC On-chain Netflow (Daily)
- 21689:   DEX Aggregator Volume (Real-time)
- 4319:    DEX Volume (Real-time)
- 3383110: Lido Staking (Daily)

Usage:
  python tools/dune_file_saver.py
  python tools/dune_file_saver.py --save-dir ./dune_data
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE_URL = "https://api.dune.com/api/v1"

# Optimized List (Safe & Fresh)
DEFAULT_QUERY_IDS = [
    6638261,  # ETH Netflow (Top 3 CEX)
    5924114,  # BTC Netflow
    21689,    # Aggregator Volume
    4319,     # DEX Volume
    3383110   # Lido Staking
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

    def _request(self, method: str, endpoint: str) -> dict[str, Any]:
        url = f"{BASE_URL}{endpoint}"
        req = Request(url, headers={"X-Dune-API-Key": self.api_key}, method=method)
        with urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)

    def fetch_results(self, query_id: int, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        params = urlencode({"limit": limit, "offset": offset})
        return self._request("GET", f"/query/{query_id}/results?{params}")

    def execute_query(self, query_id: int) -> str:
        data = self._request("POST", f"/query/{query_id}/execute")
        return data["execution_id"]

    def get_execution_status(self, execution_id: str) -> dict[str, Any]:
        return self._request("GET", f"/execution/{execution_id}/status")

    def get_execution_results(self, execution_id: str, limit: int = 50) -> dict[str, Any]:
        return self._request("GET", f"/execution/{execution_id}/results?limit={limit}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Dune results fetch and save to files")
    parser.add_argument("--query-ids", nargs="+", type=int, default=DEFAULT_QUERY_IDS, help="Dune Query IDs")
    parser.add_argument("--limit", type=int, default=50, help="Rows to fetch per query (default: 50)")
    parser.add_argument("--offset", type=int, default=0, help="Pagination offset (default: 0)")
    parser.add_argument("--save-dir", type=str, default=".", help="Directory to save JSON files (default: current dir)")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds (default: 30)")
    return parser.parse_args()

def check_freshness(payload: dict[str, Any]) -> tuple[str, bool]:
    exec_end = payload.get("execution_ended_at")
    if not exec_end:
        return "⚠️ Unknown execution time", True

    try:
        last_run = datetime.fromisoformat(exec_end.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        diff = now - last_run
        hours_ago = diff.total_seconds() / 3600
        is_stale = hours_ago > 24
        return f"Last Run: {hours_ago:.1f} hours ago", is_stale
    except Exception as e:
        return f"⚠️ Date Parse Error: {e}", True

def run_execution_cycle(client: DuneClient, qid: int, limit: int) -> dict[str, Any]:
    print(f"[query_id={qid}] 🔄 Starting execution (Cost credits!)...")
    exec_id = client.execute_query(qid)
    
    while True:
        status = client.get_execution_status(exec_id)
        state = status.get("state")
        
        if state == "QUERY_STATE_COMPLETED":
            break
        elif state in ["QUERY_STATE_FAILED", "QUERY_STATE_CANCELLED"]:
            raise Exception(f"Query execution failed: {state}")
        
        time.sleep(3)
        
    return client.get_execution_results(exec_id, limit=limit)

def main() -> int:
    load_env_file()
    args = parse_args()

    api_key = os.getenv("DUNE_API_KEY", "").strip()
    if not api_key:
        print("ERROR: DUNE_API_KEY not found. Put it in .env or env vars.")
        return 1

    client = DuneClient(api_key=api_key, timeout=args.timeout)

    # 저장 경로 설정
    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Saving data to: {save_dir}")

    for qid in args.query_ids:
        try:
            payload = None
            try:
                # 1. Fetch
                payload = client.fetch_results(query_id=qid, limit=args.limit, offset=args.offset)
                _, is_stale = check_freshness(payload)
                
                # 2. Check Freshness & Re-execute if needed
                if is_stale:
                    print(f"[query_id={qid}] ⚠️ Data is stale (>24h). Re-executing...")
                    payload = run_execution_cycle(client, qid, args.limit)
                
            except HTTPError as exc:
                if exc.code == 404:
                    print(f"[query_id={qid}] ⚠️ No existing results (404). Re-executing...")
                    payload = run_execution_cycle(client, qid, args.limit)
                else:
                    raise exc

            # 3. Save to File
            if payload:
                rows = payload.get("result", {}).get("rows", [])
                
                # 파일명: dune_data_6638261.json
                filename = f"dune_data_{qid}.json"
                file_path = save_dir / filename
                
                # JSON 저장
                file_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[query_id={qid}] ✅ Saved {len(rows)} rows to {filename}")

        except Exception as exc:
            print(f"[query_id={qid}] ❌ Error: {exc}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())

