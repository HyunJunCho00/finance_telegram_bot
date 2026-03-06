"""Inspect stored realtime-pressure snapshots from market_status_events.

Examples:
  python tools/inspect_market_status_pressure.py --symbol BTCUSDT --limit 8
  python tools/inspect_market_status_pressure.py --symbol ETHUSDT --hours 24
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.database import db  # noqa: E402


def _extract_pressure(snapshot: dict) -> dict:
    if not isinstance(snapshot, dict):
        return {}
    return (
        snapshot.get("realtime_pressure")
        or (snapshot.get("swing") or {}).get("realtime_pressure")
        or (snapshot.get("position") or {}).get("realtime_pressure")
        or {}
    )


def _extract_structure(snapshot: dict) -> str | None:
    if not isinstance(snapshot, dict):
        return None
    swing = snapshot.get("swing") or {}
    tf = swing.get("primary_tf", "4h")
    ms = ((swing.get("market_structure") or {}).get(tf) or {})
    return ms.get("trend")


def _extract_evaluation(snapshot: dict) -> dict:
    if not isinstance(snapshot, dict):
        return {}
    return snapshot.get("evaluation") if isinstance(snapshot.get("evaluation"), dict) else {}


def _fmt_num(value, digits: int = 2) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return "-"


def fetch_rows(symbol: str, limit: int, hours: int | None) -> list[dict]:
    query = (
        db.client.table("market_status_events")
        .select("created_at,symbol,regime,price,technical_snapshot")
        .eq("symbol", symbol)
        .order("created_at", desc=True)
        .limit(limit)
    )
    if hours is not None:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        query = query.gte("created_at", cutoff)
    response = query.execute()
    return response.data if response.data else []


def print_rows(rows: list[dict], raw: bool) -> None:
    if not rows:
        print("No market_status_events found.")
        return

    for row in rows:
        snapshot = row.get("technical_snapshot") if isinstance(row.get("technical_snapshot"), dict) else {}
        pressure = _extract_pressure(snapshot)
        evaluation = _extract_evaluation(snapshot)
        metrics = pressure.get("metrics", {}) if isinstance(pressure.get("metrics"), dict) else {}
        details = pressure.get("details", []) if isinstance(pressure.get("details"), list) else []
        structure = _extract_structure(snapshot) or "-"

        print(
            f"{row.get('created_at')}  {row.get('symbol')}  "
            f"structure={structure}  pressure={pressure.get('summary', '-')}  "
            f"price={_fmt_num(row.get('price'))}"
        )
        if details:
            print(f"  details: {', '.join(str(x) for x in details[:3])}")
        print(
            "  metrics: "
            f"1m={_fmt_num(metrics.get('price_change_1m_pct'), 4)}%  "
            f"3m={_fmt_num(metrics.get('price_change_3m_pct'), 4)}%  "
            f"5m={_fmt_num(metrics.get('price_change_5m_pct'), 4)}%  "
            f"cvd3={_fmt_num(metrics.get('cvd_delta_3m'), 2)}  "
            f"whale5=${_fmt_num(metrics.get('whale_delta_5m_usd'), 2)}  "
            f"longLiq5=${_fmt_num(metrics.get('long_liq_5m_usd'), 2)}  "
            f"shortLiq5=${_fmt_num(metrics.get('short_liq_5m_usd'), 2)}"
        )
        if evaluation:
            print(
                "  eval: "
                f"5m={_fmt_num(evaluation.get('forward_5m_return_pct'), 4)}%/{evaluation.get('outcome_5m', '-')}  "
                f"15m={_fmt_num(evaluation.get('forward_15m_return_pct'), 4)}%/{evaluation.get('outcome_15m', '-')}  "
                f"30m={_fmt_num(evaluation.get('forward_30m_return_pct'), 4)}%/{evaluation.get('outcome_30m', '-')}"
            )
        if raw:
            print("  raw:")
            print(json.dumps(pressure, ensure_ascii=False, indent=2))
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect realtime pressure stored in market_status_events.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Canonical symbol, e.g. BTCUSDT")
    parser.add_argument("--limit", type=int, default=10, help="Max rows to show")
    parser.add_argument("--hours", type=int, default=None, help="Filter rows newer than N hours")
    parser.add_argument("--raw", action="store_true", help="Print full realtime_pressure JSON")
    args = parser.parse_args()

    rows = fetch_rows(symbol=args.symbol, limit=args.limit, hours=args.hours)
    print_rows(rows, raw=args.raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
