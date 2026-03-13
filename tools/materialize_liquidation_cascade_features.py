"""Materialize liquidation cascade feature snapshots into DB."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.database import db  # noqa: E402
from liquidation_cascade.dataset import load_minute_panel  # noqa: E402
from liquidation_cascade.features import compute_feature_panel  # noqa: E402
from liquidation_cascade.schema import DEFAULT_FEATURE_COLUMNS  # noqa: E402


CORE_COLUMNS = [
    "timestamp",
    "symbol",
    "side",
    "event_candidate",
    "vulnerability_score",
    "ignition_score",
    "ignition_ema",
    "ignition_slope",
    "ignition_r2",
    "vulnerability_pct_rank",
    "ignition_pct_rank",
    "slope_pct_rank",
    "r2_pct_rank",
    "feature_version",
]


def _to_rows(feature_df, limit: int):
    rows = []
    tail = feature_df.sort_values("timestamp").tail(limit)
    for _, row in tail.iterrows():
        payload = {key: row.get(key) for key in CORE_COLUMNS}
        payload["event_candidate"] = bool(int(payload.get("event_candidate") or 0))
        payload["features"] = {
            col: (None if row.get(col) is None else float(row.get(col)))
            for col in DEFAULT_FEATURE_COLUMNS
            if col in row.index
        }
        rows.append(payload)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize liquidation cascade features into DB.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--limit", type=int, default=120)
    args = parser.parse_args()

    panel = load_minute_panel(symbol=args.symbol, lookback_minutes=args.days * 24 * 60)
    if panel.empty:
        print(json.dumps({"status": "empty_panel", "symbol": args.symbol}, indent=2))
        return 1

    all_rows = []
    for side in ("down", "up"):
        feature_df = compute_feature_panel(panel, side=side)
        all_rows.extend(_to_rows(feature_df, limit=args.limit))

    if not all_rows:
        print(json.dumps({"status": "no_rows"}, indent=2))
        return 1

    db.batch_upsert_liquidation_cascade_features(all_rows)
    print(json.dumps({"status": "ok", "rows": len(all_rows), "symbol": args.symbol}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
