"""Evaluate stored realtime-pressure signals and write results back into technical_snapshot.evaluation.

Examples:
  python tools/evaluate_market_status_pressure.py --symbol BTCUSDT --hours 72 --limit 100
  python tools/evaluate_market_status_pressure.py --symbol ETHUSDT --hours 24 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.database import db  # noqa: E402

THRESHOLD_VERSION = "realtime_pressure_v1"
DEFAULT_HORIZONS = (5, 15, 30)


def _extract_pressure(snapshot: dict) -> dict:
    if not isinstance(snapshot, dict):
        return {}
    return (
        snapshot.get("realtime_pressure")
        or (snapshot.get("swing") or {}).get("realtime_pressure")
        or (snapshot.get("position") or {}).get("realtime_pressure")
        or {}
    )


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _first_price_at_or_after(df, target_time: datetime) -> float | None:
    if df is None or df.empty:
        return None
    matched = df[df["timestamp"] >= target_time]
    if matched.empty:
        return None
    try:
        return float(matched.iloc[0]["close"])
    except Exception:
        return None


def _calc_return(base_price: float | None, future_price: float | None) -> float | None:
    if not isinstance(base_price, (int, float)) or not isinstance(future_price, (int, float)) or base_price == 0:
        return None
    return float((future_price - base_price) / base_price * 100.0)


def _classify_outcome(signal: str | None, ret_pct: float | None) -> str | None:
    if signal not in ("bullish", "bearish") or not isinstance(ret_pct, (int, float)):
        return None
    if signal == "bullish":
        return "correct" if ret_pct > 0 else "wrong" if ret_pct < 0 else "flat"
    return "correct" if ret_pct < 0 else "wrong" if ret_pct > 0 else "flat"


def _has_all_horizons(evaluation: dict, horizons: tuple[int, ...]) -> bool:
    if not isinstance(evaluation, dict):
        return False
    for minutes in horizons:
        if f"forward_{minutes}m_return_pct" not in evaluation:
            return False
    return True


def build_evaluation(row: dict, horizons: tuple[int, ...]) -> tuple[dict | None, str]:
    snapshot = row.get("technical_snapshot") if isinstance(row.get("technical_snapshot"), dict) else {}
    pressure = _extract_pressure(snapshot)
    signal = pressure.get("signal")
    event_time = _parse_ts(row.get("created_at"))
    if event_time is None:
        return None, "invalid_timestamp"

    now_utc = datetime.now(timezone.utc)
    if event_time > now_utc - timedelta(minutes=max(horizons)):
        return None, "too_recent"

    future_df = db.get_market_data_since(
        symbol=row.get("symbol", ""),
        since=event_time,
        limit=max(horizons) + 20,
    )
    if future_df is None or future_df.empty:
        return None, "no_forward_market_data"

    base_price = row.get("price")
    if not isinstance(base_price, (int, float)):
        try:
            base_price = float(future_df.iloc[0]["close"])
        except Exception:
            return None, "missing_base_price"

    evaluation = {
        "evaluated_at": now_utc.isoformat(),
        "threshold_version": THRESHOLD_VERSION,
        "price_at_event": round(float(base_price), 4),
        "signal": signal,
        "summary": pressure.get("summary"),
    }

    for minutes in horizons:
        target_time = event_time + timedelta(minutes=minutes)
        future_price = _first_price_at_or_after(future_df, target_time)
        ret_pct = _calc_return(float(base_price), future_price)
        if ret_pct is None:
            continue
        evaluation[f"forward_{minutes}m_price"] = round(float(future_price), 4)
        evaluation[f"forward_{minutes}m_return_pct"] = round(ret_pct, 4)
        outcome = _classify_outcome(signal, ret_pct)
        if outcome:
            evaluation[f"outcome_{minutes}m"] = outcome

    if len(evaluation) <= 4:
        return None, "insufficient_forward_points"
    return evaluation, "ok"


def run_evaluation(
    symbol: str | None = None,
    limit: int = 200,
    hours: int = 72,
    dry_run: bool = False,
) -> dict:
    horizons = DEFAULT_HORIZONS
    rows = db.get_market_status_events(symbol=symbol, limit=limit, hours=hours)
    if not rows:
        print("No market_status_events found.")
        return {"checked": 0, "updated": 0, "skipped": 0}

    checked = 0
    updated = 0
    skipped = 0

    for row in reversed(rows):
        snapshot = row.get("technical_snapshot") if isinstance(row.get("technical_snapshot"), dict) else {}
        current_eval = snapshot.get("evaluation") if isinstance(snapshot.get("evaluation"), dict) else {}
        if _has_all_horizons(current_eval, horizons):
            skipped += 1
            continue

        checked += 1
        evaluation, status = build_evaluation(row, horizons)
        if not evaluation:
            print(f"skip id={row.get('id')} symbol={row.get('symbol')} reason={status}")
            skipped += 1
            continue

        merged_eval = dict(current_eval)
        merged_eval.update(evaluation)
        snapshot["evaluation"] = merged_eval

        if dry_run:
            print(
                f"dry-run id={row.get('id')} symbol={row.get('symbol')} "
                f"summary={evaluation.get('summary')} "
                f"5m={evaluation.get('forward_5m_return_pct')} "
                f"15m={evaluation.get('forward_15m_return_pct')} "
                f"30m={evaluation.get('forward_30m_return_pct')}"
            )
            updated += 1
            continue

        db.update_market_status_event_technical_snapshot(int(row["id"]), snapshot)
        print(
            f"updated id={row.get('id')} symbol={row.get('symbol')} "
            f"summary={evaluation.get('summary')} "
            f"5m={evaluation.get('forward_5m_return_pct')} "
            f"15m={evaluation.get('forward_15m_return_pct')} "
            f"30m={evaluation.get('forward_30m_return_pct')}"
        )
        updated += 1

    print(f"checked={checked} updated={updated} skipped={skipped}")
    return {"checked": checked, "updated": updated, "skipped": skipped}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate realtime pressure signals stored in market_status_events.")
    parser.add_argument("--symbol", default=None, help="Canonical symbol, e.g. BTCUSDT. Default: all symbols")
    parser.add_argument("--limit", type=int, default=200, help="Max recent rows to inspect")
    parser.add_argument("--hours", type=int, default=72, help="Only consider rows newer than N hours")
    parser.add_argument("--dry-run", action="store_true", help="Compute only, do not write updates")
    args = parser.parse_args()
    run_evaluation(symbol=args.symbol, limit=args.limit, hours=args.hours, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
