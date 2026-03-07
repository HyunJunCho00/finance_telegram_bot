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
from config.settings import settings  # noqa: E402

THRESHOLD_VERSION = "realtime_pressure_v1"
DEFAULT_HORIZONS = (5, 15, 30)
ROUND_TRIP_FEE_PCT = float(getattr(settings, "EVAL_ROUND_TRIP_FEE_PCT", 0.10))


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


def _first_row_at_or_after(df, target_time: datetime) -> dict | None:
    if df is None or df.empty:
        return None
    matched = df[df["timestamp"] >= target_time]
    if matched.empty:
        return None
    try:
        row = matched.iloc[0]
        return {
            "close": float(row["close"]),
            "timestamp": row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"],
        }
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


def _signal_to_decision(signal: str | None) -> str:
    if signal == "bullish":
        return "LONG"
    if signal == "bearish":
        return "SHORT"
    return "HOLD"


def _confidence_from_pressure(summary: str | None, signal: str | None) -> float:
    summary = str(summary or "").lower()
    if signal not in ("bullish", "bearish"):
        return 50.0
    if summary.startswith("strong_"):
        return 85.0
    if summary.startswith("early_"):
        return 60.0
    if summary in ("bullish", "bearish"):
        return 72.0
    return 65.0


def _realized_return(signal: str | None, benchmark_return_pct: float) -> float:
    if signal == "bullish":
        return benchmark_return_pct
    if signal == "bearish":
        return -benchmark_return_pct
    return 0.0


def _actual_direction(benchmark_return_pct: float) -> str:
    if benchmark_return_pct > 0:
        return "LONG"
    if benchmark_return_pct < 0:
        return "SHORT"
    return "HOLD"


def _path_metrics(base_price: float, path_prices: list[float], signal: str | None) -> dict:
    if not path_prices or base_price == 0:
        return {"mfe_pct": None, "mae_pct": None}

    realized_path = []
    for price in path_prices:
        if signal == "bearish":
            realized_path.append(((base_price - price) / base_price) * 100.0)
        elif signal == "bullish":
            realized_path.append(((price - base_price) / base_price) * 100.0)
        else:
            realized_path.append(0.0)

    return {
        "mfe_pct": round(max(realized_path), 4) if realized_path else None,
        "mae_pct": round(min(realized_path), 4) if realized_path else None,
    }


def _ensure_prediction_row(row: dict, pressure: dict) -> dict | None:
    event_id = row.get("id")
    if not isinstance(event_id, int):
        return None

    existing = db.get_evaluation_prediction_by_source("market_status_event", event_id, "realtime_pressure")
    if existing:
        return existing

    signal = pressure.get("signal")
    payload = {
        "source_type": "market_status_event",
        "source_id": event_id,
        "prediction_time": row.get("created_at"),
        "symbol": row.get("symbol"),
        "mode": "realtime_pressure",
        "decision": _signal_to_decision(signal),
        "prediction_label": str(pressure.get("summary") or signal or "mixed"),
        "confidence": _confidence_from_pressure(pressure.get("summary"), signal),
        "entry_price": row.get("price"),
        "regime": row.get("regime"),
        "model_version": "rule:realtime_pressure_v1",
        "prompt_version": "deterministic_rule_v1",
        "rag_version": "none",
        "strategy_version": THRESHOLD_VERSION,
        "consensus_rate": None,
        "anomalies_detected": pressure.get("details") if isinstance(pressure.get("details"), list) else [],
        "input_context": {
            "pressure_metrics": pressure.get("metrics") if isinstance(pressure.get("metrics"), dict) else {},
        },
        "metadata": {
            "summary": pressure.get("summary"),
            "signal": signal,
        },
    }
    return db.upsert_evaluation_prediction(payload)


def build_evaluation(row: dict, horizons: tuple[int, ...]) -> tuple[dict | None, list[dict], str]:
    snapshot = row.get("technical_snapshot") if isinstance(row.get("technical_snapshot"), dict) else {}
    pressure = _extract_pressure(snapshot)
    signal = pressure.get("signal")
    event_time = _parse_ts(row.get("created_at"))
    if event_time is None:
        return None, [], "invalid_timestamp"

    now_utc = datetime.now(timezone.utc)
    if event_time > now_utc - timedelta(minutes=max(horizons)):
        return None, [], "too_recent"

    future_df = db.get_market_data_since(
        symbol=row.get("symbol", ""),
        since=event_time,
        limit=max(horizons) + 30,
    )
    if future_df is None or future_df.empty:
        return None, [], "no_forward_market_data"

    base_price = row.get("price")
    if not isinstance(base_price, (int, float)):
        first_row = _first_row_at_or_after(future_df, event_time)
        if not first_row:
            return None, [], "missing_base_price"
        base_price = float(first_row["close"])

    evaluation = {
        "evaluated_at": now_utc.isoformat(),
        "threshold_version": THRESHOLD_VERSION,
        "price_at_event": round(float(base_price), 4),
        "signal": signal,
        "summary": pressure.get("summary"),
    }
    outcome_rows: list[dict] = []

    for minutes in horizons:
        target_time = event_time + timedelta(minutes=minutes)
        future_row = _first_row_at_or_after(future_df, target_time)
        future_price = future_row["close"] if future_row else None
        ret_pct = _calc_return(float(base_price), future_price)
        if ret_pct is None or not future_row:
            continue

        evaluation[f"forward_{minutes}m_price"] = round(float(future_price), 4)
        evaluation[f"forward_{minutes}m_return_pct"] = round(ret_pct, 4)
        outcome = _classify_outcome(signal, ret_pct)
        if outcome:
            evaluation[f"outcome_{minutes}m"] = outcome

        path_df = future_df[future_df["timestamp"] <= future_row["timestamp"]]
        path_prices = [float(v) for v in path_df["close"].tolist()] if not path_df.empty else []
        path_metrics = _path_metrics(float(base_price), path_prices, signal)
        realized_return_pct = _realized_return(signal, ret_pct)
        fee_adjusted_pnl_pct = realized_return_pct - (ROUND_TRIP_FEE_PCT if signal in ("bullish", "bearish") else 0.0)
        delay_minutes = round(max((future_row["timestamp"] - target_time).total_seconds() / 60.0, 0.0), 2)

        outcome_rows.append({
            "horizon_minutes": minutes,
            "target_time": target_time.isoformat(),
            "entry_price": round(float(base_price), 4),
            "exit_price": round(float(future_price), 4),
            "actual_direction": _actual_direction(ret_pct),
            "realized_return_pct": round(realized_return_pct, 4),
            "fee_adjusted_pnl_pct": round(fee_adjusted_pnl_pct, 4),
            "benchmark_return_pct": round(ret_pct, 4),
            "excess_return_pct": round(fee_adjusted_pnl_pct - ret_pct, 4),
            "correct": outcome == "correct",
            "tp_hit": None,
            "sl_hit": None,
            "mfe_pct": path_metrics.get("mfe_pct"),
            "mae_pct": path_metrics.get("mae_pct"),
            "sample_count": max(len(path_prices), 1),
            "data_delay_minutes": delay_minutes,
            "metadata": {
                "source_type": "market_status_event",
                "event_id": row.get("id"),
                "summary": pressure.get("summary"),
                "signal": signal,
                "outcome_label": outcome,
            },
        })

    if len(evaluation) <= 4:
        return None, [], "insufficient_forward_points"
    return evaluation, outcome_rows, "ok"


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
    outcome_upserts = 0

    for row in reversed(rows):
        snapshot = row.get("technical_snapshot") if isinstance(row.get("technical_snapshot"), dict) else {}
        pressure = _extract_pressure(snapshot)
        checked += 1

        evaluation, outcome_rows, status = build_evaluation(row, horizons)
        if not evaluation:
            print(f"skip id={row.get('id')} symbol={row.get('symbol')} reason={status}")
            skipped += 1
            continue

        current_eval = snapshot.get("evaluation") if isinstance(snapshot.get("evaluation"), dict) else {}
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

        prediction_row = _ensure_prediction_row(row, pressure)
        prediction_id = prediction_row.get("id") if isinstance(prediction_row, dict) else None

        db.update_market_status_event_technical_snapshot(int(row["id"]), snapshot)
        if prediction_id is not None:
            for outcome_row in outcome_rows:
                payload = {
                    "prediction_id": int(prediction_id),
                    **outcome_row,
                    "evaluated_at": datetime.now(timezone.utc).isoformat(),
                }
                db.upsert_evaluation_outcome(payload)
                outcome_upserts += 1

        print(
            f"updated id={row.get('id')} symbol={row.get('symbol')} "
            f"summary={evaluation.get('summary')} "
            f"5m={evaluation.get('forward_5m_return_pct')} "
            f"15m={evaluation.get('forward_15m_return_pct')} "
            f"30m={evaluation.get('forward_30m_return_pct')}"
        )
        updated += 1

    print(f"checked={checked} updated={updated} skipped={skipped} outcome_upserts={outcome_upserts}")
    return {"checked": checked, "updated": updated, "skipped": skipped, "outcome_upserts": outcome_upserts}


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
