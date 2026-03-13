"""Score the latest liquidation cascade feature row and optionally persist prediction."""

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
from liquidation_cascade.inference import score_latest_feature_row  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Score the latest liquidation cascade snapshot.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--side", choices=["down", "up"], default="down")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--artifact-dir", default="data/models/liquidation_cascade")
    parser.add_argument("--persist", action="store_true")
    args = parser.parse_args()

    panel = load_minute_panel(symbol=args.symbol, lookback_minutes=args.days * 24 * 60)
    if panel.empty:
        print(json.dumps({"status": "empty_panel", "symbol": args.symbol}, indent=2))
        return 1

    feature_panel = compute_feature_panel(panel, side=args.side)
    result = score_latest_feature_row(
        feature_panel=feature_panel,
        artifact_dir=args.artifact_dir,
        symbol=args.symbol,
        side=args.side,
        horizon_minutes=args.horizon,
    )

    if args.persist and result.get("status") == "ok":
        db.insert_liquidation_cascade_prediction(
            {
                "timestamp": result["timestamp"],
                "symbol": result["symbol"],
                "side": result["side"],
                "horizon_minutes": result["horizon_minutes"],
                "model_name": "liquidation_cascade_gbm",
                "model_version": result.get("model_version") or "unknown",
                "feature_version": result.get("feature_version") or "unknown",
                "probability": result["probability"],
                "diagnostics": result.get("diagnostics") or {},
            }
        )

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
