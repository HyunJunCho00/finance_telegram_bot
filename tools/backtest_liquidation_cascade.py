"""Backtest an already-trained liquidation cascade classifier on recent DB data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from liquidation_cascade.dataset import build_training_dataset  # noqa: E402
from liquidation_cascade.model import load_artifact, predict_probabilities  # noqa: E402
from liquidation_cascade.schema import DEFAULT_FEATURE_COLUMNS  # noqa: E402


def _report(y_true, y_prob, threshold: float = 0.5) -> dict:
    import numpy as np

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {
        "threshold": threshold,
        "samples": int(len(y_true)),
        "base_rate": float(y_true.mean()) if len(y_true) else 0.0,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "false_alarms": fp,
        "true_positives": tp,
        "false_negatives": fn,
        "true_negatives": tn,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Score recent data with a trained liquidation cascade model.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--side", choices=["down", "up"], default="down")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--artifact-dir", default="data/models/liquidation_cascade")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--events-only", action="store_true")
    args = parser.parse_args()

    dataset = build_training_dataset(
        symbol=args.symbol,
        side=args.side,
        horizon_minutes=args.horizon,
        lookback_minutes=args.days * 24 * 60,
        features=list(DEFAULT_FEATURE_COLUMNS),
    )
    if dataset.empty:
        print(json.dumps({"status": "empty_dataset"}, indent=2))
        return 1
    if args.events_only:
        dataset = dataset[dataset["event_candidate"] == 1].copy()

    booster, metadata = load_artifact(
        artifact_dir=args.artifact_dir,
        symbol=args.symbol,
        side=args.side,
        horizon_minutes=args.horizon,
    )
    probabilities = predict_probabilities(
        booster,
        metadata["backend"],
        dataset,
        metadata.get("feature_columns") or list(DEFAULT_FEATURE_COLUMNS),
    )
    result = {
        "status": "ok",
        "symbol": args.symbol,
        "side": args.side,
        "rows": int(len(dataset)),
        "report": _report(dataset["cascade_label"].astype(int).to_numpy(), probabilities, threshold=args.threshold),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

