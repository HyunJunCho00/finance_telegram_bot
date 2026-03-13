"""Train a liquidation cascade classifier from stored minute data.

Examples:
  python tools/train_liquidation_cascade.py --symbol BTCUSDT --side down --days 30
  python tools/train_liquidation_cascade.py --symbol ETHUSDT --side up --backend xgboost
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from liquidation_cascade.dataset import build_training_dataset  # noqa: E402
from liquidation_cascade.model import save_artifact, train_gradient_boosting  # noqa: E402
from liquidation_cascade.schema import DEFAULT_FEATURE_COLUMNS  # noqa: E402


def _time_split(df, valid_fraction: float = 0.2):
    if df.empty:
        return df, df
    cutoff = max(int(len(df) * (1.0 - valid_fraction)), 1)
    train = df.iloc[:cutoff].copy()
    valid = df.iloc[cutoff:].copy()
    return train, valid


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a liquidation cascade classifier.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--side", choices=["down", "up"], default="down")
    parser.add_argument("--horizon", type=int, default=5, help="Forward labeling horizon in minutes")
    parser.add_argument("--days", type=int, default=30, help="History window to load from DB")
    parser.add_argument("--backend", choices=["lightgbm", "xgboost"], default="lightgbm")
    parser.add_argument("--artifact-dir", default="data/models/liquidation_cascade")
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--events-only", action="store_true", help="Train only on event_candidate rows")
    args = parser.parse_args()

    dataset = build_training_dataset(
        symbol=args.symbol,
        side=args.side,
        horizon_minutes=args.horizon,
        lookback_minutes=args.days * 24 * 60,
        features=list(DEFAULT_FEATURE_COLUMNS),
    )
    if dataset.empty:
        print(json.dumps({"status": "empty_dataset", "symbol": args.symbol, "side": args.side}, indent=2))
        return 1

    if args.events_only:
        dataset = dataset[dataset["event_candidate"] == 1].copy()
    dataset = dataset.dropna(subset=["cascade_label"]).reset_index(drop=True)
    train_df, valid_df = _time_split(dataset, valid_fraction=args.valid_fraction)
    booster, metrics, feature_columns, params = train_gradient_boosting(
        train_df=train_df,
        valid_df=valid_df if not valid_df.empty else None,
        feature_columns=list(DEFAULT_FEATURE_COLUMNS),
        backend=args.backend,
    )
    artifact = save_artifact(
        booster=booster,
        artifact_dir=args.artifact_dir,
        backend=args.backend,
        feature_columns=feature_columns,
        params=params,
        symbol=args.symbol,
        side=args.side,
        horizon_minutes=args.horizon,
        metrics=metrics,
    )
    result = {
        "status": "ok",
        "dataset_rows": int(len(dataset)),
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "artifacts": {
            "model_path": str(artifact.model_path),
            "metadata_path": str(artifact.metadata_path),
        },
        "metrics": metrics,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
