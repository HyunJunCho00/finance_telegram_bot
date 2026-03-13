from __future__ import annotations

from pathlib import Path

import pandas as pd

from .model import load_artifact, predict_probabilities
from .schema import DEFAULT_FEATURE_COLUMNS


def score_latest_feature_row(
    feature_panel: pd.DataFrame,
    artifact_dir: str | Path,
    symbol: str,
    side: str,
    horizon_minutes: int,
) -> dict:
    if feature_panel is None or feature_panel.empty:
        return {"status": "empty_panel"}

    booster, metadata = load_artifact(artifact_dir=artifact_dir, symbol=symbol, side=side, horizon_minutes=horizon_minutes)
    feature_columns = metadata.get("feature_columns") or list(DEFAULT_FEATURE_COLUMNS)
    latest = feature_panel.sort_values("timestamp").tail(1).copy()
    probability = float(predict_probabilities(booster, metadata["backend"], latest, feature_columns)[0])
    return {
        "status": "ok",
        "timestamp": latest["timestamp"].iloc[-1].isoformat() if hasattr(latest["timestamp"].iloc[-1], "isoformat") else latest["timestamp"].iloc[-1],
        "symbol": symbol,
        "side": side,
        "horizon_minutes": int(horizon_minutes),
        "probability": probability,
        "diagnostics": {
            "event_candidate": int(latest.get("event_candidate", pd.Series([0])).iloc[-1]),
            "vulnerability_pct_rank": float(latest.get("vulnerability_pct_rank", pd.Series([0.0])).iloc[-1]),
            "ignition_pct_rank": float(latest.get("ignition_pct_rank", pd.Series([0.0])).iloc[-1]),
            "slope_pct_rank": float(latest.get("slope_pct_rank", pd.Series([0.0])).iloc[-1]),
            "r2_pct_rank": float(latest.get("r2_pct_rank", pd.Series([0.0])).iloc[-1]),
        },
        "model_version": metadata.get("model_version"),
        "feature_version": latest.get("feature_version", pd.Series(["unknown"])).iloc[-1],
    }
