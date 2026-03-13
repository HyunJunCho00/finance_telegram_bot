from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schema import DEFAULT_FEATURE_COLUMNS, MODEL_VERSION


@dataclass
class ModelArtifact:
    backend: str
    model_path: Path
    metadata_path: Path
    feature_columns: list[str]
    params: dict[str, Any]
    model_version: str


def _require_backend(backend: str):
    if backend == "lightgbm":
        return importlib.import_module("lightgbm")
    if backend == "xgboost":
        return importlib.import_module("xgboost")
    raise ValueError(f"Unsupported backend={backend!r}")


def _as_float32(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    return df[feature_columns].astype(np.float32).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _binary_report(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    accuracy = (tp + tn) / max(len(y_true), 1)
    false_alarm_rate = fp / max(fp + tn, 1)
    brier = float(np.mean((y_prob - y_true) ** 2)) if len(y_true) else 0.0
    return {
        "samples": int(len(y_true)),
        "base_rate": float(np.mean(y_true)) if len(y_true) else 0.0,
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "false_alarm_rate": float(false_alarm_rate),
        "brier_score": brier,
    }


def train_gradient_boosting(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame | None = None,
    feature_columns: list[str] | None = None,
    label_column: str = "cascade_label",
    backend: str = "lightgbm",
    params: dict[str, Any] | None = None,
):
    if train_df is None or train_df.empty:
        raise ValueError("train_df is empty")

    feature_columns = feature_columns or list(DEFAULT_FEATURE_COLUMNS)
    params = params or {}
    backend_lib = _require_backend(backend)

    x_train = _as_float32(train_df, feature_columns)
    y_train = train_df[label_column].astype(int).to_numpy()

    if backend == "lightgbm":
        train_set = backend_lib.Dataset(x_train, label=y_train, feature_name=feature_columns, free_raw_data=False)
        valid_sets = [train_set]
        callbacks = []
        valid_names = ["train"]
        if valid_df is not None and not valid_df.empty:
            x_valid = _as_float32(valid_df, feature_columns)
            y_valid = valid_df[label_column].astype(int).to_numpy()
            valid_set = backend_lib.Dataset(x_valid, label=y_valid, feature_name=feature_columns, free_raw_data=False)
            valid_sets.append(valid_set)
            valid_names.append("valid")
            callbacks.append(backend_lib.early_stopping(stopping_rounds=50, verbose=False))
        default_params = {
            "objective": "binary",
            "metric": ["binary_logloss"],
            "learning_rate": 0.05,
            "num_leaves": 31,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "min_data_in_leaf": 40,
            "verbosity": -1,
            "seed": 42,
        }
        default_params.update(params)
        num_boost_round = int(default_params.pop("num_boost_round", 300))
        booster = backend_lib.train(
            default_params,
            train_set=train_set,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        predict_fn = lambda df: booster.predict(_as_float32(df, feature_columns))
        model_params = default_params | {"num_boost_round": num_boost_round}
    else:
        dtrain = backend_lib.DMatrix(x_train, label=y_train, feature_names=feature_columns)
        evals = [(dtrain, "train")]
        if valid_df is not None and not valid_df.empty:
            x_valid = _as_float32(valid_df, feature_columns)
            y_valid = valid_df[label_column].astype(int).to_numpy()
            dvalid = backend_lib.DMatrix(x_valid, label=y_valid, feature_names=feature_columns)
            evals.append((dvalid, "valid"))
        default_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "eta": 0.05,
            "max_depth": 5,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "seed": 42,
        }
        default_params.update(params)
        num_boost_round = int(default_params.pop("num_boost_round", 300))
        booster = backend_lib.train(
            default_params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=50 if len(evals) > 1 else None,
            verbose_eval=False,
        )
        predict_fn = lambda df: booster.predict(backend_lib.DMatrix(_as_float32(df, feature_columns), feature_names=feature_columns))
        model_params = default_params | {"num_boost_round": num_boost_round}

    metrics = {"train": _binary_report(y_train, predict_fn(train_df))}
    if valid_df is not None and not valid_df.empty:
        y_valid = valid_df[label_column].astype(int).to_numpy()
        metrics["valid"] = _binary_report(y_valid, predict_fn(valid_df))

    return booster, metrics, feature_columns, model_params


def save_artifact(
    booster,
    artifact_dir: str | Path,
    backend: str,
    feature_columns: list[str],
    params: dict[str, Any],
    symbol: str,
    side: str,
    horizon_minutes: int,
    metrics: dict[str, Any],
    model_version: str = MODEL_VERSION,
) -> ModelArtifact:
    artifact_root = Path(artifact_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)
    stem = f"{symbol.lower()}_{side}_{horizon_minutes}m"
    model_suffix = "txt" if backend == "lightgbm" else "json"
    model_path = artifact_root / f"{stem}.{model_suffix}"
    metadata_path = artifact_root / f"{stem}.metadata.json"
    booster.save_model(str(model_path))
    metadata = {
        "backend": backend,
        "feature_columns": feature_columns,
        "params": params,
        "symbol": symbol,
        "side": side,
        "horizon_minutes": int(horizon_minutes),
        "model_version": model_version,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return ModelArtifact(
        backend=backend,
        model_path=model_path,
        metadata_path=metadata_path,
        feature_columns=feature_columns,
        params=params,
        model_version=model_version,
    )


def load_artifact(artifact_dir: str | Path, symbol: str, side: str, horizon_minutes: int):
    artifact_root = Path(artifact_dir)
    stem = f"{symbol.lower()}_{side}_{horizon_minutes}m"
    metadata_path = artifact_root / f"{stem}.metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    backend = metadata["backend"]
    backend_lib = _require_backend(backend)
    model_suffix = "txt" if backend == "lightgbm" else "json"
    model_path = artifact_root / f"{stem}.{model_suffix}"
    if backend == "lightgbm":
        booster = backend_lib.Booster(model_file=str(model_path))
    else:
        booster = backend_lib.Booster()
        booster.load_model(str(model_path))
    return booster, metadata


def predict_probabilities(booster, backend: str, frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    x = _as_float32(frame, feature_columns)
    if backend == "lightgbm":
        return np.asarray(booster.predict(x), dtype=float)
    xgboost = _require_backend("xgboost")
    return np.asarray(booster.predict(xgboost.DMatrix(x, feature_names=feature_columns)), dtype=float)
