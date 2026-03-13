from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import SIDE_CONFIGS


EPS = 1e-9


def _future_sum(series: pd.Series, horizon: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    for idx in range(len(values)):
        end = idx + horizon + 1
        if end > len(values):
            continue
        out[idx] = float(values[idx + 1 : end].sum())
    return pd.Series(out, index=series.index)


def attach_labels(
    features: pd.DataFrame,
    side: str,
    horizon_minutes: int = 5,
    barrier_lookback: int = 720,
    liq_quantile: float = 0.95,
    oi_release_quantile: float = 0.90,
    price_sigma_mult: float = 1.0,
) -> pd.DataFrame:
    if side not in SIDE_CONFIGS:
        raise ValueError(f"Unsupported side={side!r}")
    if features is None or features.empty:
        return pd.DataFrame()

    cfg = SIDE_CONFIGS[side]
    df = features.copy()
    same_liq = pd.to_numeric(df[cfg.same_side_liq_col], errors="coerce").fillna(0.0)
    close = pd.to_numeric(df["close"], errors="coerce").ffill().fillna(0.0)
    oi = pd.to_numeric(df["open_interest_value"], errors="coerce").ffill().fillna(0.0)
    rv_5m = pd.to_numeric(df["rv_5m_z"], errors="coerce").abs().fillna(0.0)

    df["future_same_side_liq_sum"] = _future_sum(same_liq, horizon_minutes)
    df["future_signed_return"] = cfg.direction_sign * (close.shift(-horizon_minutes) / (close + EPS) - 1.0)
    df["future_oi_release"] = -(oi.shift(-horizon_minutes) / (oi + EPS) - 1.0)

    liq_ref = same_liq.rolling(horizon_minutes, min_periods=horizon_minutes).sum()
    min_periods = max(60, barrier_lookback // 5)
    df["liq_barrier"] = liq_ref.shift(1).rolling(barrier_lookback, min_periods=min_periods).quantile(liq_quantile)
    oi_release_hist = -(oi.pct_change(horizon_minutes).replace([np.inf, -np.inf], 0.0).fillna(0.0))
    df["oi_release_barrier"] = oi_release_hist.shift(1).rolling(barrier_lookback, min_periods=min_periods).quantile(oi_release_quantile)

    valid_barriers = df["liq_barrier"].notna() & df["oi_release_barrier"].notna()
    cascade = (
        valid_barriers
        & (df["future_same_side_liq_sum"] >= df["liq_barrier"])
        & (df["future_signed_return"] >= price_sigma_mult * rv_5m)
        & (df["future_oi_release"] >= df["oi_release_barrier"])
    )
    df["cascade_label"] = cascade.astype(int)
    df["label_horizon_minutes"] = int(horizon_minutes)
    return df
