from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .schema import FEATURE_VERSION, SIDE_CONFIGS


EPS = 1e-9


def _safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default).astype(float)


def _rolling_mad(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    def _mad(values: np.ndarray) -> float:
        if len(values) == 0:
            return 0.0
        med = float(np.median(values))
        return float(np.median(np.abs(values - med)))

    return series.rolling(window=window, min_periods=min_periods).apply(_mad, raw=True)


def robust_zscore(series: pd.Series, window: int = 240, min_periods: int = 60) -> pd.Series:
    history = _safe_numeric(series).shift(1)
    median = history.rolling(window=window, min_periods=min_periods).median()
    mad = _rolling_mad(history, window=window, min_periods=min_periods)
    scale = (1.4826 * mad).replace(0, np.nan)
    z = (_safe_numeric(series) - median) / (scale + EPS)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def rolling_percentile_rank(series: pd.Series, window: int = 240, min_periods: int = 60) -> pd.Series:
    values = _safe_numeric(series)

    def _rank(arr: np.ndarray) -> float:
        if len(arr) == 0:
            return 0.0
        last = arr[-1]
        return float(np.mean(arr <= last))

    pct = values.rolling(window=window, min_periods=min_periods).apply(_rank, raw=True)
    return pct.fillna(0.0)


def rolling_slope_and_r2(series: pd.Series, window: int = 5) -> tuple[pd.Series, pd.Series]:
    y = _safe_numeric(series).to_numpy(dtype=float)
    slopes = np.zeros_like(y, dtype=float)
    r2s = np.zeros_like(y, dtype=float)
    x = np.arange(window, dtype=float)

    for idx in range(window - 1, len(y)):
        segment = y[idx - window + 1 : idx + 1]
        if np.isnan(segment).any():
            continue
        x_centered = x - x.mean()
        y_centered = segment - segment.mean()
        denom = float(np.dot(x_centered, x_centered))
        if denom <= 0:
            continue
        slope = float(np.dot(x_centered, y_centered) / denom)
        corr = float(np.corrcoef(x, segment)[0, 1]) if np.std(segment) > 0 else 0.0
        slopes[idx] = slope
        r2s[idx] = corr * corr if np.isfinite(corr) else 0.0

    return pd.Series(slopes, index=series.index), pd.Series(r2s, index=series.index)


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = 0.0
    return out


def compute_feature_panel(
    panel: pd.DataFrame,
    side: str,
    lookback_window: int = 240,
    percentile_window: int = 720,
    ema_span: int = 5,
    slope_window: int = 5,
) -> pd.DataFrame:
    if side not in SIDE_CONFIGS:
        raise ValueError(f"Unsupported side={side!r}")
    if panel is None or panel.empty:
        return pd.DataFrame()

    cfg = SIDE_CONFIGS[side]
    df = panel.copy()
    df = _ensure_columns(
        df,
        [
            cfg.same_side_liq_col,
            cfg.opposite_side_liq_col,
            cfg.same_side_count_col,
            cfg.opposite_side_count_col,
            "open_interest_value",
            "funding_rate",
            "basis_pct",
            "spread_bps",
            "slippage_buy_100k_bps",
            "volume_delta",
            "close",
        ],
    )
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["symbol"] = df["symbol"].astype(str)
    df["side"] = side

    same_liq = _safe_numeric(df[cfg.same_side_liq_col])
    opp_liq = _safe_numeric(df[cfg.opposite_side_liq_col])
    same_count = _safe_numeric(df[cfg.same_side_count_col])
    oi = _safe_numeric(df["open_interest_value"])
    funding = _safe_numeric(df["funding_rate"])
    basis = _safe_numeric(df["basis_pct"])
    spread = _safe_numeric(df["spread_bps"])
    slippage = _safe_numeric(df["slippage_buy_100k_bps"])
    cvd_delta = _safe_numeric(df["volume_delta"])
    close = _safe_numeric(df["close"])

    returns_1m = close.pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    rv_5m_raw = returns_1m.rolling(5, min_periods=3).std().fillna(0.0)
    oi_change_1m = oi.pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    oi_change_5m = oi.pct_change(5).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    same_liq_ratio_1m = np.log1p(same_liq / (oi + EPS))
    same_liq_ratio_5m = np.log1p(same_liq.rolling(5, min_periods=1).sum() / (oi + EPS))
    same_count_log = np.log1p(same_count)

    df["same_side_liq_surprise_1m"] = robust_zscore(same_liq_ratio_1m, window=lookback_window)
    df["same_side_liq_surprise_5m"] = robust_zscore(same_liq_ratio_5m, window=lookback_window)
    df["same_side_liq_count_surprise_1m"] = robust_zscore(same_count_log, window=lookback_window)
    df["liq_imbalance_1m"] = ((same_liq - opp_liq) / (same_liq + opp_liq + EPS)).clip(-1.0, 1.0)
    df["oi_level_z"] = robust_zscore(oi, window=lookback_window)
    df["oi_change_5m_z"] = robust_zscore(oi_change_5m, window=lookback_window)
    df["funding_crowding_z"] = robust_zscore(cfg.funding_sign * funding, window=lookback_window)
    df["basis_crowding_z"] = robust_zscore(cfg.basis_sign * basis, window=lookback_window)
    df["spread_z"] = robust_zscore(spread, window=lookback_window)
    df["slippage_100k_z"] = robust_zscore(slippage, window=lookback_window)
    df["cvd_delta_3m_z"] = robust_zscore(cvd_delta.rolling(3, min_periods=1).sum(), window=lookback_window)
    df["rv_5m_z"] = robust_zscore(rv_5m_raw, window=lookback_window)

    df["vulnerability_score"] = (
        df["oi_level_z"]
        + df["funding_crowding_z"]
        + df["basis_crowding_z"]
        + df["spread_z"]
    ) / 4.0

    oi_release = -oi_change_1m
    oi_release_z = robust_zscore(oi_release, window=lookback_window)
    df["ignition_score"] = (
        0.5 * df["same_side_liq_surprise_1m"]
        + 0.2 * df["same_side_liq_count_surprise_1m"]
        + 0.3 * oi_release_z
    )
    df["ignition_ema"] = df["ignition_score"].ewm(span=ema_span, adjust=False).mean()
    df["ignition_slope"], df["ignition_r2"] = rolling_slope_and_r2(df["ignition_ema"], window=slope_window)

    min_periods = max(60, percentile_window // 5)
    df["vulnerability_pct_rank"] = rolling_percentile_rank(df["vulnerability_score"], window=percentile_window, min_periods=min_periods)
    df["ignition_pct_rank"] = rolling_percentile_rank(df["ignition_ema"], window=percentile_window, min_periods=min_periods)
    df["slope_pct_rank"] = rolling_percentile_rank(df["ignition_slope"], window=percentile_window, min_periods=min_periods)
    df["r2_pct_rank"] = rolling_percentile_rank(df["ignition_r2"], window=percentile_window, min_periods=min_periods)

    df["event_candidate"] = ((df["same_side_liq_surprise_1m"] > 0) & (df["ignition_pct_rank"] >= 0.97)).astype(int)
    df["feature_version"] = FEATURE_VERSION
    return df
