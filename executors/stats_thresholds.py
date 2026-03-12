from __future__ import annotations

from typing import Iterable, Dict

import pandas as pd


def symbol_liq_min_std_usd(
    symbol: str,
    *,
    btc_floor: float,
    eth_floor: float,
    default_floor: float,
) -> float:
    upper_symbol = str(symbol or "").upper()
    if "BTC" in upper_symbol:
        return float(btc_floor)
    if "ETH" in upper_symbol:
        return float(eth_floor)
    return float(default_floor)


def adaptive_std_floor(
    values: Iterable[float],
    *,
    base_floor: float,
    rolling_window: int = 24,
    rolling_quantile: float = 0.20,
    min_periods: int = 12,
    min_rolling_samples: int = 24,
) -> Dict[str, float]:
    series = pd.Series(list(values), dtype="float64").dropna()
    if series.empty:
        return {
            "floor": float(base_floor),
            "base_floor": float(base_floor),
            "rolling_floor": 0.0,
            "sample_count": 0.0,
            "rolling_samples": 0.0,
            "series_std": 0.0,
        }

    window = max(2, min(int(rolling_window), len(series)))
    effective_min_periods = max(2, min(int(min_periods), window))
    rolling_std = series.rolling(window=window, min_periods=effective_min_periods).std().dropna()

    rolling_floor = 0.0
    if len(rolling_std) >= int(min_rolling_samples):
        rolling_floor = float(rolling_std.quantile(float(rolling_quantile)))
    elif not rolling_std.empty:
        rolling_floor = float(rolling_std.median())

    return {
        "floor": float(max(float(base_floor), rolling_floor)),
        "base_floor": float(base_floor),
        "rolling_floor": float(rolling_floor),
        "sample_count": float(len(series)),
        "rolling_samples": float(len(rolling_std)),
        "series_std": float(series.std()),
    }
