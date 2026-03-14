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
    base_floor_ratio: float = 0.30,
) -> Dict[str, float]:
    """rolling 데이터가 충분할 때는 floor가 양방향으로 adaptive 동작.

    - 데이터 부족 시 : floor = base_floor  (보수적 고정값)
    - 데이터 충분 시 : floor = max(base_floor * base_floor_ratio, rolling_floor)
        → 시장이 지속적으로 조용하면 floor도 함께 낮아짐
        → base_floor_ratio 는 base_floor 의 절대 최저선 (기본 30%)

    Args:
        base_floor_ratio: 데이터 충분 시 base_floor 를 몇% 까지 허용할지 (0.0~1.0).
            0.30 이면 base_floor 의 30% 까지 내려갈 수 있음.
    """
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

    has_enough_data = len(rolling_std) >= int(min_rolling_samples)

    rolling_floor = 0.0
    if has_enough_data:
        rolling_floor = float(rolling_std.quantile(float(rolling_quantile)))
    elif not rolling_std.empty:
        rolling_floor = float(rolling_std.median())

    # 데이터 충분 시: base_floor 도 downward adaptive (rolling_floor 기준, 단 base_floor*ratio 이상은 보장)
    # 데이터 부족 시: 보수적으로 base_floor 고정 사용
    if has_enough_data and rolling_floor > 0.0:
        effective_base = float(base_floor) * float(base_floor_ratio)
        floor = max(effective_base, rolling_floor)
    else:
        floor = float(base_floor)

    return {
        "floor": float(floor),
        "base_floor": float(base_floor),
        "rolling_floor": float(rolling_floor),
        "has_enough_data": bool(has_enough_data),
        "sample_count": float(len(series)),
        "rolling_samples": float(len(rolling_std)),
        "series_std": float(series.std()),
    }
