from __future__ import annotations

from dataclasses import dataclass


FEATURE_VERSION = "liq_cascade_features_v1"
MODEL_VERSION = "liq_cascade_lgbm_v1"


@dataclass(frozen=True)
class SideConfig:
    side: str
    same_side_liq_col: str
    opposite_side_liq_col: str
    same_side_count_col: str
    opposite_side_count_col: str
    funding_sign: float
    basis_sign: float
    direction_sign: float


SIDE_CONFIGS = {
    "down": SideConfig(
        side="down",
        same_side_liq_col="long_liq_usd",
        opposite_side_liq_col="short_liq_usd",
        same_side_count_col="long_liq_count",
        opposite_side_count_col="short_liq_count",
        funding_sign=1.0,
        basis_sign=1.0,
        direction_sign=-1.0,
    ),
    "up": SideConfig(
        side="up",
        same_side_liq_col="short_liq_usd",
        opposite_side_liq_col="long_liq_usd",
        same_side_count_col="short_liq_count",
        opposite_side_count_col="long_liq_count",
        funding_sign=-1.0,
        basis_sign=-1.0,
        direction_sign=1.0,
    ),
}


DEFAULT_FEATURE_COLUMNS = [
    "same_side_liq_surprise_1m",
    "same_side_liq_surprise_5m",
    "same_side_liq_count_surprise_1m",
    "liq_imbalance_1m",
    "oi_level_z",
    "oi_change_5m_z",
    "funding_crowding_z",
    "basis_crowding_z",
    "spread_z",
    "slippage_100k_z",
    "cvd_delta_3m_z",
    "rv_5m_z",
    "vulnerability_score",
    "ignition_score",
    "ignition_ema",
    "ignition_slope",
    "ignition_r2",
    "vulnerability_pct_rank",
    "ignition_pct_rank",
    "slope_pct_rank",
    "r2_pct_rank",
]
