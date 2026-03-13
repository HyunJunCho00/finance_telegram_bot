from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from config.database import db

from .features import compute_feature_panel
from .labels import attach_labels
from .schema import DEFAULT_FEATURE_COLUMNS


MARKET_COLUMNS = "timestamp,symbol,close"
LIQUIDATION_COLUMNS = "timestamp,symbol,long_liq_usd,short_liq_usd,long_liq_count,short_liq_count"
FUNDING_COLUMNS = "timestamp,symbol,funding_rate,open_interest_value,basis_pct"
MICRO_COLUMNS = "timestamp,symbol,spread_bps,slippage_buy_100k_bps"
CVD_COLUMNS = "timestamp,symbol,volume_delta"


def _merge_panel(
    market_df: pd.DataFrame,
    liq_df: pd.DataFrame,
    funding_df: pd.DataFrame,
    micro_df: pd.DataFrame,
    cvd_df: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    for frame in (market_df, liq_df, funding_df, micro_df, cvd_df):
        if frame is not None and not frame.empty:
            tmp = frame.copy()
            tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], utc=True, errors="coerce")
            tmp = tmp.sort_values("timestamp")
            frames.append(tmp)
    if not frames:
        return pd.DataFrame()

    merged = frames[0]
    for nxt in frames[1:]:
        merged = pd.merge_asof(
            merged.sort_values("timestamp"),
            nxt.sort_values("timestamp"),
            on="timestamp",
            by="symbol" if "symbol" in merged.columns and "symbol" in nxt.columns else None,
            direction="backward",
            tolerance=pd.Timedelta("5m"),
            suffixes=("", "_dup"),
        )
        dup_cols = [c for c in merged.columns if c.endswith("_dup")]
        if dup_cols:
            merged = merged.drop(columns=dup_cols)

    merged = merged.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    return merged


def load_minute_panel(symbol: str, lookback_minutes: int = 360) -> pd.DataFrame:
    since = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
    row_limit = lookback_minutes + 5
    market_df = db.get_market_data_since(symbol=symbol, since=since, limit=row_limit, columns=MARKET_COLUMNS)
    liq_df = db.get_liquidation_data(symbol=symbol, limit=row_limit, since=since, columns=LIQUIDATION_COLUMNS)
    funding_df = db.get_funding_history(symbol=symbol, limit=row_limit, since=since, columns=FUNDING_COLUMNS)
    cvd_df = db.get_cvd_data(symbol=symbol, limit=row_limit, since=since, columns=CVD_COLUMNS)
    micro_df = db.get_microstructure_history(symbol=symbol, limit=row_limit, since=since, columns=MICRO_COLUMNS)
    panel = _merge_panel(market_df, liq_df, funding_df, micro_df, cvd_df)
    if panel.empty:
        return panel
    panel["symbol"] = panel["symbol"].astype(str)
    return panel


def build_training_dataset(
    symbol: str,
    side: str,
    horizon_minutes: int = 5,
    lookback_minutes: int = 60 * 24 * 30,
    features: list[str] | None = None,
) -> pd.DataFrame:
    panel = load_minute_panel(symbol=symbol, lookback_minutes=lookback_minutes)
    if panel.empty:
        return pd.DataFrame()

    feature_panel = compute_feature_panel(panel=panel, side=side)
    labeled = attach_labels(feature_panel, side=side, horizon_minutes=horizon_minutes)
    keep_cols = ["timestamp", "symbol", "side", "event_candidate", "cascade_label", "label_horizon_minutes"]
    keep_cols.extend(features or DEFAULT_FEATURE_COLUMNS)
    keep_cols.extend(
        [
            "future_same_side_liq_sum",
            "future_signed_return",
            "future_oi_release",
            "feature_version",
        ]
    )
    keep_cols = [c for c in keep_cols if c in labeled.columns]
    dataset = labeled[keep_cols].copy()
    dataset = dataset.dropna(subset=["timestamp"])
    return dataset.reset_index(drop=True)
