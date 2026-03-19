from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from typing import Optional

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

# Incremental panel cache: keyed by "SYMBOL:lookback_minutes"
# Only used for short-horizon real-time calls (≤ 1 day).
# Large one-off training/materialisation calls bypass the cache entirely.
_CACHE_MAX_LOOKBACK = 1440   # minutes — anything larger is a one-off batch call
_CACHE_OVERLAP      = 3      # minutes of overlap on each incremental fetch (safety margin)
_panel_cache: dict[str, tuple[pd.DataFrame, datetime]] = {}
_panel_cache_lock = threading.Lock()


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


def _fetch_panel_since(symbol: str, since: datetime, row_limit: int) -> pd.DataFrame:
    """Fetch and merge all panel tables from `since` up to `row_limit` rows each."""
    market_df  = db.get_market_data_since(symbol=symbol, since=since, limit=row_limit, columns=MARKET_COLUMNS)
    liq_df     = db.get_liquidation_data(symbol=symbol, limit=row_limit, since=since, columns=LIQUIDATION_COLUMNS)
    funding_df = db.get_funding_history(symbol=symbol, limit=row_limit, since=since, columns=FUNDING_COLUMNS)
    cvd_df     = db.get_cvd_data(symbol=symbol, limit=row_limit, since=since, columns=CVD_COLUMNS)
    micro_df   = db.get_microstructure_history(symbol=symbol, limit=row_limit, since=since, columns=MICRO_COLUMNS)
    panel = _merge_panel(market_df, liq_df, funding_df, micro_df, cvd_df)
    if not panel.empty:
        panel["symbol"] = panel["symbol"].astype(str)
        panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True, errors="coerce")
    return panel


def load_minute_panel(symbol: str, lookback_minutes: int = 360) -> pd.DataFrame:
    """Load a merged OHLCV+liq+funding+micro+cvd panel.

    For real-time callers (lookback ≤ 1 day) the first call does a full fetch and
    subsequent calls only pull the incremental rows since the last fetch, then
    append+trim in memory.  This cuts egress by ~100-300x versus fetching the
    entire window every minute.

    Large one-off calls (training / feature materialisation) bypass the cache so
    they always receive fresh, complete data.
    """
    now    = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=lookback_minutes)

    # Bypass cache for large batch/training calls
    if lookback_minutes > _CACHE_MAX_LOOKBACK:
        return _fetch_panel_since(symbol, cutoff, lookback_minutes + 5)

    cache_key = f"{symbol.upper()}:{lookback_minutes}"

    with _panel_cache_lock:
        cached_entry: Optional[tuple[pd.DataFrame, datetime]] = _panel_cache.get(cache_key)

        if cached_entry is None:
            # Cold start: full fetch
            panel = _fetch_panel_since(symbol, cutoff, lookback_minutes + 5)
            _panel_cache[cache_key] = (panel.copy() if not panel.empty else panel, now)
            return panel

        cached_df, cached_at = cached_entry

        # Incremental: fetch only rows newer than (last fetch - overlap)
        since_incr = cached_at - timedelta(minutes=_CACHE_OVERLAP)
        incr_limit = int((now - since_incr).total_seconds() / 60) + 10
        new_panel  = _fetch_panel_since(symbol, since_incr, incr_limit)

        if not new_panel.empty:
            combined = (
                pd.concat([cached_df, new_panel], ignore_index=True)
                .drop_duplicates(subset=["timestamp"], keep="last")
                .sort_values("timestamp")
            )
        else:
            combined = cached_df.copy()

        # Trim to lookback window
        combined = combined[combined["timestamp"] >= cutoff].reset_index(drop=True)
        _panel_cache[cache_key] = (combined.copy(), now)
        return combined


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
