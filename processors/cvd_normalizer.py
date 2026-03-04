from __future__ import annotations

from typing import Optional

import pandas as pd


def _to_utc_timestamp_col(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    if df is None or df.empty or col not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
    out = out.dropna(subset=[col]).sort_values(col).reset_index(drop=True)
    out = out.loc[:, ~out.columns.duplicated()].copy()
    return out


def build_price_timeline(
    df_1m: Optional[pd.DataFrame],
    df_1d: Optional[pd.DataFrame],
    fallback_price: float = 60000.0,
) -> pd.DataFrame:
    """Build unified timestamp->close timeline from 1m (preferred) and 1d (fallback)."""
    frames = []
    if df_1d is not None and not df_1d.empty and {"timestamp", "close"}.issubset(df_1d.columns):
        d1 = _to_utc_timestamp_col(df_1d[["timestamp", "close"]])
        if not d1.empty:
            frames.append(d1)
    if df_1m is not None and not df_1m.empty and {"timestamp", "close"}.issubset(df_1m.columns):
        d0 = _to_utc_timestamp_col(df_1m[["timestamp", "close"]])
        if not d0.empty:
            frames.append(d0)

    if not frames:
        return pd.DataFrame({"timestamp": [], "close": []})

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)
    merged["close"] = pd.to_numeric(merged["close"], errors="coerce").fillna(fallback_price)
    return merged


def merge_cvd_sources(
    hist_cvd: Optional[pd.DataFrame],
    bridge_cvd: Optional[pd.DataFrame],
    recent_cvd: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Merge CVD sources with deterministic precedence: hist < bridge < recent."""
    frames = []
    for priority, frame in enumerate([hist_cvd, bridge_cvd, recent_cvd]):
        if frame is None or frame.empty:
            continue
        f = _to_utc_timestamp_col(frame)
        if f.empty:
            continue
        f["_priority"] = priority
        frames.append(f)

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["timestamp", "_priority"]).drop_duplicates(subset=["timestamp"], keep="last")
    merged = merged.drop(columns=["_priority"]).sort_values("timestamp").reset_index(drop=True)
    return merged


def normalize_cvd_to_usd(
    cvd_df: Optional[pd.DataFrame],
    price_timeline: Optional[pd.DataFrame],
    fallback_price: float = 60000.0,
) -> pd.DataFrame:
    """Normalize mixed CVD schema into explicit base/usd fields.

    Rules:
    - `taker_*` columns are treated as base-unit volumes and converted to USD.
    - `whale_*` columns are treated as already-USD (websocket whale feed).
    - If taker columns are present, they are the canonical source for chart CVD.
      Otherwise whale columns are used.
    """
    if cvd_df is None or cvd_df.empty:
        return pd.DataFrame()

    cvd = _to_utc_timestamp_col(cvd_df)
    if cvd.empty:
        return pd.DataFrame()

    for col in ["taker_buy_volume", "taker_sell_volume", "whale_buy_vol", "whale_sell_vol", "volume_delta"]:
        if col not in cvd.columns:
            cvd[col] = 0.0
        cvd[col] = pd.to_numeric(cvd[col], errors="coerce").fillna(0.0)

    has_taker = ("taker_buy_volume" in cvd_df.columns) or ("taker_sell_volume" in cvd_df.columns)
    has_whale = ("whale_buy_vol" in cvd_df.columns) or ("whale_sell_vol" in cvd_df.columns)

    prices = _to_utc_timestamp_col(price_timeline) if price_timeline is not None else pd.DataFrame()
    if prices is not None and not prices.empty and {"timestamp", "close"}.issubset(prices.columns):
        priced = pd.merge_asof(
            cvd[["timestamp"]].sort_values("timestamp"),
            prices[["timestamp", "close"]].sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
        px = pd.to_numeric(priced["close"], errors="coerce").ffill().bfill().fillna(fallback_price).values
    else:
        px = pd.Series(fallback_price, index=cvd.index).values

    cvd["taker_buy_base"] = cvd["taker_buy_volume"]
    cvd["taker_sell_base"] = cvd["taker_sell_volume"]
    cvd["delta_base"] = cvd["taker_buy_base"] - cvd["taker_sell_base"]

    cvd["buy_usd_from_base"] = cvd["taker_buy_base"] * px
    cvd["sell_usd_from_base"] = cvd["taker_sell_base"] * px
    cvd["delta_usd_from_base"] = cvd["buy_usd_from_base"] - cvd["sell_usd_from_base"]

    cvd["buy_usd_from_whale"] = cvd["whale_buy_vol"]
    cvd["sell_usd_from_whale"] = cvd["whale_sell_vol"]
    cvd["delta_usd_from_whale"] = cvd["buy_usd_from_whale"] - cvd["sell_usd_from_whale"]

    if has_taker:
        cvd["cvd_buy_usd"] = cvd["buy_usd_from_base"]
        cvd["cvd_sell_usd"] = cvd["sell_usd_from_base"]
        cvd["delta_usd"] = cvd["delta_usd_from_base"]
    elif has_whale:
        cvd["cvd_buy_usd"] = cvd["buy_usd_from_whale"]
        cvd["cvd_sell_usd"] = cvd["sell_usd_from_whale"]
        cvd["delta_usd"] = cvd["delta_usd_from_whale"]
    else:
        cvd["cvd_buy_usd"] = 0.0
        cvd["cvd_sell_usd"] = 0.0
        cvd["delta_usd"] = 0.0

    # Keep compatibility with chart_generator expected columns.
    cvd["whale_buy_vol"] = cvd["cvd_buy_usd"]
    cvd["whale_sell_vol"] = cvd["cvd_sell_usd"]

    return cvd.sort_values("timestamp").reset_index(drop=True)
