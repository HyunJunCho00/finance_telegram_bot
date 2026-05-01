# -*- coding: utf-8 -*-
from loguru import logger
import pandas as pd
from datetime import datetime, timezone
from processors.math_engine import math_engine
from processors.gcs_parquet import gcs_parquet_store
from config.database import db
from config.settings import settings, TradingMode
from utils.math_utils import pct_change, project_trendline_price, distance_pct


def build_realtime_pressure(symbol: str, df) -> dict:
    """Summarize near-real-time directional pressure from recent price, CVD, whale flow, and liquidations."""
    result = {
        "summary": None,
        "signal": None,
        "details": [],
        "metrics": {},
    }
    try:
        if df is None or df.empty or len(df) < 6:
            return result

        closes = df["close"].astype(float).reset_index(drop=True)
        current_price = float(closes.iloc[-1])
        chg_1m = pct_change(current_price, float(closes.iloc[-2])) if len(closes) >= 2 else None
        chg_3m = pct_change(current_price, float(closes.iloc[-4])) if len(closes) >= 4 else None
        chg_5m = pct_change(current_price, float(closes.iloc[-6])) if len(closes) >= 6 else None

        cvd_df = gcs_parquet_store.load_timeseries("cvd", symbol, months_back=1).tail(15).reset_index(drop=True)
        liq_df = db.get_liquidation_data(symbol, limit=15)

        cvd_3m = cvd_5m = whale_delta_5m = None
        whale_buy_5m = whale_sell_5m = 0.0
        if cvd_df is not None and not cvd_df.empty:
            cvd_df = cvd_df.sort_values("timestamp").reset_index(drop=True)
            last_3 = cvd_df.tail(min(3, len(cvd_df)))
            last_5 = cvd_df.tail(min(5, len(cvd_df)))
            if "volume_delta" in cvd_df.columns:
                cvd_3m = float(last_3["volume_delta"].fillna(0).sum())
                cvd_5m = float(last_5["volume_delta"].fillna(0).sum())
            if "whale_buy_vol" in cvd_df.columns and "whale_sell_vol" in cvd_df.columns:
                whale_buy_5m = float(last_5["whale_buy_vol"].fillna(0).sum())
                whale_sell_5m = float(last_5["whale_sell_vol"].fillna(0).sum())
                whale_delta_5m = whale_buy_5m - whale_sell_5m

        long_liq_5m = short_liq_5m = 0.0
        if liq_df is not None and not liq_df.empty:
            liq_df = liq_df.sort_values("timestamp").reset_index(drop=True)
            last_5_liq = liq_df.tail(min(5, len(liq_df)))
            if "long_liq_usd" in last_5_liq.columns:
                long_liq_5m = float(last_5_liq["long_liq_usd"].fillna(0).sum())
            if "short_liq_usd" in last_5_liq.columns:
                short_liq_5m = float(last_5_liq["short_liq_usd"].fillna(0).sum())

        price_up = (
            isinstance(chg_3m, (int, float)) and chg_3m > 0.12
        ) or (
            isinstance(chg_5m, (int, float)) and chg_5m > 0.20
        )
        price_down = (
            isinstance(chg_3m, (int, float)) and chg_3m < -0.12
        ) or (
            isinstance(chg_5m, (int, float)) and chg_5m < -0.20
        )

        flow_up = (
            isinstance(cvd_3m, (int, float)) and cvd_3m > 0
        ) and (
            isinstance(whale_delta_5m, (int, float)) and whale_delta_5m > 0
        )
        flow_down = (
            isinstance(cvd_3m, (int, float)) and cvd_3m < 0
        ) and (
            isinstance(whale_delta_5m, (int, float)) and whale_delta_5m < 0
        )

        squeeze_up = short_liq_5m > max(long_liq_5m * 1.35, 250000.0)
        squeeze_down = long_liq_5m > max(short_liq_5m * 1.35, 250000.0)

        details = []
        if price_up:
            details.append("3-5분 격 상방 속")
        elif price_down:
            details.append("3-5분 격 하방 속")
        if flow_up:
            details.append("CVD·whale 순매수 우세")
        elif flow_down:
            details.append("CVD·whale 순매도 우세")
        if squeeze_up:
            details.append("숏 청산 우세")
        elif squeeze_down:
            details.append("롱 청산 우세")

        if price_up and flow_up and squeeze_up:
            signal = "bullish"
            summary = "strong_bullish"
        elif price_down and flow_down and squeeze_down:
            signal = "bearish"
            summary = "strong_bearish"
        elif price_up and (flow_up or squeeze_up):
            signal = "bullish"
            summary = "bullish"
        elif price_down and (flow_down or squeeze_down):
            signal = "bearish"
            summary = "bearish"
        elif flow_up and squeeze_up:
            signal = "bullish"
            summary = "early_bullish"
        elif flow_down and squeeze_down:
            signal = "bearish"
            summary = "early_bearish"
        else:
            signal = "mixed"
            summary = "mixed"

        result["signal"] = signal
        result["summary"] = summary
        result["details"] = details[:3]
        result["metrics"] = {
            "price_change_1m_pct": round(chg_1m, 4) if isinstance(chg_1m, (int, float)) else None,
            "price_change_3m_pct": round(chg_3m, 4) if isinstance(chg_3m, (int, float)) else None,
            "price_change_5m_pct": round(chg_5m, 4) if isinstance(chg_5m, (int, float)) else None,
            "cvd_delta_3m": round(cvd_3m, 4) if isinstance(cvd_3m, (int, float)) else None,
            "cvd_delta_5m": round(cvd_5m, 4) if isinstance(cvd_5m, (int, float)) else None,
            "whale_buy_5m_usd": round(whale_buy_5m, 2),
            "whale_sell_5m_usd": round(whale_sell_5m, 2),
            "whale_delta_5m_usd": round(whale_delta_5m, 2) if isinstance(whale_delta_5m, (int, float)) else None,
            "long_liq_5m_usd": round(long_liq_5m, 2),
            "short_liq_5m_usd": round(short_liq_5m, 2),
        }
        return result
    except Exception as e:
        logger.warning(f"Failed to build realtime pressure for {symbol}: {e}")
        return result


def build_mode_technical_snapshot(symbol: str, mode: TradingMode) -> dict:
    """Build deterministic mode-specific technical snapshot for hourly Telegram status."""
    snapshot = {}
    try:
        import pandas as pd
        df = pd.DataFrame()
        try:
            df = gcs_parquet_store.load_ohlcv("1m", symbol, months_back=0.2)
        except Exception:
            pass
        if not df.empty and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            last_ts = df["timestamp"].max()
            from datetime import datetime, timezone
            gap_minutes = int((datetime.now(timezone.utc) - last_ts).total_seconds() / 60) + 60
            gap_limit = max(gap_minutes, 60)  # no hard cap — fetch actual gap size
            try:
                df_gap = db.get_market_data_gap(symbol, since=last_ts, limit=gap_limit)
                if df_gap is not None and not df_gap.empty:
                    df_gap["timestamp"] = pd.to_datetime(df_gap["timestamp"], utc=True, errors="coerce")
                    df = pd.concat([df, df_gap]).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            except Exception:
                pass
        if df is None or df.empty:
            df = db.get_latest_market_data(symbol, limit=settings.SWING_CANDLE_LIMIT)  # fallback
        if df is None or df.empty:
            return snapshot

        # GCS 4h/1d/1w 로드 — 구조 분석(Fib, trendline, S/R)에 충분한 히스토리 확보
        # node_collect_data와 동일한 패턴. 1m 6일치만으로는 4h 36개로 부족
        df_4h, df_1d, df_1w = None, None, None
        try:
            m_back = settings.SWING_HISTORY_MONTHS
            df_4h = gcs_parquet_store.load_ohlcv("4h", symbol, months_back=m_back)
            df_1d = gcs_parquet_store.load_ohlcv("1d", symbol, months_back=m_back)
            df_1w = gcs_parquet_store.load_ohlcv("1w", symbol, months_back=m_back)
        except Exception:
            pass

        analysis = math_engine.analyze_market(df, mode, df_4h=df_4h, df_1d=df_1d, df_1w=df_1w)
        current_price = float(df["close"].iloc[-1])

        primary_tf = "4h" if mode == TradingMode.SWING else "1d"
        higher_tf = "1d" if mode == TradingMode.SWING else "1w"

        swing_primary = (analysis.get("swing_levels", {}) or {}).get(primary_tf, {}) or {}
        fib_primary = (analysis.get("fibonacci", {}) or {}).get(primary_tf, {}) or {}
        ms_primary = (analysis.get("market_structure", {}) or {}).get(primary_tf, {}) or {}
        ms_higher = (analysis.get("market_structure", {}) or {}).get(higher_tf, {}) or {}
        structure = analysis.get("structure", {}) or {}
        tq = analysis.get("trendline_quality", {}) or {}
        zones = (analysis.get("confluence_zones", []) or [])[:2]
        scenario_engine = analysis.get("scenario_engine", {}) or {}
        active_setup = scenario_engine.get("active_setup", {}) or {}
        liquidity_map = scenario_engine.get("liquidity_map", {}) or {}
        volume_profile = (analysis.get("volume_profile", {}) or {}).get(primary_tf, {}) or {}
        fvg_primary = (analysis.get("fvg", {}) or {}).get(primary_tf, []) or []

        sup_line = structure.get(f"support_{primary_tf}")
        res_line = structure.get(f"resistance_{primary_tf}")
        tf_df_len = len(math_engine.resample_to_timeframe(df, primary_tf))
        sup_price = project_trendline_price(sup_line, tf_df_len)
        res_price = project_trendline_price(res_line, tf_df_len)

        tf_ind = ((analysis.get("timeframes", {}) or {}).get(primary_tf, {}) or {})
        atr_val = tf_ind.get("atr")
        atr_pct = None
        if isinstance(atr_val, (int, float)) and current_price:
            atr_pct = abs(float(atr_val) / float(current_price) * 100.0)

        snapshot = {
            "mode": mode.value,
            "primary_tf": primary_tf,
            "higher_tf": higher_tf,
            "current_price": round(current_price, 2),
            "atr_percent": round(atr_pct, 4) if isinstance(atr_pct, (int, float)) else None,
            "market_structure": {
                primary_tf: {
                    "trend": ms_primary.get("structure"),
                    "choch": (ms_primary.get("choch") or {}).get("type"),
                    "msb": (ms_primary.get("msb") or {}).get("type"),
                    "last_swing_high": ms_primary.get("last_swing_high"),
                    "last_swing_low": ms_primary.get("last_swing_low"),
                },
                higher_tf: {
                    "trend": ms_higher.get("structure"),
                    "choch": (ms_higher.get("choch") or {}).get("type"),
                    "msb": (ms_higher.get("msb") or {}).get("type"),
                    "last_swing_high": ms_higher.get("last_swing_high"),
                    "last_swing_low": ms_higher.get("last_swing_low"),
                },
            },
            f"trendlines_{primary_tf}": {
                "diagonal_support": round(sup_price, 2) if isinstance(sup_price, (int, float)) else None,
                "diagonal_resistance": round(res_price, 2) if isinstance(res_price, (int, float)) else None,
                "support_quality": tq.get(f"support_{primary_tf}"),
                "resistance_quality": tq.get(f"resistance_{primary_tf}"),
            },
            f"swing_levels_{primary_tf}": {
                "nearest_support": swing_primary.get("nearest_support"),
                "nearest_resistance": swing_primary.get("nearest_resistance"),
            },
            f"fibonacci_{primary_tf}": {
                "trend": fib_primary.get("trend"),
                "nearest_fib": fib_primary.get("nearest_fib"),
                "fib_500": fib_primary.get("fib_500"),
                "fib_618": fib_primary.get("fib_618"),
                "fib_705": fib_primary.get("fib_705"),
                "fib_786": fib_primary.get("fib_786"),
            },
            "realtime_pressure": build_realtime_pressure(symbol, df),
            "confluence_zones": zones,
            "philosophy_snapshot": {
                "profile": scenario_engine.get("profile", getattr(settings, "PHILOSOPHY_PROFILE", "inbum_shipalnam")),
                "higher_timeframe_bias": scenario_engine.get("higher_timeframe_bias"),
                "execution_bias": scenario_engine.get("execution_bias"),
                "scenario_revision_reason": scenario_engine.get("scenario_revision_reason"),
                "active_setup": {
                    "side": active_setup.get("side"),
                    "trigger": active_setup.get("trigger"),
                    "status": active_setup.get("status"),
                    "entry_zone_low": active_setup.get("entry_zone_low"),
                    "entry_zone_high": active_setup.get("entry_zone_high"),
                    "entry_reference": active_setup.get("entry_reference"),
                    "invalidation": active_setup.get("invalidation"),
                    "tp1": active_setup.get("tp1"),
                    "tp2": active_setup.get("tp2"),
                    "risk_box_pct": active_setup.get("risk_box_pct"),
                    "breakeven_rule": active_setup.get("breakeven_rule"),
                    "split_entries": (active_setup.get("split_entries") or [])[:3],
                    "trigger_conditions": (active_setup.get("trigger_conditions") or [])[:2],
                    "invalidation_conditions": (active_setup.get("invalidation_conditions") or [])[:1],
                },
                "liquidity_map": {
                    "liquidity_sweep": liquidity_map.get("liquidity_sweep", {}),
                    "sr_flip": liquidity_map.get("sr_flip", {}),
                    "bpr_zone": liquidity_map.get("bpr_zone", {}),
                },
                "confluence_count": len(zones),
                "volume_profile": {
                    "poc": volume_profile.get("poc"),
                    "value_area_high": volume_profile.get("value_area_high"),
                    "value_area_low": volume_profile.get("value_area_low"),
                },
                "fvg_summary": [
                    {
                        "type": gap.get("type"),
                        "gap_low": gap.get("gap_low"),
                        "gap_high": gap.get("gap_high"),
                        "filled": gap.get("filled"),
                    }
                    for gap in fvg_primary[:2]
                    if isinstance(gap, dict)
                ],
            },
        }
    except Exception as e:
        logger.warning(f"Failed to build {mode.value} technical snapshot for {symbol}: {e}")

    return snapshot


def detect_technical_events(symbol: str, swing: dict, position: dict, funding: float | None,
                             volatility: float | None, regime: str = "UNKNOWN") -> dict:
    """Detect event flags for conditional detailed commentary."""
    events = []
    swing_atr = swing.get("atr_percent") if isinstance(swing, dict) else None
    position_atr = position.get("atr_percent") if isinstance(position, dict) else None
    atr_candidates = [x for x in [swing_atr, position_atr] if isinstance(x, (int, float)) and x > 0]
    base_atr_pct = sum(atr_candidates) / len(atr_candidates) if atr_candidates else 1.0

    # ATR-adaptive thresholds with safety clamps.
    level_touch = min(1.2, max(0.25, base_atr_pct * 0.45))
    diag_touch = min(1.5, max(0.30, base_atr_pct * 0.60))
    fib_touch = min(1.0, max(0.20, base_atr_pct * 0.35))
    volatility_evt = min(5.0, max(1.2, base_atr_pct * 2.0))
    funding_evt = 0.01  # keep fixed until per-symbol funding distribution calibration

    # Regime-aware multiplier (from MetaAgent output persisted in ai_reports.market_regime).
    regime = str(regime or "UNKNOWN").upper()
    regime_mult_map = {
        "RANGE_BOUND": 0.85,
        "SIDEWAYS_ACCUMULATION": 0.90,
        "BULL_MOMENTUM": 1.10,
        "BEAR_MOMENTUM": 1.10,
        "VOLATILITY_PANIC": 1.35,
    }
    regime_mult = regime_mult_map.get(regime, 1.00)
    level_touch = min(1.8, max(0.18, level_touch * regime_mult))
    diag_touch = min(2.0, max(0.22, diag_touch * regime_mult))
    fib_touch = min(1.4, max(0.15, fib_touch * regime_mult))
    volatility_evt = min(7.0, max(0.9, volatility_evt * regime_mult))

    thresholds = {
        "base_atr_pct": round(base_atr_pct, 4),
        "regime_multiplier": round(regime_mult, 3),
        "level_touch_pct": round(level_touch, 4),
        "diag_touch_pct": round(diag_touch, 4),
        "fib_touch_pct": round(fib_touch, 4),
        "funding_abs_pct": funding_evt,
        "volatility_abs_pct": round(volatility_evt, 4),
    }

    def _scan_mode(snapshot: dict, mode_name: str):
        if not isinstance(snapshot, dict):
            return
        primary_tf = snapshot.get("primary_tf", "4h")
        price = snapshot.get("current_price")
        ms = (snapshot.get("market_structure", {}) or {}).get(primary_tf, {}) or {}
        sw = snapshot.get(f"swing_levels_{primary_tf}", {}) or {}
        tr = snapshot.get(f"trendlines_{primary_tf}", {}) or {}
        fib = snapshot.get(f"fibonacci_{primary_tf}", {}) or {}

        if ms.get("choch"):
            events.append(f"{mode_name.upper()}: CHoCH 감 ({primary_tf})")
        if ms.get("msb"):
            events.append(f"{mode_name.upper()}: MSB 감 ({primary_tf})")

        for label, level, th in [
            (f"{mode_name.upper()} nearest_support", sw.get("nearest_support"), thresholds["level_touch_pct"]),
            (f"{mode_name.upper()} nearest_resistance", sw.get("nearest_resistance"), thresholds["level_touch_pct"]),
            (f"{mode_name.upper()} diagonal_support", tr.get("diagonal_support"), thresholds["diag_touch_pct"]),
            (f"{mode_name.upper()} diagonal_resistance", tr.get("diagonal_resistance"), thresholds["diag_touch_pct"]),
            (f"{mode_name.upper()} fib_618", fib.get("fib_618"), thresholds["fib_touch_pct"]),
            (f"{mode_name.upper()} fib_705", fib.get("fib_705"), thresholds["fib_touch_pct"]),
        ]:
            d = distance_pct(price, level)
            if isinstance(d, (int, float)) and d <= th:
                events.append(f"{label} 근접 ({d:.2f}%)")

    _scan_mode(swing, "swing")
    _scan_mode(position, "position")

    if isinstance(funding, (int, float)) and abs(funding) >= thresholds["funding_abs_pct"]:
        events.append(f"펀딩 극단치 ({funding:+.5f}%)")
    if isinstance(volatility, (int, float)) and abs(volatility) >= thresholds["volatility_abs_pct"]:
        events.append(f"유동성 이벤트 ({volatility:+.2f}%)")

    return {
        "regime": regime,
        "has_event": len(events) > 0,
        "event_count": len(events),
        "thresholds": thresholds,
        "event_items": events[:8],
    }
