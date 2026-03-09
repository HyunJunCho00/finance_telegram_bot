from apscheduler.triggers.cron import CronTrigger
from collectors.price_collector import collector
from collectors.funding_collector import funding_collector
from collectors.volatility_monitor import volatility_monitor
from collectors.dune_collector import dune_collector
from collectors.microstructure_collector import microstructure_collector
from collectors.macro_collector import macro_collector
from collectors.deribit_collector import deribit_collector
from collectors.fear_greed_collector import fear_greed_collector
from collectors.crypto_news_collector import collector as news_collector
from collectors.coinmetrics_collector import coinmetrics_collector
from executors.orchestrator import orchestrator
from evaluators.feedback_generator import feedback_generator
from evaluators.evaluation_rollup import evaluation_rollup_service
from processors.light_rag import light_rag
from processors.gcs_archive import gcs_archive_exporter
from processors.math_engine import math_engine
from agents.market_monitor_agent import market_monitor_agent
from tools.evaluate_market_status_pressure import run_evaluation as run_pressure_signal_evaluation
from config import scheduler_config
from config.settings import settings, TradingMode
from config.database import db
from executors.order_manager import execution_desk
from executors.evaluator_daemon import EvaluatorDaemon
from executors.paper_exchange import paper_engine
from loguru import logger
import sys
import threading
import json
import base64
from datetime import datetime, timezone


def _project_trendline_price(line_info: dict, candle_len: int) -> float | None:
    """Project diagonal trendline value at the latest candle index."""
    if not isinstance(line_info, dict):
        return None
    try:
        x1 = line_info.get("_x1")
        x2 = line_info.get("_x2")
        p1 = line_info.get("point1")
        p2 = line_info.get("point2")
        if x1 is None or x2 is None or not p1 or not p2 or x2 == x1:
            return None
        y1 = float(p1[1])
        y2 = float(p2[1])
        slope = (y2 - y1) / (x2 - x1)
        current_val = y2 + slope * (candle_len - 1 - x2)
        return float(current_val)
    except Exception:
        return None


def _pct_change(latest: float | None, prev: float | None) -> float | None:
    if not isinstance(latest, (int, float)) or not isinstance(prev, (int, float)) or prev == 0:
        return None
    return float((latest - prev) / prev * 100.0)


def _build_realtime_pressure(symbol: str, df) -> dict:
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
        chg_1m = _pct_change(current_price, float(closes.iloc[-2])) if len(closes) >= 2 else None
        chg_3m = _pct_change(current_price, float(closes.iloc[-4])) if len(closes) >= 4 else None
        chg_5m = _pct_change(current_price, float(closes.iloc[-6])) if len(closes) >= 6 else None

        cvd_df = db.get_cvd_data(symbol, limit=15)
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
            details.append("3-5분 가격 상방 가속")
        elif price_down:
            details.append("3-5분 가격 하방 가속")
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


def _build_mode_technical_snapshot(symbol: str, mode: TradingMode) -> dict:
    """Build deterministic mode-specific technical snapshot for hourly Telegram status."""
    snapshot = {}
    try:
        limit = settings.SWING_CANDLE_LIMIT if mode == TradingMode.SWING else settings.POSITION_CANDLE_LIMIT
        df = db.get_latest_market_data(symbol, limit=limit)
        if df is None or df.empty:
            return snapshot

        analysis = math_engine.analyze_market(df, mode)
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

        sup_line = structure.get(f"support_{primary_tf}")
        res_line = structure.get(f"resistance_{primary_tf}")
        tf_df_len = len(math_engine.resample_to_timeframe(df, primary_tf))
        sup_price = _project_trendline_price(sup_line, tf_df_len)
        res_price = _project_trendline_price(res_line, tf_df_len)

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
            "realtime_pressure": _build_realtime_pressure(symbol, df),
            "confluence_zones": zones,
        }
    except Exception as e:
        logger.warning(f"Failed to build {mode.value} technical snapshot for {symbol}: {e}")

    return snapshot


def _distance_pct(price: float | None, level: float | None) -> float | None:
    if not isinstance(price, (int, float)) or not isinstance(level, (int, float)) or price == 0:
        return None
    return abs(price - level) / abs(price) * 100.0


def _get_recent_market_regime(symbol: str) -> str:
    """Read latest persisted market regime from ai_reports."""
    try:
        report = db.get_latest_report(symbol=symbol)
        if isinstance(report, dict):
            regime = str(report.get("market_regime", "") or "").upper().strip()
            if regime:
                return regime
    except Exception:
        pass
    return "UNKNOWN"


def _detect_technical_events(symbol: str, swing: dict, position: dict, funding: float | None,
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
            events.append(f"{mode_name.upper()}: CHoCH 감지 ({primary_tf})")
        if ms.get("msb"):
            events.append(f"{mode_name.upper()}: MSB 감지 ({primary_tf})")

        for label, level, th in [
            (f"{mode_name.upper()} nearest_support", sw.get("nearest_support"), thresholds["level_touch_pct"]),
            (f"{mode_name.upper()} nearest_resistance", sw.get("nearest_resistance"), thresholds["level_touch_pct"]),
            (f"{mode_name.upper()} diagonal_support", tr.get("diagonal_support"), thresholds["diag_touch_pct"]),
            (f"{mode_name.upper()} diagonal_resistance", tr.get("diagonal_resistance"), thresholds["diag_touch_pct"]),
            (f"{mode_name.upper()} fib_618", fib.get("fib_618"), thresholds["fib_touch_pct"]),
            (f"{mode_name.upper()} fib_705", fib.get("fib_705"), thresholds["fib_touch_pct"]),
        ]:
            d = _distance_pct(price, level)
            if isinstance(d, (int, float)) and d <= th:
                events.append(f"{label} 근접 ({d:.2f}%)")

    _scan_mode(swing, "swing")
    _scan_mode(position, "position")

    if isinstance(funding, (int, float)) and abs(funding) >= thresholds["funding_abs_pct"]:
        events.append(f"펀딩 극단치 ({funding:+.5f}%)")
    if isinstance(volatility, (int, float)) and abs(volatility) >= thresholds["volatility_abs_pct"]:
        events.append(f"변동성 이벤트 ({volatility:+.2f}%)")

    return {
        "regime": regime,
        "has_event": len(events) > 0,
        "event_count": len(events),
        "thresholds": thresholds,
        "event_items": events[:8],
    }


def job_1min_tick():
    # 1. Price Collection (Critical)
    try:
        collector.run()
    except Exception as e:
        logger.error(f"Price collection error: {e}")

    # 2. Funding Rate (8h signal, 1m cadence)
    try:
        funding_collector.run()
    except Exception as e:
        logger.error(f"Funding collection error: {e}")

    # 3. Microstructure (Ephemeral)
    try:
        microstructure_collector.run()
    except Exception as e:
        logger.error(f"Microstructure collection error: {e}")

    # 4. Volatility Monitor
    try:
        volatility_monitor.run()
    except Exception as e:
        logger.error(f"Volatility monitor error: {e}")

def job_1min_execution():
    """V5: Process Orders, V7: Check Margin Calls + TP/SL"""
    try:
        execution_desk.process_intents()
        
        if settings.PAPER_TRADING_MODE:
            prices = {}
            # [FIX HIGH-13] Reuse existing authenticated CCXT instance
            from executors.trade_executor import trade_executor
            ex = trade_executor.binance
            for symbol in settings.trading_symbols_slash:
                try:
                    t = ex.fetch_ticker(symbol)
                    # Map back to canonical format (BTC/USDT -> BTCUSDT)
                    canonical = symbol.replace('/', '')
                    prices[canonical] = float(t['last'])
                except Exception:
                    pass
            paper_engine.check_liquidations(prices)
            # [FIX CRITICAL-2] Actually call check_tp_sl ??was implemented but never invoked
            paper_engine.check_tp_sl(prices)
            
    except Exception as e:
        logger.error(f"1-minute execution job error: {e}")


def job_15min_dune():
    """Collect cadence-aware Dune snapshots and persist to DB."""
    if dune_collector is None:
        return
    try:
        dune_collector.run_due_queries(limit=200, offset=0)
    except Exception as e:
        logger.error(f"15-minute Dune job error: {e}")


def job_1hour_deribit():
    """Collect Deribit options data: DVOL, PCR, IV Term Structure, 25d Skew."""
    try:
        deribit_collector.run()
    except Exception as e:
        logger.error(f"Deribit collection job error: {e}")

def job_8hour_funding_fee():
    """V8: Simulate funding fee deduction every 8 hours."""
    try:
        from executors.paper_exchange import paper_engine
        from executors.trade_executor import trade_executor
        
        rates = {}
        prices = {}
        positions = paper_engine.get_open_positions()
        if not positions: return
            
        for pos in positions:
            symbol = pos["symbol"]
            if symbol not in rates:
                try:
                    f_info = trade_executor.binance.fetch_funding_rate(symbol)
                    rates[symbol] = float(f_info['fundingRate'])
                    t = trade_executor.binance.fetch_ticker(symbol)
                    prices[symbol] = float(t['last'])
                except Exception as e:
                    logger.warning(f"Failed to fetch funding rate for {symbol}: {e}")
                    
        paper_engine.apply_funding_fees(rates, prices)
    except Exception as e:
        logger.error(f"8-hour funding fee job error: {e}")


def job_daily_fear_greed():
    """Collect Crypto Fear & Greed Index (alternative.me, daily)."""
    try:
        fear_greed_collector.run()
    except Exception as e:
        logger.error(f"Fear & Greed collection job error: {e}")


def job_daily_coinmetrics():
    """Collect daily Coin Metrics snapshots used for regime and risk gating."""
    try:
        if not settings.COINMETRICS_ENABLED:
            logger.info("Coin Metrics job skipped (disabled)")
            return
        coinmetrics_collector.run()
    except Exception as e:
        logger.error(f"Coin Metrics daily job error: {e}")


def job_routine_market_status():
    """V13.3: Routine Market Status check (Free-First) with Multi-Coin & Telegram Intel."""
    try:
        logger.info("Running routine market status check (Free-First)")

        def _parse_reference_source(source_tag: str) -> tuple[str, str]:
            raw = str(source_tag or "").strip()
            if raw.startswith("[") and raw.endswith("]"):
                raw = raw[1:-1].strip()
            if " - " in raw:
                source_name, source_ref = raw.split(" - ", 1)
                return source_name.strip() or "Unknown", source_ref.strip()
            return raw or "Unknown", ""

        def _build_reference_message(selected_news: list[dict]) -> str:
            lines = []
            for idx, item in enumerate(selected_news[:6], start=1):
                headline = str(item.get("headline") or f"뉴스 {idx}").strip()
                lines.append(f"{idx}. {headline}")

                raw_sources = item.get("sources", [])
                if not isinstance(raw_sources, list):
                    raw_sources = [raw_sources]

                seen_sources = set()
                for raw_source in raw_sources:
                    source_name, source_ref = _parse_reference_source(raw_source)
                    dedupe_key = (source_name, source_ref)
                    if dedupe_key in seen_sources:
                        continue
                    seen_sources.add(dedupe_key)

                    if source_ref:
                        lines.append(f"{source_name} : {source_ref}")
                    else:
                        lines.append(source_name)

                if idx < min(len(selected_news), 6):
                    lines.append("")

            return "\n".join(lines).strip()

        indicators = {}
        for symbol in settings.trading_symbols:
            indicators[symbol] = {}
            # Price
            try:
                df = db.get_latest_market_data(symbol, limit=1)
                if not df.empty and "close" in df.columns:
                    indicators[symbol]["price"] = float(df["close"].iloc[-1])
            except Exception:
                pass

            # Funding Rate
            try:
                f_df = db.get_funding_history(symbol, limit=1)
                if not f_df.empty and "funding_rate" in f_df.columns:
                    indicators[symbol]["funding_rate"] = float(f_df["funding_rate"].iloc[-1])
            except Exception:
                pass

            # Volatility
            indicators[symbol]["volatility"] = volatility_monitor.calculate_price_change(symbol)
            swing_snapshot = _build_mode_technical_snapshot(symbol, TradingMode.SWING)
            position_snapshot = _build_mode_technical_snapshot(symbol, TradingMode.POSITION)
            latest_regime = _get_recent_market_regime(symbol)
            indicators[symbol]["technical_snapshot"] = {
                "swing": swing_snapshot,
                "position": position_snapshot,
                "realtime_pressure": (
                    swing_snapshot.get("realtime_pressure")
                    or position_snapshot.get("realtime_pressure")
                    or {}
                ),
                "events": _detect_technical_events(
                    symbol=symbol,
                    swing=swing_snapshot,
                    position=position_snapshot,
                    funding=indicators[symbol].get("funding_rate"),
                    volatility=indicators[symbol].get("volatility"),
                    regime=latest_regime,
                ),
            }
            try:
                onchain = db.get_latest_onchain_snapshot(symbol, max_age_hours=48)
                if onchain:
                    indicators[symbol]["onchain_snapshot"] = {
                        "risk_bias": onchain.get("risk_bias"),
                        "bias_score": onchain.get("bias_score"),
                        "regime_flags": onchain.get("regime_flags", {}),
                        "is_stale": onchain.get("is_stale"),
                    }
            except Exception:
                pass
            try:
                db.insert_market_status_event({
                    "symbol": symbol,
                    "regime": latest_regime,
                    "price": indicators[symbol].get("price"),
                    "funding_rate": indicators[symbol].get("funding_rate"),
                    "volatility": indicators[symbol].get("volatility"),
                    "technical_snapshot": indicators[symbol]["technical_snapshot"],
                })
            except Exception as e:
                logger.warning(f"market_status_events insert skipped for {symbol}: {e}")
                
        # News Intel (Last 1 hour): Telegram + external crypto news synthesized by LLM
        telegram_intel = "최근 1시간 내 주요 뉴스 없음"
        try:
            from agents.ai_router import ai_client
            import time as _time

            tg_messages = db.get_recent_telegram_messages(hours=1) or []
            tg_items = []
            for msg in tg_messages[:20]:
                tg_items.append({
                    "source_type": "telegram",
                    "source": msg.get("channel", "telegram"),
                    "text": str(msg.get("text", "")),
                    "timestamp": msg.get("timestamp") or msg.get("created_at", ""),
                })

            ext_raw = news_collector.fetch_news(
                categories=["bitcoin", "ethereum", "macro", "trading", "institutional", "defi"],
                limit=4,
                lang="en",
            ) or []
            ext_items = []
            seen_links = set()
            for a in ext_raw:
                link = a.get("link", "")
                if not link or link in seen_links:
                    continue
                seen_links.add(link)
                ext_items.append({
                    "source_type": "external",
                    "source": a.get("source", "unknown"),
                    "title": str(a.get("title", "")),
                    "description": str(a.get("description", "")),
                    "url": link,
                })
                if len(ext_items) >= 12:
                    break

            if tg_items or ext_items:
                logger.info(
                    f"Routine news synthesis inputs: telegram={len(tg_items)} external={len(ext_items)}"
                )
                payload = {
                    "telegram_messages": tg_items,
                    "external_news": ext_items,
                    "utc_now": datetime.now(timezone.utc).isoformat(),
                }
                cluster_prompt = (
                    "Select high-impact crypto news from the input and merge duplicates.\n"
                    "Return STRICT JSON only with this schema:\n"
                    "{\n"
                    "  \"selected\": [\n"
                    "    {\n"
                    "      \"headline\": \"short title\",\n"
                    "      \"claim\": \"single factual claim\",\n"
                    "      \"sources\": [\"[source - url_or_telegram]\"],\n"
                    "      \"impact\": 1-5,\n"
                    "      \"why\": \"one short reason\"\n"
                    "    }\n"
                    "  ]\n"
                    "}\n"
                    "Rules:\n"
                    "- Keep at most 6 items.\n"
                    "- Merge duplicate events and union their sources.\n"
                    "- Use only provided evidence. No fabrication."
                )
                cluster_raw = ai_client.generate_response(
                    system_prompt="You are a strict JSON market-news selector.",
                    user_message=f"{cluster_prompt}\n\nINPUT_JSON:\n{json.dumps(payload, ensure_ascii=False)}",
                    temperature=0.1,
                    max_tokens=1300,
                    role="news_cluster",
                ) or ""

                def _extract_json_object(raw: str) -> dict:
                    if not raw:
                        return {}
                    raw = raw.strip()
                    try:
                        obj = json.loads(raw)
                        return obj if isinstance(obj, dict) else {}
                    except Exception:
                        pass
                    start = raw.find("{")
                    if start < 0:
                        return {}
                    depth = 0
                    in_string = False
                    escape = False
                    for i in range(start, len(raw)):
                        ch = raw[i]
                        if escape:
                            escape = False
                            continue
                        if ch == "\\":
                            escape = True
                            continue
                        if ch == '"':
                            in_string = not in_string
                            continue
                        if in_string:
                            continue
                        if ch == "{":
                            depth += 1
                        elif ch == "}":
                            depth -= 1
                            if depth == 0:
                                block = raw[start:i + 1]
                                try:
                                    obj = json.loads(block)
                                    return obj if isinstance(obj, dict) else {}
                                except Exception:
                                    return {}
                    return {}

                cluster_obj = _extract_json_object(cluster_raw)
                selected_items = cluster_obj.get("selected", []) if isinstance(cluster_obj, dict) else []
                if not isinstance(selected_items, list):
                    selected_items = []
                if not selected_items:
                    selected_items = (ext_items[:4] + tg_items[:2])[:6]

                normalized_items = []
                for item in selected_items[:6]:
                    if not isinstance(item, dict):
                        continue
                    headline = str(
                        item.get("headline")
                        or item.get("title")
                        or item.get("source")
                        or "Untitled"
                    ).strip()
                    claim = str(
                        item.get("claim")
                        or item.get("description")
                        or item.get("text")
                        or headline
                    ).strip()
                    why = str(item.get("why") or "").strip()
                    impact = item.get("impact", 3)
                    try:
                        impact = int(impact)
                    except Exception:
                        impact = 3

                    sources = item.get("sources", [])
                    if not isinstance(sources, list) or not sources:
                        source_name = str(item.get("source", "unknown")).strip() or "unknown"
                        source_ref = "telegram"
                        if item.get("url"):
                            source_ref = str(item.get("url", "")).strip()
                        sources = [f"[{source_name} - {source_ref}]"]
                    else:
                        sources = [str(src).strip() for src in sources if str(src).strip()]

                    normalized_items.append({
                        "headline": headline,
                        "claim": claim,
                        "impact": max(1, min(5, impact)),
                        "why": why,
                        "sources": sources[:4],
                    })

                if not normalized_items:
                    normalized_items = [{
                        "headline": "최근 1시간 주요 뉴스 없음",
                        "claim": "유의미한 신규 이벤트가 확인되지 않았습니다.",
                        "impact": 1,
                        "why": "",
                        "sources": [],
                    }]

                _time.sleep(1.2)  # avoid burst-call spikes between stage 1 and stage 2

                final_payload = {
                    "selected_news": normalized_items[:6],
                    "utc_now": datetime.now(timezone.utc).isoformat(),
                }
                final_prompt = (
                    "Write a concise Korean market briefing based ONLY on selected_news.\n"
                    "Output plain text only, under 220 words.\n"
                    "Summarize the top 6 items as short numbered lines.\n"
                    "Do NOT include raw URLs or long source tags in the summary body.\n"
                    "Keep each line focused on event + likely market implication.\n"
                    "If signals conflict, mention the conflict clearly.\n"
                    "The final sentence MUST end with a full stop."
                )
                telegram_intel = ai_client.generate_response(
                    system_prompt="You are a crypto market briefing writer. No markdown fences.",
                    user_message=f"{final_prompt}\n\nINPUT_JSON:\n{json.dumps(final_payload, ensure_ascii=False)}",
                    temperature=0.2,
                    max_tokens=900,
                    role="news_brief_final",
                ) or ""

                bad_ending = ("에서", "및", "또는", "-", ":", "(", "[", "{", "/", ",")
                if not telegram_intel.strip() or telegram_intel.strip().endswith(bad_ending):
                    logger.warning("news_brief_final returned empty/partial text, retrying once.")
                    _time.sleep(1.0)
                    telegram_intel = ai_client.generate_response(
                        system_prompt="You are a crypto market briefing writer. No markdown fences.",
                        user_message=f"{final_prompt}\n\nINPUT_JSON:\n{json.dumps(final_payload, ensure_ascii=False)}",
                        temperature=0.1,
                        max_tokens=900,
                        role="news_brief_final",
                    ) or "최근 1시간 내 주요 뉴스 없음"
            else:
                telegram_intel = "최근 1시간 내 주요 뉴스 없음"
        except Exception as e:
            logger.warning(f"Failed to synthesize market news intel: {e}")
            
        indicators["TELEGRAM_INTEL"] = telegram_intel

        # [FEATURE-3] Split reports for better readability
        from bot.telegram_bot import trading_bot
        if trading_bot:
            import asyncio
            
            # Message 1: News Briefing (only if news exists)
            if telegram_intel and "주요 뉴스 없음" not in telegram_intel:
                news_header = "<b>📰 최근 1시간 뉴스 브리핑 (Synthesized)</b>"
                try:
                    asyncio.run(trading_bot.send_message(settings.TELEGRAM_CHAT_ID, f"{news_header}\n\n{telegram_intel}"))
                except Exception as e:
                    logger.warning(f"Routine news briefing send failed: {e}")

                try:
                    import html
                    refs_text = _build_reference_message(final_payload.get("selected_news", [])[:6])
                    refs_header = "<b>🔗 최근 1시간 뉴스 참고 링크</b>"
                    asyncio.run(
                        trading_bot.send_message(
                            settings.TELEGRAM_CHAT_ID,
                            f"{refs_header}\n\n{html.escape(refs_text)}",
                        )
                    )
                except Exception as e:
                    logger.warning(f"Routine news reference send failed: {e}")
            
            # Message 2: Market Status Summary (Indicators)
            market_header = "<b>📊 주요 시장 지표 업데이트</b>"
            summary = market_monitor_agent.summarize_current_status(indicators)
            logger.success(f"Market Summary Generated:\n{summary}")
            try:
                asyncio.run(trading_bot.send_message(settings.TELEGRAM_CHAT_ID, f"{market_header}\n\n{summary}"))
            except Exception as e:
                logger.warning(f"Routine market status send failed: {e}")


    except Exception as e:
        logger.error(f"Routine market status job error: {e}")


def job_hourly_swing_charts():
    """Hourly Swing chart push for BTC/ETH to Telegram."""
    try:
        from bot.telegram_bot import trading_bot
        if not trading_bot:
            logger.warning("Hourly swing chart skipped: trading_bot unavailable")
            return

        from mcp_server.tools import mcp_tools
        import asyncio

        target_symbols = [s for s in settings.trading_symbols if s in ("BTCUSDT", "ETHUSDT")]
        if not target_symbols:
            target_symbols = settings.trading_symbols[:2]

        for symbol in target_symbols:
            try:
                result = mcp_tools.get_chart_images(symbol, lane="swing")
                if not isinstance(result, dict) or "charts" not in result:
                    logger.warning(f"Hourly swing chart failed for {symbol}: {result.get('error') if isinstance(result, dict) else 'unknown error'}")
                    continue

                charts = result.get("charts", [])
                total = len(charts)
                for idx, chart in enumerate(charts, start=1):
                    chart_bytes = base64.b64decode(chart["chart_base64"])
                    tf = str(chart.get("timeframe", "4h")).upper()
                    caption = (
                        f"📈 <b>{symbol} SWING 차트 (정기 1시간)</b>\n"
                        f"Lane: <code>swing</code>\n"
                        f"Timeframe: <code>{tf}</code>\n"
                        f"Panel: <code>{idx}/{total}</code>\n"
                        f"Lookback: <code>12M</code>"
                    )
                    asyncio.run(trading_bot.send_photo(settings.TELEGRAM_CHAT_ID, chart_bytes, caption))
            except Exception as e:
                logger.warning(f"Hourly swing chart send failed for {symbol}: {e}")
    except Exception as e:
        logger.error(f"Hourly swing charts job error: {e}")


def job_24hour_evaluation():
    try:
        logger.info("Running 24-hour evaluation job")
        feedback_generator.run_feedback_cycle()
    except Exception as e:
        logger.error(f"24-hour evaluation job error: {e}")


def job_daily_evaluation_rollup():
    try:
        logger.info("Running daily evaluation rollup job")
        result = evaluation_rollup_service.run_daily_rollup(lookback_days=3)
        logger.info(f"Daily evaluation rollup result: {result}")
    except Exception as e:
        logger.error(f"Daily evaluation rollup job error: {e}")

def job_1hour_evaluation():
    """V6: Self-Healing RAG evaluation of completed trades."""
    try:
        logger.info("Running 1-hour episodic memory evaluation")
        daemon = EvaluatorDaemon()
        daemon.evaluate_recent_trades()
    except Exception as e:
        logger.error(f"1-hour evaluation job error: {e}")

def job_1hour_telegram():
    """Batch stored Telegram messages into LightRAG every 1 hour (Real-time listener handles collection)."""
    try:
        logger.info("Running 1-hour Telegram batching job")

        # 1. Synthesize and ingest to LightRAG
        from processors.telegram_batcher import telegram_batcher
        telegram_batcher.process_and_ingest(lookback_hours=1)

        # 3. Truth Engine: Triangulate corroborated claims via Perplexity (Web)
        from config.local_state import state_manager
        if state_manager.is_analysis_enabled():
            light_rag.run_triangulation_worker(limit=3)
        else:
            logger.info("Triangulation worker skipped (AI analysis disabled)")
    except Exception as e:
        logger.error(f"1-hour Telegram job error: {e}")

def job_1hour_crypto_news():
    """Fetch Free Crypto News API and ingest to LightRAG every 1 hour."""
    try:
        logger.info("Running 1-hour Crypto News API fetch job")
        news_collector.fetch_and_ingest()
    except Exception as e:
        logger.error(f"1-hour Crypto News API job error: {e}")


def job_pressure_signal_evaluation():
    """Backfill forward-return evaluation into market_status_events.technical_snapshot.evaluation."""
    try:
        logger.info("Running realtime pressure signal evaluation")
        result = run_pressure_signal_evaluation(limit=200, hours=72, dry_run=False)
        logger.info(f"Realtime pressure evaluation result: {result}")
    except Exception as e:
        logger.error(f"Realtime pressure evaluation job error: {e}")


def job_daily_archive_to_gcs():
    """Archive expiring rows to GCS Parquet and verify manifests."""
    try:
        logger.info("Running daily GCS archive job")
        archive_result = gcs_archive_exporter.run_daily_archive()
        logger.info(f"GCS Parquet archive: {archive_result}")
    except Exception as e:
        logger.error(f"Daily GCS archive job error: {e}")


def job_daily_safe_cleanup():
    """Delete only rows covered by verified archive manifests."""
    try:
        logger.info("Running daily safe cleanup job")
        result = gcs_archive_exporter.run_safe_cleanup(limit=1000)
        logger.info(f"Safe cleanup result: {result}")

        graph_days = settings.RETENTION_GRAPH_DAYS
        if graph_days > 0:
            light_rag.cleanup_old(days=graph_days)
        stats = light_rag.get_stats()
        logger.info(f"LightRAG stats: {stats}")
    except Exception as e:
        logger.error(f"Daily safe cleanup job error: {e}")





def job_daily_precision():
    """00:00 UTC serial: BTC POSITION -> ETH POSITION.
    Runs high-quality analysis once per symbol and persists dual-lane playbooks.
    """
    try:
        from config.local_state import state_manager
        if not state_manager.is_analysis_enabled():
            logger.info("Daily precision skipped (analysis disabled)")
            return
        job_daily_coinmetrics()
        macro_collector.run()
        orchestrator.run_daily_playbook()
    except Exception as e:
        logger.error(f"Daily precision job error: {e}")


def job_hourly_monitor():
    """Hourly: evaluate all symbol/mode pairs against Daily Playbook.
    Outputs NO_ACTION / WATCH / TRIGGER.
    TRIGGER ??run analysis + allow order execution (capped at 2/day/symbol).
    """
    try:
        from config.local_state import state_manager
        if not state_manager.is_analysis_enabled():
            logger.info("Hourly monitor skipped (analysis disabled)")
            return
        orchestrator.run_hourly_monitor()
    except Exception as e:
        logger.error(f"Hourly monitor job error: {e}")


def main():
    mode = settings.trading_mode

    logger.info(f"Starting Trading System (mode={mode.value})")
    logger.info("  Primary analysis cadence (UTC): daily_precision=00:00 | hourly_monitor=hh:15 | market_status=hh:20")
    logger.info(f"  Timeframes: {settings.analysis_timeframes}")
    logger.info(f"  Chart timeframe: {settings.chart_timeframe}")
    logger.info(f"  Candle limit: {settings.candle_limit}")
    logger.info(f"  Data lookback: {settings.data_lookback_hours}h")
    logger.info(f"  Chart for VLM: {settings.should_use_chart}")
    logger.info(f"  Symbols: {', '.join(settings.trading_symbols)}")
    logger.info(f"  AI: Gemini Judge/VLM (Project A/B) + Cerebras (meta/risk) + Groq (news/rag) + OpenRouter (monitor)")
    logger.info(f"  Data: Global OI + OI Divergence + MFI Proxy + Liquidations + Coin Metrics + Perplexity + LightRAG")
    logger.info(f"  Dune: {'enabled' if dune_collector else 'disabled'}")
    logger.info(f"  LightRAG: Neo4j {'connected' if settings.NEO4J_URI else 'in-memory'} + "
                f"Milvus {'connected' if settings.MILVUS_URI else 'in-memory'}")

    # Start WebSocket collector for liquidation + whale data
    try:
        from collectors.websocket_collector import websocket_collector
        websocket_collector.start_background()
        logger.info("WebSocket collector started (liquidation + whale trades)")
    except Exception as e:
        logger.warning(f"WebSocket collector unavailable: {e}")

    # [FIX] WebSocket thread health check ??every 5 minutes
    def job_5m_ws_health_check():
        try:
            from collectors.websocket_collector import websocket_collector
            if hasattr(websocket_collector, '_thread'):
                if not websocket_collector._thread.is_alive():
                    logger.warning("WebSocket thread died ??restarting...")
                    websocket_collector.start_background()
        except Exception as e:
            logger.error(f"WS health check error: {e}")

    scheduler_config.scheduler.add_job(
        job_5m_ws_health_check,
        'interval',
        minutes=5,
        id='job_5m_ws_health',
        max_instances=1
    )

    # [FIX Cold Start] Run Telegram bot in a daemon thread
    def _run_telegram_bot():
        try:
            from bot.telegram_bot import trading_bot
            trading_bot.run()  # blocking within this thread
        except Exception as e:
            logger.error(f"Telegram bot crashed: {e}")
            logger.info("Trading system continues WITHOUT Telegram bot commands.")

    bot_thread = threading.Thread(target=_run_telegram_bot, name="telegram-bot", daemon=True)
    bot_thread.start()
    logger.info("Telegram bot started in background thread.")

    # [V13.1] Persistent Telegram Listener (Real-Time Alpha Ingestion)
    def _run_telegram_listener():
        try:
            from collectors.telegram_listener import telegram_listener
            import asyncio
            asyncio.run(telegram_listener.start())
        except Exception as e:
            logger.error(f"Telegram listener crashed: {e}")

    listener_thread = threading.Thread(target=_run_telegram_listener, name="telegram-listener", daemon=True)
    listener_thread.start()
    logger.info("Real-time Telegram Listener (Alpha V13.1) started.")

    # 1-minute tick: price, funding, microstructure, volatility
    scheduler_config.scheduler.add_job(
        job_1min_tick,
        'interval',
        minutes=1,
        id='job_1min_tick',
        max_instances=1
    )
    
    # 1-minute execution tick: ExecutionDesk and Paper Engine Liquidations
    scheduler_config.scheduler.add_job(
        job_1min_execution,
        'interval',
        minutes=1,
        id='job_1min_execution',
        max_instances=1
    )

    # 15-minute Dune
    scheduler_config.scheduler.add_job(
        job_15min_dune,
        'interval',
        minutes=15,
        id='job_15min_dune',
        max_instances=1
    )

    # Deribit options data: DVOL, PCR, IV Term Structure, 25d Skew ??every 1 hour
    scheduler_config.scheduler.add_job(
        job_1hour_deribit,
        CronTrigger(minute=0),
        id='job_1hour_deribit',
        max_instances=1
    )

    # Funding Fee Simulation - every 8 hours (00:00, 08:00, 16:00 UTC)
    if settings.PAPER_TRADING_MODE:
        scheduler_config.scheduler.add_job(
            job_8hour_funding_fee,
            CronTrigger(hour='0,8,16', minute=0),
            id='job_8hour_funding_fee',
            max_instances=1
        )

    # Fear & Greed Index ??daily at 00:15 UTC (after data refreshes)
    scheduler_config.scheduler.add_job(
        job_daily_fear_greed,
        CronTrigger(hour=0, minute=15),
        id='job_daily_fear_greed',
        max_instances=1
    )

    # Coin Metrics daily snapshot refresh
    scheduler_config.scheduler.add_job(
        job_daily_coinmetrics,
        CronTrigger(hour=0, minute=12),
        id='job_daily_coinmetrics',
        max_instances=1
    )

    # 1-Hour Telegram Batching & Ingestion
    scheduler_config.scheduler.add_job(
        job_1hour_telegram,
        CronTrigger(minute=5),
        id='job_1hour_telegram',
        max_instances=1
    )

    # 1-Hour Crypto News API Fetch & Ingestion
    scheduler_config.scheduler.add_job(
        job_1hour_crypto_news,
        CronTrigger(minute=10),
        id='job_1hour_crypto_news',
        max_instances=1
    )

    # Realtime pressure signal evaluation -> write 5m/15m/30m outcomes into market_status_events JSONB
    scheduler_config.scheduler.add_job(
        job_pressure_signal_evaluation,
        CronTrigger(minute='*/15'),
        id='job_pressure_signal_evaluation',
        max_instances=1,
    )

    # ✨✨ Daily Precision (00:00 UTC) [BTC/ETH] SWING/POSITION Playbook generation ✨✨
    scheduler_config.scheduler.add_job(
        job_daily_precision,
        CronTrigger(hour=0, minute=0),
        id='job_daily_precision',
        max_instances=1,
    )

    # ✨✨ Hourly Monitor [NO_ACTION / WATCH / TRIGGER] against Daily Playbook ✨✨
    scheduler_config.scheduler.add_job(
        job_hourly_monitor,
        CronTrigger(minute=15),
        id='job_hourly_monitor',
        max_instances=1,
    )

    # Routine Market Status (Free-First) ??kept for passive hourly Telegram update
    scheduler_config.scheduler.add_job(
        job_routine_market_status,
        CronTrigger(minute=20),
        id='job_market_status',
        max_instances=1,
    )

    # Hourly Swing charts (BTC/ETH)
    scheduler_config.scheduler.add_job(
        job_hourly_swing_charts,
        CronTrigger(minute=22),
        id='job_hourly_swing_charts',
        max_instances=1,
    )

    # Daily evaluation at 00:30 UTC = 09:30 KST
    scheduler_config.scheduler.add_job(
        job_24hour_evaluation,
        CronTrigger(hour=0, minute=30),
        id='job_24hour_evaluation',
        max_instances=1
    )

    scheduler_config.scheduler.add_job(
        job_daily_evaluation_rollup,
        CronTrigger(hour=0, minute=40),
        id='job_daily_evaluation_rollup',
        max_instances=1
    )
    
    # 1-Hour RAG Episodic Memory Evaluation (V6)
    scheduler_config.scheduler.add_job(
        job_1hour_evaluation,
        CronTrigger(minute=45),
        id='job_1hour_evaluation',
        max_instances=1
    )

    # Daily archive at 01:00 UTC = 10:00 KST
    scheduler_config.scheduler.add_job(
        job_daily_archive_to_gcs,
        CronTrigger(hour=1, minute=0),
        id='job_daily_archive_to_gcs',
        max_instances=1
    )

    # Safe cleanup after archive verification
    scheduler_config.scheduler.add_job(
        job_daily_safe_cleanup,
        CronTrigger(hour=1, minute=20),
        id='job_daily_safe_cleanup',
        max_instances=1
    )



    scheduler_config.scheduler.start()
    logger.info("Scheduler started.")
    logger.info(
        "Cadence(UTC): daily_precision=00:00 | telegram_batch=hh:05 | crypto_news=hh:10 | "
        "hourly_monitor=hh:15 | market_status=hh:20 | daily_rollup=00:40 | "
        "hourly_eval=hh:45 | gcs_archive=01:00 | safe_cleanup=01:20"
    )

    # [FIX Cold Start] Run initial data collection immediately so first analysis has data
    logger.info("Running initial data collection (cold start bootstrap)...")
    def _startup_telegram_catchup():
        """Handles initial RAG synthesis for the last 24h.
        Raw collection is handled by the telegram_listener thread's backfill task.
        """
        import time
        # Increased delay to 30s to let the Listener finish its metadata-only backfill
        time.sleep(30)
        
        from processors.telegram_batcher import telegram_batcher
        # Reduced to 6h for synthesized catch-up to avoid heavy LLM load on cold start
        telegram_batcher.process_and_ingest(lookback_hours=6)

    _initial_collectors = [
        ("Price + Funding + Microstructure", lambda: (collector.run(), funding_collector.run(), microstructure_collector.run())),
        ("Volatility", lambda: volatility_monitor.run()),
        ("Deribit", lambda: deribit_collector.run()),
        ("Fear & Greed", lambda: fear_greed_collector.run()),
        ("Coin Metrics", lambda: job_daily_coinmetrics()),
        ("Pressure signal evaluation", lambda: job_pressure_signal_evaluation()),
        ("Telegram catch-up (24h)", _startup_telegram_catchup),
    ]
    for name, fn in _initial_collectors:
        try:
            fn()
            logger.info(f"  ✅ {name} collected")
        except Exception as e:
            logger.warning(f"  ⚠️ {name} collection failed (non-fatal): {e}")

    # Main thread: keep alive + graceful shutdown
    try:
        import time
        while True:
            time.sleep(60)
            # Optional: restart bot thread if it dies
            if not bot_thread.is_alive():
                logger.warning("Telegram bot thread died ??restarting in 30s...")
                time.sleep(30)
                bot_thread = threading.Thread(target=_run_telegram_bot, name="telegram-bot", daemon=True)
                bot_thread.start()

            if not listener_thread.is_alive():
                logger.warning("Telegram listener thread died ??restarting in 30s...")
                time.sleep(30)
                listener_thread = threading.Thread(target=_run_telegram_listener, name="telegram-listener", daemon=True)
                listener_thread.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down...")
        # Upload Telegram session to Secret Manager before exit
        try:
            from collectors.telegram_listener import upload_session_to_cloud
            upload_session_to_cloud()
        except Exception:
            pass
        try:
            from collectors.websocket_collector import websocket_collector
            websocket_collector.stop()
        except Exception:
            pass
        scheduler_config.scheduler.shutdown()
        sys.exit(0)


if __name__ == "__main__":
    main()
