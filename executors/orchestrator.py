# -*- coding: utf-8 -*-
"""Orchestrator: LangGraph StateGraph multi-agent analysis pipeline.

Architecture:
  LangGraph StateGraph manages the analysis flow as a directed graph.
  Each node is a processing step. Edges define the execution order.

Graph:
  collect_data -> context_gathering -> meta_agent -> triage
  -> generate_chart -> rule_based_chart -> vlm_expert (conditional runtime)
  -> judge_agent -> risk_manager -> execute_trade -> generate_report

Benefits over sequential:
  - Explicit state management (TypedDict)
  - Error isolation per node
  - Easy to add/remove/reorder steps
  - Built-in retry support

Cost optimization:
  - Default path is 3-agent core: Meta -> Judge -> Risk
  - VLM is only invoked when triage/rule-based chart signals indicate uncertainty or stress
"""

from typing import Dict, Optional, TypedDict, Annotated
import threading
from config.database import db
from config.settings import settings, TradingMode
from processors.math_engine import math_engine
from processors.chart_generator import chart_generator
from processors.light_rag import light_rag
from processors.onchain_signal_engine import onchain_signal_engine
# CVD normalizer removed - CVD pipeline deprecated
from collectors.perplexity_collector import perplexity_collector
from collectors.macro_collector import macro_collector
from agents.vlm_geometric_agent import vlm_geometric_agent
from agents.meta_agent import meta_agent
from agents.judge_agent import judge_agent
from agents.risk_manager_agent import risk_manager_agent
from agents.liquidity_agent import liquidity_agent
from agents.microstructure_agent import microstructure_agent
from agents.macro_options_agent import macro_options_agent
from config.local_state import state_manager
from executors.report_generator import report_generator
from executors.trade_executor import trade_executor
from executors.policy_engine import policy_engine
from executors.agent_snapshot_refresher import ensure_registered, refresh_snapshot_bundle
from executors.performance_telemetry import performance_telemetry
from executors.report_hot_path import run_report_hot_path, run_snapshot_analysis_hot_path
from executors.post_mortem import write_post_mortem
from executors.data_synthesizer import synthesize_training_data
from executors.stats_thresholds import adaptive_std_floor, symbol_liq_min_std_usd
from utils.cooldown import is_on_cooldown, set_cooldown
from loguru import logger
import gc
import hashlib
import numpy as np
import pandas as pd
import json
import time
from datetime import datetime, timezone

# [FIX CRASH-2] Module-level cache to avoid 4x duplicate DB queries per cycle
# Cleared at the start of each analysis in run_analysis()
_df_cache: dict = {}        # {symbol:mode: DataFrame}
_market_data_cache: dict = {}  # {symbol:mode: market_data_dict}
_cvd_cache: dict = {}       # {symbol:mode: DataFrame}

_liq_cache: dict = {}       # {symbol:mode: DataFrame}
_funding_cache: dict = {}   # {symbol:mode: DataFrame}

# VLM result cache: keyed by chart content hash, TTL 4 hours
# Avoids re-running expensive Gemini Pro VLM call when chart hasn't changed
_vlm_result_cache: dict = {}   # {chart_hash: (result_dict, timestamp)}
_VLM_CACHE_TTL_S = 4 * 3600


try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("langgraph not available, using sequential fallback")


import operator

def merge_dicts(a: dict, b: dict) -> dict:
    out = a.copy() if a else {}
    if b: out.update(b)
    return out


def _cache_key(symbol: str, mode: TradingMode | str) -> str:
    mode_value = mode.value if isinstance(mode, TradingMode) else str(mode).lower()
    return f"{symbol}:{mode_value}"


def _clear_symbol_mode_caches(symbol: str, mode: TradingMode | str) -> None:
    cache_key = _cache_key(symbol, mode)
    for cache in (_df_cache, _market_data_cache, _cvd_cache, _liq_cache, _funding_cache):
        cache.pop(cache_key, None)
        cache.pop(symbol, None)


def _fmt_level(value) -> str:
    if not isinstance(value, (int, float)) or pd.isna(value):
        return "N/A"
    return f"{float(value):,.2f}"


def _fmt_directional_level(structure: dict, key: str, fallback_key: str) -> str:
    if not isinstance(structure, dict):
        return "N/A"
    value = structure.get(key)
    if not isinstance(value, (int, float)):
        value = structure.get(fallback_key)
    return _fmt_level(value)


def _build_vlm_context_text(market_data: dict, mode: TradingMode) -> str:
    primary_tf = "4H" if mode == TradingMode.SWING else "1D"
    higher_tf = "1D" if mode == TradingMode.SWING else "1W"

    analysis = market_data or {}
    current_price = _fmt_level(analysis.get("current_price"))
    market_structure = (analysis.get("market_structure", {}) or {}).get(higher_tf.lower(), {}) or {}
    structure = analysis.get("structure", {}) or {}
    fibonacci = (analysis.get("fibonacci", {}) or {}).get(higher_tf.lower(), {}) or {}
    swing = (analysis.get("swing_levels", {}) or {}).get(higher_tf.lower(), {}) or {}
    zones = analysis.get("confluence_zones", []) or []
    scenario_engine = analysis.get("scenario_engine", {}) or {}

    diag_support = structure.get(f"support_{higher_tf.lower()}") or {}
    diag_resistance = structure.get(f"resistance_{higher_tf.lower()}") or {}

    filtered_zones = []
    for zone in zones:
        if not isinstance(zone, dict):
            continue
        zone_tfs = {str(tf).lower() for tf in zone.get("timeframes", [])}
        if higher_tf.lower() in zone_tfs:
            filtered_zones.append(zone)

    lines = [
        f"Primary TF: {primary_tf}",
        f"Higher TF: {higher_tf}",
        f"Current Price: {current_price}",
        "Rule: Use the higher timeframe as a directional filter. If the chart image conflicts with it, lower confidence.",
    ]

    if market_structure:
        lines.extend([
            f"{higher_tf} Bias: {str(market_structure.get('structure', 'unknown')).upper()}",
            f"{higher_tf} Last Swing High: {_fmt_level(market_structure.get('last_swing_high'))}",
            f"{higher_tf} Last Swing Low: {_fmt_level(market_structure.get('last_swing_low'))}",
        ])
        choch = market_structure.get("choch") or {}
        if isinstance(choch.get("price"), (int, float)):
            lines.append(f"{higher_tf} CHoCH: {str(choch.get('type', 'unknown')).upper()} @ {_fmt_level(choch.get('price'))}")
        msb = market_structure.get("msb") or {}
        msb_price = msb.get("price")
        if not isinstance(msb_price, (int, float)):
            msb_price = msb.get("broken_level")
        if isinstance(msb_price, (int, float)):
            lines.append(f"{higher_tf} MSB: {str(msb.get('type', 'unknown')).upper()} @ {_fmt_level(msb_price)}")

    lines.extend([
        f"{higher_tf} Diagonal Support: {_fmt_directional_level(diag_support, 'support_price', 'price')}",
        f"{higher_tf} Diagonal Resistance: {_fmt_directional_level(diag_resistance, 'resistance_price', 'price')}",
    ])

    if fibonacci:
        lines.extend([
            f"{higher_tf} Fib 50.0: {_fmt_level(fibonacci.get('fib_500'))}",
            f"{higher_tf} Fib 61.8: {_fmt_level(fibonacci.get('fib_618'))}",
            f"{higher_tf} Fib 70.5: {_fmt_level(fibonacci.get('fib_705'))}",
            f"{higher_tf} Fib 78.6: {_fmt_level(fibonacci.get('fib_786'))}",
        ])

    swing_highs = [v for v in (swing.get("swing_highs", []) or []) if isinstance(v, (int, float))]
    swing_lows = [v for v in (swing.get("swing_lows", []) or []) if isinstance(v, (int, float))]
    if swing_highs or swing_lows:
        lines.extend([
            f"{higher_tf} Swing Highs: {', '.join(_fmt_level(v) for v in swing_highs[:2]) or 'N/A'}",
            f"{higher_tf} Swing Lows: {', '.join(_fmt_level(v) for v in swing_lows[:2]) or 'N/A'}",
            f"{higher_tf} Nearest Resistance: {_fmt_level(swing.get('nearest_resistance'))}",
            f"{higher_tf} Nearest Support: {_fmt_level(swing.get('nearest_support'))}",
        ])

    if filtered_zones:
        zone_lines = []
        for zone in filtered_zones[:2]:
            sources = [str(src) for src in (zone.get("sources", []) or [])[:2]]
            zone_lines.append(
                f"{_fmt_level(zone.get('price_low'))}~{_fmt_level(zone.get('price_high'))} "
                f"(strength={zone.get('strength', 'N/A')}, sources={','.join(sources) or 'N/A'})"
            )
        lines.append(f"{higher_tf} Confluence: {'; '.join(zone_lines)}")

    active_setup = scenario_engine.get("active_setup", {}) or {}
    if active_setup:
        lines.extend([
            f"Scenario Bias: HTF={scenario_engine.get('higher_timeframe_bias', 'neutral')} | EXEC={scenario_engine.get('execution_bias', 'neutral')}",
            f"Active Setup: {active_setup.get('side', 'N/A')} {active_setup.get('trigger', 'wait')}",
            f"Entry Zone: {_fmt_level(active_setup.get('entry_zone_low'))} ~ {_fmt_level(active_setup.get('entry_zone_high'))}",
            f"Invalidation: {_fmt_level(active_setup.get('invalidation'))}",
            f"Targets: {_fmt_level(active_setup.get('tp1'))} / {_fmt_level(active_setup.get('tp2'))}",
        ])
        trap_context = active_setup.get("trap_context", {}) or {}
        if trap_context.get("confirmed"):
            lines.append(
                f"Liquidity Event: {str(trap_context.get('status', 'none')).upper()} "
                f"around {_fmt_level(trap_context.get('reference_level'))}"
            )
        revision_reason = scenario_engine.get("scenario_revision_reason")
        if revision_reason:
            lines.append(f"Scenario Revision: {revision_reason}")

    return "\n".join(lines)

# -------- State definition (LangGraph TypedDict) --------

class AnalysisState(TypedDict):
    symbol: str
    mode: str  # "swing" or "position"
    is_emergency: bool
    execute_trades: bool
    allow_perplexity: bool
    notification_context: str
    playbook_context: dict
    emergency_context: dict

    # Data collection results
    df_size: int
    market_data_compact: str
    narrative_text: str
    funding_context: str
    cvd_context: str
    liquidation_context: str
    rag_context: str
    telegram_news: str
    unified_narrative: str  # Deduplicated merge of narrative_text + rag_context + telegram_news
    feedback_text: str
    microstructure_context: str
    macro_context: str
    deribit_context: str
    fear_greed_context: str
    active_orders: list
    open_positions: str
    onchain_snapshot: dict
    onchain_context: str
    onchain_gate: dict

    # Blackboard Pattern State
    budget: int
    turn_count: int
    anomalies: list
    blackboard: Annotated[Dict[str, dict], merge_dicts]
    conviction_score: float
    raw_funding: dict  # [FIX] Cached from node_funding_context for report reuse

    # Chart
    chart_image_b64: Optional[str]
    chart_bytes: Optional[bytes]
    vlm_context_text: str
    current_bias: str
    scenario_revision_reason: str
    active_setup: dict

    # Final output
    market_regime: str
    regime_context: dict
    final_decision: Dict
    daily_dual_plan: dict
    policy_snapshot: dict
    report: Optional[Dict]

    # ------------- Z Score (context_gathering ) -------------
    stats_context: dict  # {liq_mean, liq_std, imbalance_mean, ..., dvol_mean, dvol_std, ...}

    # Error tracking
    errors: Annotated[list, operator.add]



# -------------------- Node functions --------------------

def node_collect_data(state: AnalysisState) -> dict:
    """1m OHLCV: GCS 로컬 캐시(역사적 데이터) → Supabase gap fill(최근 갭) → 머지.

    데이터 레이어:
      - 1m bulk  : GCS 로컬 캐시 (cache/gcs_parquet/) — Swing 6개월 / Position 4년
      - 1m 갭    : Supabase — 캐시 마지막 ts ~ 현재, 최대 43,200개(30일 상한)
      - 4h/1d/1w : GCS 로컬 캐시 (리샘플링된 고해상도 히스토리)
    """
    symbol = state["symbol"]
    mode = TradingMode(state["mode"])
    cache_key = _cache_key(symbol, mode)
    months_back = settings.history_lookback_months_for_mode(mode)

    from processors.gcs_parquet import gcs_parquet_store

    # ── 1. GCS 로컬 캐시에서 1m 역사 데이터 로드 ─────────────────────────
    df_cached = pd.DataFrame()
    try:
        df_cached = gcs_parquet_store.load_ohlcv("1m", symbol, months_back=months_back)
        if not df_cached.empty:
            logger.info(f"GCS local cache: {len(df_cached):,} rows for {symbol} ({months_back}m)")
    except Exception as e:
        logger.warning(f"GCS local cache load failed for {symbol}: {e}")

    # ── 2. Supabase에서 갭 구간만 보충 ────────────────────────────────────
    if not df_cached.empty:
        last_cached_ts = df_cached["timestamp"].max()
        if hasattr(last_cached_ts, 'to_pydatetime'):
            last_cached_ts = last_cached_ts.to_pydatetime()

        gap_minutes = int((datetime.now(timezone.utc) - last_cached_ts).total_seconds() / 60) + 10
        gap_limit = min(gap_minutes, settings.SWING_CANDLE_LIMIT)  # 상한: 43,200 (30일)

        logger.info(f"Supabase gap fill: {gap_minutes:,}min gap → fetching up to {gap_limit:,} rows")
        df_recent = db.get_market_data_gap(symbol, since=last_cached_ts, limit=gap_limit)

        if not df_recent.empty:
            df = pd.concat([df_cached, df_recent], ignore_index=True)
            df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            logger.info(f"Merged: {len(df_cached):,} (cache) + {len(df_recent):,} (gap) = {len(df):,} rows")
        else:
            df = df_cached
            logger.info(f"No gap data from Supabase, using cache only: {len(df):,} rows")
    else:
        # 캐시 없음 → Supabase 전체 fallback (최대 30일)
        fallback_limit = settings.SWING_CANDLE_LIMIT
        logger.warning(f"GCS local cache empty for {symbol}, Supabase fallback (limit={fallback_limit:,})")
        df = db.get_latest_market_data(symbol, limit=fallback_limit)

    if df.empty:
        return {"df_size": 0, "errors": state.get("errors", []) + [f"No market data for {symbol}"]}

    # [FIX CRASH-2] Cache for reuse by node_triage, node_generate_chart, node_generate_report
    _df_cache[cache_key] = df

    # ── 3. 고해상도 TF: GCS 로컬 캐시에서 로드 (4h / 1d / 1w) ──────────
    df_4h, df_1d, df_1w = None, None, None
    try:
        # Swing: 4h/1d for execution, 1w for big-picture structural context
        df_4h = gcs_parquet_store.load_ohlcv("4h", symbol, months_back=months_back)
        df_1d = gcs_parquet_store.load_ohlcv("1d", symbol, months_back=months_back)
        df_1w = gcs_parquet_store.load_ohlcv("1w", symbol, months_back=months_back)
        logger.debug(
            f"GCS TF data — 4h:{len(df_4h) if df_4h is not None and not df_4h.empty else 0} "
            f"1d:{len(df_1d) if df_1d is not None and not df_1d.empty else 0} "
            f"1w:{len(df_1w) if df_1w is not None and not df_1w.empty else 0}"
        )
    except Exception as e:
        logger.warning(f"GCS higher TF load skipped: {e}")

    market_data = math_engine.analyze_market(df, mode, df_4h=df_4h, df_1d=df_1d, df_1w=df_1w)
    compact = math_engine.format_compact(market_data)

    # [OOM FIX] GCS historical DataFrames 분석 완료 후 즉시 해제 (수백MB)
    try:
        del df_4h, df_1d, df_1w
        gc.collect()
    except Exception:
        pass

    # [FIX RESOURCE-1] Cache market_data so node_generate_chart doesn't recompute
    _market_data_cache[cache_key] = market_data

    # V5: Fetch currently active orders for this symbol to inject into the PM context
    all_active = state_manager.get_active_orders()
    active_for_symbol = [o for o in all_active if o['symbol'] == symbol]

    # V7: Fetch open positions to prevent double-ordering
    open_position_text = "No open positions."
    if settings.PAPER_TRADING_MODE:
        try:
            from executors.paper_exchange import paper_engine
            pos = paper_engine._fetch_all(
                "SELECT * FROM paper_positions WHERE is_open = 1 AND symbol = ?", (symbol,)
            )
            if pos:
                open_position_text = "\n".join([f"[{p['exchange']}] {p['side']} {p['size']:.4f} coins @ ${p['entry_price']:.2f}" for p in pos])
        except Exception as e:
            logger.error(f"Failed to fetch paper positions: {e}")

    return {
        "df_size": len(df),
        "market_data_compact": compact,
        "active_orders": active_for_symbol,
        "open_positions": open_position_text,
        "errors": state.get("errors", [])
    }
def node_context_gathering(state: AnalysisState) -> dict:
    """Consolidated node to gather all external context (Perplexity, RAG, DB, etc.)
    Uses ThreadPoolExecutor to run expensive I/O operations in parallel.
    """
    import concurrent.futures
    updates = {}
    symbol = state["symbol"]
    mode = TradingMode(state["mode"])
    cache_key = _cache_key(symbol, mode)
    is_emergency = state.get("is_emergency", False)
    allow_perplexity = bool(state.get("allow_perplexity", False))
    asset = "BTC" if "BTC" in symbol else "ETH"
    coin_name = "Bitcoin BTC" if "BTC" in symbol else "Ethereum ETH"

    def fetch_perplexity():
        perp_updates = {}
        if not allow_perplexity:
            logger.info(f"Perplexity disabled for {symbol} in this analysis path.")
            perp_updates["narrative_text"] = "Market Narrative: Disabled for this analysis path"
            return perp_updates
        try:
            narrative = perplexity_collector.search_market_narrative(symbol, is_emergency=is_emergency)
            narrative_text = perplexity_collector.format_for_agents(narrative)
            logger.info(f"Narrative for {asset}: {narrative.get('sentiment', '?')}")
            perp_updates["narrative_text"] = narrative_text
            
            # RAG Ingest
            if narrative_text and "Unavailable" not in narrative_text and len(narrative_text) > 50:
                import hashlib
                from datetime import datetime, timezone
                ts = datetime.now(timezone.utc).isoformat()
                doc_id = hashlib.md5(f"perplexity:{symbol}:{ts[:13]}".encode()).hexdigest()
                light_rag.ingest_message(text=narrative_text, channel=f"perplexity_{symbol}", timestamp=ts, message_id=doc_id)
                logger.info(f"RAG: Perplexity narrative ingested for {symbol}")
        except Exception as e:
            logger.error(f"Perplexity error: {e}")
            perp_updates["narrative_text"] = "Market Narrative: Unavailable"
        return perp_updates

    def fetch_rag():
        try:
            return {"rag_context": light_rag.format_context_for_agents(light_rag.query(coin_name), max_length=1500)}
        except Exception as e:
            logger.error(f"RAG query error: {e}")
            return {}

    def fetch_db_contexts():
        db_updates = {}
        # 3. Funding Context
        try:
            res = db.client.table("funding_data").select("*").eq("symbol", symbol).order("timestamp", desc=True).limit(1).execute()
            raw_funding = res.data[0] if res.data else {}
            funding_data = math_engine.analyze_funding_context(raw_funding) if raw_funding else {}
            if raw_funding:
                for k in ['oi_binance', 'oi_bybit', 'oi_okx']: funding_data[k] = raw_funding.get(k, 0)
            db_updates["funding_context"] = json.dumps(funding_data, default=str) if funding_data else "No funding data."
            db_updates["raw_funding"] = raw_funding
        except Exception as e:
            logger.error(f"Funding context error: {e}")

        # 4. OI Divergence + MFI Summary (appended to funding_context)
        try:
            oi_rows = db.client.table("funding_data").select("oi_binance", "oi_bybit", "oi_okx", "timestamp").eq("symbol", symbol).order("timestamp", desc=True).limit(12).execute()
            if oi_rows.data and len(oi_rows.data) >= 3:
                oi_series = [float(r.get("oi_binance", 0) or 0) + float(r.get("oi_bybit", 0) or 0) + float(r.get("oi_okx", 0) or 0) for r in oi_rows.data]
                oi_now, oi_prev = oi_series[0], oi_series[-1]
                oi_chg_pct = ((oi_now - oi_prev) / oi_prev * 100) if oi_prev else 0
                df_snap = _df_cache.get(cache_key)
                price_chg_pct = 0.0
                if df_snap is not None and not df_snap.empty and len(df_snap) >= 12:
                    p_now = float(df_snap['close'].iloc[-1])
                    p_prev = float(df_snap['close'].iloc[-12])
                    price_chg_pct = ((p_now - p_prev) / p_prev * 100) if p_prev else 0
                oi_div = "DIVERGENCE" if (oi_chg_pct > 1.5 and price_chg_pct < -0.5) or (oi_chg_pct < -1.5 and price_chg_pct > 0.5) else "ALIGNED"
                mfi_proxy = "INFLOW" if oi_chg_pct > 0.5 and price_chg_pct > 0 else "OUTFLOW" if oi_chg_pct < -0.5 and price_chg_pct < 0 else "NEUTRAL"
                oi_summary = f" | [OI_DIV] OI_chg={oi_chg_pct:+.2f}% Price_chg={price_chg_pct:+.2f}% Status={oi_div} | [MFI_PROXY] {mfi_proxy}"
                db_updates["funding_context"] = db_updates.get("funding_context", "") + oi_summary
        except Exception as e:
            logger.error(f"OI divergence/MFI error: {e}")

        # 5. Liquidation Context
        try:
            liq_df = db.get_liquidation_data(symbol, limit=settings.data_lookback_hours * 60)
            _liq_cache[cache_key] = liq_df  # Cache for chart generation
            if not liq_df.empty:
                t_long, t_short = float(liq_df['long_liq_usd'].sum()), float(liq_df['short_liq_usd'].sum())
                db_updates["liquidation_context"] = f"[LIQUIDATION] Total=${t_long+t_short:,.0f} (Long=${t_long:,.0f}, Short=${t_short:,.0f})"
        except Exception as e: logger.error(f"Liq error: {e}")


        # 7. Telegram News
        try:
            news = db.get_recent_telegram_messages(hours=1 if is_emergency else 4)
            db_updates["telegram_news"] = "\n".join([f"[{m['channel']}] {m['text'][:200]}" for m in news[:10]]) if news else "No news."
        except Exception as e: logger.error(f"Telegram news error: {e}")

        # 8-12. Other DB Contexts
        try:
            fb = db.get_feedback_history(limit=5)
            if fb: db_updates["feedback_text"] = "\n\n[PAST MISTAKES]\n" + "\n".join([f"- {f.get('mistake_summary', 'N/A')[:150]}" for f in fb[:3]])
        except Exception: pass
        try:
            snap = db.get_latest_microstructure(symbol)
            if snap: db_updates["microstructure_context"] = f"[MICRO] spread={snap.get('spread_bps', 0):.2f}bps imbalance={snap.get('orderbook_imbalance', 0):.4f}"
        except Exception: pass
        try:
            m = db.get_latest_macro_data()
            if m: db_updates["macro_context"] = f"[MACRO] DGS10={m.get('dgs10', 'N/A')} DXY={m.get('dxy', 'N/A')} NASDAQ={m.get('nasdaq', 'N/A')}"
        except Exception: pass
        try:
            currency = symbol[:-4] if symbol.endswith('USDT') else symbol
            d = db.get_latest_deribit_data(currency)
            if d: db_updates["deribit_context"] = f"[DERIBIT {currency}] DVOL={d.get('dvol')} PCR={d.get('pcr_oi')}"
        except Exception: pass
        try:
            fg = db.get_latest_fear_greed()
            if fg: db_updates["fear_greed_context"] = f"[FEAR&GREED] {fg.get('value')}/100"
        except Exception: pass

        return db_updates

    def fetch_stats_context():
        """Z-Score 기반 통계 컨텍스트 수집 로직.

        통계 임계값 계산 최근 7일 시계열 데이터의 rolling std 기반으로 도출
        asset/regime 별 adaptive floor를 사용한다.
        """
        stats = {}
        currency = symbol[:-4] if symbol.endswith("USDT") else symbol
        adaptive_kwargs = {
            "rolling_window": int(getattr(settings, "STATS_ADAPTIVE_STD_WINDOW_HOURS", 24)),
            "rolling_quantile": float(getattr(settings, "STATS_ADAPTIVE_STD_FLOOR_QUANTILE", 0.20)),
            "min_periods": int(getattr(settings, "STATS_ADAPTIVE_STD_MIN_PERIODS", 12)),
            "min_rolling_samples": int(getattr(settings, "STATS_ADAPTIVE_STD_MIN_ROLLING_SAMPLES", 24)),
        }

        def _age_hours(value) -> float | None:
            try:
                ts = pd.to_datetime(value, utc=True, errors="coerce")
                if pd.isna(ts):
                    return None
                return float((pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 3600.0)
            except Exception:
                return None

        # 단계 1. Liquidation 7-day hourly stats 수집
        try:
            liq_df_wide = db.get_liquidation_data(
                symbol,
                limit=int(getattr(settings, "ORCHESTRATOR_LIQ_STATS_LIMIT", 2880)),
                columns="timestamp,long_liq_usd,short_liq_usd",
            )
            if not liq_df_wide.empty and "timestamp" in liq_df_wide.columns:
                liq_ts = pd.to_datetime(liq_df_wide["timestamp"], utc=True, errors="coerce")
                latest_liq_ts = liq_ts.max()
                liq_age_hours = _age_hours(latest_liq_ts)
                liq_is_stale = (
                    liq_age_hours is not None
                    and liq_age_hours > float(getattr(settings, "STATS_MAX_LIQ_STALE_HOURS", 6.0))
                )
                stats["liq_data_age_hours"] = liq_age_hours
                stats["liq_data_stale"] = bool(liq_is_stale)
                hourly = (
                    liq_df_wide.set_index("timestamp")[["long_liq_usd", "short_liq_usd"]]
                    .resample("1h").sum()
                )
                hourly["total"] = hourly["long_liq_usd"] + hourly["short_liq_usd"]
                liq_mean = float(hourly["total"].mean())
                liq_std  = float(hourly["total"].std())
                liq_median = float(hourly["total"].median())
                liq_mad = float((hourly["total"] - liq_median).abs().median())
                stats["liq_p90"] = float(hourly["total"].quantile(0.90))
                stats["liq_p95"] = float(hourly["total"].quantile(0.95))
                stats["liq_p99"] = float(hourly["total"].quantile(0.99))
                stats["liq_median"] = liq_median
                stats["liq_mad"] = liq_mad
                liq_floor_meta = adaptive_std_floor(
                    hourly["total"].tolist(),
                    base_floor=symbol_liq_min_std_usd(
                        symbol,
                        btc_floor=float(getattr(settings, "STATS_LIQ_MIN_STD_BTC_USD", 1_500_000.0)),
                        eth_floor=float(getattr(settings, "STATS_LIQ_MIN_STD_ETH_USD", 600_000.0)),
                        default_floor=float(getattr(settings, "STATS_LIQ_MIN_STD_DEFAULT_USD", 500_000.0)),
                    ),
                    **adaptive_kwargs,
                )
                liq_floor = float(liq_floor_meta["floor"])
                stats["liq_std_floor"] = liq_floor
                stats["liq_std_floor_base"] = float(liq_floor_meta["base_floor"])
                stats["liq_std_floor_rolling"] = float(liq_floor_meta["rolling_floor"])
                if liq_is_stale:
                    logger.warning(
                        f"[Stats] liquidation data stale age={liq_age_hours:.2f}h "
                        f"> {float(getattr(settings, 'STATS_MAX_LIQ_STALE_HOURS', 6.0)):.2f}h ? static fallback"
                    )
                elif liq_std >= liq_floor:
                    stats["liq_mean"] = liq_mean
                    stats["liq_std"]  = liq_std
                    logger.debug(
                        f"[Stats] liq mean=${liq_mean:,.0f} std=${liq_std:,.0f} "
                        f"floor=${liq_floor:,.0f}"
                    )
                else:
                    _liq_data_ok = liq_floor_meta.get("has_enough_data", False)
                    logger.warning(
                        f"[Stats] liq_std ${liq_std:,.0f} < adaptive floor ${liq_floor:,.0f} "
                        f"(base=${liq_floor_meta['base_floor']:,.0f}, rolling=${liq_floor_meta['rolling_floor']:,.0f}) "
                        f"data_ok={_liq_data_ok} → static fallback 사용"
                    )
        except Exception as e:
            logger.warning(f"[Stats] liq error: {e}")

        # 단계 2. Microstructure 7-day stats (최소 30개 hourly snapshot)
        try:
            rows = (
                db.client.table("microstructure_data")
                .select("spread_bps,orderbook_imbalance,timestamp")
                .eq("symbol", symbol)
                .order("timestamp", desc=True)
                .limit(168)
                .execute()
            )
            if rows.data and len(rows.data) >= 30:  # 30개 미만이면 std 불
                latest_micro_ts = max((r.get("timestamp") for r in rows.data if r.get("timestamp")), default=None)
                micro_age_hours = _age_hours(latest_micro_ts)
                micro_is_stale = (
                    micro_age_hours is not None
                    and micro_age_hours > float(getattr(settings, "STATS_MAX_MICRO_STALE_HOURS", 3.0))
                )
                stats["micro_data_age_hours"] = micro_age_hours
                stats["micro_data_stale"] = bool(micro_is_stale)
                imbs    = [abs(float(r.get("orderbook_imbalance") or 0)) for r in rows.data]
                spreads = [float(r.get("spread_bps") or 0) for r in rows.data]
                if micro_is_stale:
                    logger.warning(
                        f"[Stats] microstructure data stale age={micro_age_hours:.2f}h "
                        f"> {float(getattr(settings, 'STATS_MAX_MICRO_STALE_HOURS', 3.0)):.2f}h - static fallback"
                    )
                else:
                    stats["imbalance_mean"] = float(np.mean(imbs))
                    stats["imbalance_std"]  = float(np.std(imbs))
                    stats["spread_mean"]    = float(np.mean(spreads))
                    stats["spread_std"]     = float(np.std(spreads))
            elif rows.data:
                logger.warning(
                    f"[Stats] micro samples={len(rows.data)} < 30 → std 불, static fallback"
                )
        except Exception as e:
            logger.warning(f"[Stats] micro error: {e}")

        # ---- 3. DVOL/PCR 7 day hourly stats ( 7~30 lookback) ----
        # ------------------ 24h lookback regime ------------------
        try:
            rows = (
                db.client.table("deribit_data")
                .select("dvol,pcr_oi,timestamp")
                .eq("symbol", currency)
                .order("timestamp", desc=True)
                .limit(168)   # 7일치 24h (최 표 lookback)
                .execute()
            )
            if rows.data and len(rows.data) >= 24:  # 최소 1주치 hourly 필요
                latest_deribit_ts = max((r.get("timestamp") for r in rows.data if r.get("timestamp")), default=None)
                deribit_age_hours = _age_hours(latest_deribit_ts)
                deribit_is_stale = (
                    deribit_age_hours is not None
                    and deribit_age_hours > float(getattr(settings, "STATS_MAX_DERIBIT_STALE_HOURS", 6.0))
                )
                stats["deribit_data_age_hours"] = deribit_age_hours
                stats["deribit_data_stale"] = bool(deribit_is_stale)
                dvols = [float(r["dvol"])   for r in rows.data if r.get("dvol")]
                pcrs  = [float(r["pcr_oi"]) for r in rows.data if r.get("pcr_oi")]
                if deribit_is_stale:
                    logger.warning(
                        f"[Stats] deribit data stale age={deribit_age_hours:.2f}h "
                        f"> {float(getattr(settings, 'STATS_MAX_DERIBIT_STALE_HOURS', 6.0)):.2f}h - static fallback"
                    )
                elif dvols:
                    dvol_mean = float(np.mean(dvols))
                    dvol_std  = float(np.std(dvols))
                    dvol_floor_meta = adaptive_std_floor(
                        dvols,
                        base_floor=float(getattr(settings, "STATS_DVOL_MIN_STD", 0.8)),
                        **adaptive_kwargs,
                    )
                    dvol_floor = float(dvol_floor_meta["floor"])
                    stats["dvol_std_floor"] = dvol_floor
                    if dvol_std >= dvol_floor:
                        stats["dvol_mean"] = dvol_mean
                        stats["dvol_std"]  = dvol_std
                    else:
                        _dvol_data_ok = dvol_floor_meta.get("has_enough_data", False)
                        logger.warning(
                            f"[Stats] dvol_std={dvol_std:.2f} < adaptive floor {dvol_floor:.2f} "
                            f"(base={dvol_floor_meta['base_floor']:.2f}, rolling={dvol_floor_meta['rolling_floor']:.2f}) "
                            f"data_ok={_dvol_data_ok} - DVOL flat, static fallback"
                        )
                if not deribit_is_stale and pcrs:
                    pcr_mean = float(np.mean(pcrs))
                    pcr_std  = float(np.std(pcrs))
                    pcr_floor_meta = adaptive_std_floor(
                        pcrs,
                        base_floor=float(getattr(settings, "STATS_PCR_MIN_STD", 0.01)),
                        **adaptive_kwargs,
                    )
                    pcr_floor = float(pcr_floor_meta["floor"])
                    stats["pcr_std_floor"] = pcr_floor
                    if pcr_std >= pcr_floor:
                        stats["pcr_mean"] = pcr_mean
                        stats["pcr_std"]  = pcr_std
                    else:
                        _pcr_data_ok = pcr_floor_meta.get("has_enough_data", False)
                        logger.warning(
                            f"[Stats] pcr_std={pcr_std:.4f} < adaptive floor {pcr_floor:.4f} "
                            f"(base={pcr_floor_meta['base_floor']:.4f}, rolling={pcr_floor_meta['rolling_floor']:.4f}) "
                            f"data_ok={_pcr_data_ok} - PCR flat, static fallback"
                        )
            elif rows.data:
                logger.warning(
                    f"[Stats] deribit samples={len(rows.data)} < 24 - static fallback"
                )
        except Exception as e:
            logger.warning(f"[Stats] dvol error: {e}")

        return {"stats_context": stats}

    def fetch_onchain_context():
        try:
            snapshot = db.get_latest_onchain_snapshot(symbol, max_age_hours=48)
            return {
                "onchain_snapshot": snapshot or {},
                "onchain_context": onchain_signal_engine.format_context(snapshot),
                "onchain_gate": onchain_signal_engine.build_gate(snapshot),
            }
        except Exception as e:
            logger.warning(f"On-chain context load error for {symbol}: {e}")
            return {
                "onchain_snapshot": {},
                "onchain_context": "On-chain Context: unavailable",
                "onchain_gate": onchain_signal_engine.build_gate(None),
            }

    # Run network/DB calls in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        f_perp  = executor.submit(fetch_perplexity)
        f_rag   = executor.submit(fetch_rag)
        f_db    = executor.submit(fetch_db_contexts)
        f_stats = executor.submit(fetch_stats_context)
        f_onchain = executor.submit(fetch_onchain_context)

        updates.update(f_perp.result())
        updates.update(f_rag.result())
        updates.update(f_db.result())
        updates.update(f_stats.result())
        updates.update(f_onchain.result())

    return updates


def _merge_narrative_sources(
    narrative_text: str,
    rag_context: str,
    telegram_news: str,
    *,
    max_chars: int = 1400,
) -> str:
    """Deduplicate and merge 3 narrative sources into one compact context string.

    Priority: Perplexity (primary) > LightRAG (secondary) > Telegram (tertiary).
    Dedup method: 4-char character-level sliding-window fingerprints.
    Works for both Korean and English without tokenization.

    Savings vs concatenation: ~67% fewer chars (4300 → ~1400).
    """
    import re as _re

    def _fingerprints(text: str) -> set:
        clean = _re.sub(r"\s+", "", text.lower())
        return {clean[i:i + 4] for i in range(len(clean) - 3)} if len(clean) > 3 else set()

    def _is_duplicate(line: str, seen: set, threshold: float) -> bool:
        fp = _fingerprints(line)
        return bool(fp) and len(fp & seen) / len(fp) >= threshold

    seen: set = set()
    parts: list = []
    total = 0

    # 1. Perplexity narrative — always primary, hard cap 700 chars
    if narrative_text:
        chunk = narrative_text.strip()[:700]
        parts.append(chunk)
        seen.update(_fingerprints(chunk))
        total += len(chunk)

    # 2. LightRAG context — line-level dedup vs Perplexity, cap 400 chars
    if rag_context and total < max_chars:
        new_lines: list = []
        rag_total = 0
        for line in rag_context.splitlines():
            line = line.strip()
            if not line:
                continue
            if not _is_duplicate(line, seen, threshold=0.40):
                new_lines.append(line)
                seen.update(_fingerprints(line))
                rag_total += len(line)
            if rag_total >= 400:
                break
        if new_lines:
            block = "\n".join(new_lines)
            parts.append(f"[RAG]\n{block}")
            total += len(block)

    # 3. Telegram news — top 3 new lines only, cap 300 chars
    if telegram_news and total < max_chars:
        new_lines = []
        for line in telegram_news.splitlines():
            line = line.strip()
            if not line:
                continue
            if not _is_duplicate(line, seen, threshold=0.50):
                new_lines.append(line)
                seen.update(_fingerprints(line))
            if len(new_lines) >= 3:
                break
        if new_lines:
            block = "\n".join(new_lines)
            parts.append(f"[TG]\n{block}")

    result = "\n\n".join(parts)
    return result[:max_chars]


def node_unify_narrative(state: AnalysisState) -> dict:
    """Merge narrative_text + rag_context + telegram_news into one deduplicated string.
    Runs once after context_gathering; result reused by meta_agent and judge_agent."""
    unified = _merge_narrative_sources(
        state.get("narrative_text", ""),
        state.get("rag_context", ""),
        state.get("telegram_news", ""),
    )
    logger.debug(
        f"unified_narrative: {len(unified)} chars "
        f"(narrative={len(state.get('narrative_text',''))} "
        f"rag={len(state.get('rag_context',''))} "
        f"tg={len(state.get('telegram_news',''))})"
    )
    return {"unified_narrative": unified}


def node_meta_agent(state: AnalysisState) -> dict:
    """Run Meta-Agent to classify market regime and provide trust directives."""
    try:
        result = meta_agent.classify_regime(
            market_data_compact=state.get("market_data_compact", ""),
            deribit_context=state.get("deribit_context", ""),
            funding_context=state.get("funding_context", ""),
            macro_context=state.get("macro_context", ""),
            rag_context=state.get("unified_narrative", "") or state.get("rag_context", ""),
            telegram_news="",  # already merged into unified_narrative
            onchain_context=state.get("onchain_context", ""),
            mode=TradingMode(state["mode"])
        )
        logger.info(f"Market Regime classified: {result.get('regime', 'UNKNOWN')}")
        return {
            "market_regime": result.get("regime", "RANGE_BOUND"),
            "regime_context": result
        }
    except Exception as e:
        logger.error(f"Meta-Agent node error: {e}")
        return {"market_regime": "RANGE_BOUND", "regime_context": {}}


def _compute_confluence_score(
    market_data: dict,
    anomalies: list,
    raw_funding: dict,
) -> dict:
    """쉽알남 '근거 중첩 필수' — count structural factors pointing in the same direction.
    Requires ≥3 aligned factors before LONG/SHORT is considered actionable.
    Returns: {score, direction, factors, gate_passed, long_score, short_score}
    """
    long_factors: list = []
    short_factors: list = []

    current_price = market_data.get("current_price")
    if not isinstance(current_price, (int, float)) or current_price <= 0:
        return {"score": 0, "direction": "NEUTRAL", "factors": [], "gate_passed": False, "long_score": 0, "short_score": 0}

    # 1. HTF Structure: 1d/1w CHoCH or MSB direction
    market_struct = market_data.get("market_structure", {}) or {}
    for tf in ("1d", "1w"):
        info = market_struct.get(tf)
        if not isinstance(info, dict):
            continue
        for struct_key in ("choch", "msb"):
            s = info.get(struct_key)
            if not isinstance(s, dict):
                continue
            stype = str(s.get("type", "")).lower()
            if any(kw in stype for kw in ("bull", "high", "up")):
                long_factors.append(f"htf_{tf}_structure")
                break
            elif any(kw in stype for kw in ("bear", "low", "down")):
                short_factors.append(f"htf_{tf}_structure")
                break

    # 2. Active Setup Bias (scenario_engine confirmed directional setup)
    scenario_engine = market_data.get("scenario_engine", {}) or {}
    active_setup = scenario_engine.get("active_setup", {}) or {}
    side = str(active_setup.get("side", "")).upper()
    if side == "LONG":
        long_factors.append("active_setup")
    elif side == "SHORT":
        short_factors.append("active_setup")

    # 3. Near Confluence Zone with multi-TF agreement (level_count >= 2, within 1.5%)
    confluence_zones = market_data.get("confluence_zones", []) or []
    for zone in confluence_zones:
        price = zone.get("price")
        if not isinstance(price, (int, float)):
            continue
        dist_pct = abs(float(current_price) - float(price)) / float(current_price) * 100.0
        if dist_pct <= 1.5 and int(zone.get("level_count", 0)) >= 2:
            zone_type = str(zone.get("type", "")).lower()
            if "support" in zone_type:
                long_factors.append("near_confluence_support")
            elif "resist" in zone_type:
                short_factors.append("near_confluence_resistance")
            else:
                # Ambiguous zone — assign to whichever directional side is active
                if side == "SHORT":
                    short_factors.append("near_confluence_zone")
                else:
                    long_factors.append("near_confluence_zone")
            break

    # 4. Fibonacci Confluence (price within 1% of 38.2/50/61.8 retracement)
    fib_data = market_data.get("fibonacci", {}) or {}
    key_fibs = {"fib_382", "fib_500", "fib_618"}
    nearest_fib_dist = None
    for _tf, fib_levels in fib_data.items():
        if not isinstance(fib_levels, dict):
            continue
        for key, val in fib_levels.items():
            if key not in key_fibs or not isinstance(val, (int, float)):
                continue
            dist_pct = abs(float(current_price) - float(val)) / float(current_price) * 100.0
            if nearest_fib_dist is None or dist_pct < nearest_fib_dist:
                nearest_fib_dist = dist_pct
    if nearest_fib_dist is not None and nearest_fib_dist <= 1.0:
        if side == "SHORT":
            short_factors.append("fib_confluence")
        else:
            long_factors.append("fib_confluence")

    # 5. Funding Rate Extreme (contrarian reversal signal)
    funding_rate = float(raw_funding.get("funding_rate", 0) or 0)
    if funding_rate > 0.05:        # Crowded long → SHORT reversal candidate
        short_factors.append("funding_extreme_high")
    elif funding_rate < -0.02:     # Crowded short → LONG reversal candidate
        long_factors.append("funding_extreme_low")

    # 6. Volume-Backed Breakout (already detected in triage Vol+OI check)
    if "true_breakout" in anomalies:
        if side == "SHORT":
            short_factors.append("volume_breakout")
        else:
            long_factors.append("volume_breakout")

    # Deduplicate while preserving order
    long_factors = list(dict.fromkeys(long_factors))
    short_factors = list(dict.fromkeys(short_factors))

    long_score = len(long_factors)
    short_score = len(short_factors)

    if long_score > short_score:
        direction = "LONG"
        score = long_score
        active_factors = long_factors
    elif short_score > long_score:
        direction = "SHORT"
        score = short_score
        active_factors = short_factors
    else:
        direction = "NEUTRAL"
        score = 0
        active_factors = []

    return {
        "score": score,
        "direction": direction,
        "factors": active_factors,
        "gate_passed": score >= 3,
        "long_score": long_score,
        "short_score": short_score,
    }


def node_triage(state: AnalysisState) -> dict:
    """Evaluate Python/Pandas rules for Anomaly Detection before waking LLMs."""
    anomalies = []
    symbol = state["symbol"]
    mode = TradingMode(state["mode"])
    cache_key = _cache_key(symbol, mode)
    
    # 1. Volume-Backed Breakout (Mathematical)
    try:
        # [FIX CRASH-2] Use cached DataFrame from node_collect_data
        df = _df_cache.get(cache_key)
        if df is None:
            df = db.get_latest_market_data(
                symbol,
                limit=int(getattr(settings, "ORCHESTRATOR_MARKET_FALLBACK_LIMIT", 2880)),
                columns="timestamp,open,high,low,close,volume",
            )  # fallback if cache miss
        if not df.empty:
            df_1h = df.resample('1h', on='timestamp').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
            }).dropna()
            
            if len(df_1h) >= 30:
                df_1h['tr'] = np.maximum(
                    df_1h['high'] - df_1h['low'],
                    np.maximum(
                        abs(df_1h['high'] - df_1h['close'].shift(1)),
                        abs(df_1h['low'] - df_1h['close'].shift(1))
                    )
                )
                df_1h['atr'] = df_1h['tr'].rolling(14).mean()
                df_1h['vol_ma'] = df_1h['volume'].rolling(24).mean() # 24h avg vol
                
                recent_atr = df_1h['atr'].iloc[-3:].mean()
                hist_atr = df_1h['atr'].iloc[:-3].mean()
                recent_vol = df_1h['volume'].iloc[-1]
                hist_vol = df_1h['vol_ma'].iloc[-2]
                
                # Fetch recent OI to detect if volume is from new positions or just liquidations
                try:
                    oi_res = db.client.table("funding_data").select("oi_binance", "timestamp").eq("symbol", symbol).order("timestamp", desc=True).limit(2).execute()
                    if oi_res.data and len(oi_res.data) >= 2:
                        recent_oi = float(oi_res.data[0].get("oi_binance", 0))
                        prev_oi = float(oi_res.data[1].get("oi_binance", 0))
                        oi_delta_pct = ((recent_oi - prev_oi) / prev_oi) * 100 if prev_oi else 0
                    else:
                        oi_delta_pct = 0
                except Exception:
                    oi_delta_pct = 0
                
                # Condition: Volatility up 50% + Volume up 100%
                if recent_atr > (hist_atr * 1.5) and recent_vol > (hist_vol * 2.0):
                    # THE WHIPSAW FILTER: 
                    # If OI dropped heavily (< -2%), it's a liquidation cascade (whipsaw). 
                    # We might want to fade it, but it's not a "true breakout".
                    # If OI increased (> 1%), it means NEW MONEY is driving the move. Truly actionable.
                    if oi_delta_pct > 0.5:
                        if not is_on_cooldown("true_breakout", symbol):
                            anomalies.append("true_breakout")
                            set_cooldown("true_breakout", symbol)
                    elif oi_delta_pct < -2.0:
                        if not is_on_cooldown("liquidation_cascade", symbol):
                            anomalies.append("liquidation_cascade")
                            set_cooldown("liquidation_cascade", symbol)
    except Exception as e:
        logger.error(f"Triage volatility math error: {e}")

    # 단계 Z-Score Enhanced Agent Checks (2, 3, 4)
    _stats = state.get("stats_context") or {}

    # ------------- 2. Liquidity Agent (Z Score ) -------------
    try:
        _liq_stats = None
        if any(
            _stats.get(key) is not None
            for key in ("liq_mean", "liq_std", "liq_median", "liq_mad", "liq_p90", "liq_p95", "liq_p99")
        ):
            _liq_stats = {
                "mean": _stats.get("liq_mean"),
                "std": _stats.get("liq_std"),
                "median": _stats.get("liq_median"),
                "mad": _stats.get("liq_mad"),
                "p90": _stats.get("liq_p90"),
                "p95": _stats.get("liq_p95"),
                "p99": _stats.get("liq_p99"),
            }
        raw_funding = state.get("raw_funding", {}) or {}
        total_oi = sum(
            float(raw_funding.get(key, 0) or 0)
            for key in ("oi_binance", "oi_bybit", "oi_okx")
        )
        _liq_result = liquidity_agent.analyze(
            cvd_context=state.get("funding_context", ""),
            liquidation_context=state.get("liquidation_context", ""),
            mode=state.get("mode", "SWING"),
            total_oi=total_oi if total_oi > 0 else None,
            liq_stats=_liq_stats,
        )
        _liq_anomaly = _liq_result.get("anomaly", "none")
        if _liq_anomaly != "none" and _liq_result.get("confidence", 0) > 0.15:
            if not is_on_cooldown(_liq_anomaly, symbol):
                anomalies.append(_liq_anomaly)
                set_cooldown(_liq_anomaly, symbol)
                logger.info(
                    f"[Triage/Liq] anomaly={_liq_anomaly} "
                    f"conf={_liq_result['confidence']:.2f} "
                    f"z={_liq_result.get('liq_z_score')}"
                )
    except Exception as e:
        logger.error(f"Triage liquidity agent error: {e}")

    # ---------- 3. Microstructure Agent (Z Score ) ----------
    try:
        _micro_stats = None
        _micro_keys = ("imbalance_mean", "imbalance_std", "spread_mean", "spread_std")
        if all(k in _stats for k in _micro_keys):
            _micro_stats = {k: _stats[k] for k in _micro_keys}
        _micro_result = microstructure_agent.analyze(
            microstructure_context=state.get("microstructure_context", ""),
            mode=state.get("mode", "SWING"),
            micro_stats=_micro_stats,
        )
        _micro_anomaly = _micro_result.get("anomaly", "none")
        if _micro_anomaly != "none" and _micro_result.get("confidence", 0) > 0.2:
            anomalies.append(_micro_anomaly)
            logger.info(
                f"[Triage/Micro] anomaly={_micro_anomaly} "
                f"conf={_micro_result['confidence']:.2f} "
                f"imb_z={_micro_result.get('imbalance_z')} "
                f"spd_z={_micro_result.get('spread_z')}"
            )
    except Exception as e:
        logger.error(f"Triage microstructure agent error: {e}")

    # ------ 4. Macro Options Agent (Z Score DVOL/PCR ) ------
    try:
        _opts_stats = None
        _opts_keys = ("dvol_mean", "dvol_std")
        if all(k in _stats for k in _opts_keys):
            _opts_stats = {k: _stats[k] for k in ("dvol_mean", "dvol_std", "pcr_mean", "pcr_std") if k in _stats}
        _opts_result = macro_options_agent.analyze(
            deribit_context=state.get("deribit_context", ""),
            macro_context=state.get("macro_context", ""),
            mode=state.get("mode", "SWING"),
            options_stats=_opts_stats,
        )
        _opts_anomaly = _opts_result.get("anomaly", "none")
        if _opts_anomaly != "none" and _opts_result.get("confidence", 0) > 0.2:
            if not is_on_cooldown(_opts_anomaly, symbol):
                anomalies.append(_opts_anomaly)
                set_cooldown(_opts_anomaly, symbol)
                logger.info(
                    f"[Triage/Opts] anomaly={_opts_anomaly} regime={_opts_result.get('regime')} "
                    f"dvol_z={_opts_result.get('dvol_z')}"
                )
    except Exception as e:
        logger.error(f"Triage macro options agent error: {e}")

    # If is_emergency was forced via UI/Telegram command
    if state.get("is_emergency") and not anomalies:
        anomalies.append("manual_emergency_trigger")

    # ── Confluence Score (쉽알남 "근거 중첩 필수") ──────────────────────────
    # Computed here so it's available on the blackboard when Judge reads it.
    deduplicated_anomalies = list(set(anomalies))
    try:
        _cs_market = _market_data_cache.get(_cache_key(symbol, mode), {}) or {}
        _cs_funding = state.get("raw_funding", {}) or {}
        confluence_score_payload = _compute_confluence_score(
            market_data=_cs_market,
            anomalies=deduplicated_anomalies,
            raw_funding=_cs_funding,
        )
        logger.info(
            f"[Triage/Confluence] {symbol} score={confluence_score_payload['score']} "
            f"dir={confluence_score_payload['direction']} "
            f"gate={'OPEN' if confluence_score_payload['gate_passed'] else 'CLOSED'} "
            f"factors={confluence_score_payload['factors']}"
        )
    except Exception as e:
        logger.error(f"Confluence score computation error: {e}")
        confluence_score_payload = {"score": 0, "direction": "NEUTRAL", "factors": [], "gate_passed": False, "long_score": 0, "short_score": 0}

    return {
        "anomalies": deduplicated_anomalies,
        "blackboard": {"confluence_score": confluence_score_payload},
    }

def node_rule_based_chart(state: AnalysisState) -> dict:
    """Deterministic chart-rule expert (no LLM cost)."""
    symbol = state.get("symbol", "BTCUSDT")
    mode = TradingMode(state["mode"])
    cache_key = _cache_key(symbol, mode)
    market_data = _market_data_cache.get(cache_key, {}) or {}
    current_price = market_data.get("current_price")

    if not market_data or current_price is None:
        return {"blackboard": {"chart_rules": {"status": "unavailable"}}}

    scenario_engine = market_data.get("scenario_engine", {}) or {}
    active_setup = scenario_engine.get("active_setup", {}) or {}

    nearest_zone = None
    confluence_zones = market_data.get("confluence_zones", []) or []
    for zone in confluence_zones:
        price = zone.get("price")
        if not isinstance(price, (int, float)):
            continue
        dist_pct = abs(float(current_price) - float(price)) / max(float(current_price), 1e-9) * 100.0
        cand = {
            "price": round(float(price), 2),
            "strength": int(zone.get("strength", 0)),
            "level_count": int(zone.get("level_count", 0)),
            "dist_pct": round(dist_pct, 3),
            "timeframes": zone.get("timeframes", []),
        }
        if nearest_zone is None or cand["dist_pct"] < nearest_zone["dist_pct"]:
            nearest_zone = cand

    nearest_fib = None
    fib_data = market_data.get("fibonacci", {}) or {}
    for tf, fib_levels in fib_data.items():
        if not isinstance(fib_levels, dict):
            continue
        for key, val in fib_levels.items():
            if not str(key).startswith("fib_") or not isinstance(val, (int, float)):
                continue
            dist_pct = abs(float(current_price) - float(val)) / max(float(current_price), 1e-9) * 100.0
            cand = {
                "timeframe": tf,
                "level": key,
                "price": round(float(val), 2),
                "dist_pct": round(dist_pct, 3),
            }
            if nearest_fib is None or cand["dist_pct"] < nearest_fib["dist_pct"]:
                nearest_fib = cand

    structure_alerts = []
    market_struct = market_data.get("market_structure", {}) or {}
    for tf, info in market_struct.items():
        if not isinstance(info, dict):
            continue
        choch = info.get("choch")
        msb = info.get("msb")
        if isinstance(choch, dict):
            structure_alerts.append({"timeframe": tf, "type": choch.get("type"), "price": choch.get("price")})
        if isinstance(msb, dict):
            structure_alerts.append({"timeframe": tf, "type": msb.get("type"), "price": msb.get("broken_level")})

    side = str(active_setup.get("side", "")).upper()
    invalidation = active_setup.get("invalidation")
    invalidation_near = (
        isinstance(invalidation, (int, float))
        and abs(float(current_price) - float(invalidation)) / max(float(current_price), 1e-9) * 100.0 <= 0.7
    )

    chart_rules = {
        "status": "ok",
        "current_price": round(float(current_price), 2),
        "nearest_confluence": nearest_zone,
        "nearest_fibonacci": nearest_fib,
        "structure_alerts": structure_alerts,
        "scenario_engine": scenario_engine,
        "active_setup": active_setup,
        "signals": {
            "at_confluence": bool(nearest_zone and nearest_zone["dist_pct"] <= 0.7),
            "at_fibonacci": bool(nearest_fib and nearest_fib["dist_pct"] <= 0.5),
            "has_structure_alert": bool(structure_alerts),
            "scenario_ready": bool(side and active_setup.get("entry_zone_low") is not None and invalidation is not None),
            "has_trap_signal": bool((active_setup.get("trap_context", {}) or {}).get("confirmed")),
            "invalidation_near": bool(invalidation_near),
        },
    }
    return {
        "blackboard": {"chart_rules": chart_rules},
        "current_bias": str(scenario_engine.get("higher_timeframe_bias", "neutral")),
        "scenario_revision_reason": str(scenario_engine.get("scenario_revision_reason", "")),
        "active_setup": active_setup,
    }


def _should_run_vlm(state: AnalysisState) -> bool:
    """Gate expensive visual reasoning to high-value conditions only."""
    if not settings.should_use_chart:
        return False
    if getattr(settings, "use_execution_philosophy", False):
        return True
    if state.get("is_emergency"):
        return True

    anomalies = set(state.get("anomalies", []) or [])
    high_impact = {
        "true_breakout",
        "liquidation_cluster",
        "liquidation_cascade",
        "options_panic",
        "manual_emergency_trigger",
    }
    if anomalies.intersection(high_impact):
        return True

    chart_rules = (state.get("blackboard", {}) or {}).get("chart_rules", {}) or {}
    signals = chart_rules.get("signals", {}) or {}
    return bool(
        signals.get("has_structure_alert")
        or signals.get("scenario_ready")
        or signals.get("has_trap_signal")
    )

def node_vlm_geometric_expert(state: AnalysisState) -> dict:
    """Run VLM Geometric visual analysis when high-impact conditions are present."""
    if not _should_run_vlm(state):
        return {"blackboard": {"vlm_geometry": {
            "anomaly": "skipped",
            "directional_bias": "NEUTRAL",
            "confidence": 0,
            "rationale": "VLM gated off by triage/rule-based conditions"
        }}}

    chart = state.get("chart_image_b64", "")
    symbol = state.get("symbol", "BTCUSDT")
    mode = TradingMode(state["mode"])
    cache_key = _cache_key(symbol, mode)

    if not chart:
        return {"blackboard": {"vlm_geometry": {
            "anomaly": "none", "directional_bias": "NEUTRAL",
            "confidence": 0, "rationale": "No chart available"
        }}}

    # VLM result cache: skip expensive Gemini Pro call if chart hasn't changed
    _chart_hash = hashlib.sha256(chart.encode()).hexdigest()[:20]
    _cached = _vlm_result_cache.get(_chart_hash)
    if _cached is not None:
        _cached_result, _cached_ts = _cached
        if time.time() - _cached_ts < _VLM_CACHE_TTL_S:
            logger.info(f"VLM cache hit for {symbol}/{mode.value} (hash={_chart_hash})")
            return {"blackboard": {"vlm_geometry": _cached_result}, "chart_image_b64": None, "chart_bytes": None}

    # Extract current price from cached market_data for prompt context
    market_data = _market_data_cache.get(cache_key, {})
    current_price = market_data.get('current_price', None)
    primary_tf = "4H" if mode == TradingMode.SWING else "1D"
    vlm_context_text = state.get("vlm_context_text", "")

    result = vlm_geometric_agent.analyze(
        chart,
        mode=state.get("mode", "SWING").upper(),
        symbol=symbol,
        current_price=current_price,
        primary_timeframe=primary_tf,
        higher_timeframe_context=vlm_context_text,
    )
    # Store in cache (only if analysis succeeded — don't cache default/error results)
    if result.get("anomaly") not in ("none", None) or result.get("confidence", 0) > 0:
        _vlm_result_cache[_chart_hash] = (result, time.time())

    # [OOM FIX] VLM 분석 완료 후 이미지 데이터를 State에서 즉시 해제
    return {
        "blackboard": {"vlm_geometry": result},
        "chart_image_b64": None,
        "chart_bytes": None,
    }



def node_generate_chart(state: AnalysisState) -> dict:
    """Generate structure chart for all modes (for Judge VLM)."""
    symbol = state["symbol"]
    mode = TradingMode(state["mode"])
    cache_key = _cache_key(symbol, mode)
    candle_limit = settings.SWING_CANDLE_LIMIT

    # [FIX CRASH-2] Use cached DataFrame from node_collect_data
    df = _df_cache.get(cache_key)
    if df is None:
        df = db.get_latest_market_data(symbol, limit=candle_limit)  # fallback
    if df.empty:
        return {"chart_image_b64": None, "chart_bytes": None, "vlm_context_text": ""}

    # Load higher TF data from GCS
    df_4h, df_1d, df_1w = None, None, None
    try:
        from processors.gcs_parquet import gcs_parquet_store
        if gcs_parquet_store.enabled:
            m_back = settings.history_lookback_months_for_mode(mode)
            df_4h = gcs_parquet_store.load_ohlcv("4h", symbol, months_back=m_back)
            df_1d = gcs_parquet_store.load_ohlcv("1d", symbol, months_back=m_back)
            df_1w = gcs_parquet_store.load_ohlcv("1w", symbol, months_back=m_back)
    except Exception as e:
        logger.warning(f"GCS load for chart skipped: {e}")

    # [FIX RESOURCE-1] Use cached market_data from node_collect_data
    market_data = _market_data_cache.get(cache_key)
    if market_data is None:
        market_data = math_engine.analyze_market(df, mode, df_4h=df_4h, df_1d=df_1d, df_1w=df_1w)
    vlm_context_text = _build_vlm_context_text(market_data, mode)

    # CVD pipeline removed - cvd_df always None; chart_generator handles None gracefully
    cvd_df = None
    if settings.CHART_SHOW_CVD_PANEL or settings.CHART_SHOW_CVD_OVERLAY:
        try:
            cvd_df = _cvd_cache.get(cache_key)
            if cvd_df is None:
                cvd_limit = settings.data_lookback_hours * 60
                cvd_df = db.get_cvd_data(symbol, limit=cvd_limit)
                _cvd_cache[cache_key] = cvd_df
        except Exception as e:
            logger.warning(f"CVD data load for chart skipped: {e}")


    # Load liquidation data for chart markers
    liquidation_df = None
    try:
        liquidation_df = _liq_cache.get(cache_key)
        if liquidation_df is None:
            liq_limit = settings.data_lookback_hours * 60
            liquidation_df = db.get_liquidation_data(symbol, limit=liq_limit)
            _liq_cache[cache_key] = liquidation_df
    except Exception:
        pass

    # Load funding/OI data for chart sync (optional panels)
    funding_df = None
    if settings.CHART_SHOW_OI_PANEL or settings.CHART_SHOW_FUNDING_PANEL:
        try:
            funding_df = _funding_cache.get(cache_key)
            if funding_df is None:
                # Fetch funding data (includes OI, LSR) matching the lookback
                funding_limit = settings.data_lookback_hours * 60
                funding_df = db.get_funding_history(symbol, limit=funding_limit)
                if funding_df is not None and not funding_df.empty:
                    funding_df = funding_df.loc[:, ~funding_df.columns.duplicated()].reset_index(drop=True)
                
                # Merge with GCS historical funding for long-term charts
                from processors.gcs_parquet import gcs_parquet_store
                if gcs_parquet_store.enabled:
                    m_back = settings.history_lookback_months_for_mode(mode)
                    hist_fnd = gcs_parquet_store.load_timeseries("funding", symbol, months_back=m_back)
                    if not hist_fnd.empty:
                        hist_fnd = hist_fnd.loc[:, ~hist_fnd.columns.duplicated()].reset_index(drop=True)
                        hist_fnd = hist_fnd.rename(columns={
                            'open_interest_value': 'open_interest'
                        })
                        hist_fnd['timestamp'] = pd.to_datetime(hist_fnd['timestamp'].astype(str), format='mixed', utc=True, errors='coerce').bfill()
                        since_fnd = hist_fnd['timestamp'].max()
                        bridge_fnd = db.get_funding_history(
                            symbol,
                            limit=int(getattr(settings, "ORCHESTRATOR_FUNDING_BRIDGE_LIMIT", 5000)),
                            since=since_fnd,
                            columns="timestamp,funding_rate,open_interest,open_interest_value,oi_binance,oi_bybit,oi_okx,basis_pct,long_short_ratio",
                        )
                        
                        dfs = [hist_fnd]
                        if bridge_fnd is not None and not bridge_fnd.empty:
                            dfs.append(bridge_fnd)
                        if funding_df is not None and not funding_df.empty:
                            dfs.append(funding_df)
                        
                        funding_df = pd.concat(dfs).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                    else:
                        funding_df = hist_fnd
                _funding_cache[cache_key] = funding_df
        except Exception as e:
            logger.warning(f"Funding/OI data load for chart skipped/merged: {e}")


    try:
        fixed_timeframe = "4h" if mode == TradingMode.SWING else "1d"
        chart_bytes = chart_generator.generate_chart(
            df, market_data, symbol,
            mode=mode,
            timeframe=fixed_timeframe,
            liquidation_df=liquidation_df,
            cvd_df=cvd_df,
            funding_df=funding_df,
            df_4h=df_4h,
            df_1d=df_1d,
            df_1w=df_1w,
            prefer_lane=False,
        )
    except Exception as e:
        logger.error(f'Chart generation FAILED: {e}')
        chart_bytes = None

    # [OOM FIX] chart 생성 직후 GCS historical DataFrames 즉시 해제 (수백MB)
    try:
        del df_4h, df_1d, df_1w
        gc.collect()
    except Exception:
        pass

    if not chart_bytes:
        logger.warning(f"Chart generation FAILED for {symbol} ({mode.value})")
        return {"chart_image_b64": None, "chart_bytes": None, "vlm_context_text": vlm_context_text}

    chart_image_b64 = None
    if chart_bytes:
        chart_bytes_for_vlm = chart_generator.resize_for_low_res(chart_bytes)
        chart_image_b64 = chart_generator.chart_to_base64(chart_bytes_for_vlm)
        logger.info(f"Generated chart for {symbol}: {len(chart_bytes)} bytes. VLM B64 size: {len(chart_image_b64)}")

    return {"chart_image_b64": chart_image_b64, "chart_bytes": chart_bytes, "vlm_context_text": vlm_context_text}


def node_judge_agent(state: AnalysisState) -> dict:
    """Run Judge (Claude Opus 4.6). Reads Blackboard."""
    mode = TradingMode(state["mode"])
    symbol = state.get("symbol", "BTCUSDT")
    cache_key = _cache_key(symbol, mode)
    
    # Extract necessary state variables
    compact = state.get("market_data_compact", "")
    blackboard = state.get("blackboard", {})
    funding_context = state.get("funding_context", "")
    feedback_text = state.get("feedback_text", "")
    open_positions = state.get("open_positions", "")
    # unified_narrative is pre-deduplicated (narrative_text + rag_context + telegram_news).
    # Fall back to raw join if node_unify_narrative was skipped (snapshot/hot-path calls).
    narrative_context = state.get("unified_narrative") or "\n\n".join(filter(None, [
        state.get("narrative_text", ""),
        state.get("rag_context", ""),
    ]))

    # Pass blackboard (includes synthesis + vlm_geometry) and regime context.
    # Chart image is NOT forwarded - VLMGeometricAgent is the sole visual analyst.
    # Judge reads VLM's structured text output from the blackboard instead.
    regime_ctx = state.get("regime_context", {})
    decision = judge_agent.make_decision(
        market_data_compact=compact,
        blackboard=blackboard,
        funding_context=funding_context,
        chart_image_b64=None,
        mode=mode,
        feedback_text=feedback_text,
        active_orders=state.get("active_orders", []),
        open_positions=open_positions,
        symbol=symbol,
        regime_context=regime_ctx,
        narrative_context=narrative_context,
        onchain_context=state.get("onchain_context", ""),
    )
    
    # Deterministic EV/RR gating (no additional LLM call)
    def _to_float(v, default=0.0):
        try:
            return float(v)
        except Exception:
            return default

    def _infer_direction_from_context(bb: dict, regime: dict) -> str:
        score = 0
        for k in ("macro", "liquidity", "microstructure", "vlm_geometry", "chart_rules"):
            node = bb.get(k, {}) if isinstance(bb, dict) else {}
            d = str(node.get("decision", "")).upper()
            if d == "LONG":
                score += 1
            elif d == "SHORT":
                score -= 1

        if score > 0:
            return "LONG"
        if score < 0:
            return "SHORT"

        regime_name = str((regime or {}).get("regime", "")).upper()
        if "BULL" in regime_name:
            return "LONG"
        if "BEAR" in regime_name:
            return "SHORT"
        return "HOLD"

    if not isinstance(decision.get("reasoning"), dict):
        decision["reasoning"] = {"final_logic": str(decision.get("reasoning", ""))}
    if "final_logic" not in decision["reasoning"]:
        decision["reasoning"]["final_logic"] = ""
    if not isinstance(decision.get("key_factors"), list):
        decision["key_factors"] = []

    win_prob_pct = max(0.0, _to_float(decision.get("win_probability_pct", 0.0)))
    win_prob = win_prob_pct / 100.0
    profit_pct = max(0.0, _to_float(decision.get("expected_profit_pct", 0.0)))
    loss_pct = max(0.0, _to_float(decision.get("expected_loss_pct", 0.0)))
    rr = (profit_pct / loss_pct) if loss_pct > 0 else (999.0 if profit_pct > 0 else 0.0)
    ev = (win_prob * profit_pct) - ((1.0 - win_prob) * loss_pct)

    min_win = float(getattr(settings, "JUDGE_MIN_WIN_PROB_PCT", 52.0))
    min_rr = float(getattr(settings, "JUDGE_MIN_RR_FOR_ENTRY", 1.35))
    min_ev = float(getattr(settings, "JUDGE_MIN_EV_FOR_ENTRY_PCT", 0.20))
    hold_override = bool(getattr(settings, "JUDGE_ENABLE_HOLD_OVERRIDE", True))
    direction = str(decision.get("decision", "HOLD")).upper()

    # Veto low-quality entries
    if direction in ("LONG", "SHORT"):
        failed = []
        if ev <= 0:
            failed.append(f"EV={ev:.2f}<=0")
        if win_prob_pct < min_win:
            failed.append(f"WinProb={win_prob_pct:.1f}%<{min_win:.1f}%")
        if rr < min_rr:
            failed.append(f"RR={rr:.2f}<{min_rr:.2f}")
        if failed:
            logger.warning(f"Judge entry vetoed: {', '.join(failed)}")
            decision["decision"] = "HOLD"
            decision["allocation_pct"] = 0
            decision["leverage"] = 1
            decision["reasoning"]["final_logic"] = (
                f"[ENTRY VETO] {'; '.join(failed)}. " + decision["reasoning"].get("final_logic", "")
            )
            decision["key_factors"].append("EV/RR/리스크 필터 미통과로 진입 거")

    # Promote HOLD to small-size entry only when edge is mathematically strong
    direction = str(decision.get("decision", "HOLD")).upper()
    if direction == "HOLD" and hold_override and ev >= min_ev and win_prob_pct >= min_win and rr >= min_rr:
        inferred = _infer_direction_from_context(blackboard, regime_ctx)
        if inferred in ("LONG", "SHORT"):
            decision["decision"] = inferred
            decision["allocation_pct"] = float(getattr(settings, "JUDGE_OVERRIDE_ALLOC_SWING_PCT", 8.0))
            decision["leverage"] = max(1, int(_to_float(decision.get("leverage", 1), 1)))
            current_price = _to_float((_market_data_cache.get(cache_key, {}) or {}).get("current_price"), 0.0)
            if current_price > 0:
                decision["entry_price"] = _to_float(decision.get("entry_price"), current_price) or current_price
                ep = float(decision["entry_price"])
                safe_loss = max(loss_pct, 1.0)
                safe_profit = max(profit_pct, 1.5)
                if inferred == "LONG":
                    decision["stop_loss"] = _to_float(decision.get("stop_loss"), ep * (1 - safe_loss / 100.0))
                    decision["take_profit"] = _to_float(decision.get("take_profit"), ep * (1 + safe_profit / 100.0))
                else:
                    decision["stop_loss"] = _to_float(decision.get("stop_loss"), ep * (1 + safe_loss / 100.0))
                    decision["take_profit"] = _to_float(decision.get("take_profit"), ep * (1 - safe_profit / 100.0))
            decision["reasoning"]["final_logic"] = (
                f"[HOLD OVERRIDE] EV={ev:.2f}, WinProb={win_prob_pct:.1f}%, RR={rr:.2f} >= gates. "
                + decision["reasoning"].get("final_logic", "")
            )
            decision["key_factors"].append("수학적 우위(EV/리스크/RR) 불충분으로 소규모 진입 거")

    # Scale allocation by Meta risk budget
    risk_budget = _to_float(regime_ctx.get("risk_budget_pct", 100), 100.0)
    original_alloc = _to_float(decision.get("allocation_pct", 0), 0.0)
    scaled_alloc = original_alloc * (risk_budget / 100.0)
    if str(decision.get("decision", "HOLD")).upper() in ("LONG", "SHORT"):
        decision["allocation_pct"] = scaled_alloc
        logger.info(f"Scaled allocation from {original_alloc}% to {scaled_alloc:.2f}% (Risk Budget: {risk_budget}%)")
        decision["reasoning"]["final_logic"] = (
            f"[SCALING] Allocation constrained by {risk_budget}% Risk Budget. EV={ev:.2f}, RR={rr:.2f}. "
            + decision["reasoning"].get("final_logic", "")
        )

    # For compatibility with legacy metrics/logging, set conviction to win_probability_pct
    dual_plan = decision.get("daily_dual_plan", {}) if isinstance(decision, dict) else {}
    if isinstance(decision, dict):
        decision["daily_dual_plan"] = dual_plan if isinstance(dual_plan, dict) else {}
        decision["confidence"] = win_prob_pct
    return {
        "final_decision": decision,
        "daily_dual_plan": dual_plan if isinstance(dual_plan, dict) else {},
        "conviction_score": win_prob * 100
    }


def node_risk_manager(state: AnalysisState) -> dict:
    """Run Risk Manager (CRO). VETO power over Judge's draft."""
    draft = state.get("final_decision", {})
    symbol = state.get("symbol", "BTCUSDT")
    mode = TradingMode(state.get("mode", "swing"))
    cache_key = _cache_key(symbol, mode)
    market_data = _market_data_cache.get(cache_key, {}) or {}
    scenario_context = market_data.get("scenario_engine", {}) or {}
    
    # Needs macro and funding context for risk overlays
    funding = state.get("funding_context", "")
    deribit = state.get("deribit_context", "")
    
    final_decision = risk_manager_agent.evaluate_trade(
        draft,
        funding,
        deribit,
        mode=mode,
        onchain_context=state.get("onchain_context", ""),
        onchain_gate=state.get("onchain_gate", {}),
        scenario_context=scenario_context,
    )
    cvd_df = _cvd_cache.get(cache_key)
    if cvd_df is None:
        try:
            cvd_df = db.get_cvd_data(symbol, limit=60)
        except Exception:
            cvd_df = None

    liq_df = _liq_cache.get(cache_key)
    if liq_df is None:
        try:
            liq_df = db.get_liquidation_data(symbol, limit=60)
        except Exception:
            liq_df = None

    final_decision = policy_engine.enforce(
        decision=final_decision,
        market_data=market_data,
        mode=mode,
        raw_funding=state.get("raw_funding", {}),
        cvd_df=cvd_df,
        liq_df=liq_df,
        playbook_context=state.get("playbook_context", {}),
    )
    return {
        "final_decision": final_decision,
        "policy_snapshot": final_decision.get("policy_checks", {}),
    }

def node_portfolio_leverage_guard(state: AnalysisState) -> dict:
    """Mathematically enforce 2.0x total account leverage limit."""
    decision = state.get("final_decision", {})
    if decision.get("decision") not in ["LONG", "SHORT"]:
        return {}

    try:
        from executors.paper_exchange import paper_engine
        
        # 1. Fetch total equity and exposure (Paper mode for now as per strategy)
        target_exchange = decision.get("target_exchange", "BINANCE").lower()
        if target_exchange == "split":
            wallet_balance = (
                paper_engine.get_wallet_balance("binance_spot")
                + paper_engine.get_wallet_balance("upbit")
            )
        else:
            wallet_balance = paper_engine.get_wallet_balance(target_exchange)
        
        # In a real setup, we'd query API for open positions. 
        # For our specific retail guard, we calculate if THIS trade + current positions exceeds 2.0x.
        # [MOCK FOR MVP] Assume 2.0x Hard Cap
        MAX_LEVERAGE_CAP = 2.0
        
        # Calculate proposed trade notional
        allocation_pct = decision.get("allocation_pct", 0)
        leverage = decision.get("leverage", 1)
        proposed_notional = wallet_balance * (allocation_pct / 100.0) * leverage
        
        # Check current total exposure (Simulated calculation)
        # In production: exposure = sum(abs(pos.notional) for pos in all_positions)
        current_exposure = 0.0 # Placeholder: build extraction logic if needed
        
        total_projected_exposure = current_exposure + proposed_notional
        total_leverage = total_projected_exposure / wallet_balance if wallet_balance > 0 else 0
        
        if total_leverage > MAX_LEVERAGE_CAP:
            # Scale down
            reduction_factor = MAX_LEVERAGE_CAP / total_leverage
            new_allocation = allocation_pct * reduction_factor
            
            logger.warning(f"[LEVERAGE GUARD] Projected leverage {total_leverage:.2f}x exceeds {MAX_LEVERAGE_CAP}x cap. Scaling allocation from {allocation_pct}% to {new_allocation:.2f}%")
            
            decision["allocation_pct"] = new_allocation
            decision["reasoning"]["final_logic"] = f"[LEVERAGE GUARD] Scaled down from {allocation_pct}% due to 2.0x account leverage cap. " + decision["reasoning"].get("final_logic", "")
            
        return {"final_decision": decision}
        
    except Exception as e:
        logger.error(f"Leverage Guard error: {e}")
        return {}

def node_execute_trade(state: AnalysisState) -> dict:
    """Execute the final approved trade autonomously (Live or Paper)."""
    decision = state.get("final_decision", {})
    symbol = state["symbol"]
    mode = TradingMode(state["mode"]).value.upper()
    if state.get("playbook_context"):
        decision["playbook_context"] = state.get("playbook_context", {})
    if state.get("emergency_context"):
        decision["emergency_context"] = state.get("emergency_context", {})

    # Optional guard: skip execution when explicitly disabled by caller.
    if not state.get("execute_trades", True):
        decision["execution_receipt"] = {
            "status": "SKIPPED",
            "reason": "execute_trades=False",
        }
        return {"final_decision": decision}
    
    # Bridge to execution
    # This will simulate DCA or single entries based on the mode and CRO sizing
    execution_result = trade_executor.execute_from_decision(
        decision,
        mode,
        symbol,
        playbook_context=state.get("playbook_context", {}),
    )
    
    # Attach execution receipt back to the decision for reporting
    decision["execution_receipt"] = execution_result
    return {"final_decision": decision}


def _safe_float(value) -> Optional[float]:
    try:
        if value in (None, "", "N/A"):
            return None
        return float(value)
    except Exception:
        return None


def _compute_consensus_rate(blackboard: dict, direction: str) -> Optional[float]:
    if not isinstance(blackboard, dict):
        return None
    direction = str(direction or "").upper()
    votes = []
    for key in ("market", "onchain", "macro"):
        payload = blackboard.get(key)
        if isinstance(payload, dict):
            votes.append(str(payload.get("decision", "HOLD")).upper())
    if not votes:
        return None
    agreed = sum(1 for vote in votes if direction and direction in vote)
    return round((agreed / len(votes)) * 100.0, 2)


def _build_evaluation_prediction_payload(state: AnalysisState, report: Dict, mode: TradingMode) -> Dict:
    decision = state.get("final_decision", {}) if isinstance(state.get("final_decision", {}), dict) else {}
    blackboard = state.get("blackboard", {}) if isinstance(state.get("blackboard", {}), dict) else {}
    report_id = report.get("report_id") or report.get("id")
    prediction_time = report.get("created_at") or report.get("timestamp") or datetime.now(timezone.utc).isoformat()
    anomalies = state.get("anomalies", [])
    if not isinstance(anomalies, list):
        anomalies = []

    return {
        "source_type": "ai_report",
        "source_id": report_id,
        "ai_report_id": report_id,
        "prediction_time": prediction_time,
        "symbol": state["symbol"],
        "mode": mode.value,
        "decision": str(decision.get("decision", "HOLD")).upper(),
        "prediction_label": str(decision.get("decision", "HOLD")).upper(),
        "confidence": _safe_float(decision.get("confidence")),
        "entry_price": _safe_float(decision.get("entry_price")),
        "take_profit": _safe_float(decision.get("take_profit")),
        "stop_loss": _safe_float(decision.get("stop_loss")),
        "regime": state.get("market_regime"),
        "model_version": str(getattr(settings, "MODEL_JUDGE", "")),
        "prompt_version": str(getattr(settings, "PROMPT_VERSION", "judge_default_v1")),
        "rag_version": str(getattr(settings, "RAG_VERSION", "lightrag_v1")),
        "strategy_version": str(getattr(settings, "STRATEGY_VERSION", "orchestrator_v8")),
        "consensus_rate": _compute_consensus_rate(blackboard, decision.get("decision", "HOLD")),
        "anomalies_detected": anomalies,
        "input_context": {
            "blackboard_agents": sorted(list(blackboard.keys()))[:12],
            "rag_chars": len(str(state.get("rag_context", "") or "")),
            "telegram_news_chars": len(str(state.get("telegram_news", "") or "")),
            "narrative_chars": len(str(state.get("narrative_text", "") or "")),
            "feedback_chars": len(str(state.get("feedback_text", "") or "")),
            "onchain_context_chars": len(str(state.get("onchain_context", "") or "")),
        },
        "metadata": {
            "models": {
                "judge": getattr(settings, "MODEL_JUDGE", ""),
                "meta_regime": getattr(settings, "MODEL_META_REGIME", ""),
                "risk_eval": getattr(settings, "MODEL_RISK_EVAL", ""),
                "rag_extraction": getattr(settings, "MODEL_RAG_EXTRACTION", ""),
                "vlm_geometric": getattr(settings, "MODEL_VLM_GEOMETRIC", ""),
            },
            "market_regime": state.get("market_regime"),
            "risk_budget_pct": (state.get("regime_context", {}) or {}).get("risk_budget_pct"),
            "report_created_at": report.get("created_at"),
            "policy_status": (state.get("policy_snapshot", {}) or {}).get("status"),
            "policy_rr": (state.get("policy_snapshot", {}) or {}).get("rr"),
            "policy_stop_basis": (state.get("policy_snapshot", {}) or {}).get("stop_basis"),
        },
    }


def _build_component_score_rows(prediction_id: int, state: AnalysisState) -> list[dict]:
    decision = state.get("final_decision", {}) if isinstance(state.get("final_decision", {}), dict) else {}
    blackboard = state.get("blackboard", {}) if isinstance(state.get("blackboard", {}), dict) else {}

    def _score(component_type: str, metric_name: str, metric_value, metric_label: str):
        value = _safe_float(metric_value)
        if value is None:
            return None
        return {
            "prediction_id": prediction_id,
            "component_type": component_type,
            "metric_name": metric_name,
            "metric_value": value,
            "metric_label": metric_label,
            "scope_key": "",
            "metadata": {},
        }

    win_prob_pct = _safe_float(decision.get("win_probability_pct"))
    expected_profit_pct = _safe_float(decision.get("expected_profit_pct"))
    expected_loss_pct = _safe_float(decision.get("expected_loss_pct"))
    rr_ratio = None
    ev_pct = None
    if expected_profit_pct is not None and expected_loss_pct not in (None, 0):
        rr_ratio = expected_profit_pct / expected_loss_pct
    if win_prob_pct is not None and expected_profit_pct is not None and expected_loss_pct is not None:
        win_prob = win_prob_pct / 100.0
        ev_pct = (win_prob * expected_profit_pct) - ((1.0 - win_prob) * expected_loss_pct)

    rows = [
        _score("system", "consensus_rate_pct", _compute_consensus_rate(blackboard, decision.get("decision", "HOLD")), "Consensus Rate (%)"),
        _score("system", "anomaly_count", len(state.get("anomalies", []) or []), "Anomaly Count"),
        _score("system", "blackboard_agent_count", len(blackboard), "Blackboard Agent Count"),
        _score("llm", "confidence_pct", decision.get("confidence"), "Decision Confidence (%)"),
        _score("llm", "win_probability_pct", win_prob_pct, "Win Probability (%)"),
        _score("llm", "expected_profit_pct", expected_profit_pct, "Expected Profit (%)"),
        _score("llm", "expected_loss_pct", expected_loss_pct, "Expected Loss (%)"),
        _score("llm", "expected_value_pct", ev_pct, "Expected Value (%)"),
        _score("llm", "rr_ratio", rr_ratio, "Risk Reward Ratio"),
        _score("rag", "rag_context_chars", len(str(state.get("rag_context", "") or "")), "RAG Context Length"),
        _score("rag", "telegram_news_chars", len(str(state.get("telegram_news", "") or "")), "Telegram News Length"),
        _score("rag", "narrative_chars", len(str(state.get("narrative_text", "") or "")), "Narrative Length"),
        _score("rag", "feedback_chars", len(str(state.get("feedback_text", "") or "")), "Feedback Context Length"),
        _score("rag", "onchain_context_chars", len(str(state.get("onchain_context", "") or "")), "On-chain Context Length"),
    ]
    return [row for row in rows if row]

def node_generate_report(state: AnalysisState) -> dict:
    """Generate and send report to Telegram."""
    from executors.metrics_logger import metrics_logger
    symbol = state["symbol"]
    mode = TradingMode(state["mode"])
    cache_key = _cache_key(symbol, mode)

    candle_limit = settings.SWING_CANDLE_LIMIT
    # [FIX CRASH-2] Use cached DataFrame from node_collect_data
    df = _df_cache.get(cache_key)
    if df is None:
        df = db.get_latest_market_data(symbol, limit=candle_limit)  # fallback
    # [FIX RESOURCE-1] Use cached market_data
    market_data = _market_data_cache.get(cache_key)
    if market_data is None:
        market_data = math_engine.analyze_market(df, mode) if not df.empty else {}

    # [FIX] Reuse raw_funding from node_funding_context (no duplicate Supabase query)
    raw_funding = state.get("raw_funding", {})
    if not raw_funding:
        try:
            response = db.client.table("funding_data")\
                .select("*").eq("symbol", symbol)\
                .order("timestamp", desc=True).limit(1).execute()
            raw_funding = response.data[0] if response.data else {}
        except Exception:
            raw_funding = {}

    bb_str = json.dumps(state.get("blackboard", {}), indent=2)
    report = report_generator.generate_report(
        symbol=symbol,
        market_data=market_data,
        bull_opinion=f"Blackboard:\n{bb_str}",
        bear_opinion=f"Anomalies Detected: {state.get('anomalies', [])}",
        risk_assessment="",
        final_decision=state.get("final_decision", {}),
        funding_data=raw_funding,
        mode=mode,
        onchain_context=state.get("onchain_context", ""),
        onchain_snapshot=state.get("onchain_snapshot", {}),
    )
    
    # [FIX CRITICAL-1] Pass actual last close price (was 0.0 - all episodic memory inverted)
    # [FIX MEDIUM-19]  Include blackboard so post-mortem LLM has decision context
    last_close = 0.0
    if not df.empty:
        last_close = float(df.iloc[-1]['close'])
    write_post_mortem({
        "symbol": symbol,
        "final_decision": state.get("final_decision", {}),
        "blackboard": state.get("blackboard", {}),
        "bear_opinion": f"Anomalies: {state.get('anomalies', [])}"
    }, current_price=last_close)

    if report:
        try:
            prediction_row = db.upsert_evaluation_prediction(
                _build_evaluation_prediction_payload(state, report, mode)
            )
            if prediction_row and prediction_row.get("id") is not None:
                component_rows = _build_component_score_rows(int(prediction_row["id"]), state)
                if component_rows:
                    db.batch_upsert_evaluation_component_scores(component_rows)
        except Exception as e:
            logger.error(f"Failed to persist evaluation prediction for {symbol}: {e}")

        chart_bytes = state.get("chart_bytes")
        if chart_bytes:
            logger.info(f"Passing {len(chart_bytes)} bytes of chart data to notify.")
        else:
            logger.warning("No chart data found in state at generate_report node.")
        report_generator.notify(
            report,
            chart_bytes=chart_bytes,
            mode=mode,
            notification_context=str(state.get("notification_context", "analysis") or "analysis"),
        )
        
        # Log prediction for academic/quantitative evaluation
        metrics_logger.log_prediction(
            symbol=symbol,
            mode=mode.value,
            final_decision=state.get("final_decision", {}),
            blackboard=state.get("blackboard", {}),
            anomalies=state.get("anomalies", []),
            report_id=report.get("report_id"),
        )

    decision = state.get("final_decision", {}).get("decision", "N/A")
    logger.info(f"Analysis completed for {symbol}: {decision} ({mode.value})")
    return {"report": report}


def node_data_synthesis(state: AnalysisState) -> dict:
    """Run Data-Synthesis pipeline to extract high-quality training examples."""
    report = state.get("report")
    if report:
        # Inject additional context for synthesis
        report["market_regime"] = state.get("market_regime", "UNKNOWN")
        report["blackboard"] = state.get("blackboard", {})
        report["mode"] = state.get("mode", "SWING")
        synthesize_training_data(report)
    return {}


# -  - Helper - 

def _build_full_context(state: AnalysisState) -> str:
    """Build full context string for agents."""
    parts = [
        state.get("narrative_text", ""),
        state.get("liquidation_context", ""),  # cvd_context removed (OI_DIV in funding_context)
        state.get("microstructure_context", ""),
        state.get("macro_context", ""),
        state.get("deribit_context", ""),      # DVOL / PCR / IV term / 25d skew
        state.get("fear_greed_context", ""),   # F&G daily sentiment
        state.get("onchain_context", ""),      # Coin Metrics daily regime overlay
        state.get("rag_context", ""),
        f"Telegram News:\n{state.get('telegram_news', '')}",
        state.get("feedback_text", ""),
    ]
    return "\n\n".join(p for p in parts if p)

# -  - Build the LangGraph StateGraph - 



# --------------- Daily Playbook Generation ---------------

def node_generate_playbook(state) -> dict:
    """Persist Daily Playbook(s) from Judge output without extra LLM calls."""
    import datetime as _dt

    symbol = state["symbol"]
    mode_hint = str(state.get("mode", "position")).lower()
    final_decision = state.get("final_decision", {})
    dual_plan = state.get("daily_dual_plan", {})
    if not isinstance(dual_plan, dict):
        dual_plan = final_decision.get("daily_dual_plan", {}) if isinstance(final_decision, dict) else {}
    if not isinstance(dual_plan, dict):
        dual_plan = {}

    def _normalize_playbook(playbook: dict, source_decision: str, allocation_pct: object) -> dict:
        if not isinstance(playbook, dict):
            playbook = {}
        entry = playbook.get("entry_conditions", [])
        invalid = playbook.get("invalidation_conditions", [])
        allowed_sides = playbook.get("allowed_sides", [])
        if not isinstance(allowed_sides, list):
            allowed_sides = []
        source_side = str(source_decision or "").upper()
        if not allowed_sides and source_side in ("LONG", "SHORT"):
            allowed_sides = [source_side]
        fallback_allocation = None
        try:
            parsed_allocation = float(allocation_pct)
            if source_side in ("LONG", "SHORT") and parsed_allocation > 0:
                fallback_allocation = parsed_allocation
        except Exception:
            fallback_allocation = None
        try:
            if playbook.get("max_allocation_pct") is not None:
                max_allocation_pct = float(playbook.get("max_allocation_pct"))
            else:
                max_allocation_pct = fallback_allocation
        except Exception:
            max_allocation_pct = None
        return {
            "entry_conditions": entry if isinstance(entry, list) else [],
            "invalidation_conditions": invalid if isinstance(invalid, list) else [],
            "allowed_sides": allowed_sides,
            "max_allocation_pct": max_allocation_pct,
            "strategy_version": str(playbook.get("strategy_version") or "daily_playbook_v1"),
        }

    def _has_meaningful_conditions(playbook: dict) -> bool:
        """Require at least one entry condition to avoid persisting empty/invalid playbooks."""
        entry = playbook.get("entry_conditions", []) if isinstance(playbook, dict) else []
        return isinstance(entry, list) and len(entry) > 0

    now_iso = _dt.datetime.now(_dt.timezone.utc).isoformat()
    records = []

    swing_plan = dual_plan.get("swing_plan")
    position_plan = dual_plan.get("position_plan")
    if isinstance(swing_plan, dict) or isinstance(position_plan, dict):
        for lane_mode, raw_plan in [("swing", swing_plan)]:
            if isinstance(raw_plan, dict):
                if "monitoring_playbook" in raw_plan and isinstance(raw_plan["monitoring_playbook"], dict):
                    raw_plan = raw_plan["monitoring_playbook"]
                normalized = _normalize_playbook(
                    raw_plan,
                    str(final_decision.get("decision", "HOLD")),
                    final_decision.get("allocation_pct"),
                )
                if not _has_meaningful_conditions(normalized):
                    logger.warning(
                        f"node_generate_playbook: skip empty playbook for {symbol}/{lane_mode} "
                        "(entry_conditions is empty)"
                    )
                    continue
                records.append({
                    "symbol": symbol,
                    "mode": lane_mode,
                    "playbook": normalized,
                    "created_at": now_iso,
                    "ttl_hours": 24,
                    "source_decision": str(final_decision.get("decision", "HOLD")),
                })
    else:
        fallback = final_decision.get("monitoring_playbook", {}) if isinstance(final_decision, dict) else {}
        if isinstance(fallback, dict) and "monitoring_playbook" in fallback and isinstance(fallback["monitoring_playbook"], dict):
            fallback = fallback["monitoring_playbook"]
        if not isinstance(fallback, dict) or not fallback:
            logger.warning(f"node_generate_playbook: no dual plan or monitoring_playbook for {symbol}; skip save")
            return {}
        normalized = _normalize_playbook(
            fallback,
            str(final_decision.get("decision", "HOLD")),
            final_decision.get("allocation_pct"),
        )
        if not _has_meaningful_conditions(normalized):
            logger.warning(
                f"node_generate_playbook: skip empty fallback playbook for {symbol}/{mode_hint} "
                "(entry_conditions is empty)"
            )
            return {}
        records.append({
            "symbol": symbol,
            "mode": mode_hint if mode_hint in ("swing", "position") else "position",
            "playbook": normalized,
            "created_at": now_iso,
            "ttl_hours": 24,
            "source_decision": str(final_decision.get("decision", "HOLD")),
        })

    if not records:
        logger.warning(f"node_generate_playbook: no valid playbook records to save for {symbol}")
        return {}

    for rec in records:
        try:
            db.client.table("daily_playbooks").upsert(
                rec, on_conflict="symbol,mode"
            ).execute()
            logger.info(f"Daily Playbook saved: {rec['symbol']}/{rec['mode']}")
        except Exception as db_err:
            logger.warning(f"Playbook DB upsert failed ({rec['symbol']}/{rec['mode']}): {db_err}")
    return {}

def build_analysis_graph():
    """Build the multi-agent analysis graph with parallel execution."""
    graph = StateGraph(AnalysisState)

    # Add nodes
    graph.add_node("collect_data", node_collect_data)
    graph.add_node("context_gathering", node_context_gathering)
    graph.add_node("unify_narrative", node_unify_narrative)
    graph.add_node("meta_agent", node_meta_agent)
    
    # Core decision path
    graph.add_node("triage", node_triage)
    graph.add_node("generate_chart", node_generate_chart)
    graph.add_node("rule_based_chart", node_rule_based_chart)
    graph.add_node("vlm_expert", node_vlm_geometric_expert)
    graph.add_node("judge_agent", node_judge_agent)
    graph.add_node("risk_manager", node_risk_manager)
    graph.add_node("portfolio_leverage_guard", node_portfolio_leverage_guard)
    graph.add_node("execute_trade", node_execute_trade)
    graph.add_node("generate_report", node_generate_report)
    graph.add_node("data_synthesis", node_data_synthesis)

    graph.set_entry_point("collect_data")

    # Sequential preprocessing
    graph.add_edge("collect_data", "context_gathering")
    graph.add_edge("context_gathering", "unify_narrative")
    graph.add_edge("unify_narrative", "meta_agent")
    graph.add_edge("meta_agent", "triage")

    graph.add_edge("triage", "generate_chart")
    graph.add_edge("generate_chart", "rule_based_chart")

    # VLM node exists in the graph but internally decides whether to run
    # based on triage + deterministic chart rules.
    # IMPORTANT: Keep a single upstream path to avoid double-triggering
    # downstream judge/risk nodes and concurrent writes to final_decision.
    graph.add_edge("rule_based_chart", "vlm_expert")

    # Sequential finalization
    graph.add_edge("vlm_expert", "judge_agent")
    graph.add_edge("judge_agent", "risk_manager")
    graph.add_edge("risk_manager", "portfolio_leverage_guard")
    graph.add_edge("portfolio_leverage_guard", "execute_trade")
    graph.add_edge("execute_trade", "generate_report")
    graph.add_edge("generate_report", "data_synthesis")
    graph.add_edge("data_synthesis", END)

    return graph.compile()



#  Orchestrator class (maintains backward compatibility) 

class Orchestrator:
    _MAX_DAILY_PRECISION_SECONDS = 7200  # 2시간 초과 시 자동 해제

    def __init__(self):
        self.symbols = settings.trading_symbols
        self._graph = None
        self._analysis_locks: Dict[str, threading.Lock] = {}
        self._daily_precision_active_count = 0
        self._daily_precision_state_lock = threading.Lock()
        self._daily_precision_started_at: Optional[datetime] = None
        ensure_registered()

    @property
    def mode(self) -> TradingMode:
        return settings.trading_mode

    @property
    def graph(self):
        """Lazy-init the LangGraph compiled graph."""
        if self._graph is None and LANGGRAPH_AVAILABLE:
            try:
                self._graph = build_analysis_graph()
                logger.info("LangGraph analysis pipeline compiled")
            except Exception as e:
                logger.error(f"LangGraph build error: {e}")
        return self._graph

    def is_daily_precision_running(self) -> bool:
        with self._daily_precision_state_lock:
            if self._daily_precision_active_count <= 0:
                return False
            # 2시간 초과 시 stuck 간주하고 강제 해제
            if self._daily_precision_started_at is not None:
                elapsed = (datetime.now(timezone.utc) - self._daily_precision_started_at).total_seconds()
                if elapsed > self._MAX_DAILY_PRECISION_SECONDS:
                    logger.warning(
                        f"Daily precision stuck detected ({elapsed:.0f}s elapsed). Force-resetting count."
                    )
                    self._daily_precision_active_count = 0
                    self._daily_precision_started_at = None
                    return False
            return True

    def _enter_daily_precision_run(self) -> None:
        with self._daily_precision_state_lock:
            self._daily_precision_active_count += 1
            if self._daily_precision_started_at is None:
                self._daily_precision_started_at = datetime.now(timezone.utc)

    def _exit_daily_precision_run(self) -> None:
        with self._daily_precision_state_lock:
            self._daily_precision_active_count = max(0, self._daily_precision_active_count - 1)
            if self._daily_precision_active_count == 0:
                self._daily_precision_started_at = None

    @staticmethod
    def _daily_precision_summary_key() -> str:
        return "daily_precision_last_summary"

    def get_last_daily_precision_summary(self) -> Dict:
        default = {"updated_at": "", "symbols": {}}
        try:
            raw = state_manager.get_system_config(self._daily_precision_summary_key(), "")
            if not raw:
                return dict(default)
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                return dict(default)
            symbols = payload.get("symbols")
            if isinstance(symbols, dict):
                return {
                    "updated_at": str(payload.get("updated_at", "") or ""),
                    "symbols": symbols,
                }
            results = payload.get("results")
            if isinstance(results, list):
                converted = {}
                for item in results:
                    if isinstance(item, dict) and item.get("symbol"):
                        converted[str(item["symbol"])] = item
                return {
                    "updated_at": str(payload.get("finished_at", "") or payload.get("updated_at", "") or ""),
                    "symbols": converted,
                }
        except Exception as e:
            logger.warning(f"Failed to load daily precision summary: {e}")
        return dict(default)

    def _persist_daily_precision_summary(self, summary: Dict) -> None:
        try:
            state_manager.set_system_config(
                self._daily_precision_summary_key(),
                json.dumps(summary, ensure_ascii=False),
            )
        except Exception as e:
            logger.warning(f"Failed to persist daily precision summary: {e}")

    def _update_daily_precision_summary(self, result: Dict) -> Dict:
        summary = self.get_last_daily_precision_summary()
        symbols = summary.setdefault("symbols", {})
        symbols[str(result.get("symbol", "") or "")] = dict(result or {})
        summary["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._persist_daily_precision_summary(summary)
        return summary

    def _notify_daily_precision_failure(self, result: Dict, summary: Dict) -> None:
        try:
            from executors.execution_repository import execution_repository
            from executors.outbox_dispatcher import outbox_dispatcher

            status_upper = str((result or {}).get("status", "")).upper()
            started_at = str((result or {}).get("started_at", "") or "")

            if status_upper == "SUCCESS":
                # 성공 시에도 완료 요약 알림 전송
                sym = str((result or {}).get("symbol", "?") or "?")
                mode_str = str((result or {}).get("mode", "?") or "?").upper()
                decision = str((result or {}).get("decision", "-") or "-")
                report_id = str((result or {}).get("report_id", "-") or "-")
                lines = [
                    "<b>\u2705 Daily Precision \uc644\ub8cc</b>",
                    f"<code>{sym} {mode_str}</code>",
                    f"Decision: <b>{decision}</b> | report_id: <code>{report_id}</code>",
                    f"Started: <code>{started_at[:16].replace('T', ' ') if started_at else 'N/A'} UTC</code>",
                ]
            else:
                lines = [
                    "<b>\u26a0\ufe0f Daily Precision Partial Failure</b>",
                    f"Started: <code>{started_at[:16].replace('T', ' ') if started_at else 'N/A'} UTC</code>",
                ]
                symbol_rows = (summary.get("symbols", {}) if isinstance(summary, dict) else {}) or {}
                ordered_symbols = list(settings.trading_symbols) or list(symbol_rows.keys())
                for symbol_key in ordered_symbols:
                    item = symbol_rows.get(symbol_key)
                    if not isinstance(item, dict):
                        continue
                    sym = str(item.get("symbol", "?") or "?")
                    mode_str = str(item.get("mode", "?") or "?").upper()
                    st = str(item.get("status", "UNKNOWN") or "UNKNOWN").upper()
                    report_id = str(item.get("report_id", "") or "-")
                    decision = str(item.get("decision", "") or "-")
                    error = str(item.get("error", "") or "").strip()
                    line = (
                        f"- <b>{sym} {mode_str}</b> | status=<code>{st}</code> "
                        f"| decision=<code>{decision}</code> | report_id=<code>{report_id}</code>"
                    )
                    if error:
                        safe_error = error[:240].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                        line += f"\n  error: <code>{safe_error}</code>"
                    lines.append(line)

            target_chat_id = state_manager.get_telegram_chat_id(settings.TELEGRAM_CHAT_ID) or settings.TELEGRAM_CHAT_ID
            execution_repository.enqueue_outbox_event(
                "telegram_message",
                {
                    "chat_id": target_chat_id,
                    "text": "\n".join(lines),
                    "parse_mode": "HTML",
                },
                idempotency_key=f"telegram:daily_precision_summary:{result.get('symbol')}:{started_at}:{result.get('status')}",
            )
            outbox_dispatcher.publish_pending(limit=20)
        except Exception as e:
            logger.warning(f"Failed to notify daily precision result: {e}")


    def run_analysis(
        self,
        symbol: str,
        is_emergency: bool = False,
        execute_trades: bool = True,
        allow_perplexity: bool = False,
        notification_context: str = "manual",
    ) -> Dict:
        mode = self.mode
        return self.run_analysis_with_mode(
            symbol,
            mode,
            is_emergency=is_emergency,
            execute_trades=execute_trades,
            allow_perplexity=allow_perplexity,
            notification_context=notification_context,
        )

    def run_analysis_with_mode(
        self,
        symbol: str,
        mode: TradingMode,
        is_emergency: bool = False,
        execute_trades: bool = True,
        allow_perplexity: bool = False,
        notification_context: str = "manual",
        playbook_context: Optional[Dict] = None,
        emergency_context: Optional[Dict] = None,
    ) -> Dict:
        lock_key = f"{symbol}:{mode.value}"
        lock = self._analysis_locks.setdefault(lock_key, threading.Lock())
        if not lock.acquire(blocking=False):
            logger.warning(f"Analysis already running for {lock_key}; skipping duplicate run.")
            return {"decision": "HOLD", "reasoning": f"Duplicate analysis blocked for {lock_key}", "confidence": 0}

        logger.info(f"Starting {'EMERGENCY' if is_emergency else 'SCHEDULED'} {mode.value.upper()} analysis for {symbol}")

        started = time.perf_counter()
        execution_path = "langgraph" if self.graph else "sequential"
        try:
            # [FIX CRASH-2] Clear per-symbol+mode cache at the start of each analysis
            _clear_symbol_mode_caches(symbol, mode)

            if self.graph:
                result = self._run_with_langgraph(
                    symbol,
                    mode,
                    is_emergency,
                    execute_trades=execute_trades,
                    allow_perplexity=allow_perplexity,
                    notification_context=notification_context,
                    playbook_context=playbook_context,
                    emergency_context=emergency_context,
                )
            else:
                result = self._run_sequential(
                    symbol,
                    mode,
                    is_emergency,
                    execute_trades=execute_trades,
                    allow_perplexity=allow_perplexity,
                    notification_context=notification_context,
                    playbook_context=playbook_context,
                    emergency_context=emergency_context,
                )
            performance_telemetry.log_full_path_run(
                symbol=symbol,
                mode=mode.value,
                notification_context=notification_context,
                execution_path=execution_path,
                allow_perplexity=allow_perplexity,
                execute_trades=execute_trades,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                final_decision=result if isinstance(result, dict) else {},
            )
            return result
        finally:
            lock.release()

    def refresh_snapshot_with_mode(
        self,
        symbol: str,
        mode: TradingMode,
        *,
        allow_perplexity: bool = False,
        include_meta: bool = True,
        include_vlm: bool = True,
    ) -> None:
        refresh_snapshot_bundle(
            symbol,
            mode,
            allow_perplexity=allow_perplexity,
            include_meta=include_meta,
            include_vlm=include_vlm,
        )

    def run_report_hot_path_with_mode(
        self,
        symbol: str,
        mode: TradingMode,
        *,
        allow_perplexity: bool = False,
        wait_budget_s: float = 5.0,
    ) -> Dict:
        self.refresh_snapshot_with_mode(symbol, mode, allow_perplexity=allow_perplexity)
        return run_report_hot_path(
            symbol,
            mode,
            wait_budget_s=wait_budget_s,
            notification_context="analysis",
        )

    def run_snapshot_analysis_with_mode(
        self,
        symbol: str,
        mode: TradingMode,
        *,
        allow_perplexity: bool = False,
        wait_budget_s: float = 5.0,
        notification_context: str = "analysis",
        playbook_context: Optional[Dict] = None,
    ) -> Dict:
        logger.info(
            f"Snapshot analysis start: {symbol}/{mode.value} "
            f"context={notification_context} perplexity={'on' if allow_perplexity else 'off'}"
        )
        self.refresh_snapshot_with_mode(symbol, mode, allow_perplexity=allow_perplexity)
        logger.info(f"Snapshot analysis refreshed: {symbol}/{mode.value}")
        result = run_snapshot_analysis_hot_path(
            symbol,
            mode,
            wait_budget_s=wait_budget_s,
            notification_context=notification_context,
            playbook_context=playbook_context,
        )
        logger.info(f"Snapshot analysis done: {symbol}/{mode.value} context={notification_context}")
        return result

    def run_snapshot_trigger_with_mode(
        self,
        symbol: str,
        mode: TradingMode,
        *,
        allow_perplexity: bool = False,
        wait_budget_s: float = 5.0,
        notification_context: str = "scheduled_trigger",
        playbook_context: Optional[Dict] = None,
    ) -> Dict:
        self.refresh_snapshot_with_mode(symbol, mode, allow_perplexity=allow_perplexity)
        result = run_snapshot_analysis_hot_path(
            symbol,
            mode,
            wait_budget_s=wait_budget_s,
            notification_context=notification_context,
            execute_trades=True,
            playbook_context=playbook_context,
        )
        final_decision = result.get("final_decision", {}) if isinstance(result, dict) else {}
        if str(final_decision.get("trigger_action", "")).upper() == "REPLAN":
            return self.run_emergency_replan_with_mode(
                symbol,
                mode,
                playbook_context=playbook_context,
                reason=str(final_decision.get("replan_reason", "") or final_decision.get("reasoning", {}).get("final_logic", "")),
                allow_perplexity=allow_perplexity,
            )
        return result

    def run_emergency_replan_with_mode(
        self,
        symbol: str,
        mode: TradingMode,
        *,
        playbook_context: Optional[Dict] = None,
        reason: str = "",
        allow_perplexity: bool = False,
    ) -> Dict:
        from agents.emergency_replan_agent import emergency_replan_agent

        emergency_context = emergency_replan_agent.assess(
            playbook_context=playbook_context,
            trigger_validation={"trigger_action": "REPLAN", "replan_reason": reason},
        )
        if emergency_context.get("action") == "CANCEL_AND_CLOSE":
            return {
                "final_decision": {
                    "decision": "CANCEL_AND_CLOSE",
                    "allocation_pct": 0,
                    "leverage": 1,
                    "reasoning": {"final_logic": emergency_context.get("reason", "")},
                    "playbook_context": playbook_context or {},
                }
            }
        return self.run_analysis_with_mode(
            symbol,
            mode,
            is_emergency=True,
            execute_trades=False,
            allow_perplexity=allow_perplexity,
            notification_context="emergency_replan",
            playbook_context=playbook_context,
            emergency_context=emergency_context,
        )

    def _run_with_langgraph(
        self,
        symbol: str,
        mode: TradingMode,
        is_emergency: bool,
        execute_trades: bool = True,
        allow_perplexity: bool = False,
        notification_context: str = "manual",
        playbook_context: Optional[Dict] = None,
        emergency_context: Optional[Dict] = None,
    ) -> Dict:
        """Run analysis using LangGraph StateGraph."""
        initial_state: AnalysisState = {
            "symbol": symbol,
            "mode": mode.value,
            "is_emergency": is_emergency,
            "execute_trades": execute_trades,
            "allow_perplexity": allow_perplexity,
            "notification_context": notification_context,
            "playbook_context": playbook_context or {},
            "emergency_context": emergency_context or {},
            "df_size": 0,
            "market_data_compact": "",
            "narrative_text": "",
            "funding_context": "",
            "cvd_context": "",
            "liquidation_context": "",
            "rag_context": "",
            "telegram_news": "",
            "unified_narrative": "",
            "feedback_text": "",
            "microstructure_context": "",
            "macro_context": "",
            "deribit_context": "",
            "fear_greed_context": "",
            "active_orders": [],
            "open_positions": "",
            "onchain_snapshot": {},
            "onchain_context": "",
            "onchain_gate": {},
            "budget": 100,
            "turn_count": 0,
            "anomalies": [],
            "blackboard": {},
            "conviction_score": 0.0,
            "raw_funding": {},
            "market_regime": "RANGE_BOUND",
            "regime_context": {},
            "chart_image_b64": None,
            "chart_bytes": None,
            "vlm_context_text": "",
            "current_bias": "neutral",
            "scenario_revision_reason": "",
            "active_setup": {},
            "final_decision": {},
            "daily_dual_plan": {},
            "policy_snapshot": {},
            "report": None,
            "stats_context": {},
            "errors": [],
        }

        try:
            result = self.graph.invoke(initial_state)
            return result.get("final_decision", {})
        except Exception as e:
            import traceback
            logger.error(f"LangGraph execution error: {e}\n{traceback.format_exc()}")
            return {"decision": "HOLD", "reasoning": f"LangGraph error: {e}", "confidence": 0}

    def _run_sequential(
        self,
        symbol: str,
        mode: TradingMode,
        is_emergency: bool,
        execute_trades: bool = True,
        allow_perplexity: bool = False,
        notification_context: str = "manual",
        playbook_context: Optional[Dict] = None,
        emergency_context: Optional[Dict] = None,
    ) -> Dict:
        """Fallback: sequential execution (no LangGraph)."""
        state = {
            "symbol": symbol, "mode": mode.value,
            "is_emergency": is_emergency,
            "execute_trades": execute_trades,
            "allow_perplexity": allow_perplexity,
            "notification_context": notification_context,
            "playbook_context": playbook_context or {},
            "emergency_context": emergency_context or {},
            "errors": [],
            "onchain_snapshot": {}, "onchain_context": "", "onchain_gate": {},
            "vlm_context_text": "", "policy_snapshot": {},
            "current_bias": "neutral", "scenario_revision_reason": "", "active_setup": {},
        }

        # Run each node sequentially - mirrors the LangGraph DAG order.
        # node_context_gathering consolidates: perplexity, RAG ingest, funding, CVD,
        # liquidation, RAG query, telegram news, microstructure,
        # macro, deribit, and fear&greed context in a single function.
        for node_fn in [
            node_collect_data,
            node_context_gathering,
            node_unify_narrative,
            node_meta_agent,
            node_triage,
        ]:
            try:
                update = node_fn(state)
                state.update(update)
            except Exception as e:
                logger.error(f"Node {node_fn.__name__} error: {e}")

        # Chart generation + deterministic chart-rule extraction
        if settings.should_use_chart and state.get("df_size", 0) > 0:
            try:
                state.update(node_generate_chart(state))
            except Exception as e:
                logger.error(f"Chart generation error: {e}")
            try:
                state.update(node_rule_based_chart(state))
            except Exception as e:
                logger.error(f"Rule-based chart node error: {e}")

        # VLM: analyzes chart, posts structured output to blackboard["vlm_geometry"]
        try:
            state.update(node_vlm_geometric_expert(state))
        except Exception as e:
            logger.error(f"VLM expert error: {e}")

        # Judge reads blackboard (chart_rules + optional vlm_geometry). No raw chart.
        try:
            state.update(node_judge_agent(state))
        except Exception as e:
            logger.error(f"Judge error: {e}")
            state["final_decision"] = {"decision": "HOLD", "reasoning": str(e), "confidence": 0}

        # [FIX CRASH-1] Risk Manager (CRO) - was completely missing from sequential fallback
        try:
            state.update(node_risk_manager(state))
        except Exception as e:
            logger.error(f"Risk Manager error (sequential): {e}")

        try:
            state.update(node_portfolio_leverage_guard(state))
        except Exception as e:
            logger.error(f"Leverage Guard error (sequential): {e}")

        # [FIX CRASH-1] Trade Execution - was missing, so Judge decisions were never executed
        try:
            state.update(node_execute_trade(state))
        except Exception as e:
            logger.error(f"Trade execution error (sequential): {e}")

        # Report
        try:
            state.update(node_generate_report(state))
        except Exception as e:
            logger.error(f"Report error: {e}")

        try:
            state.update(node_data_synthesis(state))
        except Exception as e:
            logger.error(f"Data synthesis error (sequential): {e}")

        return state.get("final_decision", {})

    def run_scheduled_analysis(self) -> None:
        logger.info("Running scheduled Swing analysis")
        for symbol in self.symbols:
            for mode in [TradingMode.SWING]:
                try:
                    self.run_analysis_with_mode(
                        symbol,
                        mode,
                        is_emergency=False,
                        execute_trades=True,
                        allow_perplexity=False,
                        notification_context="scheduled_dual",
                    )
                except Exception as e:
                    logger.error(f"Analysis error for {symbol} ({mode.value}): {e}")
                    continue

    def run_emergency_analysis(self, symbol: str) -> None:
        logger.critical(f"Running EMERGENCY analysis for {symbol} (mode=swing)")
        self.run_analysis_with_mode(
            symbol,
            TradingMode.SWING,
            is_emergency=True,
            execute_trades=True,
            allow_perplexity=False,
            notification_context="emergency",
        )


    def _execute_daily_playbook_for_symbol(self, symbol: str, mode: TradingMode) -> Dict:
        symbol_started_at = datetime.now(timezone.utc).isoformat()
        result = {
            "symbol": symbol,
            "mode": mode.value,
            "status": "FAILED",
            "report_id": "",
            "decision": "",
            "started_at": symbol_started_at,
            "finished_at": symbol_started_at,
        }
        use_snapshot_hot_path = bool(getattr(settings, "ENABLE_SNAPSHOT_HOT_PATH_DAILY", True))
        try:
            logger.info(f"[Daily] {symbol} {mode.value.upper()} starting...")
            _clear_symbol_mode_caches(symbol, mode)

            if use_snapshot_hot_path:
                logger.info(f"[Daily] {symbol} {mode.value.upper()} refreshing snapshot + hot path...")
                hot_result = self.run_snapshot_analysis_with_mode(
                    symbol,
                    mode,
                    allow_perplexity=True,
                    wait_budget_s=5.0,
                    notification_context="scheduled_daily",
                )
                logger.info(f"[Daily] {symbol} {mode.value.upper()} hot path returned.")
                final_decision = hot_result.get("final_decision", {}) if isinstance(hot_result, dict) else {}
                state = {
                    "symbol": symbol,
                    "mode": mode.value,
                    "final_decision": final_decision,
                    "daily_dual_plan": (
                        final_decision.get("daily_dual_plan", {})
                        if isinstance(final_decision, dict)
                        else {}
                    ),
                }
            elif self.graph:
                hot_result = {}
                logger.info(f"[Daily] {symbol} {mode.value.upper()} running LangGraph path...")
                state = self._run_with_langgraph(
                    symbol,
                    mode,
                    is_emergency=False,
                    execute_trades=True,
                    allow_perplexity=True,
                    notification_context="scheduled_daily",
                )
            else:
                hot_result = {}
                logger.info(f"[Daily] {symbol} {mode.value.upper()} running sequential path...")
                state = self._run_sequential(
                    symbol,
                    mode,
                    is_emergency=False,
                    execute_trades=True,
                    allow_perplexity=True,
                    notification_context="scheduled_daily",
                )

            report_payload = hot_result.get("report", {}) if isinstance(hot_result, dict) else {}
            playbook_decision = state if isinstance(state, dict) else {}
            if isinstance(state, dict) and isinstance(state.get("final_decision"), dict):
                playbook_decision = state.get("final_decision", {}) or {}
            playbook_dual_plan = {}
            if isinstance(state, dict):
                playbook_dual_plan = state.get("daily_dual_plan", {}) or {}
            if not playbook_dual_plan and isinstance(playbook_decision, dict):
                playbook_dual_plan = playbook_decision.get("daily_dual_plan", {}) or {}
            logger.info(f"[Daily] {symbol} {mode.value.upper()} persisting playbook...")
            node_generate_playbook({
                "symbol": symbol,
                "mode": mode.value,
                "final_decision": playbook_decision,
                "daily_dual_plan": playbook_dual_plan,
            })
            logger.info(f"[Daily] {symbol} {mode.value.upper()} done.")
            report_id = ""
            if isinstance(report_payload, dict):
                report_id = str(report_payload.get("report_id") or report_payload.get("id") or "")
            decision_name = ""
            if isinstance(playbook_decision, dict):
                decision_name = str(playbook_decision.get("decision", "") or "")
            result.update({
                "status": "SUCCESS" if report_payload else "NO_REPORT",
                "report_id": report_id,
                "decision": decision_name,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            })
            return result
        except Exception:
            import traceback

            result.update({
                "status": "FAILED",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": traceback.format_exc(limit=4),
            })
            logger.exception(f"Daily playbook error {symbol}/{mode.value}")
            return result

    def run_daily_playbook_for_symbol(self, symbol: str, mode: TradingMode = TradingMode.SWING) -> Dict:
        lock_key = f"daily_precision:{str(symbol).upper()}:{mode.value}"
        lock = self._analysis_locks.setdefault(lock_key, threading.Lock())
        if not lock.acquire(blocking=False):
            logger.warning(f"Daily Precision Run already active for {symbol}/{mode.value}; skipping duplicate request.")
            return {
                "symbol": str(symbol).upper(),
                "mode": mode.value,
                "status": "SKIPPED_LOCK",
                "report_id": "",
                "decision": "",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }

        self._enter_daily_precision_run()
        try:
            result = self._execute_daily_playbook_for_symbol(str(symbol).upper(), mode)
            summary = self._update_daily_precision_summary(result)
            logger.info(f"Daily precision summary: {json.dumps(summary, ensure_ascii=False)}")
            self._notify_daily_precision_failure(result, summary)
            return result
        finally:
            self._exit_daily_precision_run()
            lock.release()

    def run_daily_playbook(self) -> None:
        """Legacy serial runner kept for compatibility and manual invocations."""
        import time as _time

        if self.is_daily_precision_running():
            logger.warning("Daily Precision Run already active; skipping duplicate request.")
            return

        schedule = [(symbol, TradingMode.SWING) for symbol in settings.trading_symbols]
        logger.info("=== Daily Precision Run ===")
        self._enter_daily_precision_run()
        daily_results = []
        try:
            inter_symbol_gap_minutes = max(0, int(getattr(settings, "DAILY_PRECISION_SYMBOL_GAP_MINUTES", 10)))
            for i, (symbol, mode) in enumerate(schedule):
                if i > 0 and inter_symbol_gap_minutes > 0:
                    logger.info(
                        f"Sleeping {inter_symbol_gap_minutes}m before {symbol}/{mode.value} "
                        "to stagger daily precision load..."
                    )
                    _time.sleep(inter_symbol_gap_minutes * 60)
                result = self._execute_daily_playbook_for_symbol(symbol, mode)
                daily_results.append(result)
                summary = self._update_daily_precision_summary(result)
                self._notify_daily_precision_failure(result, summary)
            logger.info(
                f"Daily precision batch summary: "
                f"{json.dumps({'results': daily_results}, ensure_ascii=False)}"
            )
        finally:
            self._exit_daily_precision_run()

    def run_hourly_monitor(self) -> None:
        """Hourly: evaluate each symbol/mode against its Daily Playbook.
        If TRIGGER ? run analysis and allow order execution.
        Daily entry count capped at DAILY_MAX_ENTRIES per symbol.
        """
        from agents.market_monitor_agent import market_monitor_agent
        from datetime import date
        from config.local_state import state_manager
        from bot.telegram_bot import trading_bot
        import asyncio

        DAILY_MAX_ENTRIES = 2
        _entry_count_key = f"_monitor_entries_{date.today().isoformat()}"

        def _lane_line(label: str, res: dict) -> str:
            status = res.get("status", "NO_ACTION")
            reasoning = res.get("reasoning", "No details")
            matched = res.get("matched_conditions", []) or []
            unmatched = res.get("unmatched_conditions", []) or []
            invalidated = bool(res.get("invalidated", False))
            inval_reason = res.get("invalidation_reason", "")
            parts = [
                f"- {label}: <b>{status}</b>",
                f"match {len(matched)}/{len(matched) + len(unmatched)}",
                reasoning,
            ]
            if invalidated and inval_reason:
                parts.append(f"invalidated={inval_reason}")
            if unmatched:
                parts.append(f"next={str(unmatched[0])[:90]}")
            return " | ".join(parts)

        for symbol in self.symbols:
            lane_results = {}
            analysis_enabled = state_manager.is_analysis_enabled()

            for mode in [TradingMode.SWING]:
                try:
                    result = market_monitor_agent.evaluate(symbol, mode.value)
                    status = result.get("status", "NO_ACTION")
                    lane_results[mode.value.upper()] = result
                    logger.info(
                        f"[Monitor] {symbol}/{mode.value}: {status} | "
                        f"matched={len(result.get('matched_conditions', []) or [])} "
                        f"unmatched={len(result.get('unmatched_conditions', []) or [])} "
                        f"invalidated={result.get('invalidated', False)} "
                        f"reason={result.get('reasoning', '')}"
                    )

                    if status in ["TRIGGER", "SOFT_TRIGGER"] and not result.get("invalidated", False):
                        # [FEATURE-1] Save debug log anyway for history
                        try:
                            log_entry = {
                                "symbol": symbol,
                                "mode": mode.value,
                                "status": status,
                                "matched_conditions": result.get("matched_conditions", []),
                                "unmatched_conditions": result.get("unmatched_conditions", []),
                                "live_indicators": result.get("live_indicators", {}),
                                "playbook_id": result.get("playbook_id"),
                                "reasoning": result.get("reasoning", ""),
                                "policy_checks": result.get("policy_checks", {}),
                            }
                            db.client.table("monitor_logs").insert(log_entry).execute()
                            logger.info(f"[Monitor] Log saved to DB for {symbol}/{mode.value}")
                        except Exception as e:
                            logger.warning(f"Failed to save monitor log: {e}")

                        # Trigger actual analysis only for strict TRIGGER
                        if status == "TRIGGER":
                            # Count guard
                            entries_today = int(state_manager.get_system_config(_entry_count_key + symbol, "0"))
                            if entries_today >= DAILY_MAX_ENTRIES:
                                logger.warning(f"[Monitor] {symbol} daily entry cap ({DAILY_MAX_ENTRIES}) reached, skipping.")
                                continue

                            veto_gate = market_monitor_agent.lightweight_veto_check(symbol, mode.value, result)
                            result["llm_veto_gate"] = veto_gate
                            if veto_gate.get("action") == "VETO":
                                logger.warning(
                                    f"[Monitor] {symbol}/{mode.value} vetoed by lightweight gate: "
                                    f"{veto_gate.get('reason', '')}"
                                )
                                result["status"] = "WATCH"
                                result["reasoning"] = (
                                    f"{result.get('reasoning', '')} 경량 LLM 거: {veto_gate.get('reason', '')}"
                                ).strip()
                                lane_results[mode.value.upper()] = result
                                continue
                            if veto_gate.get("action") == "REDUCE":
                                logger.info(
                                    f"[Monitor] {symbol}/{mode.value} marked REDUCE by lightweight gate: "
                                    f"{veto_gate.get('reason', '')}"
                                )
                                result["reasoning"] = (
                                    f"{result.get('reasoning', '')} 경량 LLM 감소 권고: {veto_gate.get('reason', '')}"
                                ).strip()

                            logger.info(f"[Monitor] TRIGGER! Running analysis for {symbol}/{mode.value}")
                            if bool(getattr(settings, "ENABLE_SNAPSHOT_HOT_PATH_TRIGGER", False)):
                                self.run_snapshot_trigger_with_mode(
                                    symbol,
                                    mode,
                                    allow_perplexity=False,
                                    wait_budget_s=5.0,
                                    notification_context="scheduled_trigger",
                                    playbook_context=result.get("playbook_context", {}),
                                )
                            else:
                                self.run_analysis_with_mode(
                                    symbol,
                                    mode,
                                    is_emergency=False,
                                    execute_trades=True,
                                    allow_perplexity=False,
                                    notification_context="scheduled_trigger",
                                    playbook_context=result.get("playbook_context", {}),
                                )
                            state_manager.set_system_config(_entry_count_key + symbol, str(entries_today + 1))
                        else:
                            # SOFT_TRIGGER: Just notify via Telegram (handled below in the loop)
                            logger.info(f"[Monitor] SOFT_TRIGGER detected for {symbol}/{mode.value}. Notifying user.")


                except Exception as e:
                    logger.error(f"Monitor error {symbol}/{mode.value}: {e}")

            # Send one consolidated dual-lane monitor message per symbol.
            try:
                if trading_bot:
                    from executors.outbox_dispatcher import outbox_dispatcher as _dispatcher
                    target_chat_id = state_manager.get_telegram_chat_id(settings.TELEGRAM_CHAT_ID) or settings.TELEGRAM_CHAT_ID
                    swing_res = lane_results.get("SWING", {})
                    swing_status = swing_res.get("status", "NO_ACTION")
                    msg = (
                        f"<b>Hourly Monitor</b>\n"
                        f"<code>{symbol}</code>\n"
                        f"AI Analysis: {'ON' if analysis_enabled else 'OFF'}\n"
                        f"{_lane_line('SWING', swing_res)}"
                    )
                    _dispatcher._run_async(trading_bot.send_message, target_chat_id, msg)

                    try:
                        from mcp_server.tools import mcp_tools
                        import base64
                        for lane, status in [("swing", swing_status)]:
                            if status in ["WATCH", "TRIGGER", "SOFT_TRIGGER"]:
                                chart_res = mcp_tools.get_chart_images(symbol, lane=lane)
                                if "charts" in chart_res:
                                    charts = chart_res.get("charts", [])
                                    total = len(charts)
                                    lookback_label = "12M" if lane == "swing" else "60M"
                                    for idx, chart in enumerate(charts, start=1):
                                        chart_bytes = base64.b64decode(chart["chart_base64"])
                                        tf = str(chart.get("timeframe", "-")).upper()
                                        caption = (
                                            f" - <b>{symbol} {lane.upper()} Chart - {status}</b>\n"
                                            f"Timeframe: <code>{tf}</code>\n"
                                            f"Panel: <code>{idx}/{total}</code>\n"
                                            f"Lookback: <code>{lookback_label}</code>"
                                        )
                                        _dispatcher._run_async(trading_bot.send_photo, target_chat_id, chart_bytes, caption=caption)
                    except Exception as e:
                        logger.warning(f"Failed to send hourly monitor chart for {symbol}: {e}")
            except Exception as e:
                logger.warning(f"Failed to send consolidated monitor message for {symbol}: {e}")


orchestrator = Orchestrator()





