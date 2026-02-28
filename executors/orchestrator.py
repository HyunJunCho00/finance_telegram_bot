"""Orchestrator: LangGraph StateGraph multi-agent analysis pipeline.

Architecture:
  LangGraph StateGraph manages the analysis flow as a directed graph.
  Each node is a processing step. Edges define the execution order.

Graph:
  collect_data -> perplexity_search -> rag_ingest -> funding/cvd/liquidation/rag_query/macro/deribit -> triage
  -> [liquidity_expert, microstructure_expert, macro_expert] 
  -> generate_chart -> judge_agent -> risk_manager -> execute_trade -> generate_report

Benefits over sequential:
  - Explicit state management (TypedDict)
  - Error isolation per node
  - Easy to add/remove/reorder steps
  - Built-in retry support

Cost optimization:
  - Experts (Liquidity, Microstructure, Macro): Gemini Flash, TEXT ONLY, compact data format
  - Judge: Claude Opus 4.6, gets chart image (512x512)
"""

from typing import Dict, Optional, TypedDict, Annotated
from config.database import db
from config.settings import settings, TradingMode
from processors.math_engine import math_engine
from processors.chart_generator import chart_generator
from processors.light_rag import light_rag
from collectors.telegram_collector import telegram_collector
from collectors.perplexity_collector import perplexity_collector
from collectors.macro_collector import macro_collector
from agents.liquidity_agent import liquidity_agent
from agents.microstructure_agent import microstructure_agent
from agents.macro_options_agent import macro_options_agent
from agents.vlm_geometric_agent import vlm_geometric_agent
from agents.meta_agent import meta_agent
from agents.judge_agent import judge_agent
from agents.risk_manager_agent import risk_manager_agent
from config.local_state import state_manager
from executors.report_generator import report_generator
from executors.trade_executor import trade_executor
from executors.post_mortem import write_post_mortem
from executors.data_synthesizer import synthesize_training_data
from utils.retry import api_retry
from utils.cooldown import is_on_cooldown, set_cooldown
from loguru import logger
import numpy as np
import pandas as pd
import json

# [FIX CRASH-2] Module-level cache to avoid 4x duplicate DB queries per cycle
# Cleared at the start of each analysis in run_analysis()
_df_cache: dict = {}        # {symbol: DataFrame}
_market_data_cache: dict = {}  # {symbol: market_data_dict}

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("langgraph not available, using sequential fallback")


# ── State definition (LangGraph TypedDict) ──

class AnalysisState(TypedDict):
    symbol: str
    mode: str  # "swing" or "position"
    is_emergency: bool

    # Data collection results
    df_size: int
    market_data_compact: str
    narrative_text: str
    funding_context: str
    cvd_context: str
    liquidation_context: str
    rag_context: str
    telegram_news: str
    feedback_text: str
    microstructure_context: str
    macro_context: str
    deribit_context: str
    fear_greed_context: str
    active_orders: list
    open_positions: str

    # Blackboard Pattern State
    budget: int
    turn_count: int
    anomalies: list
    blackboard: Dict[str, dict]
    conviction_score: float
    raw_funding: dict  # [FIX] Cached from node_funding_context for report reuse

    # Chart
    chart_image_b64: Optional[str]
    chart_bytes: Optional[bytes]

    # Final output
    market_regime: str
    regime_context: dict
    final_decision: Dict
    report: Optional[Dict]

    # Error tracking
    errors: list


# ── Node functions ──

def node_collect_data(state: AnalysisState) -> dict:
    """Fetch 1m OHLCV data from Supabase + higher TFs from GCS if available."""
    symbol = state["symbol"]
    mode = TradingMode(state["mode"])
    candle_limit = settings.candle_limit

    df = db.get_latest_market_data(symbol, limit=candle_limit)
    if df.empty:
        return {"df_size": 0, "errors": state.get("errors", []) + [f"No market data for {symbol}"]}

    # [FIX CRASH-2] Cache for reuse by node_triage, node_generate_chart, node_generate_report
    _df_cache[symbol] = df

    # Load higher timeframe data from GCS for deeper indicator history
    df_4h, df_1d, df_1w = None, None, None
    try:
        from processors.gcs_parquet import gcs_parquet_store
        if gcs_parquet_store.enabled:
            # Provide VLM with more context (18 months for broad macro trends)
            if mode == TradingMode.SWING:
                df_1d = gcs_parquet_store.load_ohlcv("1d", symbol, months_back=18)
            elif mode == TradingMode.POSITION:
                df_1d = gcs_parquet_store.load_ohlcv("1d", symbol, months_back=24)
                df_1w = gcs_parquet_store.load_ohlcv("1w", symbol, months_back=120)
    except Exception as e:
        logger.warning(f"GCS load skipped: {e}")

    market_data = math_engine.analyze_market(df, mode, df_4h=df_4h, df_1d=df_1d, df_1w=df_1w)
    compact = math_engine.format_compact(market_data)

    # [FIX RESOURCE-1] Cache market_data so node_generate_chart doesn't recompute
    _market_data_cache[symbol] = market_data

    # V5: Fetch currently active orders for this symbol to inject into the PM context
    all_active = state_manager.get_active_orders()
    active_for_symbol = [o for o in all_active if o['symbol'] == symbol]
    
    # V7: Fetch open positions to prevent double-ordering
    open_position_text = "No open positions."
    if settings.PAPER_TRADING_MODE:
        try:
            from executors.paper_exchange import paper_engine
            cursor = paper_engine._conn.cursor()
            cursor.execute("SELECT * FROM paper_positions WHERE is_open = 1 AND symbol = ?", (symbol,))
            pos = cursor.fetchall()
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
@api_retry(max_attempts=3, delay_seconds=2)
def node_perplexity_search(state: AnalysisState) -> dict:
    """Search Perplexity for market narrative."""
    symbol = state["symbol"]
    asset = "BTC" if "BTC" in symbol else "ETH"
    try:
        # Save with full trading symbol (e.g., BTCUSDT) for DB/report traceability
        narrative = perplexity_collector.search_market_narrative(symbol)
        narrative_text = perplexity_collector.format_for_agents(narrative)
        logger.info(f"Narrative for {asset}: {narrative.get('sentiment', '?')}")
        return {"narrative_text": narrative_text}
    except Exception as e:
        logger.error(f"Perplexity error: {e}")
        return {"narrative_text": "Market Narrative: Unavailable"}


def node_rag_ingest(state: AnalysisState) -> dict:
    """Ingest recent data into LightRAG:
    1. Telegram messages (Synthesized via Categorized Batching)
    2. Perplexity narrative (current analysis cycle, single doc)
    """
    from processors.telegram_batcher import telegram_batcher
    
    # ── 1. Batch Telegram synthesis (categorized by source) ──
    # Note: Telegram batching is now handled asynchronously every 1 hour via scheduler.py.
    # We no longer block the 4-hour analysis cycle to fetch and summarize Telegram here.

    # ── 2. Perplexity narrative (fact-based, good for entity extraction) ──
    try:
        narrative_text = state.get("narrative_text", "")
        symbol = state.get("symbol", "")
        if narrative_text and "Unavailable" not in narrative_text and len(narrative_text) > 50:
            import hashlib
            from datetime import datetime, timezone
            ts = datetime.now(timezone.utc).isoformat()
            doc_id = hashlib.md5(f"perplexity:{symbol}:{ts[:13]}".encode()).hexdigest()
            light_rag.ingest_message(
                text=narrative_text,
                channel=f"perplexity_{symbol}",
                timestamp=ts,
                message_id=doc_id,
            )
            logger.info(f"RAG: Perplexity narrative ingested for {symbol}")
    except Exception as e:
        logger.error(f"RAG Perplexity ingestion error: {e}")

    return {}


@api_retry(max_attempts=3, delay_seconds=1)
def node_funding_context(state: AnalysisState) -> dict:
    """Fetch funding rate and Global OI data."""
    symbol = state["symbol"]
    try:
        response = db.client.table("funding_data")\
            .select("*")\
            .eq("symbol", symbol)\
            .order("timestamp", desc=True)\
            .limit(1)\
            .execute()

        raw_funding = response.data[0] if response.data else {}
        funding_data = math_engine.analyze_funding_context(raw_funding) if raw_funding else {}

        if raw_funding:
            funding_data['oi_binance'] = raw_funding.get('oi_binance', 0)
            funding_data['oi_bybit'] = raw_funding.get('oi_bybit', 0)
            funding_data['oi_okx'] = raw_funding.get('oi_okx', 0)

        funding_str = json.dumps(funding_data, default=str) if funding_data else "No funding data."
        # [FIX] Store raw_funding for node_generate_report reuse (eliminates duplicate query)
        return {"funding_context": funding_str, "raw_funding": raw_funding}
    except Exception as e:
        logger.error(f"Funding context error: {e}")
        return {"funding_context": "No funding data.", "raw_funding": {}}


def node_cvd_context(state: AnalysisState) -> dict:
    """Fetch CVD (Cumulative Volume Delta) + Whale CVD context."""
    symbol = state["symbol"]
    mode = TradingMode(state["mode"])
    lookback = settings.data_lookback_hours
    limit = lookback * 60  # 1m resolution

    try:
        cvd_df = db.get_cvd_data(symbol, limit=limit)
        if cvd_df.empty:
            return {"cvd_context": "CVD: No data available"}

        total_delta = float(cvd_df['volume_delta'].sum())
        recent_delta = float(cvd_df.tail(60)['volume_delta'].sum())
        cvd_final = float(cvd_df['cvd'].iloc[-1])

        cvd_direction = "BUYING" if total_delta > 0 else "SELLING"
        recent_direction = "BUYING" if recent_delta > 0 else "SELLING"

        parts = [
            f"[CVD] {lookback}H Delta={total_delta:.2f} ({cvd_direction}) | "
            f"1H Delta={recent_delta:.2f} ({recent_direction}) | "
            f"CVD={cvd_final:.2f}"
        ]

        # Whale CVD if available
        if 'whale_cvd' in cvd_df.columns:
            whale_buy = cvd_df['whale_buy_vol'].sum()
            whale_sell = cvd_df['whale_sell_vol'].sum()
            whale_delta = whale_buy - whale_sell
            whale_dir = "ACCUMULATING" if whale_delta > 0 else "DISTRIBUTING"
            ratio = whale_buy / max(whale_sell, 1)
            parts.append(
                f"[WHALE_CVD] Buy=${whale_buy:,.0f} Sell=${whale_sell:,.0f} "
                f"Ratio={ratio:.2f} ({whale_dir})"
            )

        return {"cvd_context": " | ".join(parts)}
    except Exception as e:
        logger.error(f"CVD context error: {e}")
        return {"cvd_context": "CVD: Error"}


def node_liquidation_context(state: AnalysisState) -> dict:
    """Fetch recent liquidation data for text context."""
    symbol = state["symbol"]
    lookback = settings.data_lookback_hours
    limit = lookback * 60

    try:
        liq_df = db.get_liquidation_data(symbol, limit=limit)
        if liq_df.empty:
            return {"liquidation_context": "Liquidation: No data (WebSocket may not be running)"}

        total_long = float(liq_df['long_liq_usd'].sum())
        total_short = float(liq_df['short_liq_usd'].sum())
        total = total_long + total_short

        # Recent 1h
        recent = liq_df.tail(60)
        recent_long = float(recent['long_liq_usd'].sum())
        recent_short = float(recent['short_liq_usd'].sum())

        # Largest single event
        if 'largest_single_usd' in liq_df.columns:
            max_row = liq_df.loc[liq_df['largest_single_usd'].idxmax()]
            largest = f"${max_row['largest_single_usd']:,.0f} {max_row.get('largest_single_side', '?')} @{max_row.get('largest_single_price', '?')}"
        else:
            largest = "N/A"

        dominant = "LONGS_REKT" if total_long > total_short else "SHORTS_REKT"

        text = (
            f"[LIQUIDATION] {lookback}H: Total=${total:,.0f} ({dominant}) | "
            f"Long_Liq=${total_long:,.0f} Short_Liq=${total_short:,.0f} | "
            f"1H: Long=${recent_long:,.0f} Short=${recent_short:,.0f} | "
            f"Largest={largest}"
        )
        return {"liquidation_context": text}

    except Exception as e:
        logger.debug(f"Liquidation context: {e}")
        return {"liquidation_context": "Liquidation: No data"}


def node_rag_query(state: AnalysisState) -> dict:
    """Query LightRAG for relationship-aware news context."""
    symbol = state["symbol"]
    try:
        coin_name = "Bitcoin BTC" if "BTC" in symbol else "Ethereum ETH"
        result = light_rag.query(coin_name)
        rag_text = light_rag.format_context_for_agents(result, max_length=1500)
        return {"rag_context": rag_text}
    except Exception as e:
        logger.error(f"RAG query error: {e}")
        return {"rag_context": "RAG Context: Unavailable"}


def node_telegram_news(state: AnalysisState) -> dict:
    """Fetch recent telegram news."""
    hours = 1 if state.get("is_emergency") else 4
    news_data = db.get_recent_telegram_messages(hours=hours)
    telegram_news = "\n".join([
        f"[{msg['channel']}] {msg['text'][:200]}"
        for msg in news_data[:10]
    ]) if news_data else "No telegram news."
    return {"telegram_news": telegram_news}


def node_self_correction(state: AnalysisState) -> dict:
    """Fetch past trading mistakes for all agents."""
    try:
        feedback_history = db.get_feedback_history(limit=5)
        if feedback_history:
            feedback_text = "\n\n[PAST MISTAKES TO AVOID]\n" + "\n".join([
                f"- {f.get('mistake_summary', 'N/A')[:150]}"
                for f in feedback_history[:3]
            ])
            return {"feedback_text": feedback_text}
    except Exception as e:
        logger.error(f"Self-correction error: {e}")
    return {"feedback_text": ""}




def node_microstructure_context(state: AnalysisState) -> dict:
    """Fetch latest microstructure snapshot (spread/imbalance/slippage)."""
    symbol = state["symbol"]
    try:
        snap = db.get_latest_microstructure(symbol)
        if not snap:
            return {"microstructure_context": "Microstructure: No data"}

        text = (
            f"[MICRO] spread={snap.get('spread_bps', 0):.2f}bps | "
            f"imbalance={snap.get('orderbook_imbalance', 0):.4f} | "
            f"slippage_100k={snap.get('slippage_buy_100k_bps', 0):.2f}bps"
        )
        return {"microstructure_context": text}
    except Exception as e:
        logger.error(f"Microstructure context error: {e}")
        return {"microstructure_context": "Microstructure: Error"}


def node_macro_context(state: AnalysisState) -> dict:
    """Fetch latest macro snapshot (FRED + cross-asset proxies)."""
    try:
        macro = db.get_latest_macro_data()
        if not macro:
            return {"macro_context": "Macro: No data"}

        text = (
            f"[MACRO] DGS2={macro.get('dgs2', 'N/A')} DGS10={macro.get('dgs10', 'N/A')} "
            f"2s10s={macro.get('ust_2s10s_spread', 'N/A')} | "
            f"DXY={macro.get('dxy', 'N/A')} NASDAQ={macro.get('nasdaq', 'N/A')} "
            f"GOLD={macro.get('gold', 'N/A')} OIL={macro.get('oil', 'N/A')}"
        )
        return {"macro_context": text}
    except Exception as e:
        logger.error(f"Macro context error: {e}")
        return {"macro_context": "Macro: Error"}

def node_deribit_context(state: AnalysisState) -> dict:
    """Fetch latest Deribit options data: DVOL, PCR, IV Term Structure, 25d Skew.
    Returns empty string for symbols without Deribit options (non-BTC/ETH)."""
    symbol = state["symbol"]
    currency = symbol[:-4] if symbol.endswith('USDT') else symbol  # BTCUSDT → BTC
    try:
        data = db.get_latest_deribit_data(currency)
        if not data:
            return {"deribit_context": f"Deribit({currency}): No data"}

        parts = [f"[DERIBIT {currency}]"]

        if data.get('dvol') is not None:
            parts.append(f"DVOL={data['dvol']}")

        if data.get('pcr_oi') is not None:
            parts.append(f"PCR_OI={data['pcr_oi']} PCR_VOL={data.get('pcr_vol', 'N/A')}")

        iv_parts = []
        for b in ['1w', '2w', '1m', '3m', '6m']:
            iv = data.get(f'iv_{b}')
            if iv is not None:
                iv_parts.append(f"{b}={iv}")
        if iv_parts:
            inverted_flag = " [INVERTED=panic]" if data.get('term_inverted') else ""
            parts.append(f"IV_Term: {' '.join(iv_parts)}{inverted_flag}")

        skew_parts = []
        for b in ['1w', '2w', '1m', '3m']:
            sk = data.get(f'skew_{b}')
            if sk is not None:
                skew_parts.append(f"{b}={sk:+.2f}")
        if skew_parts:
            parts.append(f"25dSkew(put-call): {' '.join(skew_parts)}")

        return {"deribit_context": " | ".join(parts)}
    except Exception as e:
        logger.error(f"Deribit context error: {e}")
        return {"deribit_context": f"Deribit({currency}): Error"}


def node_fear_greed_context(state: AnalysisState) -> dict:
    """Fetch latest Crypto Fear & Greed Index (market-wide daily sentiment)."""
    try:
        data = db.get_latest_fear_greed()
        if not data:
            return {"fear_greed_context": "Fear&Greed: No data"}

        value = data.get('value', 'N/A')
        classification = data.get('classification', 'Unknown')
        change = data.get('change')
        change_str = f" Δ{change:+d}" if change is not None else ""

        return {"fear_greed_context": f"[FEAR&GREED] {value}/100 — {classification}{change_str}"}
    except Exception as e:
        logger.error(f"Fear & Greed context error: {e}")
        return {"fear_greed_context": "Fear&Greed: Error"}


def node_meta_agent(state: AnalysisState) -> dict:
    """Run Meta-Agent to classify market regime and provide trust directives."""
    try:
        # [NEW] Inject RAG context and Telegram news for Narrative-Aware classification
        rag_context = state.get("rag_context", "")
        telegram_news = state.get("telegram_news", "")
        
        result = meta_agent.classify_regime(
            market_data_compact=state.get("market_data_compact", ""),
            deribit_context=state.get("deribit_context", ""),
            funding_context=state.get("funding_context", ""),
            macro_context=state.get("macro_context", ""),
            rag_context=rag_context,
            telegram_news=telegram_news,
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


def node_triage(state: AnalysisState) -> dict:
    """Evaluate Python/Pandas rules for Anomaly Detection before waking LLMs."""
    anomalies = []
    symbol = state["symbol"]
    
    # 1. Volume-Backed Breakout (Mathematical)
    try:
        # [FIX CRASH-2] Use cached DataFrame from node_collect_data
        df = _df_cache.get(symbol)
        if df is None:
            df = db.get_latest_market_data(symbol, limit=4320)  # fallback if cache miss
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

    # 2. The Big Squeeze 
    liq_ctx = state.get("liquidation_context", "")
    if "LONGS_REKT" in liq_ctx or "SHORTS_REKT" in liq_ctx:
        # Require >$100k liquidation in 1H for emergency trigger
        import re
        match = re.search(r"Total=\$?([\d\.]+)", liq_ctx.replace(',', ''))
        if match and float(match.group(1)) > 100000:
            if not is_on_cooldown("liquidation_cluster", symbol):
                anomalies.append("liquidation_cluster")
                set_cooldown("liquidation_cluster", symbol)

    # 3. Microstructure Breakdown
    micro_ctx = state.get("microstructure_context", "")
    if "imbalance" in micro_ctx:
        # Require > 70% (0.7) imbalance
        import re
        match = re.search(r"imbalance[:=]\s*([-0-9\.]+)", micro_ctx)
        if match and abs(float(match.group(1))) > 0.7:
            anomalies.append("microstructure_imbalance")
        
    # 4. Options Panic (DVOL Spike or Term Inversion)
    if "INVERTED" in state.get("deribit_context", "") or "DVOL=" in state.get("deribit_context", ""):
        if not is_on_cooldown("options_panic", symbol):
            anomalies.append("options_panic")
            set_cooldown("options_panic", symbol)

    # If is_emergency was forced via UI/Telegram command
    if state.get("is_emergency") and not anomalies:
        anomalies.append("manual_emergency_trigger")

    return {
        "budget": 100,
        "turn_count": 0,
        "anomalies": list(set(anomalies)), # deduplicate
    }

def node_liquidity_expert(state: AnalysisState) -> dict:
    """Run Liquidity Agent."""
    bb = state.get("blackboard", {})
    if state.get("budget", 0) <= 0: return {}
    
    result = liquidity_agent.analyze(
        state.get("cvd_context", ""),
        state.get("liquidation_context", ""),
        mode=state.get("mode", "SWING").upper()
    )
    bb["liquidity"] = result
    return {"blackboard": bb, "budget": state["budget"] - 20, "turn_count": state.get("turn_count", 0) + 1}

def node_microstructure_expert(state: AnalysisState) -> dict:
    """Run Microstructure Agent."""
    bb = state.get("blackboard", {})
    if state.get("budget", 0) <= 0: return {}
    
    result = microstructure_agent.analyze(
        state.get("microstructure_context", ""),
        mode=state.get("mode", "SWING").upper()
    )
    bb["microstructure"] = result
    return {"blackboard": bb, "budget": state["budget"] - 15, "turn_count": state.get("turn_count", 0) + 1}

def node_macro_options_expert(state: AnalysisState) -> dict:
    """Run Macro Options Agent."""
    bb = state.get("blackboard", {})
    if state.get("budget", 0) <= 0: return {}
    
    result = macro_options_agent.analyze(
        state.get("deribit_context", ""),
        state.get("macro_context", ""),
        mode=state.get("mode", "SWING").upper()
    )
    bb["macro"] = result
    return {"blackboard": bb, "budget": state["budget"] - 15, "turn_count": state.get("turn_count", 0) + 1}

def node_vlm_geometric_expert(state: AnalysisState) -> dict:
    """Run VLM Geometric visual analysis."""
    bb = state.get("blackboard", {})
    if state.get("budget", 0) <= 0: return {}
    
    # Needs chart image generated previously
    result = vlm_geometric_agent.analyze(
        state.get("chart_image_b64", ""),
        mode=state.get("mode", "SWING").upper()
    )
    bb["vlm_geometry"] = result
    return {"blackboard": bb, "budget": state["budget"] - 25, "turn_count": state.get("turn_count", 0) + 1}


def node_generate_chart(state: AnalysisState) -> dict:
    """Generate structure chart for all modes (for Judge VLM)."""
    symbol = state["symbol"]
    mode = TradingMode(state["mode"])
    candle_limit = settings.candle_limit

    # [FIX CRASH-2] Use cached DataFrame from node_collect_data
    df = _df_cache.get(symbol)
    if df is None:
        df = db.get_latest_market_data(symbol, limit=candle_limit)  # fallback
    if df.empty:
        return {"chart_image_b64": None, "chart_bytes": None}

    # Load higher TF data from GCS
    df_4h, df_1d, df_1w = None, None, None
    try:
        from processors.gcs_parquet import gcs_parquet_store
        if gcs_parquet_store.enabled:
            # Provide VLM with more context (18 months for broad macro trends)
            if mode == TradingMode.SWING:
                df_1d = gcs_parquet_store.load_ohlcv("1d", symbol, months_back=18)
            elif mode == TradingMode.POSITION:
                df_1d = gcs_parquet_store.load_ohlcv("1d", symbol, months_back=24)
                df_1w = gcs_parquet_store.load_ohlcv("1w", symbol, months_back=120)
    except Exception as e:
        logger.warning(f"GCS load for chart skipped: {e}")

    # [FIX RESOURCE-1] Use cached market_data from node_collect_data
    market_data = _market_data_cache.get(symbol)
    if market_data is None:
        market_data = math_engine.analyze_market(df, mode, df_4h=df_4h, df_1d=df_1d, df_1w=df_1w)

    # Load CVD (Volume Delta) data for chart sync
    cvd_df = None
    try:
        # Fetch 1m CVD data matching the candle lookback
        cvd_limit = settings.data_lookback_hours * 60
        cvd_df = db.get_cvd_data(symbol, limit=cvd_limit)
        
        # [NEW] Merge with GCS historical CVD for long-term charts
        from processors.gcs_parquet import gcs_parquet_store
        if gcs_parquet_store.enabled:
            m_back = 6 if mode == TradingMode.SWING else 12
            hist_cvd = gcs_parquet_store.load_timeseries("cvd", symbol, months_back=m_back)
            if not hist_cvd.empty:
                hist_cvd['timestamp'] = pd.to_datetime(hist_cvd['timestamp'], utc=True)
                cvd_df = pd.concat([hist_cvd, cvd_df]).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    except Exception as e:
        logger.warning(f"CVD data load for chart skipped/merged: {e}")

    # Load liquidation data for chart markers
    liquidation_df = None
    try:
        liq_limit = settings.data_lookback_hours * 60
        liquidation_df = db.get_liquidation_data(symbol, limit=liq_limit)
    except Exception:
        pass

    # Load funding/OI data for chart sync
    funding_df = None
    try:
        # Fetch funding data (includes OI, LSR) matching the lookback
        funding_limit = settings.data_lookback_hours * 60
        funding_df = db.get_funding_history(symbol, limit=funding_limit)
        
        # [NEW] Merge with GCS historical funding for long-term charts
        from processors.gcs_parquet import gcs_parquet_store
        if gcs_parquet_store.enabled:
            m_back = 6 if mode == TradingMode.SWING else 12
            hist_fnd = gcs_parquet_store.load_timeseries("funding", symbol, months_back=m_back)
            if not hist_fnd.empty:
                hist_fnd['timestamp'] = pd.to_datetime(hist_fnd['timestamp'], utc=True)
                funding_df = pd.concat([hist_fnd, funding_df]).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    except Exception as e:
        logger.warning(f"Funding/OI data load for chart skipped/merged: {e}")

    chart_bytes = chart_generator.generate_chart(df, market_data, symbol, mode,
                                                  liquidation_df=liquidation_df,
                                                  cvd_df=cvd_df,
                                                  funding_df=funding_df,
                                                  df_1d=df_1d,
                                                  df_1w=df_1w)

    if not chart_bytes:
        logger.warning(f"Chart generation FAILED for {symbol} ({mode.value})")
        return {"chart_image_b64": None, "chart_bytes": None}

    chart_image_b64 = None
    if chart_bytes:
        chart_bytes_for_vlm = chart_generator.resize_for_low_res(chart_bytes)
        chart_image_b64 = chart_generator.chart_to_base64(chart_bytes_for_vlm)
        logger.info(f"Generated chart for {symbol}: {len(chart_bytes)} bytes. VLM B64 size: {len(chart_image_b64)}")

    return {"chart_image_b64": chart_image_b64, "chart_bytes": chart_bytes}


def node_judge_agent(state: AnalysisState) -> dict:
    """Run Judge (Claude Opus 4.6). Reads Blackboard."""
    mode = TradingMode(state["mode"])
    
    # Extract necessary state variables
    compact = state.get("market_data_compact", "")
    blackboard = state.get("blackboard", {})
    funding_context = state.get("funding_context", "")
    feedback_text = state.get("feedback_text", "")
    open_positions = state.get("open_positions", "")

    # Pass blackboard and regime context
    regime_ctx = state.get("regime_context", {})
    decision = judge_agent.make_decision(
        market_data_compact=compact,
        blackboard=blackboard,
        funding_context=funding_context,
        chart_image_b64=state.get("chart_image_b64"),
        mode=mode,
        feedback_text=feedback_text,
        active_orders=state.get("active_orders", []),
        open_positions=open_positions,
        symbol=state.get("symbol", "BTCUSDT"),
        regime_context=regime_ctx
    )
    
    # [NEW] Probabilistic EV & Strategic Scaling (SOTA 2026)
    # Calculate Expected Value
    win_prob = decision.get("win_probability_pct", 0) / 100.0
    profit_pct = decision.get("expected_profit_pct", 0.0)
    loss_pct = decision.get("expected_loss_pct", 0.0)
    
    ev = (win_prob * profit_pct) - ((1.0 - win_prob) * loss_pct)
    
    # Scale allocation by Risk Budget
    risk_budget = regime_ctx.get("risk_budget_pct", 100)
    original_alloc = decision.get("allocation_pct", 0) if isinstance(decision.get("allocation_pct"), (int, float)) else 0
    scaled_alloc = original_alloc * (risk_budget / 100.0)
    
    if ev <= 0 and decision.get("decision") != "HOLD":
        logger.warning(f"Judge EV ({ev:.2f}) is negative/zero. Forcing HOLD.")
        decision["decision"] = "HOLD"
        decision["reasoning"]["final_logic"] = f"[EV VETO] Forced HOLD because EV is {ev:.2f}. " + decision["reasoning"].get("final_logic", "")
    elif decision.get("decision") != "HOLD":
        decision["allocation_pct"] = scaled_alloc
        logger.info(f"Scaled allocation from {original_alloc}% to {scaled_alloc:.2f}% (Risk Budget: {risk_budget}%)")
        decision["reasoning"]["final_logic"] = f"[SCALING] Allocation constrained by {risk_budget}% Risk Budget. EV is {ev:.2f}. " + decision["reasoning"].get("final_logic", "")

    # For compatibility with legacy metrics/logging, set conviction to win_probability_pct
    return {"final_decision": decision, "conviction_score": win_prob * 100}

def node_risk_manager(state: AnalysisState) -> dict:
    """Run Risk Manager (CRO). VETO power over Judge's draft."""
    draft = state.get("final_decision", {})
    
    # Needs macro and funding context for risk overlays
    funding = state.get("funding_context", "")
    deribit = state.get("deribit_context", "")
    
    final_decision = risk_manager_agent.evaluate_trade(draft, funding, deribit)
    return {"final_decision": final_decision}

def node_portfolio_leverage_guard(state: AnalysisState) -> dict:
    """Mathematically enforce 2.0x total account leverage limit."""
    decision = state.get("final_decision", {})
    if decision.get("decision") not in ["LONG", "SHORT"]:
        return {}

    try:
        from executors.paper_exchange import paper_engine
        
        # 1. Fetch total equity and exposure (Paper mode for now as per strategy)
        target_exchange = decision.get("target_exchange", "BINANCE").lower()
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
    
    # Bridge to execution
    # This will simulate DCA or single entries based on the mode and CRO sizing
    execution_result = trade_executor.execute_from_decision(decision, mode, symbol)
    
    # Attach execution receipt back to the decision for reporting
    decision["execution_receipt"] = execution_result
    return {"final_decision": decision}

def node_generate_report(state: AnalysisState) -> dict:
    """Generate and send report to Telegram."""
    from executors.metrics_logger import metrics_logger
    symbol = state["symbol"]
    mode = TradingMode(state["mode"])

    candle_limit = settings.candle_limit
    # [FIX CRASH-2] Use cached DataFrame from node_collect_data
    df = _df_cache.get(symbol)
    if df is None:
        df = db.get_latest_market_data(symbol, limit=candle_limit)  # fallback
    # [FIX RESOURCE-1] Use cached market_data
    market_data = _market_data_cache.get(symbol)
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
    )
    
    # [FIX CRITICAL-1] Pass actual last close price (was 0.0 → all episodic memory inverted)
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
        chart_bytes = state.get("chart_bytes")
        if chart_bytes:
            logger.info(f"Passing {len(chart_bytes)} bytes of chart data to notify.")
        else:
            logger.warning("No chart data found in state at generate_report node.")
        report_generator.notify(report, chart_bytes=chart_bytes, mode=mode)
        
        # Log prediction for academic/quantitative evaluation
        metrics_logger.log_prediction(
            symbol=symbol,
            mode=mode.value,
            final_decision=state.get("final_decision", {}),
            blackboard=state.get("blackboard", {}),
            anomalies=state.get("anomalies", [])
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


# ── Helper ──

def _build_full_context(state: AnalysisState) -> str:
    """Build full context string for agents."""
    parts = [
        state.get("narrative_text", ""),
        state.get("cvd_context", ""),
        state.get("liquidation_context", ""),
        state.get("microstructure_context", ""),
        state.get("macro_context", ""),
        state.get("deribit_context", ""),      # DVOL / PCR / IV term / 25d skew
        state.get("fear_greed_context", ""),   # F&G daily sentiment
        state.get("rag_context", ""),
        f"Telegram News:\n{state.get('telegram_news', '')}",
        state.get("feedback_text", ""),
    ]
    return "\n\n".join(p for p in parts if p)


# ── Conditional edge ──

# ── Conditional edge ──

def route_triage(state: AnalysisState) -> str:
    """[FIX CRITICAL-3] Always run Judge — SWING trading requires evaluation every cycle,
    not only when anomalies are detected. Without this, positions are never re-evaluated
    and new trend entries are missed in quiet markets."""
    anomalies = state.get("anomalies", [])
    if not anomalies:
        return "generate_chart"  # → vlm_expert → judge_agent → risk_manager → execute_trade
    
    if "liquidation_cluster" in anomalies or "whale_cvd_divergence" in anomalies:
        return "liquidity_expert"
    elif "microstructure_imbalance" in anomalies:
        return "microstructure_expert"
    elif "options_panic" in anomalies:
        return "macro_expert"
    return "generate_chart"

def route_swarm(state: AnalysisState) -> str:
    budget = state.get("budget", 0)
    turns = state.get("turn_count", 0)
    
    if budget <= 0 or turns >= 5:
        return "generate_chart"
        
    bb = state.get("blackboard", {})
    
    # If all text experts have spoken, go to chart generation and then VLM
    if "liquidity" in bb and "microstructure" in bb and "macro" in bb:
        return "generate_chart"
        
    if "microstructure" not in bb:
        return "microstructure_expert"
    if "macro" not in bb:
        return "macro_expert"
        
    return "generate_chart"

def route_vlm(state: AnalysisState) -> str:
    """Route to VLM Expert after chart is generated, then Judge."""
    return "vlm_expert"


# ── Build the LangGraph StateGraph ──

def build_analysis_graph():
    """Build the multi-agent analysis graph."""
    graph = StateGraph(AnalysisState)

    # Add nodes
    graph.add_node("collect_data", node_collect_data)
    graph.add_node("perplexity_search", node_perplexity_search)
    graph.add_node("rag_ingest", node_rag_ingest)
    graph.add_node("funding_context", node_funding_context)
    graph.add_node("cvd_context", node_cvd_context)
    graph.add_node("liquidation_context", node_liquidation_context)
    graph.add_node("rag_query", node_rag_query)
    graph.add_node("telegram_news", node_telegram_news)
    graph.add_node("self_correction", node_self_correction)
    graph.add_node("microstructure_context", node_microstructure_context)
    graph.add_node("macro_context", node_macro_context)
    graph.add_node("deribit_context", node_deribit_context)
    graph.add_node("fear_greed_context", node_fear_greed_context)
    graph.add_node("meta_agent", node_meta_agent)
    
    # Triage and Experts
    graph.add_node("triage", node_triage)
    graph.add_node("liquidity_expert", node_liquidity_expert)
    graph.add_node("microstructure_expert", node_microstructure_expert)
    graph.add_node("macro_expert", node_macro_options_expert)
    
    graph.add_node("generate_chart", node_generate_chart)
    graph.add_node("vlm_expert", node_vlm_geometric_expert)
    graph.add_node("judge_agent", node_judge_agent)
    graph.add_node("risk_manager", node_risk_manager)
    graph.add_node("portfolio_leverage_guard", node_portfolio_leverage_guard)
    graph.add_node("execute_trade", node_execute_trade)
    graph.add_node("generate_report", node_generate_report)
    graph.add_node("data_synthesis", node_data_synthesis)

    graph.set_entry_point("collect_data")

    graph.add_edge("collect_data", "perplexity_search")
    graph.add_edge("perplexity_search", "rag_ingest")
    graph.add_edge("rag_ingest", "funding_context")
    graph.add_edge("funding_context", "cvd_context")
    graph.add_edge("cvd_context", "liquidation_context")
    graph.add_edge("liquidation_context", "rag_query")
    graph.add_edge("rag_query", "telegram_news")
    graph.add_edge("telegram_news", "self_correction")
    graph.add_edge("self_correction", "microstructure_context")
    graph.add_edge("microstructure_context", "macro_context")
    graph.add_edge("macro_context", "deribit_context")
    graph.add_edge("deribit_context", "fear_greed_context")
    graph.add_edge("fear_greed_context", "meta_agent")
    graph.add_edge("meta_agent", "triage")

    graph.add_conditional_edges("triage", route_triage, {
        "liquidity_expert": "liquidity_expert",
        "microstructure_expert": "microstructure_expert",
        "macro_expert": "macro_expert",
        "generate_chart": "generate_chart",
        # [FIX SILENT-1] Removed dead "generate_report" — route_triage no longer returns it
    })
    
    graph.add_conditional_edges("liquidity_expert", route_swarm, {
        "microstructure_expert": "microstructure_expert",
        "macro_expert": "macro_expert",
        "generate_chart": "generate_chart"
    })
    
    graph.add_conditional_edges("microstructure_expert", route_swarm, {
        "macro_expert": "macro_expert",
        "generate_chart": "generate_chart"
    })
    
    graph.add_conditional_edges("macro_expert", route_swarm, {
        "generate_chart": "generate_chart"
    })

    graph.add_edge("generate_chart", "vlm_expert")
    graph.add_edge("vlm_expert", "judge_agent")
    graph.add_edge("judge_agent", "risk_manager")
    graph.add_edge("risk_manager", "portfolio_leverage_guard")
    graph.add_edge("portfolio_leverage_guard", "execute_trade")
    graph.add_edge("execute_trade", "generate_report")
    graph.add_edge("generate_report", "data_synthesis")
    graph.add_edge("data_synthesis", END)

    return graph.compile()


# ── Orchestrator class (maintains backward compatibility) ──

class Orchestrator:
    def __init__(self):
        self.symbols = settings.trading_symbols
        self._graph = None

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

    def run_analysis(self, symbol: str, is_emergency: bool = False) -> Dict:
        mode = self.mode
        logger.info(f"Starting {'EMERGENCY' if is_emergency else 'SCHEDULED'} {mode.value.upper()} analysis for {symbol}")

        # [FIX CRASH-2] Clear per-symbol cache at the start of each analysis
        _df_cache.pop(symbol, None)
        _market_data_cache.pop(symbol, None)

        if self.graph:
            return self._run_with_langgraph(symbol, mode, is_emergency)
        else:
            return self._run_sequential(symbol, mode, is_emergency)

    def _run_with_langgraph(self, symbol: str, mode: TradingMode, is_emergency: bool) -> Dict:
        """Run analysis using LangGraph StateGraph."""
        initial_state: AnalysisState = {
            "symbol": symbol,
            "mode": mode.value,
            "is_emergency": is_emergency,
            "df_size": 0,
            "market_data_compact": "",
            "narrative_text": "",
            "funding_context": "",
            "cvd_context": "",
            "liquidation_context": "",
            "rag_context": "",
            "telegram_news": "",
            "feedback_text": "",
            "microstructure_context": "",
            "macro_context": "",
            "deribit_context": "",
            "fear_greed_context": "",
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
            "final_decision": {},
            "report": None,
            "errors": [],
        }

        try:
            result = self.graph.invoke(initial_state)
            return result.get("final_decision", {})
        except Exception as e:
            import traceback
            logger.error(f"LangGraph execution error: {e}\n{traceback.format_exc()}")
            return {"decision": "HOLD", "reasoning": f"LangGraph error: {e}", "confidence": 0}

    def _run_sequential(self, symbol: str, mode: TradingMode, is_emergency: bool) -> Dict:
        """Fallback: sequential execution (no LangGraph)."""
        state = {
            "symbol": symbol, "mode": mode.value,
            "is_emergency": is_emergency, "errors": [],
        }

        # Run each node sequentially
        # [FIX CRITICAL-4] Replace undefined node_bull_agent/node_bear_agent/node_risk_agent
        # with actual node functions defined in this module.
        for node_fn in [
            node_collect_data, node_perplexity_search, node_rag_ingest,
            node_funding_context, node_cvd_context, node_liquidation_context,
            node_rag_query, node_telegram_news, node_self_correction,
            node_microstructure_context, node_macro_context,
            node_deribit_context, node_fear_greed_context,
            node_meta_agent,
            node_triage,
            node_liquidity_expert, node_microstructure_expert,
            node_macro_options_expert,
        ]:
            try:
                update = node_fn(state)
                state.update(update)
            except Exception as e:
                logger.error(f"Node {node_fn.__name__} error: {e}")

        # Conditional chart (all modes now)
        if settings.should_use_chart and state.get("df_size", 0) > 0:
            try:
                state.update(node_generate_chart(state))
            except Exception as e:
                logger.error(f"Chart generation error: {e}")

        # Judge
        try:
            state.update(node_judge_agent(state))
        except Exception as e:
            logger.error(f"Judge error: {e}")
            state["final_decision"] = {"decision": "HOLD", "reasoning": str(e), "confidence": 0}

        # [FIX CRASH-1] Risk Manager (CRO) — was completely missing from sequential fallback
        try:
            state.update(node_risk_manager(state))
        except Exception as e:
            logger.error(f"Risk Manager error (sequential): {e}")

        # [FIX CRASH-1] Trade Execution — was missing, so Judge decisions were never executed
        try:
            state.update(node_execute_trade(state))
        except Exception as e:
            logger.error(f"Trade execution error (sequential): {e}")

        # Report
        try:
            state.update(node_generate_report(state))
        except Exception as e:
            logger.error(f"Report error: {e}")

        return state.get("final_decision", {})

    def run_scheduled_analysis(self) -> None:
        logger.info(f"Running scheduled analysis (mode={self.mode.value})")

        # Collect telegram messages first
        telegram_collector.run(hours=4)

        for symbol in self.symbols:
            try:
                self.run_analysis(symbol, is_emergency=False)
            except Exception as e:
                logger.error(f"Analysis error for {symbol}: {e}")
                continue

    def run_emergency_analysis(self, symbol: str) -> None:
        logger.critical(f"Running EMERGENCY analysis for {symbol} (mode={self.mode.value})")
        telegram_collector.run(hours=1)
        self.run_analysis(symbol, is_emergency=True)


orchestrator = Orchestrator()
