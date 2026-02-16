"""Orchestrator: LangGraph StateGraph multi-agent analysis pipeline.

Architecture:
  LangGraph StateGraph manages the analysis flow as a directed graph.
  Each node is a processing step. Edges define the execution order.
  Conditional edges route based on state (e.g., skip chart in SCALP mode).

Graph:
  collect_data -> perplexity_search -> rag_ingest -> technical_analysis
  -> funding_context -> cvd_context -> rag_query -> self_correction
  -> bull_agent -> bear_agent -> risk_agent -> should_add_chart?
     -> (yes) generate_chart -> judge_agent
     -> (no) judge_agent
  -> generate_report -> notify

Benefits over sequential:
  - Explicit state management (TypedDict)
  - Conditional routing (chart only in SWING)
  - Error isolation per node
  - Easy to add/remove/reorder steps
  - Built-in retry support

Cost optimization:
  - Bull/Bear/Risk: Gemini Flash, TEXT ONLY, compact data format
  - Judge: Claude Opus 4.6, gets chart image ONLY in SWING mode (512x512)
  - SCALP mode: zero images (speed + cost)
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
from agents.bullish_agent import bullish_agent
from agents.bearish_agent import bearish_agent
from agents.risk_agent import risk_agent
from agents.judge_agent import judge_agent
from executors.report_generator import report_generator
from loguru import logger
import json

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("langgraph not available, using sequential fallback")


# ── State definition (LangGraph TypedDict) ──

class AnalysisState(TypedDict):
    symbol: str
    mode: str  # "day_trading", "swing", or "position"
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

    # Agent opinions
    bull_opinion: str
    bear_opinion: str
    risk_assessment: str

    # Chart
    chart_image_b64: Optional[str]
    chart_bytes: Optional[bytes]

    # Final output
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

    # Load higher timeframe data from GCS for deeper indicator history
    df_4h, df_1d, df_1w = None, None, None
    try:
        from processors.gcs_parquet import gcs_parquet_store
        if gcs_parquet_store.enabled:
            if mode == TradingMode.DAY_TRADING:
                df_4h = gcs_parquet_store.load_ohlcv("4h", symbol, months_back=2)
            elif mode == TradingMode.SWING:
                df_1d = gcs_parquet_store.load_ohlcv("1d", symbol, months_back=8)
            elif mode == TradingMode.POSITION:
                df_1d = gcs_parquet_store.load_ohlcv("1d", symbol, months_back=8)
                df_1w = gcs_parquet_store.load_ohlcv("1w", symbol, months_back=60)
    except Exception as e:
        logger.warning(f"GCS load skipped: {e}")

    market_data = math_engine.analyze_market(df, mode, df_4h=df_4h, df_1d=df_1d, df_1w=df_1w)
    compact = math_engine.format_compact(market_data)

    return {
        "df_size": len(df),
        "market_data_compact": compact,
    }


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
    """Ingest recent telegram messages into LightRAG."""
    try:
        recent_messages = db.get_telegram_messages_for_rag(days=7)
        if recent_messages:
            light_rag.ingest_batch(recent_messages)
    except Exception as e:
        logger.error(f"RAG ingestion error: {e}")
    return {}


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
        return {"funding_context": funding_str}
    except Exception as e:
        logger.error(f"Funding context error: {e}")
        return {"funding_context": "No funding data."}


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
        result = light_rag.query(coin_name, mode="hybrid")
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

def node_bull_agent(state: AnalysisState) -> dict:
    """Run bullish analyst (Gemini Flash, text-only)."""
    mode = TradingMode(state["mode"])
    full_context = _build_full_context(state)

    opinion = bullish_agent.analyze(
        state.get("market_data_compact", ""),
        full_context,
        state.get("funding_context", ""),
        mode
    )
    return {"bull_opinion": opinion}


def node_bear_agent(state: AnalysisState) -> dict:
    """Run bearish analyst (Gemini Flash, text-only)."""
    mode = TradingMode(state["mode"])
    full_context = _build_full_context(state)

    opinion = bearish_agent.analyze(
        state.get("market_data_compact", ""),
        full_context,
        state.get("funding_context", ""),
        mode
    )
    return {"bear_opinion": opinion}


def node_risk_agent(state: AnalysisState) -> dict:
    """Run risk manager (Gemini Flash, text-only)."""
    mode = TradingMode(state["mode"])
    assessment = risk_agent.analyze(
        state.get("market_data_compact", ""),
        state.get("bull_opinion", ""),
        state.get("bear_opinion", ""),
        state.get("funding_context", ""),
        mode
    )
    return {"risk_assessment": assessment}


def node_generate_chart(state: AnalysisState) -> dict:
    """Generate structure chart for all modes (for Judge VLM)."""
    symbol = state["symbol"]
    mode = TradingMode(state["mode"])
    candle_limit = settings.candle_limit

    df = db.get_latest_market_data(symbol, limit=candle_limit)
    if df.empty:
        return {"chart_image_b64": None, "chart_bytes": None}

    # Load higher TF data from GCS
    df_4h, df_1d, df_1w = None, None, None
    try:
        from processors.gcs_parquet import gcs_parquet_store
        if gcs_parquet_store.enabled:
            if mode == TradingMode.DAY_TRADING:
                df_4h = gcs_parquet_store.load_ohlcv("4h", symbol, months_back=2)
            elif mode == TradingMode.SWING:
                df_1d = gcs_parquet_store.load_ohlcv("1d", symbol, months_back=8)
            elif mode == TradingMode.POSITION:
                df_1d = gcs_parquet_store.load_ohlcv("1d", symbol, months_back=8)
                df_1w = gcs_parquet_store.load_ohlcv("1w", symbol, months_back=60)
    except Exception as e:
        logger.warning(f"GCS load for chart skipped: {e}")

    market_data = math_engine.analyze_market(df, mode, df_4h=df_4h, df_1d=df_1d, df_1w=df_1w)

    # Load liquidation data for chart markers
    liquidation_df = None
    try:
        liq_limit = settings.data_lookback_hours * 60
        liquidation_df = db.get_liquidation_data(symbol, limit=liq_limit)
    except Exception:
        pass

    chart_bytes = chart_generator.generate_chart(df, market_data, symbol, mode,
                                                  liquidation_df=liquidation_df)

    chart_image_b64 = None
    if chart_bytes:
        chart_bytes_for_vlm = chart_generator.resize_for_low_res(chart_bytes)
        chart_image_b64 = chart_generator.chart_to_base64(chart_bytes_for_vlm)
        logger.info(f"Chart for Judge ({mode.value}): {len(chart_bytes_for_vlm)} bytes (low-res)")

    return {"chart_image_b64": chart_image_b64, "chart_bytes": chart_bytes}


def node_judge_agent(state: AnalysisState) -> dict:
    """Run Judge (Claude Opus 4.6). Gets chart in SWING mode."""
    mode = TradingMode(state["mode"])
    decision = judge_agent.make_decision(
        state.get("market_data_compact", ""),
        state.get("bull_opinion", ""),
        state.get("bear_opinion", ""),
        state.get("risk_assessment", ""),
        funding_context=state.get("funding_context", ""),
        chart_image_b64=state.get("chart_image_b64"),
        mode=mode,
        feedback_text=state.get("feedback_text", ""),
    )
    return {"final_decision": decision}


def node_generate_report(state: AnalysisState) -> dict:
    """Generate and send report to Telegram."""
    symbol = state["symbol"]
    mode = TradingMode(state["mode"])

    candle_limit = settings.candle_limit
    df = db.get_latest_market_data(symbol, limit=candle_limit)
    market_data = math_engine.analyze_market(df, mode) if not df.empty else {}

    # Get raw funding for report
    try:
        response = db.client.table("funding_data")\
            .select("*").eq("symbol", symbol)\
            .order("timestamp", desc=True).limit(1).execute()
        raw_funding = response.data[0] if response.data else {}
    except Exception:
        raw_funding = {}

    report = report_generator.generate_report(
        symbol=symbol,
        market_data=market_data,
        bull_opinion=state.get("bull_opinion", ""),
        bear_opinion=state.get("bear_opinion", ""),
        risk_assessment=state.get("risk_assessment", ""),
        final_decision=state.get("final_decision", {}),
        funding_data=raw_funding,
        mode=mode,
    )

    if report:
        chart_bytes = state.get("chart_bytes")
        report_generator.notify(report, chart_bytes=chart_bytes, mode=mode)

    decision = state.get("final_decision", {}).get("decision", "N/A")
    logger.info(f"Analysis completed for {symbol}: {decision} ({mode.value})")
    return {"report": report}


# ── Helper ──

def _build_full_context(state: AnalysisState) -> str:
    """Build full context string for agents (narrative + cvd + liquidation + rag + news + feedback)."""
    parts = [
        state.get("narrative_text", ""),
        state.get("cvd_context", ""),
        state.get("liquidation_context", ""),
        state.get("microstructure_context", ""),
        state.get("macro_context", ""),
        state.get("rag_context", ""),
        f"Telegram News:\n{state.get('telegram_news', '')}",
        state.get("feedback_text", ""),
    ]
    return "\n\n".join(p for p in parts if p)


# ── Conditional edge ──

def should_generate_chart(state: AnalysisState) -> str:
    """Conditional: generate chart for all modes if enabled."""
    if settings.should_use_chart and state.get("df_size", 0) > 0:
        return "generate_chart"
    return "judge_agent"


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
    graph.add_node("bull_agent", node_bull_agent)
    graph.add_node("bear_agent", node_bear_agent)
    graph.add_node("risk_agent", node_risk_agent)
    graph.add_node("generate_chart", node_generate_chart)
    graph.add_node("judge_agent", node_judge_agent)
    graph.add_node("generate_report", node_generate_report)

    # Set entry point
    graph.set_entry_point("collect_data")

    # Data collection chain
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

    # Agent chain
    graph.add_edge("macro_context", "bull_agent")
    graph.add_edge("bull_agent", "bear_agent")
    graph.add_edge("bear_agent", "risk_agent")

    # Conditional chart generation
    graph.add_conditional_edges(
        "risk_agent",
        should_generate_chart,
        {
            "generate_chart": "generate_chart",
            "judge_agent": "judge_agent",
        }
    )
    graph.add_edge("generate_chart", "judge_agent")

    # Final decision and report
    graph.add_edge("judge_agent", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()


# ── Orchestrator class (maintains backward compatibility) ──

class Orchestrator:
    def __init__(self):
        self.symbols = ['BTCUSDT', 'ETHUSDT']
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
            "bull_opinion": "",
            "bear_opinion": "",
            "risk_assessment": "",
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
            logger.error(f"LangGraph execution error: {e}")
            return {"decision": "HOLD", "reasoning": f"LangGraph error: {e}", "confidence": 0}

    def _run_sequential(self, symbol: str, mode: TradingMode, is_emergency: bool) -> Dict:
        """Fallback: sequential execution (no LangGraph)."""
        state = {
            "symbol": symbol, "mode": mode.value,
            "is_emergency": is_emergency, "errors": [],
        }

        # Run each node sequentially
        for node_fn in [
            node_collect_data, node_perplexity_search, node_rag_ingest,
            node_funding_context, node_cvd_context, node_liquidation_context,
            node_rag_query, node_telegram_news, node_self_correction,
            node_microstructure_context, node_macro_context,
            node_bull_agent, node_bear_agent, node_risk_agent,
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
