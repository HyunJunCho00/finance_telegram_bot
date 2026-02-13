from typing import Dict, Optional
from config.database import db
from config.settings import settings, TradingMode
from processors.math_engine import math_engine
from processors.chart_generator import chart_generator
from collectors.telegram_collector import telegram_collector
from agents.bullish_agent import bullish_agent
from agents.bearish_agent import bearish_agent
from agents.risk_agent import risk_agent
from agents.judge_agent import judge_agent
from executors.report_generator import report_generator
from loguru import logger
import json


class Orchestrator:
    """Orchestrates the multi-agent analysis pipeline.

    Cost optimization strategy:
    - Bull/Bear/Risk agents: TEXT ONLY (no images, compact data format)
    - Judge agent: Gets chart image ONLY in SWING mode (512x512 = ~1024 tokens)
    - SCALP mode: zero images (speed + cost)
    - Compact text format saves ~40-60% tokens vs JSON
    """

    def __init__(self):
        self.symbols = ['BTCUSDT', 'ETHUSDT']

    @property
    def mode(self) -> TradingMode:
        return settings.trading_mode

    def _get_funding_data(self, symbol: str) -> Dict:
        try:
            response = db.client.table("funding_data")\
                .select("*")\
                .eq("symbol", symbol)\
                .order("timestamp", desc=True)\
                .limit(1)\
                .execute()
            if response.data:
                return response.data[0]
        except Exception as e:
            logger.error(f"Funding data fetch error for {symbol}: {e}")
        return {}

    def run_analysis(self, symbol: str, is_emergency: bool = False) -> Dict:
        mode = self.mode
        logger.info(f"Starting {'EMERGENCY' if is_emergency else 'SCHEDULED'} {mode.value.upper()} analysis for {symbol}")

        # Fetch 1m data (amount depends on mode)
        df = db.get_latest_market_data(symbol, limit=settings.candle_limit)

        if df.empty:
            logger.error(f"No market data for {symbol}")
            return {}

        # ─── 1. Technical Analysis (mode-specific) ───
        market_data = math_engine.analyze_market(df, mode)

        # Format as compact text (saves ~40-60% tokens)
        market_data_compact = math_engine.format_compact(market_data)

        # ─── 2. Funding/OI Context ───
        raw_funding = self._get_funding_data(symbol)
        funding_context_data = math_engine.analyze_funding_context(raw_funding)
        funding_context = json.dumps(funding_context_data, default=str) if funding_context_data else "No funding data."

        # ─── 3. Chart Image (SWING mode + Judge only) ───
        chart_image_b64 = None
        chart_bytes = None
        if settings.should_use_chart:
            chart_bytes = chart_generator.generate_chart(df, market_data, symbol, mode)
            if chart_bytes:
                # Always low-res for cost (512x512 = ~1024 tokens)
                chart_bytes_for_vlm = chart_generator.resize_for_low_res(chart_bytes)
                chart_image_b64 = chart_generator.chart_to_base64(chart_bytes_for_vlm)
                logger.info(f"Chart for Judge: {len(chart_bytes_for_vlm)} bytes (low-res)")

        # ─── 4. News ───
        news_hours = 1 if is_emergency else 4
        news_data = db.get_recent_telegram_messages(hours=news_hours)
        news_summary = "\n".join([
            f"[{msg['channel']}] {msg['text'][:200]}"
            for msg in news_data[:10]
        ]) if news_data else "No significant news."

        # ─── 5. Run Agents (text-only for Bull/Bear/Risk) ───
        bull_opinion = bullish_agent.analyze(
            market_data_compact, news_summary, funding_context, mode)

        bear_opinion = bearish_agent.analyze(
            market_data_compact, news_summary, funding_context, mode)

        risk_assessment = risk_agent.analyze(
            market_data_compact, bull_opinion, bear_opinion, funding_context, mode)

        # ─── 6. Judge (Opus, gets chart in SWING mode) ───
        final_decision = judge_agent.make_decision(
            market_data_compact, bull_opinion, bear_opinion, risk_assessment,
            funding_context=funding_context,
            chart_image_b64=chart_image_b64,
            mode=mode,
        )

        # ─── 7. Report ───
        report = report_generator.generate_report(
            symbol=symbol,
            market_data=market_data,
            bull_opinion=bull_opinion,
            bear_opinion=bear_opinion,
            risk_assessment=risk_assessment,
            final_decision=final_decision,
            funding_data=raw_funding,
            mode=mode,
        )

        if report:
            # Send full-res chart to Telegram (human viewing), not the low-res VLM version
            report_generator.notify(report, chart_bytes=chart_bytes, mode=mode)

        logger.info(f"Analysis completed for {symbol}: {final_decision.get('decision', 'N/A')} ({mode.value})")
        return final_decision

    def run_scheduled_analysis(self) -> None:
        logger.info(f"Running scheduled analysis (mode={self.mode.value})")
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
