from agents.claude_client import claude_client
from config.settings import settings, TradingMode
from typing import Dict, Optional
from loguru import logger


class BearishAgent:
    """Bearish risk analyst. Receives raw market facts and finds bearish evidence.
    NO chart images - text only (cost optimization). Only Judge gets the chart."""

    SWING_PROMPT = """You are a professional crypto swing/position trader analyzing for bearish risks and downside.

You receive raw market data across 1h, 4h, and 1d timeframes. Your job:
- Find any bearish evidence using YOUR OWN expertise
- Check Fibonacci extension levels for potential resistance/reversal zones
- Evaluate funding rate data (extreme positive = overleveraged longs, potential top)
- Look at OBV and CMF for distribution signals
- Check if price is near diagonal resistance or volume profile gaps
- Assess RSI divergences (bearish) across timeframes
- Look for death cross signals (SMA50 below SMA200)
- Consider negative news impact over days/weeks
- Check OI-price divergence (rising OI + weakening momentum = fragile)
- Analyze CVD for distribution signals (falling CVD + rising price = bearish divergence)
- Check Global OI (Binance+Bybit+OKX) for crowded positioning
- Use Perplexity market narrative for negative macro catalysts
- Use RAG relationship context for connected risk events

You are a FACT INTERPRETER. We give you numbers, you give us your honest analysis.
If there is no bearish case, say so clearly. Be specific with price levels."""

    SCALP_PROMPT = """You are a professional crypto day trader analyzing for short-term bearish setups.

You receive raw market data across 1m, 5m, 15m, and 1h timeframes. Your job:
- Find quick bearish setups using YOUR OWN expertise
- Check VWAP and Keltner Channel positions for overbought mean-reversion shorts
- Evaluate volume delta proxy for selling pressure
- Look at Stochastic and RSI on 5m/15m for overbought rejections
- Check funding rate impact on short carry cost
- Assess immediate resistance levels on 15m
- Look for rapid OI buildup (potential liquidation cascade fuel)
- Analyze CVD for real-time selling pressure
- Check Global OI for crowded long positioning
- Use market narrative for imminent negative catalysts

You are a FACT INTERPRETER. We give you numbers, you give us your honest analysis.
If there is no short-term bearish setup, say so clearly. Be specific with price levels and timeframes."""

    INDEPENDENCE_APPENDIX = """
Debate discipline (critical):
- Stay INDEPENDENT from other agents' views; do not converge for consensus.
- Include at least 2 strongest bearish facts AND at least 2 invalidation risks.
- If evidence quality is weak, explicitly say conviction is low.
- Never mirror generic wording; anchor claims to concrete levels/indicators/timeframes.
"""

    def __init__(self):
        pass

    def analyze(self, market_data_compact: str, news_summary: str,
                funding_context: str, mode: TradingMode = TradingMode.SWING) -> str:
        """Analyze market for bearish evidence. TEXT ONLY - no images."""
        base_prompt = self.SWING_PROMPT if mode == TradingMode.SWING else self.SCALP_PROMPT
        prompt = f"{base_prompt}\n\n{self.INDEPENDENCE_APPENDIX}"

        user_message = f"""Market Data:
{market_data_compact}

Derivatives Context:
{funding_context}

Context & News:
{news_summary}

Provide your bearish analysis."""

        try:
            response = claude_client.generate_response(
                system_prompt=prompt,
                user_message=user_message,
                temperature=0.5,
                max_tokens=2000,
                chart_image_b64=None,  # Never send image - cost optimization
                role="bearish",
            )
            logger.info("Bearish agent analysis completed")
            return response
        except Exception as e:
            logger.error(f"Bearish agent error: {e}")
            return "Bearish analysis unavailable"


bearish_agent = BearishAgent()
