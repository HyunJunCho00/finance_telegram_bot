from agents.claude_client import claude_client
from config.settings import settings, TradingMode
from typing import Dict, Optional
from loguru import logger


class BullishAgent:
    """Bullish opportunity analyst. Receives raw market facts and finds bullish evidence.
    NO chart images - text only (cost optimization). Only Judge gets the chart."""

    SWING_PROMPT = """You are a professional crypto swing/position trader analyzing for bullish opportunities.

You receive raw market data across 1h, 4h, and 1d timeframes. Your job:
- Find any bullish evidence using YOUR OWN expertise
- Check Fibonacci retracement levels (38.2%, 50%, 61.8%) for potential entry zones
- Evaluate funding rate data (extreme negative = potential capitulation bottom)
- Look at OBV and CMF for accumulation signals
- Check if price is near diagonal support or volume profile POC
- Assess RSI divergences across timeframes
- Analyze CVD (Cumulative Volume Delta) for hidden accumulation
  - Rising CVD + flat price = whale accumulation (bullish)
- Check Global OI breakdown (Binance+Bybit+OKX) for positioning
- Use Perplexity market narrative for macro context
- Use RAG relationship context for connected events
- Consider news sentiment impact over days/weeks

You are a FACT INTERPRETER. We give you numbers, you give us your honest analysis.
If there is no bullish case, say so clearly. Be specific with price levels."""

    SCALP_PROMPT = """You are a professional crypto day trader analyzing for short-term bullish setups.

You receive raw market data across 1m, 5m, 15m, and 1h timeframes. Your job:
- Find quick bullish setups using YOUR OWN expertise
- Check VWAP and Keltner Channel positions for mean-reversion entries
- Evaluate volume delta proxy for buying pressure
- Analyze CVD data for real-time accumulation signals
- Look at Stochastic and RSI on 5m/15m for oversold bounces
- Check funding rate for carry cost/benefit direction
- Check Global OI for overall market positioning
- Assess immediate support levels on 15m
- Use market narrative for any imminent catalysts

You are a FACT INTERPRETER. We give you numbers, you give us your honest analysis.
If there is no short-term bullish setup, say so clearly. Be specific with price levels and timeframes."""

    def __init__(self):
        pass

    def analyze(self, market_data_compact: str, news_summary: str,
                funding_context: str, mode: TradingMode = TradingMode.SWING) -> str:
        """Analyze market for bullish evidence. TEXT ONLY - no images."""
        prompt = self.SWING_PROMPT if mode == TradingMode.SWING else self.SCALP_PROMPT

        user_message = f"""Market Data:
{market_data_compact}

Derivatives Context:
{funding_context}

Context & News:
{news_summary}

Provide your bullish analysis."""

        try:
            response = claude_client.generate_response(
                system_prompt=prompt,
                user_message=user_message,
                temperature=0.5,
                max_tokens=2000,
                chart_image_b64=None,  # Never send image - cost optimization
                role="bullish",
            )
            logger.info("Bullish agent analysis completed")
            return response
        except Exception as e:
            logger.error(f"Bullish agent error: {e}")
            return "Bullish analysis unavailable"


bullish_agent = BullishAgent()
