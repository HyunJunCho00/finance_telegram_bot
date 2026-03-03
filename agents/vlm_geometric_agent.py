import json
from typing import Optional
from .ai_router import ai_client
from loguru import logger

class VLMGeometricAgent:
    """Visual chart specialist. Sole agent that reads the chart image.
    Judge receives only this agent's structured text output — no raw chart.
    Model: gemini-3.1-pro-preview (via role='vlm_geometric')."""

    SYSTEM_PROMPT = """You are a master technical analyst and liquidity sniper with perfect spatial reasoning.
You are the ONLY agent that sees the chart. Your structured output is what the Judge uses for all visual context.
Be precise with price levels — vague descriptions are useless to a decision-making system.

The chart contains: price action (candlesticks), technical overlays (FVGs, Order Blocks, Fibonacci),
CVD (Cumulative Volume Delta), funding rate, and Open Interest.

Your task is to answer specific questions based on the visual evidence.
Prioritize 'ACTIVE' levels (unmitigated gaps, nearest order blocks, current price contact points).

Output strictly JSON with no markdown. All price fields must be actual numbers from the chart.

Schema:
{
  "anomaly": "geometric_trap | fake_breakout | exhaustion_top | capitulation_bottom | clean_breakout | accumulation_range | none",
  "directional_bias": "BULLISH | BEARISH | NEUTRAL",
  "confidence": float (0.0 to 1.0),
  "key_levels": {
    "nearest_order_block": float,
    "nearest_unmitigated_fvg": float,
    "liquidation_pool_above": float,
    "liquidation_pool_below": float,
    "volume_poc": float
  },
  "fibonacci_context": {
    "test_level": "e.g. 61.8%",
    "anchor_high": float,
    "anchor_low": float,
    "confluence": "string"
  },
  "pattern": "H&S | inv_H&S | double_top | double_bottom | bull_flag | bear_flag | ascending_triangle | wedge | none | other",
  "cvd_signal": "divergence_bullish | divergence_bearish | confirming | neutral",
  "rationale": "Concrete explanation answering: 1. Is price inside/above/below the nearest OB? 2. Is the nearest FVG active? 3. Are Fib anchors logical?"
}"""

    def analyze(self, chart_image_b64: str, mode: str = "SWING", symbol: str = "UNKNOWN", current_price: Optional[float] = None) -> dict:
        if not chart_image_b64:
            return self._default()

        user_message = (
            f"Symbol: {symbol}\n"
            f"Current Price: {current_price if current_price else 'See Chart'}\n"
            f"Trading Mode: {mode}\n\n"
            "Analyze every panel. Answer these specific questions in your rationale:\n"
            "1. Is current price INSIDE, ABOVE, or BELOW the nearest highlighted Order Block?\n"
            "2. Which visible Fair Value Gaps (FVG) are UNMITIGATED (price has not returned to fill them)?\n"
            "3. What are the exact anchor points (High/Low) for the drawn Fibonacci levels?\n"
            "4. Does CVD show a clear divergence at the recent swing extreme?\n"
            "\nReturn JSON only."
        )
        try:
            response = ai_client.generate_response(
                system_prompt=self.SYSTEM_PROMPT,
                user_message=user_message,
                temperature=0.1,
                max_tokens=1024,
                chart_image_b64=chart_image_b64,
                role="vlm_geometric",
            )
            s, e = response.find('{'), response.rfind('}') + 1
            if s != -1 and e > s:
                return json.loads(response[s:e])
        except Exception as exc:
            logger.error(f"VLM Geometric analyst error: {exc}")
        return self._default()

    def _default(self) -> dict:
        return {
            "anomaly": "none",
            "directional_bias": "NEUTRAL",
            "confidence": 0,
            "key_levels": {
                "nearest_order_block": None,
                "nearest_unmitigated_fvg": None,
                "liquidation_pool_above": None,
                "liquidation_pool_below": None,
                "volume_poc": None,
            },
            "fibonacci_context": {
                "test_level": "unknown",
                "anchor_high": None,
                "anchor_low": None,
                "confluence": "No chart provided",
            },
            "pattern": "none",
            "cvd_signal": "neutral",
            "rationale": "Analysis failed or no chart provided",
        }

vlm_geometric_agent = VLMGeometricAgent()
