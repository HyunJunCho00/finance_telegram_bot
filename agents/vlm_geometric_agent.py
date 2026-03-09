import json
from typing import Optional

from loguru import logger

from .ai_router import ai_client


class VLMGeometricAgent:
    """Visual chart specialist. Sole agent that reads the chart image.
    Judge receives only this agent's structured text output ??no raw chart
    """

    SYSTEM_PROMPT = """You are a master technical analyst and liquidity sniper with perfect spatial reasoning.
You are the ONLY agent that sees the chart. Your structured output is what the Judge uses for all visual context.
Be precise with price levels; vague descriptions are useless to a decision-making system.

You will receive ONE primary timeframe chart image plus a compact higher-timeframe text context.
Treat the image as the execution chart. Treat the higher-timeframe text as the directional constraint.

The chart may include price action, Fibonacci, confluence zones, liquidity levels, FVGs, Order Blocks,
Anchored VWAP, liquidations, and optional CVD/OI/Funding panels.

Your task is not to name random patterns. Your task is to decide whether this is an executable scenario:
- Is there a liquidity sweep or fake breakout?
- Is price reclaiming/rejecting a key level cleanly?
- Is there a valid execution zone with clear invalidation?
- Does the chart support the higher-timeframe scenario or force a scenario revision?

Output strictly JSON with no markdown. All price fields must be actual numbers from the chart or null.

Schema:
{
  "anomaly": "geometric_trap | fake_breakout | exhaustion_top | capitulation_bottom | clean_breakout | accumulation_range | none",
  "directional_bias": "BULLISH | BEARISH | NEUTRAL",
  "confidence": float (0.0 to 1.0),
  "trap_type": "buy_side_sweep | sell_side_sweep | fake_breakout | none",
  "sweep_side": "above_highs | below_lows | none",
  "sr_flip": "bullish_flip | bearish_flip | none",
  "retest_quality": "clean | acceptable | weak | failed | unknown",
  "scenario_shift": "confirm | revise | neutral",
  "execution": {
    "entry_zone_low": float,
    "entry_zone_high": float,
    "invalidation_level": float,
    "target_1": float,
    "target_2": float
  },
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
  "cvd_signal": "divergence_bullish | divergence_bearish | confirming | neutral | unavailable",
  "rationale": "Concrete explanation of liquidity event, execution zone, invalidation, and whether the current chart confirms or revises the higher timeframe scenario."
}"""

    def analyze(
        self,
        chart_image_b64: str,
        mode: str = "SWING",
        symbol: str = "UNKNOWN",
        current_price: Optional[float] = None,
        primary_timeframe: Optional[str] = None,
        higher_timeframe_context: str = "",
    ) -> dict:
        if not chart_image_b64:
            return self._default()

        primary_tf = primary_timeframe or ("4H" if str(mode).upper() == "SWING" else "1D")
        user_message = (
            f"Symbol: {symbol}\n"
            f"Current Price: {current_price if current_price else 'See Chart'}\n"
            f"Trading Mode: {mode}\n"
            f"Primary Chart Timeframe: {primary_tf}\n\n"
            "Higher Timeframe Context:\n"
            f"{higher_timeframe_context.strip() if higher_timeframe_context else 'None provided'}\n\n"
            "Analyze the single chart image as the execution timeframe. Use the higher timeframe context as a constraint, not as an image.\n"
            "Answer these specific questions in your rationale:\n"
            "1. Is there a buy-side sweep, sell-side sweep, or fake breakout?\n"
            "2. Is the current reaction a clean retest / SR flip or a weak acceptance?\n"
            "3. What is the best visible execution zone, invalidation, and TP1/TP2?\n"
            "4. What are the exact anchor points (High/Low) for the drawn Fibonacci levels?\n"
            "5. Does the primary chart confirm the higher timeframe scenario or force a revision?\n"
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
            "trap_type": "none",
            "sweep_side": "none",
            "sr_flip": "none",
            "retest_quality": "unknown",
            "scenario_shift": "neutral",
            "execution": {
                "entry_zone_low": None,
                "entry_zone_high": None,
                "invalidation_level": None,
                "target_1": None,
                "target_2": None,
            },
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
            "cvd_signal": "unavailable",
            "rationale": "Analysis failed or no chart provided",
        }


vlm_geometric_agent = VLMGeometricAgent()
