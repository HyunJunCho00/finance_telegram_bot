import json
from agents.claude_client import claude_client
from loguru import logger

class VLMGeometricAgent:
    """Visual chart specialist. Sole agent that reads the chart image.
    Judge receives only this agent's structured text output — no raw chart.
    Model: gemini-3.1-pro-preview (via role='vlm_geometric')."""

    SYSTEM_PROMPT = """You are a master technical analyst and liquidity sniper with perfect spatial reasoning.
You are the ONLY agent that sees the chart. Your structured output is what the Judge uses for all visual context.
Be precise with price levels — vague descriptions are useless to a decision-making system.

The chart contains: price action (candlesticks), technical overlays (Fibonacci, EMAs, volume profile),
CVD (Cumulative Volume Delta), funding rate, and a liquidation heatmap.

Keep the Current Trading Mode in mind:
- SWING: Multi-day patterns. Fibonacci retracements and swing highs/lows matter most.
- POSITION: Multi-week/month patterns. ATH boundaries, macro trend channels, and long-term POC.

Output strictly JSON with no markdown. All price fields must be actual numbers from the chart (not null, not 0 unless price is literally at 0).

Schema:
{
  "anomaly": "geometric_trap | fake_breakout | exhaustion_top | capitulation_bottom | clean_breakout | accumulation_range | none",
  "directional_bias": "BULLISH | BEARISH | NEUTRAL",
  "confidence": float (0.0 to 1.0),
  "key_levels": {
    "immediate_support": float,
    "immediate_resistance": float,
    "major_support": float,
    "major_resistance": float,
    "liquidation_pool_above": float,
    "liquidation_pool_below": float,
    "volume_poc": float
  },
  "fibonacci_context": "e.g. Price testing 61.8% retracement at $X, confluence with EMA200",
  "pattern": "H&S | inv_H&S | double_top | double_bottom | bull_flag | bear_flag | ascending_triangle | wedge | none | other",
  "cvd_signal": "divergence_bullish | divergence_bearish | confirming | neutral",
  "funding_visual": "extreme_positive | high_positive | neutral | negative | extreme_negative",
  "rationale": "Concrete explanation with specific price levels and why they matter"
}"""

    def analyze(self, chart_image_b64: str, mode: str = "SWING") -> dict:
        if not chart_image_b64:
            return self._default()

        user_message = (
            f"Trading Mode: {mode}\n"
            "Analyze every panel of this chart. Extract all visible price levels precisely. Return JSON only."
        )
        try:
            response = claude_client.generate_response(
                system_prompt=self.SYSTEM_PROMPT,
                user_message=user_message,
                temperature=0.1,
                max_tokens=900,
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
            "key_levels": {},
            "fibonacci_context": "No chart provided",
            "pattern": "none",
            "cvd_signal": "neutral",
            "funding_visual": "neutral",
            "rationale": "Analysis failed or no chart provided",
        }

vlm_geometric_agent = VLMGeometricAgent()
