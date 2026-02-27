import json
from agents.claude_client import claude_client
from loguru import logger

class VLMGeometricAgent:
    """Specialized agent for analyzing chart geometry (fibonacci, trendlines) and heatmaps using Vision capabilities."""
    
    SYSTEM_PROMPT = """You are a master technical analyst and liquidity sniper with perfect spatial reasoning.
Your job is to look at the provided chart (which contains price action, geometric lines, and a liquidation heatmap).

Identify where the retail traders are placing their stop losses (traps) and where the visual geometric confluences are.
Keep the Current Trading Mode (SWING or POSITION) in mind. A geometric confluence on a POSITION chart is much stronger and takes longer to play out than on a SWING chart.

Output strictly JSON with no markdown formatting.
Schema:
{
  "anomaly": "string (e.g., geometric_trap, fake_breakout, none)",
  "confidence": float (0.0 to 1.0),
  "visual_target": float (price level of the geometric trap/liquidation pool),
  "rationale": "short explanation of the visual setup"
}"""

    def analyze(self, chart_image_b64: str, mode: str = "SWING") -> dict:
        if not chart_image_b64:
            return {"anomaly": "none", "confidence": 0, "visual_target": 0, "rationale": "No chart provided"}

        user_message = f"Trading Mode: {mode}\nAnalyze this geometric and heatmap chart. Return JSON."
        try:
            # Using the VLM designated model
            response = claude_client.generate_response(
                system_prompt=self.SYSTEM_PROMPT,
                user_message=user_message,
                temperature=0.2,
                max_tokens=600,
                chart_image_b64=chart_image_b64,
                role="vlm_geometric"
            )
            
            s, e = response.find('{'), response.rfind('}') + 1
            if s != -1 and e > s:
                return json.loads(response[s:e])
        except Exception as e:
            logger.error(f"VLM Geometric analyst error: {e}")
        return {"anomaly": "none", "confidence": 0, "visual_target": 0, "rationale": "Analysis failed"}

vlm_geometric_agent = VLMGeometricAgent()
