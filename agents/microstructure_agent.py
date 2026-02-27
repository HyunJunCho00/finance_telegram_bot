import json
from agents.claude_client import claude_client
from loguru import logger

class MicrostructureAgent:
    """Specialized agent for analyzing orderbook depth, spread, and slippage."""
    
    SYSTEM_PROMPT = """You are an orderbook and market microstructure expert.
Your job is to identify fake walls (spoofing) and real momentum before price moves.

You will be given:
- Current Trading Mode (SWING or POSITION)
- Microstructure Context (Spread, orderbook imbalance, 100k slippage)

Output strictly JSON with no markdown formatting.
Schema:
{
  "anomaly": "string (e.g., microstructure_imbalance, fake_wall, none)",
  "confidence": float (0.0 to 1.0),
  "directional_bias": "UP", "DOWN", or "NEUTRAL",
  "rationale": "short explanation of orderbook pressure"
}"""

    def analyze(self, microstructure_context: str, mode: str = "SWING") -> dict:
        user_message = f"Trading Mode: {mode}\n\nMicrostructure Context:\n{microstructure_context}\n\nAnalyze and return JSON."
        try:
            response = claude_client.generate_response(
                system_prompt=self.SYSTEM_PROMPT,
                user_message=user_message,
                temperature=0.2,
                max_tokens=500,
                role="microstructure"
            )
            
            s, e = response.find('{'), response.rfind('}') + 1
            if s != -1 and e > s:
                return json.loads(response[s:e])
        except Exception as e:
            logger.error(f"Microstructure analyst error: {e}")
        return {"anomaly": "none", "confidence": 0, "directional_bias": "NEUTRAL", "rationale": "Analysis failed"}

microstructure_agent = MicrostructureAgent()
