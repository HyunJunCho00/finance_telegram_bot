import json
from agents.claude_client import claude_client
from loguru import logger

class LiquidityAgent:
    """Specialized agent for analyzing CVD, whales, and liquidation density."""
    
    SYSTEM_PROMPT = """You are a highly specialized crypto quant analyzing Liquidity and CVD.
Your sole job is to identify "Liquidity Sweeps" and "Orderflow Inconsistencies".

You will be given:
- Current Trading Mode (SWING or POSITION)
- CVD Context (Whale accumulations vs Retail volume)
- Liquidation Context (Density of stop-losses/liquidations near current price)

Output strictly JSON with no markdown formatting.
Schema:
{
  "anomaly": "string (e.g., whale_cvd_divergence, liquidation_hunting, none)",
  "confidence": float (0.0 to 1.0),
  "target_entry": float (price level where the sweep stops and reversal begins),
  "rationale": "short explanation of the liquidity setup"
}"""

    def analyze(self, cvd_context: str, liquidation_context: str, mode: str = "SWING") -> dict:
        user_message = f"Trading Mode: {mode}\n\nCVD Context:\n{cvd_context}\n\nLiquidation Context:\n{liquidation_context}\n\nAnalyze and return JSON."
        try:
            response = claude_client.generate_response(
                system_prompt=self.SYSTEM_PROMPT,
                user_message=user_message,
                temperature=0.2,
                max_tokens=500,
                role="liquidity"
            )
            
            s, e = response.find('{'), response.rfind('}') + 1
            if s != -1 and e > s:
                return json.loads(response[s:e])
        except Exception as e:
            logger.error(f"Liquidity analyst error: {e}")
        return {"anomaly": "none", "confidence": 0, "target_entry": 0, "rationale": "Analysis failed"}

liquidity_agent = LiquidityAgent()
