import json
from agents.claude_client import claude_client
from loguru import logger

class MacroOptionsAgent:
    """Specialized agent for analyzing Deribit options data and Macro economic indicators."""
    
    SYSTEM_PROMPT = """You are a smart-money options and macro quant.
Your job is to read institutional hedging and systemic regime changes.

You will be given:
- Deribit Context (DVOL, PCR, IV Term Structure, 25d Skew)
- Macro Context (DGS10, DXY, Nasdaq, etc.)

Output strictly JSON with no markdown formatting.
Schema:
{
  "anomaly": "string (e.g., options_panic, macro_divergence, none)",
  "options_bias": "BULLISH", "BEARISH", or "NEUTRAL",
  "confidence": float (0.0 to 1.0),
  "rationale": "short explanation of institutional positioning"
}"""

    def analyze(self, deribit_context: str, macro_context: str) -> dict:
        user_message = f"Deribit Context:\n{deribit_context}\n\nMacro Context:\n{macro_context}\n\nAnalyze and return JSON."
        try:
            response = claude_client.generate_response(
                system_prompt=self.SYSTEM_PROMPT,
                user_message=user_message,
                temperature=0.2,
                max_tokens=500,
                role="macro"
            )
            
            s, e = response.find('{'), response.rfind('}') + 1
            if s != -1 and e > s:
                return json.loads(response[s:e])
        except Exception as e:
            logger.error(f"Macro/Options analyst error: {e}")
        return {"anomaly": "none", "options_bias": "NEUTRAL", "confidence": 0, "rationale": "Analysis failed"}

macro_options_agent = MacroOptionsAgent()
