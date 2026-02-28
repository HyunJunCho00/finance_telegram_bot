from agents.claude_client import claude_client
from config.settings import settings, TradingMode
from typing import Dict, Optional
from loguru import logger
import json

class MetaAgent:
    """The Market Regime Classifier.
    Analyzes macro, volatility, and trend data to determine the current 'Regime'.
    Provides trust directives to the Judge Agent to prevent black-box weighting."""

    SYSTEM_PROMPT = """You are the Chief Market Strategist at a Tier-1 Quant Fund.
Your job is to determine the current 'Market Regime' and provide reasoning to the Portfolio Manager (Judge).

REGIME TYPES:
1. BULL_MOMENTUM: Strong uptrend, high confidence, trend-following works best.
2. BEAR_MOMENTUM: Strong downtrend, aggressive selling, trend-following works best.
3. RANGE_BOUND: No clear direction, price oscillating near Point of Control (POC). Mean reversion works best.
4. VOLATILITY_PANIC: Extreme volatility, liquidity cascades, technicals often fail. Macro/Options data is king.
5. SIDEWAYS_ACCUMULATION: Low volatility, flat price, potential whale accumulation. On-chain/Sentiment data is key.

NARRATIVE INTELLIGENCE:
- You will be provided with `RAG_CONTEXT` (Synthesized market intelligence from LightRAG) and `TELEGRAM_NEWS`.
- If the narrative indicates a systemic risk (e.g., major exchange hack, security exploit, regulatory crackdown), you MUST prioritize this over technical indicators and consider a VOLATILITY_PANIC or BEAR_MOMENTUM regime.
- If the narrative indicates strong institutional adoption (e.g., ETF approval, nation-state buying), prioritize BULL_MOMENTUM.

YOUR OUTPUT (Strict JSON):
{
    "regime": "BULL_MOMENTUM" | "BEAR_MOMENTUM" | "RANGE_BOUND" | "VOLATILITY_PANIC" | "SIDEWAYS_ACCUMULATION",
    "rationale": "Clear, data-driven explanation for this classification...",
    "trust_directive": "Specific instructions on which data sources (experts) to prioritize/discount in this regime.",
    "risk_budget_pct": 0-100, // Maximum percentage of normal allocation allowed in this regime (e.g., 10 for PANIC, 100 for BULL).
    "risk_bias": "AGGRESSIVE" | "NEUTRAL" | "CONSERVATIVE"
}"""

    def classify_regime(
        self,
        market_data_compact: str,
        deribit_context: str,
        funding_context: str,
        macro_context: str,
        rag_context: str = "",
        telegram_news: str = "",
        mode: TradingMode = TradingMode.SWING
    ) -> Dict:
        """Determines the current market regime based on multi-source data."""
        
        user_message = f"""MARKET DATA SNAPSHOT:
{market_data_compact}

DERIVATIVES & VOLATILITY (DERIBIT):
{deribit_context}

FUNDING & LIQUIDITY:
{funding_context}

MACRO CONTEXT:
{macro_context}

NARRATIVE & NEWS (RAG):
{rag_context}

RECENT TELEGRAM:
{telegram_news}

MODE: {mode.value.upper()}

Analyze the data and classify the current Market Regime. Provide a clear rationale and trust instructions for the PM."""

        try:
            # Meta-Agent uses Gemini 3.1 Flash for cost-effective but intelligent classification
            response = claude_client.generate_response(
                system_prompt=self.SYSTEM_PROMPT,
                user_message=user_message,
                temperature=0.2,
                max_tokens=800,
                role="macro", # Routes to flash/pro based on settings
            )

            # Robust JSON parsing
            return self._parse_response(response)

        except Exception as e:
            logger.error(f"Meta-Agent classification error: {e}")
            return self._default_regime()

    def _parse_response(self, response: str) -> Dict:
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(response[start:end])
        except Exception as e:
            logger.error(f"Meta-Agent JSON parse error: {e}")
        return self._default_regime()

    def _default_regime(self) -> Dict:
        return {
            "regime": "RANGE_BOUND",
            "rationale": "Defaulting to Range Bound due to system error.",
            "trust_directive": "Prioritize balanced expert consensus.",
            "risk_budget_pct": 50,
            "risk_bias": "NEUTRAL"
        }

meta_agent = MetaAgent()

if __name__ == "__main__":
    # Test block
    test_result = meta_agent.classify_regime("Price: 60000, EMA200: 55000, ADX: 30", "DVOL: 45", "Funding: 0.01%", "Nasdaq: Up")
    print(json.dumps(test_result, indent=2))
