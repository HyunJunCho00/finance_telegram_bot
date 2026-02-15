from agents.claude_client import claude_client
from config.settings import settings, TradingMode
from typing import Dict, List
from config.database import db
from loguru import logger


class RiskAgent:
    """Risk management specialist. Synthesizes bull/bear views, applies risk framework.
    NO chart images - text only. AI decides everything."""

    SWING_PROMPT = """You are a crypto trading risk management specialist for POSITION/SWING trades.

You receive a bullish analyst's view, a bearish analyst's view, market data, and past mistakes.

Your job (using YOUR OWN expertise):
- Assess overall risk of the current situation across 1h/4h/1d timeframes
- Consider fractional Kelly criterion principles for position sizing
  (if your edge seems weak, recommend smaller size; if strong, up to 10-25% of Kelly)
- Evaluate funding rate carry cost impact on hold duration
- Check Global OI (3 exchanges) for liquidation cascade risk
- Analyze CVD for hidden distribution/accumulation divergence
- Consider Perplexity narrative for macro risk factors
- Review past trading mistakes to avoid repeating them
- Identify what could go wrong with BOTH the bull and bear scenarios
- Recommend maximum leverage (1-3x for swing) and position size limits
- Be honest about uncertainty

You are not biased toward any direction. You focus PURELY on risk.
If both bull and bear cases are weak, say so. If signals conflict, explain clearly."""

    SCALP_PROMPT = """You are a crypto trading risk management specialist for DAY/SCALP trades.

You receive a bullish analyst's view, a bearish analyst's view, market data, and past mistakes.

Your job (using YOUR OWN expertise):
- Assess risk for SHORT-TERM positions (minutes to hours)
- Per-trade risk should be tight: max 0.5-1% of capital
- Daily loss limit awareness
- Factor in transaction costs and slippage on tight timeframes
- Check funding rate direction for carry cost/benefit
- Assess liquidation proximity risk from Global OI (3 exchanges) + CVD
- Review past mistakes

Scalping has DIFFERENT risk rules than swing trading. Tight stops, small size, high frequency.
Be honest about uncertainty."""

    DEBATE_APPENDIX = """
Debate protocol:
- Do NOT average opinions mechanically; challenge both sides explicitly.
- Provide a disagreement matrix: [Bull strongest point, Bull fatal flaw, Bear strongest point, Bear fatal flaw].
- If one side is weakly evidenced, state that clearly and reduce risk budget.
- Prefer no-trade/HOLD when uncertainty or conflict is unresolved.
"""

    def __init__(self):
        pass

    def get_recent_feedback(self, limit: int = 5) -> List[Dict]:
        try:
            return db.get_feedback_history(limit=limit)
        except Exception as e:
            logger.error(f"Error fetching feedback: {e}")
            return []

    def analyze(self, market_data_compact: str, bull_opinion: str, bear_opinion: str,
                funding_context: str, mode: TradingMode = TradingMode.SWING) -> str:
        base_prompt = self.SWING_PROMPT if mode == TradingMode.SWING else self.SCALP_PROMPT
        prompt = f"{base_prompt}\n\n{self.DEBATE_APPENDIX}"

        feedback_history = self.get_recent_feedback()
        feedback_text = "\n\n".join([
            f"Date: {f.get('created_at', 'N/A')}\n"
            f"Mistake: {f.get('mistake_summary', 'N/A')}\n"
            f"Lesson: {f.get('lesson_learned', 'N/A')}"
            for f in feedback_history
        ]) if feedback_history else "No past mistakes recorded yet."

        user_message = f"""Bullish Analyst's View:
{bull_opinion}

Bearish Analyst's View:
{bear_opinion}

Past Mistakes to Learn From:
{feedback_text}

Market Data:
{market_data_compact}

Derivatives Context:
{funding_context}

Provide your risk assessment."""

        try:
            response = claude_client.generate_response(
                system_prompt=prompt,
                user_message=user_message,
                temperature=0.3,
                max_tokens=2000,
                role="risk",
            )
            logger.info("Risk agent analysis completed")
            return response
        except Exception as e:
            logger.error(f"Risk agent error: {e}")
            return "Risk analysis unavailable"


risk_agent = RiskAgent()
