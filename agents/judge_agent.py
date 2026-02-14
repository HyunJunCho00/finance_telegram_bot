from agents.claude_client import claude_client
from config.settings import settings, TradingMode
from typing import Dict, Optional
from config.database import db
from loguru import logger
import json


class JudgeAgent:
    """The final decision maker. Uses the best available model (Opus).
    ONLY agent that receives chart image (SWING mode, cost optimization).
    AI decides everything - we just deliver data."""

    SWING_PROMPT = """You are a senior crypto portfolio manager making SWING/POSITION trading decisions.

You receive ALL available data:
1. Multi-timeframe indicators (1h, 4h, 1d) with Fibonacci levels
2. Structural analysis (support/resistance, divergences, volume profile)
3. Funding rate + Global OI (Binance+Bybit+OKX) + CVD (volume delta)
4. Perplexity market narrative (WHY price is moving)
5. LightRAG relationship context (connected events and entities)
6. Bull analyst argument, Bear analyst argument, Risk manager assessment
7. Your previous decision (maintain consistency unless data clearly changed)
8. Chart image (if provided - use for visual pattern confirmation)

YOUR JOB: Make a FINAL trading decision using YOUR OWN judgment.

Professional swing trading principles you should consider:
- Top-down analysis: 1d determines bias, 4h identifies setup, 1h confirms entry
- Fibonacci 38.2%/50%/61.8% are key retracement entry zones
- Extreme funding rates are contrarian signals (high positive = potential top)
- OI-price divergence warns of fragile moves (rising OI + flat price = danger)
- Volume profile POC acts as price magnet
- CVD divergence: rising CVD + flat price = accumulation; falling CVD + rising price = distribution
- Global OI: sum across 3 exchanges eliminates single-exchange noise
- Market narrative from Perplexity provides the "WHY" behind moves
- Position sizing: 10-25% of Kelly criterion (conservative)
- Leverage: 1-3x maximum for swing trades
- Hold period: days to weeks

Output your decision as JSON:
{
  "decision": "LONG" | "SHORT" | "HOLD",
  "allocation_pct": 0-100,
  "leverage": 1-3,
  "entry_price": float,
  "stop_loss": float,
  "take_profit": float,
  "hold_duration": "hours/days/weeks estimate",
  "reasoning": "detailed reasoning",
  "confidence": 0-100,
  "key_factors": ["factor1", "factor2", ...]
}

You have FULL AUTONOMY. If the market is unclear, HOLD is always valid.
Be aware of your previous decision for consistency."""

    SCALP_PROMPT = """You are a senior crypto day trader making SCALP/SHORT-TERM trading decisions.

You receive ALL available data:
1. Multi-timeframe indicators (1m, 5m, 15m, 1h)
2. Scalp-specific data (Keltner, VWAP, volume delta, fast stochastic)
3. Funding rate + Global OI (3 exchanges) + CVD (real-time flow)
4. Market narrative (Perplexity) + RAG relationship context
5. Bull/Bear/Risk analyst arguments
6. Your previous decision

YOUR JOB: Make a FINAL short-term trading decision using YOUR OWN judgment.

Professional scalping principles you should consider:
- VWAP is king for intraday mean-reversion
- Keltner Channel touches with RSI confirmation
- Volume delta shows real-time buying/selling pressure
- Per-trade risk: 0.5-1% max
- Tight stops, quick exits
- Trade WITH the funding direction when possible (earn funding instead of paying)
- Leverage: 1-5x for scalps (tight stops justify higher leverage)
- Hold: minutes to hours

Output your decision as JSON:
{
  "decision": "LONG" | "SHORT" | "HOLD",
  "allocation_pct": 0-100,
  "leverage": 1-5,
  "entry_price": float,
  "stop_loss": float,
  "take_profit": float,
  "hold_duration": "minutes/hours estimate",
  "reasoning": "detailed reasoning",
  "confidence": 0-100,
  "key_factors": ["factor1", "factor2", ...]
}

You have FULL AUTONOMY. Move fast but be precise."""

    def __init__(self):
        pass

    def get_previous_decision(self) -> Optional[Dict]:
        try:
            latest_report = db.get_latest_report()
            if latest_report and latest_report.get('final_decision'):
                fd = latest_report['final_decision']
                if isinstance(fd, str):
                    return json.loads(fd)
                return fd
            return None
        except Exception as e:
            logger.error(f"Error fetching previous decision: {e}")
            return None

    def make_decision(
        self,
        market_data_compact: str,
        bull_opinion: str,
        bear_opinion: str,
        risk_assessment: str,
        funding_context: str,
        chart_image_b64: Optional[str] = None,
        mode: TradingMode = TradingMode.SWING,
        feedback_text: str = "",
    ) -> Dict:
        prompt = self.SWING_PROMPT if mode == TradingMode.SWING else self.SCALP_PROMPT
        previous_decision = self.get_previous_decision()

        previous_context = "No previous decision available."
        if previous_decision:
            previous_context = f"""Previous Decision:
- Position: {previous_decision.get('decision', 'N/A')}
- Allocation: {previous_decision.get('allocation_pct', 0)}%
- Leverage: {previous_decision.get('leverage', 1)}x
- Hold Duration: {previous_decision.get('hold_duration', 'N/A')}
- Confidence: {previous_decision.get('confidence', 0)}%
- Reasoning: {previous_decision.get('reasoning', 'N/A')[:300]}"""

        user_message = f"""MARKET DATA:
{market_data_compact}

DERIVATIVES CONTEXT:
{funding_context}

BULLISH ANALYST:
{bull_opinion}

BEARISH ANALYST:
{bear_opinion}

RISK MANAGER:
{risk_assessment}

{previous_context}
{feedback_text}

Make your trading decision. Output as JSON."""

        try:
            # Judge uses the best model (Opus)
            # Only Judge gets the chart image (SWING mode only)
            response = claude_client.generate_response(
                system_prompt=prompt,
                user_message=user_message,
                temperature=0.2,
                max_tokens=3000,
                chart_image_b64=chart_image_b64,
                use_premium=True,
            )

            decision = self._parse_decision(response)
            logger.info(f"Judge decision: {decision.get('decision', 'UNKNOWN')} (mode={mode.value})")
            return decision

        except Exception as e:
            logger.error(f"Judge agent error: {e}")
            return self._default_decision()

    def _parse_decision(self, response: str) -> Dict:
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            return self._default_decision()
        except Exception as e:
            logger.error(f"Decision parsing error: {e}")
            return self._default_decision()

    def _default_decision(self) -> Dict:
        return {
            "decision": "HOLD",
            "allocation_pct": 0,
            "leverage": 1,
            "entry_price": 0,
            "stop_loss": 0,
            "take_profit": 0,
            "hold_duration": "N/A",
            "reasoning": "Error in decision-making process, defaulting to HOLD",
            "confidence": 0,
            "key_factors": []
        }


judge_agent = JudgeAgent()
