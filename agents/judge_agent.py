from agents.claude_client import claude_client
from config.settings import settings, TradingMode
from typing import Dict, Optional
from config.database import db
from executors.post_mortem import retrieve_similar_memories
from loguru import logger
import json


class JudgeAgent:
    """The final decision maker. Uses the best available model (Opus).
    ONLY agent that receives chart image (SWING mode, cost optimization).
    AI decides everything - we just deliver data."""

    BASE_PROMPT = """You are a senior crypto portfolio manager making {mode_upper} trading decisions.

You receive ALL available data:
1. Multi-timeframe indicators (1h, 4h, 1d) with Fibonacci levels
2. Structural analysis (support/resistance, divergences, volume profile)
3. Funding rate + Global OI (Binance+Bybit+OKX) + CVD (volume delta)
4. Perplexity market narrative (WHY price is moving)
5. LightRAG relationship context (connected events and entities)
6. Expert Agent analyses from the Blackboard (Liquidity, Microstructure, Macro/Options)
7. Your previous decision (maintain consistency unless data clearly changed)
8. Chart image (if provided) - Uses full history for indicators to prevent distortion. Visual window zooms: SWING (~6 months for trend clarity), POSITION (~5 years to capture previous ATHs and major historical resistance).

YOUR JOB: Make a FINAL trading decision using YOUR OWN judgment.

{mode_specific_rules}

Output your decision as JSON:
{
  "decision": "LONG" | "SHORT" | "HOLD" | "CANCEL_AND_CLOSE",
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
CRITICAL V5 RULE: You will be given a list of ACTIVE_ORDERS (e.g. pending DCA chunks). If the market paradigm shifts against your active orders, you MUST output "decision": "CANCEL_AND_CLOSE". If you just want to add to an existing order, you can output LONG/SHORT again.
Be aware of your previous decision for consistency."""

    SWING_RULES = """Professional SWING trading principles you should consider:
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
- Hold period: days to weeks"""

    POSITION_RULES = """Professional POSITION trading principles you should consider:
- Top-down analysis: 1w determines macro bias, 1d identifies structural shifts
- Macro cycles and previous All-Time Highs (ATH) act as absolute boundaries
- Fundamental shifts and narrative multi-month trends are more important than daily orderbook noise
- Extreme negative funding over long periods = generational bottom; Extreme positive over months = late cycle
- DVOL (Deribit Volatility Index) spikes indicate capitulation/bottoms
- Position sizing: 5-15% of Kelly criterion (highly conservative, room for deep drawdowns)
- Leverage: 1x (Spot) or maximum 1.5x. Liquidation must be nearly impossible.
- Hold period: months to years"""

    DEBATE_APPENDIX = """
Swarm reasoning controls & Data Trust Hierarchy:
- Synthesize all expert insights from the Blackboard into a cohesive probability map.
- CONVICTION HIERARCHY (Resolve deadlocks using this rule):
  1. For SCALP/Short-term: Microstructure (Orderbook/Slippage) > Liquidity (CVD) > Macro > Visual Geometry.
  2. For SWING/Position: Macro/Options > Liquidity (CVD) > Visual Geometry > Microstructure.
- Weigh conflicting evidence against the Hierarchy above.
- Explicitly list strongest pro-trade and strongest anti-trade factors in key_factors.
- Do not reward agreement itself; reward evidence quality and falsifiability.
- If conflict remains unresolved and violates Hierarchy rules, choose HOLD with clear uncertainty rationale.
"""

    def __init__(self):
        pass

    def get_previous_decision(self, symbol: str = "BTCUSDT") -> Optional[Dict]:
        """[FIX CRITICAL-6] symbol parameter added — was NameError (symbol not in scope)."""
        try:
            latest_report = db.get_latest_report(symbol=symbol)
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
        blackboard: Dict[str, dict],
        funding_context: str,
        chart_image_b64: Optional[str] = None,
        mode: TradingMode = TradingMode.SWING,
        feedback_text: str = "",
        active_orders: list = [],
        open_positions: str = "",
        symbol: str = "BTCUSDT"
    ) -> Dict:
        mode_str = mode.value.upper()
        mode_rules = self.POSITION_RULES if mode == TradingMode.POSITION else self.SWING_RULES
        
        prompt = f"{self.BASE_PROMPT.format(mode_upper=mode_str, mode_specific_rules=mode_rules)}\n\n{self.DEBATE_APPENDIX}"
        previous_decision = self.get_previous_decision(symbol=symbol)

        previous_context = "No previous decision available."
        if previous_decision:
            previous_context = f"""Previous Decision:
- Position: {previous_decision.get('decision', 'N/A')}
- Allocation: {previous_decision.get('allocation_pct', 0)}%
- Leverage: {previous_decision.get('leverage', 1)}x
- Hold Duration: {previous_decision.get('hold_duration', 'N/A')}
- Confidence: {previous_decision.get('confidence', 0)}%
- Reasoning: {previous_decision.get('reasoning', 'N/A')[:300]}"""

        # --- Episodic Memory Retrieval ---
        # Queries the JSONL vector DB placeholder for past identical patterns.
        try:
            sample_anomalies = [k for k in blackboard.keys()]
            symbol_detected = symbol
            episodic_memory = retrieve_similar_memories(sample_anomalies, symbol_detected)
        except Exception as e:
            logger.error(f"Failed to query memory: {e}")
            episodic_memory = "Vector DB RAG Search: System running in Bootstrap mode."

        bb_str = json.dumps(blackboard, indent=2)

        user_message = f"""MARKET DATA:
{market_data_compact}

DERIVATIVES CONTEXT:
{funding_context}

BLACKBOARD (EXPERT ANALYSES):
{bb_str}

EPISODIC MEMORY (PAST SIMILAR TRADES):
{episodic_memory}

OPEN POSITIONS (CURRENTLY HELD):
{open_positions if open_positions else "No open positions."}

ACTIVE DCA/TWAP ORDERS (V5 EXECUTION DESK PENDING):
{json.dumps(active_orders, indent=2) if active_orders else "No active pending orders."}

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
                role="judge",
            )

            decision = self._parse_decision(response)
            logger.info(f"Judge decision: {decision.get('decision', 'UNKNOWN')} (mode={mode.value})")
            return decision

        except Exception as e:
            logger.error(f"Judge agent error: {e}")
            return self._default_decision()

    def _parse_decision(self, response: str) -> Dict:
        """Parse JSON from LLM response. Uses bracket-depth counting to find
        the correct outermost JSON object, even if the LLM includes nested
        JSON examples inside string values like 'reasoning'."""
        try:
            # Strategy: find first '{', then count bracket depth to find matching '}'
            start_idx = response.find('{')
            if start_idx == -1:
                return self._default_decision()

            depth = 0
            in_string = False
            escape_next = False
            for i in range(start_idx, len(response)):
                c = response[i]
                if escape_next:
                    escape_next = False
                    continue
                if c == '\\':
                    escape_next = True
                    continue
                if c == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        json_str = response[start_idx:i + 1]
                        return json.loads(json_str)

            # Fallback: simple rfind (original behavior)
            end_idx = response.rfind('}') + 1
            if end_idx > start_idx:
                return json.loads(response[start_idx:end_idx])

            return self._default_decision()
        except Exception as e:
            logger.error(f"Decision parsing error: {e}")
            return self._default_decision()

    def _default_decision(self) -> Dict:
        return {
            "decision": "HOLD",
            "allocation_pct": 0,
            "leverage": 1,
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "hold_duration": "N/A",
            "reasoning": "Error in decision-making process, defaulting to HOLD",
            "confidence": 0,
            "key_factors": []
        }


judge_agent = JudgeAgent()
