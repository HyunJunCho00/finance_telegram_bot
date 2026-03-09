from .ai_router import ai_client
from config.settings import settings, TradingMode
from typing import Dict, Optional
from config.database import db
from executors.post_mortem import retrieve_similar_memories
from loguru import logger
import json
from utils.decision_parser import extract_decision_from_response


class JudgeAgent:
    """The final decision maker. Uses the best available model
    ONLY agent that receives chart image (SWING mode, cost optimization).
    AI decides everything - we just deliver data."""

    BASE_PROMPT = """You are a senior crypto portfolio manager making {mode_upper} trading decisions.

You receive ALL available data:
1. Multi-timeframe indicators (1h, 4h, 1d) with Fibonacci levels
2. Structural analysis (support/resistance, divergences, volume profile)
3. Funding rate + Global OI (Binance+Bybit+OKX) + OI Divergence + MFI (money flow index proxy)
4. Perplexity market narrative (WHY price is moving)
5. LightRAG relationship context (connected events and entities)
6. Expert Agent analyses from the Blackboard (Liquidity, Microstructure, Macro/Options)
7. REGIME CONTEXT: Market regime and trust directives from the Meta-Agent.
8. Your previous decision (maintain consistency unless data clearly changed)
9. Chart image (if provided) - Uses full history for indicators to prevent distortion. Visual window zooms: SWING (~6 months for trend clarity), POSITION (~5 years to capture previous ATHs and major historical resistance).

YOUR JOB: Make a DRAFT trading decision. 
CRITICAL POLICY RULE: Human-defined policy engine has final authority. You must NOT rely on discretionary exceptions. If structure, stop, or reward/risk is unclear, prefer HOLD and explain why.
CRITICAL V9 RULE: You MUST perform Falsifiability Analysis BEFORE your final decision. Identify the strongest evidence AGAINST your primary bias.
If the Meta-Agent indicates a high-risk regime, you MUST demand higher confidence levels for any entry.

CRITICAL V10 RULE - PLAYBOOK GENERATION: 
Regardless of your decision (even if HOLD), you MUST generate a `monitoring_playbook` that the Hourly Monitor can use. 
- If your decision is LONG/SHORT, the `entry_conditions` should be the specific triggers that define your confirmed entry. 
- If your decision is HOLD, the `entry_conditions` should be the "What needs to happen for me to enter?" scenario.
- Use metrics: `price`, `funding_rate`, `oi_chg_pct`, `price_chg_pct_1h`, `volatility`.
- Ensure operators are one of: `>`, `<`, `>=`, `<=`, `==`.

CRITICAL EXECUTION RULE:
- Prioritize evidence in this order: Higher-timeframe structure -> liquidity event / trap -> retest quality -> derivatives confirmation -> narrative.
- You MUST think in scenarios, not predictions. Always define:
  1. primary_scenario
  2. alternate_scenario
  3. trigger_to_enter
  4. trigger_to_abort
  5. partial_tp_plan
  6. stop_to_be_rule

{mode_specific_rules}

Output your decision as JSON:
{{
  "decision": "LONG" | "SHORT" | "HOLD" | "CANCEL_AND_CLOSE",
  "allocation_pct": 0-100,
  "leverage": 1-3,
  "entry_price": float,
  "stop_loss": float,
  "take_profit": float,
  "hold_duration": "hours/days/weeks estimate",
  "reasoning": {{
    "counter_scenario": "The strongest argument for why my decision might be WRONG",
    "meta_agent_context": "How the current market regime influenced your weighting of experts",
    "technical": "MTF indicators, Structure, Fibonacci levels",
    "derivatives": "Funding, Global OI (OI_DIV status), MFI proxy, Volume Profile",
    "experts": "Blackboard expert summaries (Liq, Micro, Macro)",
    "narrative": "Perplexity narrative and RAG events",
    "onchain": "On-chain valuation / flow / network activity interpretation",
    "final_logic": "Concluding synthesis"
  }},
  "win_probability_pct": 0-100,
  "expected_profit_pct": float,
  "expected_loss_pct": float,
  "ev_rationale": "Clear math-based explanation of the Expected Value calculation.",
  "key_factors": ["short bullet 1", "short bullet 2", ...],
  "scenario_plan": {{
    "primary_scenario": "Korean sentence",
    "alternate_scenario": "Korean sentence",
    "trigger_to_enter": "Korean sentence",
    "trigger_to_abort": "Korean sentence",
    "partial_tp_plan": "Korean sentence",
    "stop_to_be_rule": "Korean sentence"
  }},
  "monitoring_playbook": {{
    "entry_conditions": [
        {{"metric": "price", "operator": ">", "value": 50000}},
        {{"metric": "funding_rate", "operator": "<", "value": 0.01}},
        {{"metric": "oi_chg_pct", "operator": ">", "value": 1.5}}
    ],
    "invalidation_conditions": [
        {{"metric": "price", "operator": "<", "value": 48000}}
    ]
  }},
  "daily_dual_plan": {{
    "swing_plan": {{
      "entry_conditions": [{{"metric": "price", "operator": ">", "value": 50000}}],
      "invalidation_conditions": [{{"metric": "price", "operator": "<", "value": 48000}}]
    }},
    "position_plan": {{
      "entry_conditions": [{{"metric": "price", "operator": ">", "value": 50000}}],
      "invalidation_conditions": [{{"metric": "price", "operator": "<", "value": 47000}}]
    }}
  }}
}}

CRITICAL: All reasoning fields including onchain, final_logic, counter_scenario, and key_factors MUST be written in Korean. 
The decision, hold_duration estimate, and JSON keys remain in English.
Return exactly one JSON object only. Do not wrap it in Markdown code fences.
Do not add any text before or after the JSON object.

You have FULL AUTONOMY. HOLD is valid only when your edge is not statistically meaningful (weak EV, weak R/R, or weak win probability).
CRITICAL V5 RULE: You will be given a list of ACTIVE_ORDERS (e.g. pending DCA chunks). If the market paradigm shifts against your active orders, you MUST output "decision": "CANCEL_AND_CLOSE". If you just want to add to an existing order, you can output LONG/SHORT again.
Be aware of your previous decision for consistency."""

    SWING_RULES = """Professional SWING trading principles you should consider:
- Top-down analysis: 1d determines bias, 4h identifies setup, 1h confirms entry
- Fibonacci 38.2%/50%/61.8% are key retracement entry zones
- Invalidation is more important than precise entry. If invalidation is unclear, choose HOLD.
- Respect liquidity events: sweep -> reclaim/reject -> retest is stronger than raw breakout chasing.
- Prefer scenario wording such as "if reclaimed", "if retest holds", "if sweep fails" over unconditional directional claims.
- Extreme funding rates are contrarian signals (high positive = potential top)
- OI divergence: rising OI + flat/falling price = fragile long squeeze risk (OI_DIV=DIVERGENCE)
- MFI proxy INFLOW (OI↑ + Price↑) confirms trend; OUTFLOW (OI↓ + Price↓) confirms liquidation
- Volume profile POC acts as price magnet
- OI-price divergence warns of fragile moves (rising OI + flat price = danger)
- Global OI: sum across 3 exchanges eliminates single-exchange noise
- Market narrative from Perplexity provides the "WHY" behind moves
- Position sizing: 10-25% of Kelly criterion (conservative)
- Leverage: 1-3x maximum for swing trades
- Stop Loss & Take Profit: Allow wider stops (e.g., 5-10%) to survive crypto volatility. However, you MUST maintain a minimum 1.5:1 or 2:1 Reward/Risk ratio. Manage the absolute risk by reducing `allocation_pct` and `leverage`, NOT by tightening the stop loss to unrealistic levels that will just get hunted.
- Hold period: days to weeks"""

    POSITION_RULES = """Professional POSITION trading principles you should consider:
- Top-down analysis: 1w determines macro bias, 1d identifies structural shifts
- Invalidation is more important than precise entry. If invalidation is unclear, choose HOLD.
- Prefer structure confirmation and liquidity sweep reversals over forcing continuation entries.
- Macro cycles and previous All-Time Highs (ATH) act as absolute boundaries
- Fundamental shifts and narrative multi-month trends are more important than daily orderbook noise
- Extreme negative funding over long periods = generational bottom; Extreme positive over months = late cycle
- DVOL (Deribit Volatility Index) spikes indicate capitulation/bottoms
- CME BASIS (BTC/ETH): Monitor the premium/discount of CME Futures vs Spot. Persistent Backwardation (Spot > Futures) indicates institutional hedging or lack of buy-side conviction. Contango (Futures > Spot) indicates institutional long-bias.
- OTC FOOTPRINT: Even if OTC is hidden, watch for "footprints":
    1. Deribit Skew: Large players hedge OTC buys with put options.
- OI Trends: OTC desks rebalancing inventory show as OI divergence spikes on public exchanges.
- Position sizing: 5-15% of Kelly criterion (highly conservative, room for deep drawdowns)
- Leverage: 1x (Spot) or maximum 1.5x. Liquidation must be nearly impossible.
- Stop Loss & Take Profit: Stops must be wide enough to survive multi-week volatility and deep drawdowns (e.g., 10-25%), but the profit target should capture major cycle moves, aiming for 3:1 or higher Reward/Risk ratio. Risk is managed entirely via minimal leverage and small portfolio allocation.
- Hold period: months to years"""

    DEBATE_APPENDIX = """
Swarm reasoning controls & Data Trust Hierarchy:
- Synthesize all expert insights from the Blackboard into a cohesive probability map.
- CONVICTION HIERARCHY (Resolve deadlocks using this rule):
  1. For SCALP/Short-term: Microstructure (Orderbook/Slippage) > Liquidity (OI/Funding/MFI) > Macro > Visual Geometry.
  2. For SWING/Position: Macro/Options > OI Divergence & Funding > Volume Profile > Visual Geometry > Microstructure.
- Weigh conflicting evidence against the Hierarchy above.
- Explicitly list strongest pro-trade and strongest anti-trade factors in key_factors.
- Do not reward agreement itself; reward evidence quality and falsifiability.
- If conflict remains unresolved and violates Hierarchy rules, choose HOLD with clear uncertainty rationale.
"""

    TRIGGER_VALIDATION_PROMPT = """You are a trigger-time execution validator.

Your authority is strictly narrower than the daily strategy judge.
- You MUST NOT invent a new thesis or reverse the playbook direction.
- You MAY only choose one of:
  1. EXECUTE: playbook still valid, execute in the original direction
  2. WAIT: playbook thesis still valid but timing/quality is not good enough yet
  3. REDUCE: playbook still valid but context is mixed, reduce size
  4. CANCEL: do not execute this trigger
  5. REPLAN: the daily thesis itself appears stale/broken and needs emergency re-planning

Return strict JSON only:
{
  "trigger_action": "EXECUTE" | "WAIT" | "REDUCE" | "CANCEL" | "REPLAN",
  "allocation_pct": 0-100,
  "leverage": 1-3,
  "reasoning": "Korean sentence",
  "replan_reason": "Korean sentence"
}

Rules:
- If the playbook source direction is LONG, you cannot output SHORT logic.
- If the playbook source direction is SHORT, you cannot output LONG logic.
- If context is stale or thesis-breaking, prefer REPLAN over inventing the opposite side.
- Use Korean for reasoning and replan_reason.
- Output exactly one JSON object only.
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
        symbol: str = "BTCUSDT",
        regime_context: Optional[Dict] = None,
        narrative_context: str = "",
        onchain_context: str = "",
    ) -> Dict:
        mode_str = mode.value.upper()
        mode_rules = self.POSITION_RULES if mode == TradingMode.POSITION else self.SWING_RULES
        # Wait, format() evaluates all curly braces. self.DEBATE_APPENDIX contains raw JSON templates!
        # Instead of formatting the whole concatenated string, format the components individually.
        base_formatted = self.BASE_PROMPT.format(mode_upper=mode_str, mode_specific_rules=mode_rules)
        prompt = f"{base_formatted}\n\n{self.DEBATE_APPENDIX}"
        previous_decision = self.get_previous_decision(symbol=symbol)

        previous_context = "No previous decision available."
        if previous_decision:
            reasoning = previous_decision.get('reasoning', 'N/A')
            if isinstance(reasoning, dict):
                # If structured, extract final_logic or join summaries
                reasoning_text = reasoning.get('final_logic', str(reasoning))
            else:
                reasoning_text = str(reasoning)

            previous_context = f"""Previous Decision:
- Position: {previous_decision.get('decision', 'N/A')}
- Allocation: {previous_decision.get('allocation_pct', 0)}%
- Leverage: {previous_decision.get('leverage', 1)}x
- Hold Duration: {previous_decision.get('hold_duration', 'N/A')}
- Confidence: {previous_decision.get('confidence', 0)}%
- Reasoning: {reasoning_text[:300]}"""

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

NARRATIVE & RAG CONTEXT:
{narrative_context if narrative_context else "Narrative: Unavailable"}

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

REGIME CONTEXT (META-AGENT TRUST DIRECTIVES):
{json.dumps(regime_context, indent=2) if regime_context else "No regime context available."}

ON-CHAIN OVERLAY:
{onchain_context if onchain_context else "On-chain Context: unavailable"}

{previous_context}
{feedback_text}

Make your trading decision. Output as JSON. Ensure the counter_scenario is thoroughly explored."""

        try:
            # Judge uses the best model (Opus)
            # Only Judge gets the chart image (SWING mode only)
            response = ai_client.generate_response(
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
            return self._default_decision(f"Judge agent error: {e}")

    def make_decision_from_snapshot(self, snapshot: Dict) -> Dict:
        snapshot = snapshot or {}
        return self.make_decision(
            market_data_compact=str(snapshot.get("market_data_compact", "") or ""),
            blackboard=snapshot.get("blackboard", {}) or {},
            funding_context="\n".join(
                filter(
                    None,
                    [
                        str(snapshot.get("funding_context", "") or ""),
                        str(snapshot.get("liquidation_context", "") or ""),
                    ],
                )
            ),
            chart_image_b64=snapshot.get("chart_image_b64"),
            mode=TradingMode(str(snapshot.get("mode", TradingMode.SWING.value)).lower()),
            feedback_text=str(snapshot.get("feedback_text", "") or ""),
            active_orders=snapshot.get("active_orders", []) or [],
            open_positions=str(snapshot.get("open_positions", "") or ""),
            symbol=str(snapshot.get("symbol", "BTCUSDT") or "BTCUSDT"),
            regime_context=snapshot.get("regime_context", {}) or {},
            narrative_context="\n\n".join(
                filter(
                    None,
                    [
                        str(snapshot.get("narrative_text", "") or ""),
                        str(snapshot.get("rag_context", "") or ""),
                        str(snapshot.get("telegram_news", "") or ""),
                    ],
                )
            ),
            onchain_context=str(snapshot.get("onchain_context", "") or ""),
        )

    def validate_trigger_against_playbook(self, snapshot: Dict, playbook_context: Optional[Dict]) -> Dict:
        snapshot = snapshot or {}
        context = dict(playbook_context or {})
        source_decision = str(context.get("source_decision", "HOLD") or "HOLD").upper()
        allowed_sides = context.get("allowed_sides", [])
        max_allocation_pct = context.get("max_allocation_pct")
        compact_snapshot = {
            "symbol": str(snapshot.get("symbol", "BTCUSDT") or "BTCUSDT"),
            "mode": str(snapshot.get("mode", TradingMode.SWING.value) or TradingMode.SWING.value),
            "market_data": snapshot.get("market_data", {}) or {},
            "funding_context": str(snapshot.get("funding_context", "") or ""),
            "liquidation_context": str(snapshot.get("liquidation_context", "") or ""),
            "narrative_text": str(snapshot.get("narrative_text", "") or ""),
            "telegram_news": str(snapshot.get("telegram_news", "") or ""),
            "onchain_context": str(snapshot.get("onchain_context", "") or ""),
            "onchain_gate": snapshot.get("onchain_gate", {}) or {},
            "regime_context": snapshot.get("regime_context", {}) or {},
            "blackboard": snapshot.get("blackboard", {}) or {},
        }
        user_message = (
            f"PLAYBOOK_CONTEXT:\n{json.dumps(context, ensure_ascii=False, indent=2)}\n\n"
            f"SNAPSHOT:\n{json.dumps(compact_snapshot, ensure_ascii=False, indent=2)}\n\n"
            f"LOCKED_DIRECTION={source_decision}\n"
            f"ALLOWED_SIDES={json.dumps(allowed_sides, ensure_ascii=False)}\n"
            f"MAX_ALLOCATION_PCT={json.dumps(max_allocation_pct, ensure_ascii=False)}\n"
        )
        try:
            response = ai_client.generate_response(
                system_prompt=self.TRIGGER_VALIDATION_PROMPT,
                user_message=user_message,
                temperature=0.1,
                max_tokens=700,
                role="judge",
            )
            parsed = extract_decision_from_response(response) or {}
        except Exception as e:
            logger.error(f"Trigger validator error: {e}")
            parsed = {}

        trigger_action = str(parsed.get("trigger_action", "WAIT") or "WAIT").upper()
        if trigger_action not in {"EXECUTE", "WAIT", "REDUCE", "CANCEL", "REPLAN"}:
            trigger_action = "WAIT"

        decision = self._decision_template()
        decision["trigger_action"] = trigger_action
        decision["playbook_context"] = context
        decision["reasoning"]["final_logic"] = str(parsed.get("reasoning", "") or "").strip()
        decision["reasoning"]["counter_scenario"] = str(parsed.get("replan_reason", "") or "").strip()
        decision["entry_price"] = ((snapshot.get("market_data", {}) or {}).get("current_price"))
        allocation_cap = context.get("max_allocation_pct")
        raw_allocation = parsed.get("allocation_pct", allocation_cap if allocation_cap is not None else 0)
        try:
            allocation_pct = float(raw_allocation or 0)
        except Exception:
            allocation_pct = 0.0
        if allocation_cap is not None:
            try:
                allocation_pct = min(allocation_pct, float(allocation_cap))
            except Exception:
                pass
        try:
            decision["leverage"] = max(1, min(3, int(float(parsed.get("leverage", 1) or 1))))
        except Exception:
            decision["leverage"] = 1

        if trigger_action in {"EXECUTE", "REDUCE"} and source_decision in {"LONG", "SHORT"}:
            decision["decision"] = source_decision
            decision["allocation_pct"] = allocation_pct
            if trigger_action == "REDUCE" and allocation_pct <= 0:
                decision["allocation_pct"] = min(5.0, float(allocation_cap or 5.0))
        elif trigger_action == "REPLAN":
            decision["decision"] = "HOLD"
            decision["replan_reason"] = str(parsed.get("replan_reason", "") or "").strip()
        elif trigger_action == "CANCEL":
            decision["decision"] = "HOLD"
        else:
            decision["decision"] = "HOLD"

        return decision

    def _parse_decision(self, response: str) -> Dict:
        """Parse JSON-like LLM output with layered recovery before fallback."""
        try:
            parsed = extract_decision_from_response(response)
            if parsed:
                return self._normalize_decision(parsed)
            return self._default_decision("Decision parsing error: could not recover a valid decision payload")
        except Exception as e:
            logger.error(f"Decision parsing error: {e}")
            return self._default_decision(f"Decision parsing error: {e}")

    def _normalize_playbook(self, playbook: Optional[Dict]) -> Dict:
        if not isinstance(playbook, dict):
            playbook = {}
        entry = playbook.get("entry_conditions", [])
        invalidation = playbook.get("invalidation_conditions", [])
        allowed_sides = playbook.get("allowed_sides", [])
        if not isinstance(allowed_sides, list):
            allowed_sides = []
        try:
            max_allocation_pct = float(playbook.get("max_allocation_pct")) if playbook.get("max_allocation_pct") is not None else None
        except Exception:
            max_allocation_pct = None
        return {
            "entry_conditions": entry if isinstance(entry, list) else [],
            "invalidation_conditions": invalidation if isinstance(invalidation, list) else [],
            "allowed_sides": allowed_sides,
            "max_allocation_pct": max_allocation_pct,
            "strategy_version": str(playbook.get("strategy_version") or "daily_playbook_v1"),
        }

    def _normalize_decision(self, decision: Dict) -> Dict:
        if not isinstance(decision, dict):
            return self._default_decision()

        normalized = self._decision_template()
        for key, value in decision.items():
            if key in ("reasoning", "monitoring_playbook", "daily_dual_plan", "scenario_plan"):
                continue
            normalized[key] = value

        reasoning = decision.get("reasoning", {})
        if isinstance(reasoning, dict):
            merged_reasoning = normalized["reasoning"].copy()
            merged_reasoning.update({k: v for k, v in reasoning.items() if v is not None})
            normalized["reasoning"] = merged_reasoning
        elif reasoning:
            normalized["reasoning"]["final_logic"] = str(reasoning)

        monitoring = self._normalize_playbook(decision.get("monitoring_playbook"))
        scenario_plan = decision.get("scenario_plan", {})
        if not isinstance(scenario_plan, dict):
            scenario_plan = {}
        merged_scenario_plan = normalized["scenario_plan"].copy()
        merged_scenario_plan.update({k: v for k, v in scenario_plan.items() if v not in (None, "")})

        dual = decision.get("daily_dual_plan", {})
        if not isinstance(dual, dict):
            dual = {}

        swing_plan = self._normalize_playbook(dual.get("swing_plan", monitoring))
        position_plan = self._normalize_playbook(dual.get("position_plan", monitoring))

        normalized["decision"] = str(normalized.get("decision", "HOLD")).upper()
        if normalized["decision"] not in {"LONG", "SHORT", "HOLD", "CANCEL_AND_CLOSE"}:
            normalized["decision"] = "HOLD"

        normalized["monitoring_playbook"] = monitoring
        normalized["scenario_plan"] = merged_scenario_plan
        normalized["daily_dual_plan"] = {
            "swing_plan": swing_plan,
            "position_plan": position_plan,
        }
        return normalized

    def _decision_template(self) -> Dict:
        return {
            "decision": "HOLD",
            "allocation_pct": 0,
            "leverage": 1,
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "hold_duration": "N/A",
            "reasoning": {
                "counter_scenario": "N/A",
                "meta_agent_context": "N/A",
                "technical": "N/A",
                "derivatives": "N/A",
                "experts": "N/A",
                "narrative": "N/A",
                "onchain": "N/A",
                "final_logic": "",
            },
            "win_probability_pct": 0,
            "confidence": 0,
            "expected_profit_pct": 0.0,
            "expected_loss_pct": 0.0,
            "ev_rationale": "N/A",
            "key_factors": [],
            "scenario_plan": {
                "primary_scenario": "",
                "alternate_scenario": "",
                "trigger_to_enter": "",
                "trigger_to_abort": "",
                "partial_tp_plan": "",
                "stop_to_be_rule": "",
            },
            "monitoring_playbook": {
                "entry_conditions": [],
                "invalidation_conditions": [],
            },
            "daily_dual_plan": {
                "swing_plan": {
                    "entry_conditions": [],
                    "invalidation_conditions": [],
                },
                "position_plan": {
                    "entry_conditions": [],
                    "invalidation_conditions": [],
                },
            },
        }

    def _default_decision(self, error_message: Optional[str] = None) -> Dict:
        safe_error = (error_message or "Error in decision-making process, defaulting to HOLD").strip()
        if len(safe_error) > 240:
            safe_error = safe_error[:237] + "..."
        decision = self._decision_template()
        decision["reasoning"]["final_logic"] = safe_error
        decision["ev_rationale"] = "System failure fallback."
        return decision


judge_agent = JudgeAgent()
