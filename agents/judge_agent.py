from .ai_router import ai_client
from config.settings import settings, TradingMode
from typing import Any, Dict, Optional
from config.database import db
from executors.post_mortem import retrieve_similar_memories
from loguru import logger
import json
import re
from utils.decision_parser import extract_decision_from_response


class JudgeAgent:
    """The final decision maker using premium Pro/Ultra models.
    Receives visual context + structural summaries for multi-modal cross-validation."""

    BASE_PROMPT = """You are a senior crypto portfolio manager making {mode_upper} trading decisions.

You receive ALL available data:
1. Higher-timeframe indicators aligned to the mode (SWING: 4h/1d, POSITION: 1d/1w) with Fibonacci levels and timing context
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
- Use metrics: `price`, `funding_rate`, `oi_chg_pct`, `price_chg_pct_4h`, `price_chg_pct_1d`, `volatility`.
- Ensure operators are one of: `>`, `<`, `>=`, `<=`, `==`.

CRITICAL EXECUTION RULE:
- Prioritize evidence in this order: Higher-timeframe structure -> liquidity event / trap -> retest quality -> derivatives confirmation -> narrative.
- Entry timing matters, but only after the higher-timeframe thesis is clear. Use execution-timeframe retests/reclaims, not low-timeframe noise, to time entries.

{mode_specific_rules}

Output your decision as JSON (execution fields only — report narrative is generated separately):
{{
  "decision": "LONG" | "SHORT" | "HOLD" | "CANCEL_AND_CLOSE",
  "allocation_pct": 0-100,
  "leverage": 1-3,
  "entry_price": float,
  "stop_loss": float,
  "take_profit": float,
  "win_probability_pct": 0-100,
  "expected_profit_pct": float,
  "expected_loss_pct": float,
  "reasoning": {{
    "final_logic": "결정 근거 2-3문장 (Korean)",
    "counter_scenario": "가장 강한 반대 시나리오 1-2문장 (Korean)"
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

CRITICAL: reasoning.final_logic and reasoning.counter_scenario MUST be written in Korean.
JSON keys remain in English.
Return exactly one JSON object only. Do not wrap it in Markdown code fences.
Do not add any text before or after the JSON object.

You have FULL AUTONOMY. HOLD is valid only when your edge is not statistically meaningful (weak EV, weak R/R, or weak win probability).
CRITICAL V5 RULE: You will be given a list of ACTIVE_ORDERS (e.g. pending DCA chunks). If the market paradigm shifts against your active orders, you MUST output "decision": "CANCEL_AND_CLOSE". If you just want to add to an existing order, you can output LONG/SHORT again.
Be aware of your previous decision for consistency."""

    SWING_RULES = """Professional SWING trading principles you should consider:

ENTRY GATE — 근거 중첩 필수 (쉽알남 원칙):
The `confluence_gate` entry in your expert_verdicts shows a structural confluence count.
- gate_passed=True  (score ≥ 3): Entry is PERMITTED if all other conditions are met.
- gate_passed=False (score < 3): You MUST output "HOLD" regardless of other signals.
  State "근거 중첩 부족 (N/3)" in final_logic and explain what is missing.
  A lone narrative signal or a single timeframe breakout is NOT enough to enter.
Confluence factors counted:
  - HTF MSB (1d/1w): price already broke structure — immediate factor
  - HTF CHoCH retested (1d/1w): first reversal swing + 4h retest confirmed — factor only after retest
  - 4h Fractal Alignment: 4h CHoCH/MSB in same direction as HTF bias (쉽알남 프랙탈)
  - Active setup direction, near confluence zone, Fibonacci confluence, funding extreme, volume breakout.

- Top-down analysis: 1d determines bias, 4h defines the setup and confirms the entry
- Fibonacci 38.2%/50%/61.8% are key retracement entry zones
- Invalidation is more important than precise entry. If invalidation is unclear, choose HOLD.
- Good timing matters, but timing must come from 4h reclaim/retest quality near key levels, not from chasing short-term candles.
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
- Stop Loss & Take Profit: Use `atr_anchor` in the brief as your baseline.
  `atr_anchor.suggested_sl_pct` = 1.5×ATR14(4h) from entry — this is the minimum viable stop distance.
  `atr_anchor.suggested_tp_pct` = 3.3×ATR14(4h) from entry — this achieves RR≈2.2.
  You MAY adjust SL/TP to structural levels (support, resistance, Fibonacci), but:
  POLICY HARD FLOOR: Final RR = (TP distance / SL distance) MUST be ≥ 2.0.
  If achieving 2.0:1 R/R is impossible without an unrealistically tight stop, choose HOLD.
  Manage absolute risk by reducing allocation_pct and leverage, NOT by tightening the stop.
- Hold period: days to weeks"""

    POSITION_RULES = """Professional POSITION trading principles you should consider:
- Top-down analysis: 1w determines macro bias, 1d identifies structural shifts
- Invalidation is more important than precise entry. If invalidation is unclear, choose HOLD.
- Entry timing still matters, but it should come from 1d structure confirmation or pullback acceptance, not from intraday noise.
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
  For SWING mode: Macro/Options > OI Divergence & Funding > Volume Profile > Visual Geometry > Microstructure.
  For POSITION mode: Macro/Options > OI Divergence & Funding > Volume Profile > Visual Geometry > Microstructure.
  (Microstructure is least important for multi-day holds; most important only for intraday timing.)
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

LEVERAGE HARD CAPS (non-negotiable, mode-specific):
  - SWING mode: leverage 13x maximum
  - POSITION mode: leverage 1.01.5x maximum (liquidation must be nearly impossible)
  Do NOT output leverage > 1.5 when mode is POSITION, even if the playbook context suggests otherwise.

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
- Focus on whether price is entering the planned zone with confirmation on the execution timeframe. Do not overreact to small intraday fluctuations.
- If context is stale or thesis-breaking, prefer REPLAN over inventing the opposite side.
- Use Korean for reasoning and replan_reason.
- Output exactly one JSON object only.
"""

    def __init__(self):
        pass

    @staticmethod
    def _compact_text(value: Any, limit: int = 240) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    @classmethod
    def _summarize_lines(cls, value: Any, *, max_lines: int = 8, max_chars: int = 800) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        lines = []
        seen = set()
        total = 0
        for raw_line in text.splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip(" -")
            if not line or line in seen:
                continue
            seen.add(line)
            compact = cls._compact_text(line, 180)
            projected = total + len(compact) + 2
            if projected > max_chars:
                break
            lines.append(compact)
            total = projected
            if len(lines) >= max_lines:
                break
        if not lines:
            return cls._compact_text(text, max_chars)
        return "\n".join(f"- {line}" for line in lines)

    @classmethod
    def _summarize_previous_decision(cls, previous_decision: Optional[Dict]) -> Dict:
        if not isinstance(previous_decision, dict) or not previous_decision:
            return {"status": "none"}
        reasoning = previous_decision.get("reasoning", {})
        final_logic = ""
        if isinstance(reasoning, dict):
            final_logic = (
                reasoning.get("final_logic")
                or reasoning.get("technical")
                or reasoning.get("narrative")
                or reasoning.get("onchain")
                or ""
            )
        else:
            final_logic = str(reasoning or "")
        return {
            "status": "available",
            "decision": str(previous_decision.get("decision", "N/A") or "N/A"),
            "allocation_pct": previous_decision.get("allocation_pct", 0),
            "leverage": previous_decision.get("leverage", 1),
            "hold_duration": str(previous_decision.get("hold_duration", "N/A") or "N/A"),
            "confidence": previous_decision.get("confidence", previous_decision.get("win_probability_pct", 0)),
            "final_logic": cls._compact_text(final_logic, 180),
        }

    @classmethod
    def _summarize_active_orders(cls, active_orders: list) -> Dict:
        if not active_orders:
            return {"count": 0, "orders": []}
        items = []
        for order in active_orders[:3]:
            if not isinstance(order, dict):
                continue
            items.append({
                "symbol": str(order.get("symbol", "") or ""),
                "side": str(order.get("side", order.get("direction", "")) or ""),
                "exchange": str(order.get("exchange", "") or ""),
                "status": str(order.get("status", "") or ""),
                "style": str(order.get("execution_style", order.get("style", "")) or ""),
            })
        return {"count": len(active_orders), "orders": items}

    @classmethod
    def _summarize_regime_context(cls, regime_context: Optional[Dict]) -> Dict:
        context = regime_context if isinstance(regime_context, dict) else {}
        return {
            "regime": str(context.get("regime", "N/A") or "N/A"),
            "risk_bias": str(context.get("risk_bias", "N/A") or "N/A"),
            "risk_budget_pct": context.get("risk_budget_pct", "N/A"),
            "trust_directive": cls._compact_text(context.get("trust_directive", ""), 220),
            "rationale": cls._compact_text(context.get("rationale", ""), 220),
        }

    @classmethod
    def _summarize_blackboard(cls, blackboard: Dict[str, dict]) -> list[Dict[str, Any]]:
        if not isinstance(blackboard, dict):
            return []

        summaries: list[Dict[str, Any]] = []
        preferred_order = ["confluence_score", "macro", "liquidity", "microstructure", "chart_rules", "vlm_geometry"]
        ordered_keys = preferred_order + [k for k in blackboard.keys() if k not in preferred_order]

        for key in ordered_keys:
            payload = blackboard.get(key)
            if not isinstance(payload, dict):
                continue
            if key == "confluence_score":
                gate = payload.get("gate_passed", False)
                summaries.append({
                    "agent": "confluence_gate",
                    "score": payload.get("score", 0),
                    "max_score": 6,
                    "direction": payload.get("direction", "NEUTRAL"),
                    "factors": payload.get("factors", []),
                    "gate_passed": gate,
                    "verdict": f"{'GATE_OPEN' if gate else 'GATE_CLOSED'} — {payload.get('score', 0)}/3 minimum factors met",
                })
                continue
            if key == "chart_rules":
                signals = payload.get("signals", {}) if isinstance(payload.get("signals"), dict) else {}
                summaries.append({
                    "agent": key,
                    "status": str(payload.get("status", "N/A") or "N/A"),
                    "scenario_ready": bool(signals.get("scenario_ready")),
                    "structure_alert": bool(signals.get("has_structure_alert")),
                    "trap_signal": bool(signals.get("has_trap_signal")),
                    "bias": str(payload.get("directional_bias", signals.get("bias", "N/A")) or "N/A"),
                })
                continue
            if key == "vlm_geometry":
                # Skip if VLM was gated off or detected nothing — saves ~200 tokens
                anomaly = str(payload.get("anomaly", "none") or "none").lower()
                if anomaly in ("none", "skipped"):
                    continue
                summaries.append({
                    "agent": key,
                    "bias": str(payload.get("directional_bias", "N/A") or "N/A"),
                    "confidence": payload.get("confidence", "N/A"),
                    "anomaly": anomaly,
                    "trap_type": str(payload.get("trap_type", "N/A") or "N/A"),
                    "rationale": cls._compact_text(payload.get("rationale", ""), 160),
                })
                continue

            verdict = (
                payload.get("decision")
                or payload.get("bias")
                or payload.get("signal")
                or payload.get("status")
                or "N/A"
            )
            rationale = (
                payload.get("summary")
                or payload.get("reasoning")
                or payload.get("rationale")
                or payload.get("details")
                or ""
            )
            # Skip agents with no meaningful signal — saves ~160 tokens per empty entry
            if str(verdict).upper() in ("N/A", "NONE", "NEUTRAL") and not str(rationale).strip():
                continue
            summaries.append({
                "agent": key,
                "verdict": str(verdict or "N/A"),
                "confidence": payload.get("confidence", payload.get("score", "N/A")),
                "rationale": cls._compact_text(rationale, 160),
            })
            if len(summaries) >= 8:
                break
        return summaries

    def _build_judge_brief(
        self,
        *,
        symbol: str,
        mode: TradingMode,
        market_data_compact: str,
        blackboard: Dict[str, dict],
        funding_context: str,
        narrative_context: str,
        onchain_context: str,
        regime_context: Optional[Dict],
        previous_decision: Optional[Dict],
        active_orders: list,
        open_positions: str,
        episodic_memory: str,
        feedback_text: str,
        atr_anchor: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        brief: Dict[str, Any] = {
            "symbol": symbol,
            "mode": mode.value.upper(),
            "structure_snapshot": self._summarize_lines(market_data_compact, max_lines=18, max_chars=1800),
            "regime_snapshot": self._summarize_regime_context(regime_context),
            "derivatives_snapshot": self._summarize_lines(funding_context, max_lines=8, max_chars=700),
            "narrative_snapshot": self._summarize_lines(narrative_context, max_lines=8, max_chars=700),
            "onchain_snapshot": self._summarize_lines(onchain_context, max_lines=6, max_chars=500),
            "expert_verdicts": self._summarize_blackboard(blackboard),
            "execution_context": {
                "open_positions": self._summarize_lines(open_positions, max_lines=4, max_chars=280) or "No open positions.",
                "active_orders": self._summarize_active_orders(active_orders),
            },
            "previous_decision": self._summarize_previous_decision(previous_decision),
            "episodic_memory": self._summarize_lines(episodic_memory, max_lines=5, max_chars=420),
            "feedback_constraints": self._summarize_lines(feedback_text, max_lines=4, max_chars=260),
        }
        if atr_anchor:
            brief["atr_anchor"] = atr_anchor
        return brief

    def get_previous_decision(self, symbol: str = "BTCUSDT") -> Optional[Dict]:
        """[FIX CRITICAL-6] symbol parameter added  was NameError (symbol not in scope)."""
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
        active_orders: Optional[list] = None,
        open_positions: str = "",
        symbol: str = "BTCUSDT",
        regime_context: Optional[Dict] = None,
        narrative_context: str = "",
        onchain_context: str = "",
        atr_anchor: Optional[Dict] = None,
    ) -> Dict:
        active_orders = active_orders or []  # Guard against mutable default
        mode_str = mode.value.upper()
        mode_rules = self.SWING_RULES
        # Wait, format() evaluates all curly braces. self.DEBATE_APPENDIX contains raw JSON templates!
        # Instead of formatting the whole concatenated string, format the components individually.
        base_formatted = self.BASE_PROMPT.format(mode_upper=mode_str, mode_specific_rules=mode_rules)
        prompt = f"{base_formatted}\n\n{self.DEBATE_APPENDIX}"
        previous_decision = self.get_previous_decision(symbol=symbol)

        # --- Episodic Memory Retrieval ---
        # Queries the JSONL vector DB placeholder for past identical patterns.
        try:
            sample_anomalies = [k for k in blackboard.keys()]
            symbol_detected = symbol
            episodic_memory = retrieve_similar_memories(sample_anomalies, symbol_detected)
        except Exception as e:
            logger.error(f"Failed to query memory: {e}")
            episodic_memory = "Vector DB RAG Search: System running in Bootstrap mode."

        judge_brief = self._build_judge_brief(
            symbol=symbol,
            mode=mode,
            market_data_compact=market_data_compact,
            blackboard=blackboard,
            funding_context=funding_context,
            narrative_context=narrative_context,
            onchain_context=onchain_context,
            regime_context=regime_context,
            previous_decision=previous_decision,
            active_orders=active_orders,
            open_positions=open_positions,
            episodic_memory=episodic_memory,
            feedback_text=feedback_text,
            atr_anchor=atr_anchor or {},
        )

        user_message = (
            "Use this compressed investment-committee brief as the ground truth.\n"
            "Do not assume missing raw fields exist.\n"
            "If a field is absent or marked unavailable, treat it as unknown rather than inferring.\n\n"
            f"JUDGE_BRIEF:\n{json.dumps(judge_brief, ensure_ascii=False, indent=2)}\n\n"
            "Make your trading decision. Output as JSON. Ensure the counter_scenario is thoroughly explored."
        )
        logger.info(
            f"Judge brief prepared for {symbol}/{mode.value}: "
            f"market_chars={len(str(market_data_compact or ''))} "
            f"narrative_chars={len(str(narrative_context or ''))} "
            f"funding_chars={len(str(funding_context or ''))} "
            f"brief_chars={len(user_message)}"
        )

        try:
            # Judge uses the best model (Opus)
            # Only Judge gets the chart image (SWING mode only)
            response = ai_client.generate_response(
                system_prompt=prompt,
                user_message=user_message,
                temperature=0.2,
                max_tokens=2000,
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
            raw_lev = int(float(parsed.get("leverage", 1) or 1))
            mode_str_ctx = str(compact_snapshot.get("mode", "swing") or "swing").lower()
            # POSITION mode hard cap: POSITION_RULES explicitly states max 1.5x
            max_lev = 1 if mode_str_ctx == "position" else 3  # 1x enforced for position (floor applied below)
            lev_cap = 1.5 if mode_str_ctx == "position" else 3.0
            decision["leverage"] = max(1, min(raw_lev, max_lev)) if mode_str_ctx != "position" else max(1.0, min(float(raw_lev), lev_cap))
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

        # Reconcile confidence vs win_probability_pct.
        # LLM may output only one of the two; use whichever is non-zero so the
        # display code doesn't silently fall through to the template default of 0.
        conf = normalized.get("confidence") or 0
        win_prob = normalized.get("win_probability_pct") or 0
        if not conf and win_prob:
            normalized["confidence"] = win_prob
        elif not win_prob and conf:
            normalized["win_probability_pct"] = conf

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
