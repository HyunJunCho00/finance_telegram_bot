import json
from loguru import logger
from .ai_router import ai_client
from config.settings import TradingMode

class RiskManagerAgent:
    """The Chief Risk Officer (CRO).
    Sits AFTER the Judge (Portfolio Manager).
    Has absolute VETO power and controls the final position sizing / leverage
    based on institutional risk frameworks (Tail-risk, Kelly Criterion, Hedging).
    """
    
    SYSTEM_PROMPT = """You are the strict Chief Risk Officer (CRO) at a Tier-1 Quant Hedge Fund.

The Portfolio Manager (Judge) has submitted a DRAFT trade proposal.
Your ONLY job is Capital Preservation. You must review the draft and the market conditions to decide if the trade is safe, and if so, what the true MAXIMUM safe allocation and leverage should be.
CRITICAL POLICY RULE: A deterministic policy engine will perform final approval after you. Do not invent exceptions to weak structure or poor reward/risk. If uncertain, choose HOLD.

You operate under the following institutional rules:
1. MAX PORTFOLIO HEAT: Never approve leverage > 3x for SWING, and never > 1x for POSITION, regardless of PM confidence.
2. VOLATILITY TAX: If macroeconomic volatility (Deribit DVOL, 2s10s spread) shows panic or inverted term structure, cut the PM's requested allocation size by 50% immediately.
3. CONTRARIAN VETO: If the PM wants to LONG but funding rates are massively positive (>0.03%), or SHORT while funding is massively negative, VETO the trade (change it to HOLD).
4. LIQUIDATION TRAP AVOIDANCE: If recent liquidations >$50M in the opposite direction exist, suspect a fakeout and reduce confidence/sizing.
5. RETAIL CONSTRAINTS (V8): 
   - UPBIT (Spot only): Can ONLY go LONG. Max Allocation 2,000,000 KRW. Leverage ALWAYS 1x.
   - BINANCE (Futures): Can go LONG or SHORT. Max Allocation $2,000 USD. Max Leverage is strictly capped at 3x.
   - If PM proposes a LONG, decide whether to route to BINANCE, UPBIT, or SPLIT. If SHORT, MUST route to BINANCE.
6. EXECUTION STYLE:
   - PASSIVE_MAKER: Low volatility. Place limits to save fees.
   - SMART_DCA: Average down into support.
   - MOMENTUM_SNIPER: Pure breakout, market sweep immediately.
   - CASINO_EXIT: Panic dump.
7. EXECUTION DISCIPLINE:
   - Invalidation comes before entry precision. If invalidation is unclear, VETO to HOLD.
   - Prefer split entries and partial take profit over all-in entries.
   - If a draft conflicts with the active scenario bias or trap context, reduce or veto.

Input Format:
- Draft Decision from PM (JSON)
- Current Risk State (Funding, Deribit Volatility snapshot)

Output Format (Strict JSON):
{
    "decision": "LONG" | "SHORT" | "HOLD",
    "approved_allocation_pct": 0-100,
    "approved_leverage": 1-3,
    "target_exchange": "BINANCE" | "UPBIT" | "SPLIT",
    "recommended_execution_style": "PASSIVE_MAKER" | "SMART_DCA" | "MOMENTUM_SNIPER" | "CASINO_EXIT",
    "cro_veto_applied": true/false,
    "cro_reasoning": "Explanation from the Risk Desk..."
}"""

    def _apply_onchain_overrides(self, decision: dict, onchain_gate: dict | None, mode: TradingMode) -> dict:
        if not isinstance(decision, dict):
            return {"decision": "HOLD", "allocation_pct": 0, "leverage": 1}

        gate = onchain_gate or {}
        direction = str(decision.get("decision", "HOLD")).upper()
        if direction not in ["LONG", "SHORT"]:
            return decision

        note_parts = []
        allocation = float(decision.get("allocation_pct", 0) or 0)
        leverage = float(decision.get("leverage", 1) or 1)

        if direction == "LONG" and not gate.get("allow_long", True):
            decision["decision"] = "HOLD"
            decision["allocation_pct"] = 0
            decision["leverage"] = 1
            decision["cro_veto_applied"] = True
            decision["risk_manager_note"] = "On-chain gate veto: LONG disabled by daily regime filter."
            return decision

        if direction == "SHORT" and not gate.get("allow_short", True):
            decision["decision"] = "HOLD"
            decision["allocation_pct"] = 0
            decision["leverage"] = 1
            decision["cro_veto_applied"] = True
            decision["risk_manager_note"] = "On-chain gate veto: SHORT disabled by daily regime filter."
            return decision

        if direction == "LONG":
            mult = float(gate.get("long_size_multiplier", 1.0) or 1.0)
            if mult != 1.0:
                allocation *= mult
                note_parts.append(f"LONG size x{mult:.2f} by on-chain gate")
            if gate.get("chase_long_blocked"):
                leverage = min(leverage, 2.0)
                note_parts.append("chase-long blocked")
        elif direction == "SHORT":
            mult = float(gate.get("short_size_multiplier", 1.0) or 1.0)
            if mult != 1.0:
                allocation *= mult
                note_parts.append(f"SHORT size x{mult:.2f} by on-chain gate")
            if str(gate.get("risk_bias", "NEUTRAL")).upper() == "RISK_ON":
                leverage = min(leverage, 1.0)
                note_parts.append("RISK_ON regime caps short leverage")

        if str(gate.get("data_quality", "")).lower() == "stale":
            allocation *= 0.8
            note_parts.append("stale on-chain snapshot applied 0.8x size cap")

        decision["allocation_pct"] = round(max(0.0, allocation), 2)
        decision["leverage"] = round(max(1.0, leverage), 2)

        if note_parts:
            existing = str(decision.get("risk_manager_note", "") or "").strip()
            decision["risk_manager_note"] = " | ".join([p for p in [existing, "; ".join(note_parts)] if p])
        return decision

    def _apply_scenario_overrides(self, decision: dict, scenario_context: dict | None, mode: TradingMode) -> dict:
        if not isinstance(decision, dict):
            return {"decision": "HOLD", "allocation_pct": 0, "leverage": 1}

        scenario = scenario_context or {}
        active_setup = scenario.get("active_setup", {}) if isinstance(scenario, dict) else {}
        if not isinstance(active_setup, dict) or not active_setup:
            return decision

        direction = str(decision.get("decision", "HOLD")).upper()
        active_side = str(active_setup.get("side", "")).upper()
        note_parts = []

        if direction in ("LONG", "SHORT") and active_side and direction != active_side:
            decision["decision"] = "HOLD"
            decision["allocation_pct"] = 0
            decision["leverage"] = 1
            decision["cro_veto_applied"] = True
            decision["risk_manager_note"] = (
                f"Scenario veto: active setup side is {active_side}, draft decision was {direction}."
            )
            return decision

        invalidation = active_setup.get("invalidation")
        if direction in ("LONG", "SHORT") and invalidation in (None, "", "N/A"):
            decision["decision"] = "HOLD"
            decision["allocation_pct"] = 0
            decision["leverage"] = 1
            decision["cro_veto_applied"] = True
            decision["risk_manager_note"] = "Scenario veto: invalidation level missing."
            return decision

        trap_context = active_setup.get("trap_context", {}) or {}
        if direction in ("LONG", "SHORT") and isinstance(trap_context, dict) and trap_context.get("confirmed"):
            trap_bias = str(trap_context.get("bias_after_sweep", "")).upper()
            if trap_bias and trap_bias != direction:
                decision["decision"] = "HOLD"
                decision["allocation_pct"] = 0
                decision["leverage"] = 1
                decision["cro_veto_applied"] = True
                decision["risk_manager_note"] = (
                    f"Scenario veto: liquidity trap implies {trap_bias}, not {direction}."
                )
                return decision
            decision["allocation_pct"] = round(float(decision.get("allocation_pct", 0) or 0) * 0.7, 2)
            if mode == TradingMode.SWING:
                decision["leverage"] = min(float(decision.get("leverage", 1) or 1), 2.0)
            note_parts.append(f"trap-aware size reduction ({trap_context.get('status', 'trap')})")

        split_entries = active_setup.get("split_entries", []) or []
        if direction in ("LONG", "SHORT") and len(split_entries) >= 2:
            if decision.get("recommended_execution_style") == "MOMENTUM_SNIPER":
                decision["recommended_execution_style"] = "SMART_DCA"
            note_parts.append("split-entry discipline enforced")

        if note_parts:
            existing = str(decision.get("risk_manager_note", "") or "").strip()
            decision["risk_manager_note"] = " | ".join([p for p in [existing, "; ".join(note_parts)] if p])

        return decision

    def _parse_decision(self, response: str, draft: dict) -> dict:
        try:
            s, e = response.find('{'), response.rfind('}') + 1
            if s != -1 and e > s:
                risk_rec = json.loads(response[s:e])
                
                # Merge the CRO's overrides back into the PM's original draft
                final_decision = draft.copy()
                final_decision['decision'] = risk_rec.get('decision', draft.get('decision'))
                final_decision['allocation_pct'] = risk_rec.get('approved_allocation_pct', 0)
                final_decision['leverage'] = risk_rec.get('approved_leverage', 1)
                final_decision['target_exchange'] = risk_rec.get('target_exchange', 'BINANCE')
                final_decision['recommended_execution_style'] = risk_rec.get('recommended_execution_style', 'MOMENTUM_SNIPER')
                final_decision['cro_veto_applied'] = risk_rec.get('cro_veto_applied', False)
                final_decision['risk_manager_note'] = risk_rec.get('cro_reasoning', '')
                
                # If VETO applied, zero out sizing
                if final_decision['decision'] == 'HOLD' or final_decision.get('cro_veto_applied'):
                    final_decision['decision'] = 'HOLD'
                    final_decision['allocation_pct'] = 0
                    final_decision['leverage'] = 1
                    
                # V8 Hardcoded Safeguard: Upbit NO SHORT, NO LEVERAGE
                if final_decision.get('target_exchange') == 'UPBIT':
                    if final_decision['decision'] == 'SHORT':
                        final_decision['decision'] = 'HOLD'
                        final_decision['allocation_pct'] = 0
                        final_decision['risk_manager_note'] = "[V8 SECURE BLOCK] Attempted to SHORT on UPBIT. Trade Vetoed."
                        final_decision['cro_veto_applied'] = True
                    if final_decision.get('leverage', 1) > 1:
                        final_decision['leverage'] = 1
                        final_decision['risk_manager_note'] += " [V8 SECURE BLOCK] Upbit leverage forced to 1x."
                    
                return final_decision
        except Exception as e:
            logger.error(f"CRO Agent parsing error: {e}")
            
        # Fail safe: Default to HOLD if CRO fails to respond properly
        return {"decision": "HOLD", "allocation_pct": 0, "leverage": 1, "reasoning": "CRO System Failure: Default to HOLD", "cro_veto_applied": True}

    def evaluate_trade(self, draft_decision: dict, funding_context: str, deribit_context: str,
                       mode: TradingMode = TradingMode.SWING, onchain_context: str = "",
                       onchain_gate: dict | None = None, scenario_context: dict | None = None) -> dict:
        
        # Fast-pass: If PM already voted HOLD or CANCEL_AND_CLOSE, CRO just agrees.
        dec = draft_decision.get('decision', 'HOLD')

        # Venue policy: Swing futures only (BINANCE), LONG/SHORT both allowed
        if dec in ['LONG', 'SHORT']:
            draft_decision['target_exchange'] = 'BINANCE'

        if dec in ['HOLD', 'CANCEL_AND_CLOSE']:
            draft_decision['cro_veto_applied'] = False
            draft_decision['risk_manager_note'] = f"PM decided to {dec}. CRO concurs without risk checks."
            return draft_decision
            
        draft_str = json.dumps(draft_decision, indent=2)
        
        user_message = f"""PM DRAFT PROPOSAL:
{draft_str}

RISK CONTEXT (DERIVATIVES & MACRO):
Funding Data: {funding_context}
Deribit Data: {deribit_context}

ON-CHAIN RISK OVERLAY:
{onchain_context if onchain_context else "On-chain Context: unavailable"}

SCENARIO CONTEXT:
{json.dumps(scenario_context, ensure_ascii=False, indent=2) if scenario_context else "Scenario Context: unavailable"}

Please execute your Risk Management oversight and output the final, safe JSON."""

        try:
            # [FIX MEDIUM-14] CRO uses GPT-5.2 via role="macro" routing.
            # GPT-5.2 excels at rule enforcement and macro risk analysis.
            response = ai_client.generate_response(
                system_prompt=self.SYSTEM_PROMPT,
                user_message=user_message,
                temperature=0.1,  # Very strict, less creative
                max_tokens=600,
                role="risk_eval"  # Cerebras qwen-3-235b-a22b-instruct-2507; fallback to Groq
            )
            final = self._parse_decision(response, draft_decision)
            final = self._apply_scenario_overrides(final, scenario_context, mode)
            final = self._apply_onchain_overrides(final, onchain_gate, mode)

            # Re-apply venue policy after LLM output parsing.
            if final.get('decision') in ['LONG', 'SHORT']:
                final['target_exchange'] = 'BINANCE'
            return final
            
        except Exception as e:
            logger.error(f"CRO Agent error: {e}")
            fallback = {"decision": "HOLD", "allocation_pct": 0, "leverage": 1, "reasoning": f"CRO Error: {e}", "cro_veto_applied": True}
            return self._apply_onchain_overrides(fallback, onchain_gate, mode)

risk_manager_agent = RiskManagerAgent()
