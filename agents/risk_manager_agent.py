import json
from loguru import logger
from agents.claude_client import claude_client

class RiskManagerAgent:
    """The Chief Risk Officer (CRO).
    Sits AFTER the Judge (Portfolio Manager).
    Has absolute VETO power and controls the final position sizing / leverage
    based on institutional risk frameworks (Tail-risk, Kelly Criterion, Hedging).
    """
    
    SYSTEM_PROMPT = """You are the strict Chief Risk Officer (CRO) at a Tier-1 Quant Hedge Fund.

The Portfolio Manager (Judge) has submitted a DRAFT trade proposal.
Your ONLY job is Capital Preservation. You must review the draft and the market conditions to decide if the trade is safe, and if so, what the true MAXIMUM safe allocation and leverage should be.

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

    def evaluate_trade(self, draft_decision: dict, funding_context: str, deribit_context: str) -> dict:
        
        # Fast-pass: If PM already voted HOLD or CANCEL_AND_CLOSE, CRO just agrees.
        dec = draft_decision.get('decision', 'HOLD')
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

Please execute your Risk Management oversight and output the final, safe JSON."""

        try:
            # We use Gemini 3.1 Pro for the CRO. It's perfectly capable of enforcing rules reliably.
            # Using Judge (Opus) again would be too slow/expensive.
            response = claude_client.generate_response(
                system_prompt=self.SYSTEM_PROMPT,
                user_message=user_message,
                temperature=0.1,  # Very strict, less creative
                max_tokens=600,
                role="macro" # Recyling the macro rate limits for the risk module
            )
            return self._parse_decision(response, draft_decision)
            
        except Exception as e:
            logger.error(f"CRO Agent error: {e}")
            return {"decision": "HOLD", "allocation_pct": 0, "leverage": 1, "reasoning": f"CRO Error: {e}", "cro_veto_applied": True}

risk_manager_agent = RiskManagerAgent()
