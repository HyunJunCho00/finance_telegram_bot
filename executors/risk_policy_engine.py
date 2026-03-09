from __future__ import annotations

from copy import deepcopy
from typing import Dict

from config.settings import TradingMode
from executors.policy_engine import policy_engine


class RiskPolicyEngine:
    """Deterministic final authority for snapshot-based hot path.

    Current implementation reuses the existing policy engine and only depends on
    the normalized blackboard snapshot. This lets us remove the LLM CRO step
    from the future hot path without changing the external contract.
    """

    @staticmethod
    def _merge_market_data(bundle: Dict) -> Dict:
        agents = (bundle or {}).get("agents", {}) or {}
        market_payload = ((agents.get("market_snapshot_agent") or {}).get("payload") or {})
        chart_payload = ((agents.get("chart_prep_agent") or {}).get("payload") or {})

        merged = deepcopy(market_payload)
        if isinstance(chart_payload.get("market_data"), dict):
            merged.update(chart_payload.get("market_data") or {})
        if isinstance(chart_payload.get("chart_rules"), dict):
            merged["chart_rules"] = chart_payload.get("chart_rules") or {}
        return merged

    def apply(self, decision: Dict, bundle: Dict, *, playbook_context: Dict | None = None) -> Dict:
        merged_decision = deepcopy(decision or {})
        mode = TradingMode(str((bundle or {}).get("mode", TradingMode.SWING.value)).lower())
        agents = (bundle or {}).get("agents", {}) or {}

        funding_payload = ((agents.get("funding_liq_agent") or {}).get("payload") or {})
        market_data = self._merge_market_data(bundle)
        return policy_engine.enforce(
            decision=merged_decision,
            market_data=market_data,
            mode=mode,
            raw_funding=funding_payload.get("raw_funding", {}),
            cvd_df=None,
            liq_df=None,
            playbook_context=playbook_context,
        )


risk_policy_engine = RiskPolicyEngine()
