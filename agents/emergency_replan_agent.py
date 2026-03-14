from __future__ import annotations

from typing import Dict, Optional


class EmergencyReplanAgent:
    """Escalates a trigger into thesis replacement or forced cancel/close."""

    def assess(
        self,
        *,
        playbook_context: Optional[Dict] = None,
        trigger_validation: Optional[Dict] = None,
        snapshot: Optional[Dict] = None,
    ) -> Dict:
        context = dict(playbook_context or {})
        validation = dict(trigger_validation or {})
        snap = dict(snapshot or {})

        trigger_action = str(validation.get("trigger_action", "") or "").upper()
        direction = str(validation.get("decision", "HOLD") or "HOLD").upper()
        active_orders = snap.get("active_orders", []) or []
        open_positions = str(snap.get("open_positions", "") or "")
        has_open_exposure = bool(active_orders) or ("no open positions" not in open_positions.lower())

        if trigger_action == "REPLAN":
            return {
                "action": "REPLAN",
                "reason": str(validation.get("replan_reason") or validation.get("reasoning", {}).get("final_logic") or "").strip(),
            }
        if bool(context.get("invalidated")) or str(validation.get("trigger_action", "")).upper() == "INVALIDATED":
            return {
                "action": "REPLAN",
                "reason": "기존 플레이북이 무효화되어 전략 재수립이 필요합니다.",
            }
        if trigger_action == "CANCEL" and has_open_exposure:
            return {
                "action": "CANCEL_AND_CLOSE",
                "reason": "기존 포션/주문을 유하기 어려워 강제 청산을 요청합니다.",
            }
        if direction == "CANCEL_AND_CLOSE":
            return {
                "action": "CANCEL_AND_CLOSE",
                "reason": str(validation.get("reasoning", {}).get("final_logic", "")).strip(),
            }
        return {
            "action": "KEEP",
            "reason": "",
        }


emergency_replan_agent = EmergencyReplanAgent()
