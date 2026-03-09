from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, Optional


def _parse_iso(value: object) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _safe_float(value: object) -> Optional[float]:
    try:
        if value in (None, "", "N/A"):
            return None
        return float(value)
    except Exception:
        return None


def _normalize_side(value: object) -> str:
    side = str(value or "").upper().strip()
    if side in ("LONG", "SHORT", "HOLD", "CANCEL_AND_CLOSE"):
        return side
    return ""


def _normalize_allowed_sides(value: object, source_decision: str) -> list[str]:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        sides = []
        for item in value:
            side = _normalize_side(item)
            if side in ("LONG", "SHORT") and side not in sides:
                sides.append(side)
        if sides:
            return sides
    if source_decision in ("LONG", "SHORT"):
        return [source_decision]
    return []


def build_playbook_context(playbook_row: Optional[Dict], *, trigger_reason: str = "") -> Dict:
    row = playbook_row or {}
    payload = row.get("playbook", {}) if isinstance(row.get("playbook", {}), dict) else {}
    source_decision = _normalize_side(row.get("source_decision"))
    created_at = _parse_iso(row.get("created_at"))
    ttl_hours = _safe_float(row.get("ttl_hours"))
    expires_at = None
    if created_at and ttl_hours is not None:
        expires_at = (created_at + timedelta(hours=float(ttl_hours))).isoformat()

    return {
        "playbook_id": row.get("id"),
        "source_decision": source_decision or "HOLD",
        "allowed_sides": _normalize_allowed_sides(payload.get("allowed_sides"), source_decision),
        "max_allocation_pct": _safe_float(payload.get("max_allocation_pct")),
        "strategy_version": payload.get("strategy_version") or row.get("strategy_version") or "daily_playbook_v1",
        "thesis_id": payload.get("thesis_id") or row.get("id"),
        "created_at": row.get("created_at"),
        "expires_at": expires_at,
        "invalidated": bool(row.get("invalidated", False)),
        "trigger_reason": str(trigger_reason or ""),
    }


def evaluate_playbook_consistency(
    decision: Optional[Dict],
    playbook_context: Optional[Dict],
    *,
    now: Optional[datetime] = None,
) -> Dict:
    context = dict(playbook_context or {})
    if not context:
        return {
            "status": "SKIPPED",
            "reasons": [],
            "playbook_id": None,
            "source_decision": "HOLD",
            "allowed_sides": [],
            "max_allocation_pct": None,
        }

    current_time = now or datetime.now(timezone.utc)
    direction = _normalize_side((decision or {}).get("decision"))
    source_decision = _normalize_side(context.get("source_decision")) or "HOLD"
    allowed_sides = _normalize_allowed_sides(context.get("allowed_sides"), source_decision)
    max_allocation_pct = _safe_float(context.get("max_allocation_pct"))
    expires_at = _parse_iso(context.get("expires_at"))
    reasons: list[str] = []

    if bool(context.get("invalidated")):
        reasons.append("Playbook is already invalidated.")
    if expires_at is not None and expires_at <= current_time:
        reasons.append("Playbook TTL expired.")

    if direction in ("LONG", "SHORT"):
        if source_decision in ("LONG", "SHORT") and direction != source_decision:
            reasons.append(f"Decision direction {direction} conflicts with playbook source {source_decision}.")
        if allowed_sides and direction not in allowed_sides:
            reasons.append(f"Decision direction {direction} is outside allowed_sides={allowed_sides}.")

        requested_allocation = _safe_float((decision or {}).get("allocation_pct"))
        if (
            max_allocation_pct is not None
            and requested_allocation is not None
            and requested_allocation > (max_allocation_pct + 1e-9)
        ):
            reasons.append(
                f"Requested allocation {requested_allocation:.4f}% exceeds playbook cap {max_allocation_pct:.4f}%."
            )

    return {
        "status": "VETO" if reasons else "PASS",
        "reasons": reasons,
        "playbook_id": context.get("playbook_id"),
        "source_decision": source_decision,
        "allowed_sides": allowed_sides,
        "max_allocation_pct": max_allocation_pct,
        "strategy_version": context.get("strategy_version"),
        "thesis_id": context.get("thesis_id"),
        "trigger_reason": context.get("trigger_reason", ""),
    }
