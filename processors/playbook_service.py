# -*- coding: utf-8 -*-
from loguru import logger
from config.database import db

def get_recent_market_regime(symbol: str) -> str:
    """Read latest persisted market regime from ai_reports."""
    try:
        report = db.get_latest_report(symbol=symbol)
        if isinstance(report, dict):
            regime = str(report.get("market_regime", "") or "").upper().strip()
            if regime:
                return regime
    except Exception:
        pass
    return "UNKNOWN"

def format_playbook_conditions(items: list, limit: int = 2) -> list[str]:
    formatted: list[str] = []
    for item in (items or [])[:limit]:
        if isinstance(item, dict):
            metric = str(item.get("metric") or "metric")
            op = str(item.get("operator") or "?")
            value = item.get("value")
            formatted.append(f"{metric} {op} {value}")
        elif isinstance(item, str) and item.strip():
            formatted.append(item.strip())
    return formatted

def get_latest_playbook_snapshot(symbol: str) -> dict:
    try:
        response = (
            db.client.table("daily_playbooks")
            .select("mode,source_decision,playbook,created_at")
            .eq("symbol", symbol)
            .order("created_at", desc=True)
            .limit(6)
            .execute()
        )
        rows = response.data or []
        snapshot: dict = {}
        for row in rows:
            mode = str(row.get("mode") or "").lower()
            if mode not in {"swing", "position"} or mode in snapshot:
                continue
            playbook = row.get("playbook", {}) if isinstance(row.get("playbook", {}), dict) else {}
            snapshot[mode] = {
                "source_decision": str(row.get("source_decision") or "HOLD").upper(),
                "max_allocation_pct": playbook.get("max_allocation_pct"),
                "entry_conditions": format_playbook_conditions(playbook.get("entry_conditions", [])),
                "invalidation_conditions": format_playbook_conditions(playbook.get("invalidation_conditions", []), limit=1),
                "created_at": row.get("created_at"),
            }
        return snapshot
    except Exception as e:
        logger.warning(f"Failed to load playbook snapshot for {symbol}: {e}")
        return {}
