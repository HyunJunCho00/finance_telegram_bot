from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import time
from typing import Dict, Iterable, List, Optional, Set

from loguru import logger

from agents.judge_agent import judge_agent
from config.settings import TradingMode
from executors.agent_state_store import agent_state_store
from executors.performance_telemetry import performance_telemetry
from executors.report_generator import report_generator
from executors.risk_policy_engine import risk_policy_engine


REQUIRED_AGENT_NAMES: tuple[str, ...] = (
    "market_snapshot_agent",
    "narrative_agent",
    "funding_liq_agent",
    "onchain_agent",
    "chart_prep_agent",
)

_AGENT_REFRESHERS: Dict[str, object] = {}


def register_agent_refresher(agent_name: str, refresher) -> None:
    _AGENT_REFRESHERS[str(agent_name)] = refresher


@dataclass
class FreshnessCheck:
    required: List[str]
    stale: List[str]
    missing: List[str]

    def required_missing_agents(self) -> List[str]:
        seen: Set[str] = set()
        ordered: List[str] = []
        for name in self.missing + self.stale:
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _is_agent_fresh(agent_row: Dict) -> bool:
    status = str((agent_row or {}).get("status", "fresh")).lower()
    if status == "stale":
        return False
    expires_at = _parse_iso((agent_row or {}).get("expires_at"))
    if expires_at is None:
        return status != "missing"
    return expires_at > datetime.now(timezone.utc)


def validate_freshness(
    bundle: Dict,
    required_agents: Iterable[str] = REQUIRED_AGENT_NAMES,
) -> FreshnessCheck:
    agents = (bundle or {}).get("agents", {}) or {}
    stale: List[str] = []
    missing: List[str] = []

    for name in required_agents:
        row = agents.get(name)
        if not row:
            missing.append(name)
            continue
        if not _is_agent_fresh(row):
            stale.append(name)

    return FreshnessCheck(
        required=list(required_agents),
        stale=stale,
        missing=missing,
    )


def refresh_required_agents(
    symbol: str,
    mode: TradingMode,
    required_agents: Iterable[str],
    *,
    wait_budget_s: float = 0.0,
) -> None:
    if not _AGENT_REFRESHERS:
        try:
            from executors.agent_snapshot_refresher import ensure_registered

            ensure_registered()
        except Exception as exc:
            logger.warning(f"Failed to initialize snapshot refresh registry: {exc}")
    for agent_name in required_agents:
        refresher = _AGENT_REFRESHERS.get(agent_name)
        if refresher is None:
            logger.info(f"Hot path refresh skipped: no refresher registered for {agent_name}")
            continue
        refresher(symbol, mode, wait_budget_s=wait_budget_s)


def _build_snapshot_for_judge(bundle: Dict) -> Dict:
    agents = (bundle or {}).get("agents", {}) or {}
    market_payload = deepcopy(((agents.get("market_snapshot_agent") or {}).get("payload") or {}))
    narrative_payload = deepcopy(((agents.get("narrative_agent") or {}).get("payload") or {}))
    funding_payload = deepcopy(((agents.get("funding_liq_agent") or {}).get("payload") or {}))
    onchain_payload = deepcopy(((agents.get("onchain_agent") or {}).get("payload") or {}))
    chart_payload = deepcopy(((agents.get("chart_prep_agent") or {}).get("payload") or {}))

    snapshot = {
        "symbol": (bundle or {}).get("symbol", ""),
        "mode": (bundle or {}).get("mode", TradingMode.SWING.value),
        "market_data_compact": market_payload.get("market_data_compact", ""),
        "market_data": market_payload.get("market_data", market_payload),
        "active_orders": market_payload.get("active_orders", []),
        "open_positions": market_payload.get("open_positions", ""),
        "regime_context": market_payload.get("regime_context", {}),
        "narrative_text": narrative_payload.get("narrative_text", ""),
        "rag_context": narrative_payload.get("rag_context", ""),
        "funding_context": funding_payload.get("funding_context", ""),
        "liquidation_context": funding_payload.get("liquidation_context", ""),
        "telegram_news": narrative_payload.get("telegram_news", ""),
        "feedback_text": narrative_payload.get("feedback_text", ""),
        "onchain_context": onchain_payload.get("onchain_context", ""),
        "onchain_gate": onchain_payload.get("onchain_gate", {}),
        "blackboard": {
            "confluence_score": chart_payload.get("confluence_score", {}),
            "chart_rules": chart_payload.get("chart_rules", {}),
            "vlm_geometry": chart_payload.get("vlm_geometry", {}),
        },
        "chart_bytes": chart_payload.get("chart_bytes"),
        "chart_image_b64": chart_payload.get("chart_image_b64"),
        "raw_funding": funding_payload.get("raw_funding", {}),
    }
    return snapshot


def run_report_hot_path(
    symbol: str,
    mode: TradingMode,
    *,
    wait_budget_s: float = 0.0,
    required_agents: Iterable[str] = REQUIRED_AGENT_NAMES,
    notification_context: str = "analysis",
) -> Dict:
    result = run_snapshot_analysis_hot_path(
        symbol,
        mode,
        wait_budget_s=wait_budget_s,
        required_agents=required_agents,
        notification_context=notification_context,
        execute_trades=False,
    )
    return result.get("report", {})


def _apply_execution_bridge(
    symbol: str,
    mode: TradingMode,
    final_decision: Dict,
    *,
    execute_trades: bool,
    playbook_context: Optional[Dict] = None,
) -> Dict:
    decision = deepcopy(final_decision or {})
    decision.setdefault("reasoning", {})
    if not isinstance(decision.get("reasoning"), dict):
        decision["reasoning"] = {"final_logic": str(decision.get("reasoning", "") or "")}
    if playbook_context:
        decision["playbook_context"] = deepcopy(playbook_context)

    state = {
        "symbol": symbol,
        "mode": mode.value,
        "final_decision": decision,
        "execute_trades": execute_trades,
        "playbook_context": deepcopy(playbook_context or {}),
    }

    try:
        import executors.orchestrator as orchestrator_module

        state.update(orchestrator_module.node_portfolio_leverage_guard(state))
        state.update(orchestrator_module.node_execute_trade(state))
    except Exception as exc:
        logger.error(f"Snapshot execution bridge failed for {symbol}/{mode.value}: {exc}")
        state["final_decision"] = decision
    return state.get("final_decision", decision)


def run_snapshot_analysis_hot_path(
    symbol: str,
    mode: TradingMode,
    *,
    wait_budget_s: float = 0.0,
    required_agents: Iterable[str] = REQUIRED_AGENT_NAMES,
    notification_context: str = "analysis",
    execute_trades: bool = False,
    playbook_context: Optional[Dict] = None,
) -> Dict:
    started = time.perf_counter()
    load_started = time.perf_counter()
    bundle = agent_state_store.load_bundle(symbol, mode.value)
    load_latency_ms = (time.perf_counter() - load_started) * 1000.0
    freshness = validate_freshness(bundle, required_agents=required_agents)
    missing = freshness.required_missing_agents()
    refresh_latency_ms = 0.0

    if missing:
        refresh_started = time.perf_counter()
        refresh_required_agents(symbol, mode, missing, wait_budget_s=wait_budget_s)
        refresh_latency_ms = (time.perf_counter() - refresh_started) * 1000.0
        bundle = agent_state_store.load_bundle(symbol, mode.value)

    snapshot = _build_snapshot_for_judge(bundle)
    judge_started = time.perf_counter()
    if playbook_context:
        decision = judge_agent.validate_trigger_against_playbook(snapshot, playbook_context)
    else:
        decision = judge_agent.make_decision_from_snapshot(snapshot)
    judge_latency_ms = (time.perf_counter() - judge_started) * 1000.0
    policy_started = time.perf_counter()
    final_decision = risk_policy_engine.apply(decision, bundle, playbook_context=playbook_context)
    policy_latency_ms = (time.perf_counter() - policy_started) * 1000.0
    execution_started = time.perf_counter()
    final_decision = _apply_execution_bridge(
        symbol,
        mode,
        final_decision,
        execute_trades=execute_trades,
        playbook_context=playbook_context,
    )
    execution_latency_ms = (time.perf_counter() - execution_started) * 1000.0

    report_started = time.perf_counter()
    report = report_generator.generate_report_from_snapshot(bundle, final_decision)
    report_latency_ms = (time.perf_counter() - report_started) * 1000.0
    if report:
        report_generator.notify(
            report,
            chart_bytes=snapshot.get("chart_bytes"),
            mode=mode,
            notification_context=notification_context,
        )
    else:
        logger.error(
            f"Hot path report generation returned empty payload for {symbol}/{mode.value} "
            f"context={notification_context}"
        )
    total_latency_ms = (time.perf_counter() - started) * 1000.0
    performance_telemetry.log_hot_path_run(
        symbol=symbol,
        mode=mode.value,
        notification_context=notification_context,
        execute_trades=execute_trades,
        snapshot_hit=not missing,
        missing_agents=missing,
        latency_ms=total_latency_ms,
        stage_latencies_ms={
            "load_bundle": load_latency_ms,
            "refresh": refresh_latency_ms,
            "judge": judge_latency_ms,
            "policy": policy_latency_ms,
            "execution_bridge": execution_latency_ms,
            "report": report_latency_ms,
        },
        final_decision=final_decision,
        report_generated=bool(report),
    )
    return {
        "report": report,
        "final_decision": final_decision,
        "snapshot": snapshot,
        "bundle": bundle,
    }
