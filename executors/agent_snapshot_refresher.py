from __future__ import annotations

from datetime import datetime, timedelta, timezone
import time
from typing import Dict

from loguru import logger

from config.settings import TradingMode
from executors.agent_state_store import agent_state_store
from executors.performance_telemetry import performance_telemetry

_REGISTERED = False


def _iso_after(*, minutes: int = 0, hours: int = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=minutes, hours=hours)).isoformat()


def _base_state(symbol: str, mode: TradingMode, allow_perplexity: bool, notification_context: str) -> Dict:
    return {
        "symbol": symbol,
        "mode": mode.value,
        "is_emergency": False,
        "execute_trades": False,
        "allow_perplexity": allow_perplexity,
        "notification_context": notification_context,
        "errors": [],
        "blackboard": {},
        "active_orders": [],
        "open_positions": "",
        "onchain_snapshot": {},
        "onchain_context": "",
        "onchain_gate": {},
        "raw_funding": {},
        "market_regime": "RANGE_BOUND",
        "regime_context": {},
        "chart_image_b64": None,
        "chart_bytes": None,
        "vlm_context_text": "",
        "current_bias": "neutral",
        "scenario_revision_reason": "",
        "active_setup": {},
    }


def _apply_update(state: Dict, update: Dict) -> None:
    update = update or {}
    if not update:
        return
    blackboard_update = update.get("blackboard")
    if isinstance(blackboard_update, dict):
        current_blackboard = state.get("blackboard", {}) or {}
        merged_blackboard = current_blackboard.copy()
        merged_blackboard.update(blackboard_update)
        state["blackboard"] = merged_blackboard

    for key, value in update.items():
        if key == "blackboard":
            continue
        state[key] = value


def refresh_market_snapshot(symbol: str, mode: TradingMode, *, wait_budget_s: float = 0.0, allow_perplexity: bool = False) -> None:
    import executors.orchestrator as orchestrator_module

    started = time.perf_counter()
    state = _base_state(symbol, mode, allow_perplexity, "snapshot_market")
    orchestrator_module._clear_symbol_mode_caches(symbol, mode)
    _apply_update(state, orchestrator_module.node_collect_data(state))

    market_data = orchestrator_module._market_data_cache.get(orchestrator_module._cache_key(symbol, mode), {}) or {}
    payload = {
        "market_data_compact": state.get("market_data_compact", ""),
        "market_data": market_data,
        "active_orders": state.get("active_orders", []),
        "open_positions": state.get("open_positions", ""),
        "df_size": state.get("df_size", 0),
    }
    expires_at = _iso_after(minutes=5)
    latency_ms = (time.perf_counter() - started) * 1000.0
    agent_state_store.upsert_agent_state(
        symbol,
        mode.value,
        "market_snapshot_agent",
        payload,
        expires_at=expires_at,
        latency_ms=latency_ms,
    )
    performance_telemetry.log_snapshot_refresh(
        symbol=symbol,
        mode=mode.value,
        stage="market_snapshot_agent",
        allow_perplexity=allow_perplexity,
        latency_ms=latency_ms,
        details={"df_size": state.get("df_size", 0)},
    )


def refresh_context_bundle(
    symbol: str,
    mode: TradingMode,
    *,
    wait_budget_s: float = 0.0,
    allow_perplexity: bool = False,
    include_meta: bool = True,
) -> None:
    import executors.orchestrator as orchestrator_module

    started = time.perf_counter()
    state = _base_state(symbol, mode, allow_perplexity, "snapshot_context")
    orchestrator_module._clear_symbol_mode_caches(symbol, mode)
    _apply_update(state, orchestrator_module.node_collect_data(state))
    _apply_update(state, orchestrator_module.node_context_gathering(state))
    if include_meta:
        _apply_update(state, orchestrator_module.node_meta_agent(state))
    else:
        state["regime_context"] = {
            "regime": state.get("market_regime", "RANGE_BOUND"),
            "rationale": "Meta agent disabled for snapshot prewarm",
            "trust_directive": "Use cached/deterministic context until precision/trigger analysis runs.",
            "risk_budget_pct": 50,
            "risk_bias": "NEUTRAL",
        }

    market_data = orchestrator_module._market_data_cache.get(orchestrator_module._cache_key(symbol, mode), {}) or {}
    market_payload = {
        "market_data_compact": state.get("market_data_compact", ""),
        "market_data": market_data,
        "active_orders": state.get("active_orders", []),
        "open_positions": state.get("open_positions", ""),
        "regime_context": state.get("regime_context", {}),
        "market_regime": state.get("market_regime", "RANGE_BOUND"),
        "df_size": state.get("df_size", 0),
    }
    agent_state_store.upsert_agent_state(
        symbol,
        mode.value,
        "market_snapshot_agent",
        market_payload,
        expires_at=_iso_after(minutes=5),
        latency_ms=(time.perf_counter() - started) * 1000.0,
    )

    narrative_payload = {
        "narrative_text": state.get("narrative_text", ""),
        "rag_context": state.get("rag_context", ""),
        "telegram_news": state.get("telegram_news", ""),
        "feedback_text": state.get("feedback_text", ""),
    }
    agent_state_store.upsert_agent_state(
        symbol,
        mode.value,
        "narrative_agent",
        narrative_payload,
        expires_at=_iso_after(minutes=30),
        latency_ms=(time.perf_counter() - started) * 1000.0,
    )

    funding_payload = {
        "funding_context": state.get("funding_context", ""),
        "liquidation_context": state.get("liquidation_context", ""),
        "raw_funding": state.get("raw_funding", {}),
        "macro_context": state.get("macro_context", ""),
        "deribit_context": state.get("deribit_context", ""),
        "fear_greed_context": state.get("fear_greed_context", ""),
        "microstructure_context": state.get("microstructure_context", ""),
    }
    agent_state_store.upsert_agent_state(
        symbol,
        mode.value,
        "funding_liq_agent",
        funding_payload,
        expires_at=_iso_after(minutes=5 if mode == TradingMode.SWING else 30),
        latency_ms=(time.perf_counter() - started) * 1000.0,
    )

    onchain_payload = {
        "onchain_snapshot": state.get("onchain_snapshot", {}),
        "onchain_context": state.get("onchain_context", ""),
        "onchain_gate": state.get("onchain_gate", {}),
    }
    agent_state_store.upsert_agent_state(
        symbol,
        mode.value,
        "onchain_agent",
        onchain_payload,
        expires_at=_iso_after(hours=6),
        latency_ms=(time.perf_counter() - started) * 1000.0,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    performance_telemetry.log_snapshot_refresh(
        symbol=symbol,
        mode=mode.value,
        stage="context_bundle",
        allow_perplexity=allow_perplexity,
        latency_ms=latency_ms,
        details={
            "has_narrative": bool(narrative_payload.get("narrative_text")),
            "has_onchain": bool(onchain_payload.get("onchain_context")),
            "includes_meta": include_meta,
        },
    )


def refresh_chart_bundle(
    symbol: str,
    mode: TradingMode,
    *,
    wait_budget_s: float = 0.0,
    allow_perplexity: bool = False,
    include_meta: bool = True,
    include_vlm: bool = True,
) -> None:
    import executors.orchestrator as orchestrator_module

    started = time.perf_counter()
    state = _base_state(symbol, mode, allow_perplexity, "snapshot_chart")
    orchestrator_module._clear_symbol_mode_caches(symbol, mode)
    _apply_update(state, orchestrator_module.node_collect_data(state))
    _apply_update(state, orchestrator_module.node_context_gathering(state))
    if include_meta:
        _apply_update(state, orchestrator_module.node_meta_agent(state))
    _apply_update(state, orchestrator_module.node_triage(state))
    _apply_update(state, orchestrator_module.node_generate_chart(state))
    _apply_update(state, orchestrator_module.node_rule_based_chart(state))
    if include_vlm:
        _apply_update(state, orchestrator_module.node_vlm_geometric_expert(state))
    else:
        state.setdefault("blackboard", {})["vlm_geometry"] = {
            "anomaly": "skipped",
            "directional_bias": "NEUTRAL",
            "confidence": 0,
            "rationale": "VLM disabled for snapshot prewarm",
        }

    blackboard = state.get("blackboard", {}) or {}
    chart_payload = {
        "chart_bytes": state.get("chart_bytes"),
        "chart_image_b64": state.get("chart_image_b64"),
        "chart_rules": blackboard.get("chart_rules", {}),
        "vlm_geometry": blackboard.get("vlm_geometry", {}),
        "confluence_score": blackboard.get("confluence_score", {}),
        "vlm_context_text": state.get("vlm_context_text", ""),
        "market_data": orchestrator_module._market_data_cache.get(
            orchestrator_module._cache_key(symbol, mode),
            {},
        ) or {},
    }
    agent_state_store.upsert_agent_state(
        symbol,
        mode.value,
        "chart_prep_agent",
        chart_payload,
        expires_at=_iso_after(minutes=5),
        latency_ms=(time.perf_counter() - started) * 1000.0,
    )
    performance_telemetry.log_snapshot_refresh(
        symbol=symbol,
        mode=mode.value,
        stage="chart_prep_agent",
        allow_perplexity=allow_perplexity,
        latency_ms=(time.perf_counter() - started) * 1000.0,
        details={
            "has_chart_bytes": bool(chart_payload.get("chart_bytes")),
            "has_vlm_geometry": bool(chart_payload.get("vlm_geometry")),
        },
    )


def refresh_snapshot_bundle(
    symbol: str,
    mode: TradingMode,
    *,
    allow_perplexity: bool = False,
    include_meta: bool = True,
    include_vlm: bool = True,
) -> None:
    started = time.perf_counter()
    logger.info(
        f"Refreshing snapshot bundle for {symbol}/{mode.value} "
        f"(perplexity={'on' if allow_perplexity else 'off'}, meta={'on' if include_meta else 'off'}, vlm={'on' if include_vlm else 'off'})"
    )
    refresh_context_bundle(symbol, mode, allow_perplexity=allow_perplexity, include_meta=include_meta)
    # chart_bundle은 context_bundle이 이미 meta_agent를 실행해 저장했으므로
    # include_meta=False로 고정하여 node_meta_agent 이중 호출 방지 (Gemini Pro RPD 절감)
    try:
        refresh_chart_bundle(
            symbol,
            mode,
            allow_perplexity=allow_perplexity,
            include_meta=False,
            include_vlm=include_vlm,
        )
    except Exception as e:
        logger.warning(f"Chart bundle refresh failed for {symbol}/{mode.value}, continuing without chart: {e}")
    logger.info(f"Snapshot bundle ready for {symbol}/{mode.value}")
    performance_telemetry.log_snapshot_refresh(
        symbol=symbol,
        mode=mode.value,
        stage="snapshot_bundle",
        allow_perplexity=allow_perplexity,
        latency_ms=(time.perf_counter() - started) * 1000.0,
        details={
            "includes_chart_bundle": True,
            "includes_context_bundle": True,
            "includes_meta": include_meta,
            "includes_vlm": include_vlm,
        },
    )


def ensure_registered() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    from executors.report_hot_path import register_agent_refresher

    register_agent_refresher("market_snapshot_agent", refresh_market_snapshot)
    register_agent_refresher("narrative_agent", refresh_context_bundle)
    register_agent_refresher("funding_liq_agent", refresh_context_bundle)
    register_agent_refresher("onchain_agent", refresh_context_bundle)
    register_agent_refresher("chart_prep_agent", refresh_chart_bundle)
    _REGISTERED = True
