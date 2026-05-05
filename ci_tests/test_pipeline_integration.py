"""Pipeline integration tests — 전체 흐름이 크래시 없이 완주하는지 검증.

목표: 모든 LLM / DB / GCS 외부 호출을 mock으로 대체하고
      run_snapshot_analysis_hot_path 및 daily playbook이
      status=FAILED 없이 완주하는지 확인.
"""
from __future__ import annotations

import json
import os
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

os.environ.setdefault("USE_SECRET_MANAGER", "false")


# ── 공통 mock 데이터 ────────────────────────────────────────────────────────────

_JUDGE_RESPONSE = json.dumps({
    "decision": "HOLD",
    "confidence": 60,
    "win_probability_pct": 50,
    "reasoning": {
        "bull_case": "test",
        "bear_case": "test",
        "final_logic": "No strong signal.",
    },
    "entry_price": None,
    "stop_loss": None,
    "take_profit": None,
    "position_size_pct": 0,
    "daily_dual_plan": {},
})

_MOCK_BUNDLE = {
    "symbol": "BTCUSDT",
    "mode": "swing",
    "agents": {
        "market_snapshot_agent": {
            "status": "fresh",
            "expires_at": "2099-01-01T00:00:00+00:00",
            "payload": {
                "market_data_compact": "BTC price: 95000",
                "market_data": {"current_price": 95000},
                "active_orders": [],
                "open_positions": "",
                "regime_context": {"regime": "RANGE_BOUND"},
            },
        },
        "narrative_agent": {
            "status": "fresh",
            "expires_at": "2099-01-01T00:00:00+00:00",
            "payload": {
                "narrative_text": "Neutral market.",
                "rag_context": "",
                "telegram_news": "",
                "feedback_text": "",
            },
        },
        "funding_liq_agent": {
            "status": "fresh",
            "expires_at": "2099-01-01T00:00:00+00:00",
            "payload": {
                "funding_context": "Funding neutral.",
                "liquidation_context": "",
                "raw_funding": {},
            },
        },
        "onchain_agent": {
            "status": "fresh",
            "expires_at": "2099-01-01T00:00:00+00:00",
            "payload": {
                "onchain_context": "",
                "onchain_snapshot": {},
                "onchain_gate": {"allow_long": True, "allow_short": True},
            },
        },
        "chart_prep_agent": {
            "status": "fresh",
            "expires_at": "2099-01-01T00:00:00+00:00",
            "payload": {
                "chart_bytes": None,
                "chart_image_b64": None,
                "vlm_context_text": "",
                "confluence_score": {"score": 0, "gate_passed": False, "direction": "NEUTRAL", "factors": []},
                "chart_rules": {},
                "vlm_geometry": {},
            },
        },
    },
}


# ── 1. run_snapshot_analysis_hot_path 완주 ─────────────────────────────────────

def test_hot_path_completes_without_crash():
    from config.settings import TradingMode
    from executors.report_hot_path import run_snapshot_analysis_hot_path
    from executors.agent_state_store import agent_state_store

    with patch.object(agent_state_store, "load_bundle", return_value=_MOCK_BUNDLE), \
         patch("agents.judge_agent.ai_client.generate_response", return_value=_JUDGE_RESPONSE), \
         patch("executors.risk_policy_engine.risk_policy_engine.apply", side_effect=lambda d, b, **kw: d), \
         patch("executors.report_generator.report_generator.generate_report_from_snapshot", return_value={"report_id": "test-1"}), \
         patch("executors.report_generator.report_generator.notify", return_value=None), \
         patch("executors.performance_telemetry.performance_telemetry.log_hot_path_run", return_value=None):

        result = run_snapshot_analysis_hot_path(
            "BTCUSDT",
            TradingMode.SWING,
            wait_budget_s=0.0,
            notification_context="test",
        )

    assert isinstance(result, dict), "결과가 dict여야 함"
    assert "final_decision" in result, "final_decision 키 없음"
    assert result["final_decision"].get("decision") == "HOLD"


def test_hot_path_gate_closed_uses_lite_model():
    """gate=CLOSED 시 judge_lite 라우팅이 작동하는지 검증."""
    from config.settings import TradingMode
    from executors.report_hot_path import run_snapshot_analysis_hot_path
    from executors.agent_state_store import agent_state_store

    captured_roles = []

    def capture_role(**kwargs):
        captured_roles.append(kwargs.get("role"))
        return _JUDGE_RESPONSE

    with patch.object(agent_state_store, "load_bundle", return_value=_MOCK_BUNDLE), \
         patch("agents.judge_agent.ai_client.generate_response", side_effect=capture_role), \
         patch("executors.risk_policy_engine.risk_policy_engine.apply", side_effect=lambda d, b, **kw: d), \
         patch("executors.report_generator.report_generator.generate_report_from_snapshot", return_value={"report_id": "test-2"}), \
         patch("executors.report_generator.report_generator.notify", return_value=None), \
         patch("executors.performance_telemetry.performance_telemetry.log_hot_path_run", return_value=None):

        run_snapshot_analysis_hot_path(
            "BTCUSDT",
            TradingMode.SWING,
            wait_budget_s=0.0,
            notification_context="test",
        )

    # bundle의 confluence_score.gate_passed=False → judge_lite 사용
    assert captured_roles, "Judge가 호출되지 않음"
    assert captured_roles[0] == "judge_lite", (
        f"gate=CLOSED인데 judge_lite가 아닌 '{captured_roles[0]}' 사용됨"
    )


# ── 2. daily playbook 완주 ─────────────────────────────────────────────────────

def test_daily_playbook_does_not_return_failed():
    """run_daily_playbook_for_symbol이 status=FAILED를 반환하지 않는지 검증."""
    import executors.orchestrator as orch
    from config.settings import TradingMode
    from executors.agent_state_store import agent_state_store

    orchestrator = orch.Orchestrator()

    with patch.object(agent_state_store, "load_bundle", return_value=_MOCK_BUNDLE), \
         patch.object(orch, "refresh_snapshot_bundle", return_value=None), \
         patch("agents.judge_agent.ai_client.generate_response", return_value=_JUDGE_RESPONSE), \
         patch("executors.risk_policy_engine.risk_policy_engine.apply", side_effect=lambda d, b, **kw: d), \
         patch("executors.report_generator.report_generator.generate_report_from_snapshot", return_value={"report_id": "test-3", "id": "test-3"}), \
         patch("executors.report_generator.report_generator.notify", return_value=None), \
         patch("executors.performance_telemetry.performance_telemetry.log_hot_path_run", return_value=None), \
         patch.object(orch, "node_generate_playbook", return_value={}), \
         patch("executors.execution_repository.execution_repository.enqueue_outbox_event", return_value="ok"), \
         patch("executors.outbox_dispatcher.outbox_dispatcher.publish_pending", return_value={}):

        result = orchestrator.run_daily_playbook_for_symbol("BTCUSDT", TradingMode.SWING)

    assert result.get("status") != "FAILED", (
        f"daily playbook FAILED: {result.get('error', '')[:300]}"
    )


# ── 3. outbox dispatcher — JSONB 타입 안전성 ──────────────────────────────────

def test_outbox_dispatcher_handles_dict_payload():
    """psycopg2 JSONB 자동 역직렬화 후 dict가 와도 crash 없이 처리."""
    from executors.outbox_dispatcher import OutboxDispatcher
    from executors.execution_repository import execution_repository

    dispatcher = OutboxDispatcher()
    dict_payload = {"chat_id": "123", "text": "hello", "parse_mode": None}

    # payload_json이 이미 dict인 row (psycopg2 JSONB 자동 역직렬화 시뮬레이션)
    mock_row = {
        "event_id": "test-uuid",
        "event_type": "telegram_message",
        "payload_json": dict_payload,
    }

    with patch.object(execution_repository, "claim_pending_outbox_events", return_value=[mock_row]), \
         patch.object(execution_repository, "mark_outbox_event_published", return_value=None), \
         patch.object(dispatcher, "_send_telegram_message", return_value=None):

        result = dispatcher.publish_pending(limit=1)

    assert result["failed"] == 0, f"dict payload 처리 실패: {result['errors']}"
    assert result["published"] == 1
