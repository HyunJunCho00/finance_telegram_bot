"""Smoke tests — import & minimal node call.

목표: 외부 연결 없이 노드 함수가 크래시 없이 호출되는지만 검증.
      오늘처럼 UnboundLocalError / NameError 같은 연결 버그를 배포 전에 차단.
"""
from __future__ import annotations

import os
import types
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

os.environ.setdefault("USE_SECRET_MANAGER", "false")


# ── 공통 픽스처 ────────────────────────────────────────────────────────────────

def _minimal_state(symbol: str = "BTCUSDT", mode: str = "swing") -> dict:
    return {
        "symbol": symbol,
        "mode": mode,
        "is_emergency": False,
        "execute_trades": False,
        "allow_perplexity": False,
        "notification_context": "test",
        "market_data_compact": "",
        "narrative_text": "",
        "rag_context": "",
        "telegram_news": "",
        "feedback_text": "",
        "funding_context": "",
        "liquidation_context": "",
        "raw_funding": {},
        "onchain_context": "",
        "onchain_snapshot": {},
        "onchain_gate": {},
        "active_orders": [],
        "open_positions": "",
        "regime_context": {"regime": "RANGE_BOUND"},
        "market_regime": "RANGE_BOUND",
        "blackboard": {"confluence_score": {"score": 0, "gate_passed": False, "direction": "NEUTRAL", "factors": []}},
        "df_size": 0,
    }


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])


# ── 1. Import smoke ────────────────────────────────────────────────────────────

def test_import_orchestrator():
    import executors.orchestrator  # noqa: F401


def test_import_report_hot_path():
    import executors.report_hot_path  # noqa: F401


def test_import_agent_snapshot_refresher():
    import executors.agent_snapshot_refresher  # noqa: F401


def test_import_judge_agent():
    import agents.judge_agent  # noqa: F401


def test_import_ai_router():
    import agents.ai_router  # noqa: F401


# ── 2. node_generate_chart — UnboundLocalError 재발 방지 ───────────────────────
# _df_cache 비어있을 때(cold start) gcs_parquet_store fallback 경로가 터지지 않는지 검증.
# 이 테스트가 실패하면 오늘 버그가 재발한 것.

def test_node_generate_chart_cold_cache_no_crash():
    import executors.orchestrator as orch

    orch._df_cache.clear()

    with patch.object(orch.gcs_parquet_store, "load_ohlcv", return_value=_empty_df()), \
         patch.object(orch.gcs_parquet_store, "enabled", False):
        result = orch.node_generate_chart(_minimal_state())

    assert isinstance(result, dict)
    # empty df → early return with Nones
    assert result.get("chart_image_b64") is None
    assert result.get("chart_bytes") is None


def test_node_generate_chart_does_not_reference_local_before_assignment():
    """gcs_parquet_store가 로컬 바인딩 없이 모듈 레벨 임포트만 쓰는지 확인."""
    import ast, inspect, executors.orchestrator as orch

    source = inspect.getsource(orch.node_generate_chart)
    tree = ast.parse(source)

    local_imports = [
        node.names[0].name if isinstance(node, ast.Import) else node.module
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        if any(
            alias.name == "gcs_parquet_store" or (isinstance(node, ast.ImportFrom) and "gcs_parquet" in (node.module or ""))
            for alias in (node.names if isinstance(node, ast.Import) else node.names)
        )
    ]
    assert local_imports == [], (
        f"node_generate_chart 내부에 gcs_parquet_store 로컬 임포트 재발: {local_imports}"
    )


# ── 3. node_collect_data smoke ─────────────────────────────────────────────────

def test_node_collect_data_returns_dict():
    import executors.orchestrator as orch

    mock_df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="1min", tz="UTC"),
        "open": [1.0] * 10, "high": [1.1] * 10,
        "low": [0.9] * 10, "close": [1.0] * 10, "volume": [100.0] * 10,
    })

    with patch.object(orch.gcs_parquet_store, "load_ohlcv", return_value=mock_df), \
         patch.object(orch.gcs_parquet_store, "enabled", True), \
         patch.object(orch.db, "get_market_data_gap", return_value=pd.DataFrame()), \
         patch.object(orch.state_manager, "get_active_orders", return_value=[]):
        result = orch.node_collect_data(_minimal_state())

    assert isinstance(result, dict)


# ── 4. node_meta_agent smoke ───────────────────────────────────────────────────

def test_node_meta_agent_returns_regime():
    import executors.orchestrator as orch
    from agents.ai_router import ai_client

    with patch.object(ai_client, "generate_response", return_value="RANGE_BOUND"):
        result = orch.node_meta_agent(_minimal_state())

    assert isinstance(result, dict)
    assert "market_regime" in result


# ── 5. refresh_snapshot_bundle smoke ──────────────────────────────────────────

def test_refresh_snapshot_bundle_no_crash():
    """전체 snapshot refresh 흐름이 crash 없이 완주하는지 검증."""
    import executors.orchestrator as orch
    import executors.agent_snapshot_refresher as refresher
    from config.settings import TradingMode

    with patch.object(orch, "node_collect_data", return_value={"df_size": 0, "market_data_compact": "", "active_orders": [], "open_positions": ""}), \
         patch.object(orch, "node_context_gathering", return_value={"narrative_text": "", "rag_context": "", "telegram_news": "", "feedback_text": "", "funding_context": "", "liquidation_context": "", "raw_funding": {}, "onchain_context": "", "onchain_snapshot": {}, "onchain_gate": {}}), \
         patch.object(orch, "node_meta_agent", return_value={"market_regime": "RANGE_BOUND", "regime_context": {"regime": "RANGE_BOUND"}}), \
         patch.object(orch, "node_triage", return_value={"blackboard": {"confluence_score": {"score": 0, "gate_passed": False}}}), \
         patch.object(orch, "node_generate_chart", return_value={"chart_bytes": None, "chart_image_b64": None, "vlm_context_text": ""}), \
         patch.object(orch, "node_rule_based_chart", return_value={"blackboard": {"chart_rules": {}}}), \
         patch.object(orch, "node_vlm_geometric_expert", return_value={"blackboard": {"vlm_geometry": {}}}), \
         patch.object(orch, "_clear_symbol_mode_caches", return_value=None):
        refresher.refresh_snapshot_bundle("BTCUSDT", TradingMode.SWING, allow_perplexity=False)
