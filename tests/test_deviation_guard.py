"""
편차 검사(DeviationGuard) 단위 테스트.

Run:
    python -m pytest tests/test_deviation_guard.py -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock


def _make_decision(entry_price: float, direction: str = "LONG") -> dict:
    return {
        "decision": direction,
        "entry_price": entry_price,
        "allocation_pct": 50.0,
        "leverage": 1,
        "target_exchange": "binance",
        "recommended_execution_style": "MOMENTUM_SNIPER",
        "tp1_price": entry_price * 1.05,
        "stop_loss": entry_price * 0.97,
        "take_profit": entry_price * 1.05,
        "tp1_exit_pct": 50.0,
    }


def _run(analysis_price: float, current_price: float, threshold: float = 2.0):
    """편차 검사만 실행하는 헬퍼."""
    from executors.trade_executor import TradeExecutor

    executor = TradeExecutor.__new__(TradeExecutor)

    with patch("executors.trade_executor.settings") as mock_settings, \
         patch.object(executor, "_get_reference_price", return_value=current_price), \
         patch("executors.trade_executor.execution_repository"), \
         patch("executors.trade_executor.state_manager") as mock_state:

        mock_settings.PAPER_TRADING_MODE = True
        mock_settings.MAX_ENTRY_DEVIATION_PCT = threshold
        mock_settings.BINANCE_PAPER_BALANCE_USD = 2000.0
        mock_settings.UPBIT_PAPER_BALANCE_USD = 1500.0
        mock_state.get_active_orders.return_value = []

        # paper_engine mock
        with patch("executors.paper_exchange.paper_engine") as mock_pe:
            mock_pe.get_open_positions.return_value = []
            mock_pe.get_wallet_balance.return_value = 2000.0
            mock_state.get_reserved_margin.return_value = 0.0

            decision = _make_decision(analysis_price)
            return executor.execute_from_decision(decision, mode="SWING", symbol="BTCUSDT")


# ── 테스트 ───────────────────────────────────────────────────────────────────

def test_within_threshold_passes():
    """편차 1% — threshold 2% -> 통과해야 함."""
    result = _run(analysis_price=68_000, current_price=68_680)  # +1.0%
    # DeviationGuard가 막지 않았다면 success or budget 관련 note
    assert "price moved" not in result.get("note", ""), (
        f"Unexpected block: {result}"
    )
    print(f"  OK 1% 편차 -> 통과: {result.get('note') or result.get('success')}")


def test_exceeds_threshold_blocked():
    """편차 2.5% — threshold 2% -> 스킵되어야 함."""
    result = _run(analysis_price=68_000, current_price=69_700)  # +2.5%
    assert result["success"] is False
    assert "price moved" in result.get("note", "")
    print(f"  OK 2.5% deviation -> skipped")


def test_no_analysis_price_skips_guard():
    """entry_price 없으면 편차 검사 건너뜀 (LLM이 안 줬을 때 안전하게 진행)."""
    from executors.trade_executor import TradeExecutor
    executor = TradeExecutor.__new__(TradeExecutor)

    with patch("executors.trade_executor.settings") as mock_settings, \
         patch.object(executor, "_get_reference_price", return_value=68_000), \
         patch("executors.trade_executor.execution_repository"), \
         patch("executors.trade_executor.state_manager") as mock_state:

        mock_settings.PAPER_TRADING_MODE = True
        mock_settings.MAX_ENTRY_DEVIATION_PCT = 2.0
        mock_settings.BINANCE_PAPER_BALANCE_USD = 2000.0
        mock_settings.UPBIT_PAPER_BALANCE_USD = 1500.0
        mock_state.get_active_orders.return_value = []

        decision = _make_decision(0)   # entry_price = 0 (없음)
        decision["entry_price"] = 0

        with patch("executors.paper_exchange.paper_engine") as mock_pe:
            mock_pe.get_open_positions.return_value = []
            mock_pe.get_wallet_balance.return_value = 2000.0
            mock_state.get_reserved_margin.return_value = 0.0

            result = executor.execute_from_decision(decision, mode="SWING", symbol="BTCUSDT")

        assert "price moved" not in result.get("note", "")
        print(f"  OK entry_price 없음 -> 가드 스킵: {result.get('note') or result.get('success')}")


def test_threshold_zero_disables_guard():
    """MAX_ENTRY_DEVIATION_PCT=0 -> 편차 검사 비활성화."""
    result = _run(analysis_price=68_000, current_price=75_000, threshold=0.0)  # +10%
    assert "price moved" not in result.get("note", "")
    print(f"  OK threshold=0 -> 가드 비활성화: {result.get('note') or result.get('success')}")


if __name__ == "__main__":
    print("=" * 50)
    print("  DeviationGuard Test")
    print("=" * 50)
    tests = [
        test_within_threshold_passes,
        test_exceeds_threshold_blocked,
        test_no_analysis_price_skips_guard,
        test_threshold_zero_disables_guard,
    ]
    for t in tests:
        print(f"\n[{t.__name__}]")
        try:
            t()
        except Exception as e:
            print(f"  FAIL: {e}")
    print("\nDone.")
