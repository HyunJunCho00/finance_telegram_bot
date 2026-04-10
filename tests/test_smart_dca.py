"""
SMART_DCA 분할 지정가 + 자동 취소 타이머 테스트

Run:
    python tests/test_smart_dca.py
"""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock, call


def _make_order(total: float = 1000.0, remaining: float = None, style: str = "SMART_DCA") -> dict:
    return {
        "intent_id": "test-intent-001",
        "symbol": "BTCUSDT",
        "direction": "LONG",
        "execution_style": style,
        "total_target_amount": total,
        "remaining_amount": remaining if remaining is not None else total,
        "exchange": "binance",
        "status": "ACTIVE",
        "leverage": 1,
        "tp_price": 72000.0,
        "sl_price": 65000.0,
        "tp2_price": 75000.0,
        "tp1_exit_pct": 50.0,
        "playbook_id": "", "source_decision": "",
        "strategy_version": "", "trigger_reason": "", "thesis_id": "",
    }


# ── Test 1: 첫 실행 — 시장가 1번 + 지정가 2번 ────────────────────────────────

def test_smart_dca_first_execution():
    """첫 실행 시 1번 시장가 + 2번 지정가 등록."""
    from executors.order_manager import ExecutionDesk

    desk = ExecutionDesk()
    market_calls = []
    limit_calls = []

    def fake_execute_chunk(intent_id, symbol, direction, amount, exchange, style,
                           leverage, tp, sl, tp2, tp1_pct, lineage):
        market_calls.append(amount)

    def fake_execute_limit(symbol, side, amount, leverage, limit_price, exchange):
        limit_calls.append({"amount": amount, "price": limit_price})
        return {"success": True, "order_id": f"fake-{len(limit_calls)}"}

    def fake_get_price(symbol, exchange="binance"):
        return 68_000.0

    with patch("executors.order_manager.settings") as mock_s, \
         patch("executors.order_manager.trade_executor") as mock_te, \
         patch("executors.order_manager.state_manager") as mock_sm:

        mock_s.PAPER_TRADING_MODE = False
        mock_s.SMART_DCA_LIMIT_TTL_MINUTES = 240
        mock_te._get_reference_price = fake_get_price
        mock_te.execute_limit = fake_execute_limit
        mock_sm.get_active_orders.return_value = [_make_order(1000.0, 1000.0)]
        mock_sm.flush_expired = MagicMock()
        mock_sm.update_status = MagicMock()

        desk._execute_chunk = fake_execute_chunk
        desk.process_intents()

    # 시장가 1번: 40% = $400
    assert len(market_calls) == 1, f"Expected 1 market call, got {market_calls}"
    assert abs(market_calls[0] - 400.0) < 1.0, f"Expected $400, got ${market_calls[0]}"

    # 지정가 2번
    assert len(limit_calls) == 2, f"Expected 2 limit orders, got {limit_calls}"
    assert abs(limit_calls[0]["amount"] - 350.0) < 1.0  # 35%
    assert abs(limit_calls[1]["amount"] - 250.0) < 1.0  # 25%

    # 지정가 가격: 현재가(68000) - 0.5/1.0 * ATR(2% = 1360)
    atr = 68_000 * 0.02
    assert abs(limit_calls[0]["price"] - (68_000 - 0.5 * atr)) < 5.0
    assert abs(limit_calls[1]["price"] - (68_000 - 1.0 * atr)) < 5.0

    # 지정가 추적 등록 확인
    assert len(desk._pending_limits) == 2

    print("  OK first execution: 1 market + 2 limits registered")


# ── Test 2: 이후 실행 — 중복 주문 없음 ──────────────────────────────────────

def test_smart_dca_subsequent_call_no_duplicate():
    """remaining < total 이면 추가 주문 없음."""
    from executors.order_manager import ExecutionDesk

    desk = ExecutionDesk()
    market_calls = []

    def fake_execute_chunk(*args, **kwargs):
        market_calls.append(True)

    with patch("executors.order_manager.settings") as mock_s, \
         patch("executors.order_manager.trade_executor") as mock_te, \
         patch("executors.order_manager.state_manager") as mock_sm:

        mock_s.PAPER_TRADING_MODE = False
        mock_s.SMART_DCA_LIMIT_TTL_MINUTES = 240
        # remaining 이 total의 50% — 이미 첫 진입 완료된 상태
        mock_sm.get_active_orders.return_value = [_make_order(1000.0, remaining=500.0)]
        mock_sm.flush_expired = MagicMock()
        mock_sm.update_status = MagicMock()

        desk._execute_chunk = fake_execute_chunk
        desk.process_intents()

    assert len(market_calls) == 0, "Should not place orders on subsequent calls"
    print("  OK subsequent call: no duplicate orders")


# ── Test 3: 지정가 TTL 만료 → 자동 취소 ──────────────────────────────────────

def test_limit_order_auto_cancel_on_expiry():
    """TTL 만료된 지정가 주문이 자동으로 취소됨."""
    from executors.order_manager import ExecutionDesk

    desk = ExecutionDesk()
    cancelled = []

    # 이미 만료된 지정가 주문 추가 (expires_ts = 과거)
    desk._pending_limits["order-expired-1"] = {
        "symbol": "BTCUSDT",
        "exchange": "binance",
        "side": "LONG",
        "intent_id": "test-intent-001",
        "expires_ts": time.monotonic() - 1.0,  # 이미 만료
        "placed_at": "2026-04-08T00:00:00+00:00",
    }

    with patch("executors.order_manager.trade_executor") as mock_te, \
         patch("executors.order_manager.state_manager") as mock_sm, \
         patch("executors.order_manager.settings") as mock_s:

        mock_s.PAPER_TRADING_MODE = False
        mock_s.SMART_DCA_LIMIT_TTL_MINUTES = 240
        mock_sm.get_active_orders.return_value = []
        mock_sm.flush_expired = MagicMock()

        def fake_cancel(exchange, order_id, symbol):
            cancelled.append(order_id)
            return True
        mock_te.cancel_order = fake_cancel

        desk.process_intents()

    assert "order-expired-1" in cancelled, "Expired limit order should be cancelled"
    assert "order-expired-1" not in desk._pending_limits, "Should be removed from tracking"
    print("  OK TTL expired: limit order auto-cancelled")


# ── Test 4: TTL 미만 — 취소 안 함 ────────────────────────────────────────────

def test_limit_order_not_cancelled_before_ttl():
    """TTL 미만 지정가 주문은 취소하지 않음."""
    from executors.order_manager import ExecutionDesk

    desk = ExecutionDesk()
    desk._pending_limits["order-active-1"] = {
        "symbol": "BTCUSDT",
        "exchange": "binance",
        "side": "LONG",
        "intent_id": "test-intent-001",
        "expires_ts": time.monotonic() + 3600.0,  # 1시간 후 만료
        "placed_at": "2026-04-08T00:00:00+00:00",
    }

    with patch("executors.order_manager.trade_executor") as mock_te, \
         patch("executors.order_manager.state_manager") as mock_sm, \
         patch("executors.order_manager.settings") as mock_s:

        mock_s.PAPER_TRADING_MODE = False
        mock_te.cancel_order = MagicMock()
        mock_sm.get_active_orders.return_value = []
        mock_sm.flush_expired = MagicMock()

        desk.process_intents()

    mock_te.cancel_order.assert_not_called()
    assert "order-active-1" in desk._pending_limits
    print("  OK active limit: not cancelled before TTL")


# ── Test 5: Paper 모드 — 지정가 없이 시장가 3번 ──────────────────────────────

def test_smart_dca_paper_mode_all_market():
    """Paper 모드에서는 3번 모두 시장가 시뮬레이션."""
    from executors.order_manager import ExecutionDesk

    desk = ExecutionDesk()
    market_calls = []

    def fake_execute_chunk(intent_id, symbol, direction, amount, *args, **kwargs):
        market_calls.append(amount)

    with patch("executors.order_manager.settings") as mock_s, \
         patch("executors.order_manager.trade_executor") as mock_te, \
         patch("executors.order_manager.state_manager") as mock_sm:

        mock_s.PAPER_TRADING_MODE = True
        mock_s.SMART_DCA_LIMIT_TTL_MINUTES = 240
        mock_sm.get_active_orders.return_value = [_make_order(1000.0, 1000.0)]
        mock_sm.flush_expired = MagicMock()
        mock_sm.update_status = MagicMock()

        desk._execute_chunk = fake_execute_chunk
        desk.process_intents()

    assert len(market_calls) == 3, f"Paper: expected 3 market calls, got {market_calls}"
    assert abs(sum(market_calls) - 1000.0) < 1.0, "Total should equal $1000"
    assert len(desk._pending_limits) == 0, "Paper mode: no limit tracking"
    print(f"  OK paper mode: 3 market orders {[round(c) for c in market_calls]}")


if __name__ == "__main__":
    print("=" * 55)
    print("  SMART_DCA + LimitTracker Test")
    print("=" * 55)

    tests = [
        test_smart_dca_first_execution,
        test_smart_dca_subsequent_call_no_duplicate,
        test_limit_order_auto_cancel_on_expiry,
        test_limit_order_not_cancelled_before_ttl,
        test_smart_dca_paper_mode_all_market,
    ]

    passed = 0
    for t in tests:
        print(f"\n[{t.__name__}]")
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")

    print(f"\n{'='*55}")
    print(f"  Result: {passed}/{len(tests)} passed")
    print("=" * 55)
