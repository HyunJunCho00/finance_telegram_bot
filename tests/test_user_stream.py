"""
WS User Data Stream 단위 테스트

Run:
    python tests/test_user_stream.py
"""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock


def _make_fill_event(order_id: str, avg_price: float, status: str = "FILLED") -> str:
    return json.dumps({
        "e": "ORDER_TRADE_UPDATE",
        "E": 1712534400000,
        "T": 1712534400000,
        "o": {
            "s": "BTCUSDT",
            "i": order_id,
            "S": "BUY",
            "o": "MARKET",
            "X": status,
            "ap": str(avg_price),
            "z": "0.01",
            "L": str(avg_price),
        }
    })


# ── Test 1: FILLED 이벤트 → DB 업데이트 ───────────────────────────────────────

def test_filled_event_updates_db():
    """FILLED 이벤트 수신 시 DB fill price 즉시 업데이트."""
    from collectors.ws_user_stream import WsUserStream

    stream = WsUserStream()
    updated = {}

    with patch("collectors.ws_user_stream.db") as mock_db:
        def fake_update(order_id, price):
            updated[order_id] = price
        mock_db.update_trade_execution_fill_price = fake_update

        stream._handle(_make_fill_event("order-123", 68500.0, "FILLED"))

    assert "order-123" in updated, "DB should be updated on FILLED"
    assert updated["order-123"] == 68500.0
    print(f"  OK FILLED -> DB updated: order-123 @ {updated['order-123']}")


# ── Test 2: PARTIALLY_FILLED → DB 업데이트 ───────────────────────────────────

def test_partially_filled_updates_db():
    """PARTIALLY_FILLED도 중간 체결가를 DB에 기록."""
    from collectors.ws_user_stream import WsUserStream

    stream = WsUserStream()
    updated = {}

    with patch("collectors.ws_user_stream.db") as mock_db:
        mock_db.update_trade_execution_fill_price = lambda oid, p: updated.update({oid: p})
        stream._handle(_make_fill_event("order-456", 67900.0, "PARTIALLY_FILLED"))

    assert "order-456" in updated
    print(f"  OK PARTIALLY_FILLED -> DB updated: {updated['order-456']}")


# ── Test 3: 무관한 이벤트 무시 ────────────────────────────────────────────────

def test_irrelevant_events_ignored():
    """체결 무관 이벤트(NEW, CANCELED 등)는 DB 업데이트 안 함."""
    from collectors.ws_user_stream import WsUserStream

    stream = WsUserStream()
    called = []

    with patch("collectors.ws_user_stream.db") as mock_db:
        mock_db.update_trade_execution_fill_price = lambda *a: called.append(a)

        # NEW 상태
        stream._handle(_make_fill_event("order-789", 68000.0, "NEW"))
        # CANCELED 상태
        stream._handle(_make_fill_event("order-789", 68000.0, "CANCELED"))
        # 전혀 다른 이벤트 타입
        stream._handle(json.dumps({"e": "ACCOUNT_UPDATE", "data": {}}))

    assert len(called) == 0, f"Should not update DB for non-fill events, got: {called}"
    print("  OK irrelevant events: DB not touched")


# ── Test 4: avg_price=0 → 무시 ───────────────────────────────────────────────

def test_zero_price_ignored():
    """avg_price=0 (아직 체결 전) 이면 DB 업데이트 안 함."""
    from collectors.ws_user_stream import WsUserStream

    stream = WsUserStream()
    called = []

    with patch("collectors.ws_user_stream.db") as mock_db:
        mock_db.update_trade_execution_fill_price = lambda *a: called.append(a)
        stream._handle(_make_fill_event("order-000", 0.0, "FILLED"))

    assert len(called) == 0
    print("  OK zero price: DB not touched")


# ── Test 5: Paper 모드에서 start() skip ──────────────────────────────────────

def test_start_skipped_in_paper_mode():
    """Paper 모드 또는 API key 없을 때 스트림 시작 안 함."""
    from collectors.ws_user_stream import WsUserStream

    stream = WsUserStream()

    with patch("collectors.ws_user_stream.settings") as mock_s:
        mock_s.PAPER_TRADING_MODE = True
        mock_s.BINANCE_API_KEY = "test-key"
        stream.start()

    assert not stream._running, "Should not start in paper mode"
    print("  OK paper mode: stream not started")


# ── Test 6: 손상된 JSON 무시 ─────────────────────────────────────────────────

def test_malformed_json_ignored():
    """손상된 메시지는 조용히 무시 (스트림 죽지 않음)."""
    from collectors.ws_user_stream import WsUserStream

    stream = WsUserStream()
    try:
        stream._handle("{{not valid json")
        stream._handle("")
        stream._handle(json.dumps({}))
    except Exception as e:
        assert False, f"Should not raise: {e}"

    print("  OK malformed JSON: handled gracefully")


if __name__ == "__main__":
    print("=" * 50)
    print("  WS User Data Stream Test")
    print("=" * 50)

    tests = [
        test_filled_event_updates_db,
        test_partially_filled_updates_db,
        test_irrelevant_events_ignored,
        test_zero_price_ignored,
        test_start_skipped_in_paper_mode,
        test_malformed_json_ignored,
    ]

    passed = 0
    for t in tests:
        print(f"\n[{t.__name__}]")
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")

    print(f"\n{'='*50}")
    print(f"  Result: {passed}/{len(tests)} passed")
    print("=" * 50)
