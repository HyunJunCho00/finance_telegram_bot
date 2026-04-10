"""
OMS 성능 벤치마크 — 외부 서비스 없이 실행 가능

측정 항목:
  1. 가격 조회 레이턴시: 인메모리 vs REST mock
  2. DB 캐시 효과: 캐시 유무에 따른 쿼리 횟수
  3. Pre-trade 파이프라인: 단계별 처리 시간
  4. 주문 모니터링 처리량: 초당 SL/TP 체크 횟수

실행:
  python -m tests.bench_oms
"""

import time
import statistics
import random
import threading
from typing import Dict, List
from unittest.mock import MagicMock, patch


# ─────────────────────────────────────────────────────────
# 1. 가격 조회: 인메모리 vs REST mock
# ─────────────────────────────────────────────────────────

def bench_price_lookup(n: int = 100_000):
    """WebSocket 인메모리 조회 vs REST API mock 레이턴시 비교."""
    print("\n[1] 가격 조회 레이턴시 벤치마크")
    print(f"    반복 횟수: {n:,}회\n")

    # ── 인메모리 조회 (ws_price_feed 방식) ──
    _prices: Dict[str, float] = {"BTCUSDT": 85000.0, "ETHUSDT": 3200.0}
    _updated_at: Dict[str, float] = {"BTCUSDT": time.monotonic(), "ETHUSDT": time.monotonic()}
    STALE = 5.0

    def get_inmemory(symbol: str):
        ts = _updated_at.get(symbol, 0.0)
        if time.monotonic() - ts < STALE:
            return _prices.get(symbol)
        return None

    t0 = time.perf_counter()
    for _ in range(n):
        get_inmemory("BTCUSDT")
    inmem_total = time.perf_counter() - t0
    inmem_per_call = inmem_total / n * 1_000_000  # microseconds

    # ── REST API mock (blocking HTTP 시뮬레이션) ──
    def get_rest_mock(symbol: str):
        # 실제 REST: 평균 80~150ms, 여기선 최소한의 오버헤드만 시뮬레이션
        time.sleep(0.0001)  # 100µs = REST overhead 하한선 (로컬 mock 기준)
        return 85000.0

    sample_rest = 1000  # REST는 1000회만
    t0 = time.perf_counter()
    for _ in range(sample_rest):
        get_rest_mock("BTCUSDT")
    rest_total = time.perf_counter() - t0
    rest_per_call = rest_total / sample_rest * 1_000_000

    print(f"    인메모리 조회:  {inmem_per_call:.3f} µs/call  (총 {n:,}회)")
    print(f"    REST mock:      {rest_per_call:.1f} µs/call  (총 {sample_rest:,}회)")
    print(f"    → 속도 차이:    {rest_per_call / inmem_per_call:.0f}x 빠름\n")

    return {
        "inmemory_us": inmem_per_call,
        "rest_us": rest_per_call,
        "speedup": rest_per_call / inmem_per_call,
    }


# ─────────────────────────────────────────────────────────
# 2. DB 캐시 효과: 쿼리 횟수 측정
# ─────────────────────────────────────────────────────────

def bench_db_cache(cycles: int = 50):
    """캐시 유무에 따른 사이클당 DB 쿼리 횟수 비교."""
    print("[2] DB 캐시 효과 벤치마크")
    print(f"    분석 사이클: {cycles}회\n")

    query_counter = {"count": 0}

    def mock_db_query(symbol: str, mode: str) -> dict:
        query_counter["count"] += 1
        time.sleep(0.002)  # 2ms DB 응답 시뮬레이션
        return {"current_price": 85000.0, "data": list(range(1000))}

    # ── 캐시 없음: 노드마다 독립 조회 ──
    NODES_PER_CYCLE = 4  # triage, chart, judge, risk_manager
    query_counter["count"] = 0
    t0 = time.perf_counter()

    for _ in range(cycles):
        for _ in range(NODES_PER_CYCLE):
            mock_db_query("BTCUSDT", "swing")

    no_cache_time = time.perf_counter() - t0
    no_cache_queries = query_counter["count"]

    # ── 캐시 있음: 첫 노드만 조회, 나머지는 캐시 히트 ──
    _cache: Dict[str, dict] = {}
    query_counter["count"] = 0
    t0 = time.perf_counter()

    for _ in range(cycles):
        key = "BTCUSDT:swing"
        if key not in _cache:
            _cache[key] = mock_db_query("BTCUSDT", "swing")
        # 나머지 3개 노드는 캐시 히트
        for _ in range(NODES_PER_CYCLE - 1):
            _ = _cache[key]
        _cache.clear()  # 사이클 끝에 클리어

    cached_time = time.perf_counter() - t0
    cached_queries = query_counter["count"]

    print(f"    캐시 없음: {no_cache_queries}회 쿼리 / {no_cache_time:.3f}s")
    print(f"    캐시 있음: {cached_queries}회 쿼리 / {cached_time:.3f}s")
    print(f"    → 쿼리 감소: {no_cache_queries / cached_queries:.1f}x")
    print(f"    → 시간 단축: {no_cache_time / cached_time:.1f}x\n")

    return {
        "no_cache_queries": no_cache_queries,
        "cached_queries": cached_queries,
        "query_reduction": no_cache_queries / cached_queries,
        "time_reduction": no_cache_time / cached_time,
    }


# ─────────────────────────────────────────────────────────
# 3. Pre-trade 파이프라인 단계별 처리 시간
# ─────────────────────────────────────────────────────────

def bench_pretrade_pipeline(n: int = 1000):
    """Pre-trade risk gate 각 단계 처리 시간 측정."""
    print("[3] Pre-trade 파이프라인 레이턴시 벤치마크")
    print(f"    반복: {n}회\n")

    timings: Dict[str, List[float]] = {
        "self_correction": [],
        "risk_manager": [],
        "risk_budget": [],
        "leverage_guard": [],
        "total": [],
    }

    for _ in range(n):
        decision = {
            "decision": random.choice(["LONG", "SHORT", "HOLD"]),
            "allocation_pct": random.uniform(5, 40),
            "leverage": random.uniform(1, 3),
            "win_probability_pct": random.uniform(40, 80),
            "expected_profit_pct": random.uniform(1, 5),
            "expected_loss_pct": random.uniform(0.5, 3),
            "stop_loss": 84000.0,
            "take_profit": 87000.0,
        }

        t_total = time.perf_counter()

        # self_correction mock
        t = time.perf_counter()
        direction = decision["decision"]
        loop_count = 0
        if direction != "HOLD" and loop_count < 2:
            _ = {
                "passed": decision["win_probability_pct"] > 50,
                "challenges": [],
                "loop_back_reason": None,
            }
        timings["self_correction"].append((time.perf_counter() - t) * 1000)

        # risk_manager mock (드로다운 계산)
        t = time.perf_counter()
        alloc = decision["allocation_pct"]
        lev = decision["leverage"]
        loss_pct = decision["expected_loss_pct"]
        worst_loss = alloc * lev * loss_pct / 100
        _ = worst_loss <= 15.0  # MAX_TOLERABLE_DRAWDOWN_PCT
        timings["risk_manager"].append((time.perf_counter() - t) * 1000)

        # risk_budget mock
        t = time.perf_counter()
        _ = alloc * lev <= 40.0
        timings["risk_budget"].append((time.perf_counter() - t) * 1000)

        # leverage_guard mock
        t = time.perf_counter()
        wallet = 2000.0
        notional = wallet * (alloc / 100) * lev
        total_lev = notional / wallet
        if total_lev > 2.0:
            decision["allocation_pct"] *= 2.0 / total_lev
        timings["leverage_guard"].append((time.perf_counter() - t) * 1000)

        timings["total"].append((time.perf_counter() - t_total) * 1000)

    print(f"    {'단계':<18} {'평균(ms)':>10} {'p99(ms)':>10} {'max(ms)':>10}")
    print(f"    {'-'*50}")
    for stage, vals in timings.items():
        avg = statistics.mean(vals)
        p99 = sorted(vals)[int(len(vals) * 0.99)]
        mx = max(vals)
        print(f"    {stage:<18} {avg:>10.4f} {p99:>10.4f} {mx:>10.4f}")
    print()

    return timings


# ─────────────────────────────────────────────────────────
# 4. 주문 모니터링 처리량
# ─────────────────────────────────────────────────────────

def bench_order_monitoring(n_orders: int = 500, duration_sec: float = 1.0):
    """초당 SL/TP 체크 처리량 측정."""
    print("[4] 주문 모니터링 처리량 벤치마크")
    print(f"    오픈 포지션: {n_orders}개 / 측정 시간: {duration_sec}s\n")

    # 모의 주문 생성
    orders = [
        {
            "id": i,
            "symbol": "BTCUSDT",
            "side": random.choice(["LONG", "SHORT"]),
            "sl_price": random.uniform(82000, 84000),
            "tp1_price": random.uniform(86000, 88000),
            "entry_price": 85000.0,
        }
        for i in range(n_orders)
    ]

    check_count = 0
    hit_count = 0

    def check_orders(price: float) -> int:
        hits = 0
        for order in orders:
            side = order["side"]
            sl = order["sl_price"]
            tp = order["tp1_price"]
            if side == "LONG":
                if price <= sl or price >= tp:
                    hits += 1
            else:
                if price >= sl or price <= tp:
                    hits += 1
        return hits

    t0 = time.perf_counter()
    elapsed = 0.0
    tick_times = []

    while elapsed < duration_sec:
        price = 85000.0 + random.uniform(-3000, 3000)
        t = time.perf_counter()
        hits = check_orders(price)
        tick_times.append((time.perf_counter() - t) * 1000)
        hit_count += hits
        check_count += 1
        elapsed = time.perf_counter() - t0

    tps = check_count / elapsed
    avg_tick = statistics.mean(tick_times)

    print(f"    처리량:       {tps:,.0f} ticks/s ({n_orders}개 포지션 동시 모니터링)")
    print(f"    틱당 처리시간: {avg_tick:.3f} ms")
    print(f"    총 체크:      {check_count:,}회 / {elapsed:.2f}s\n")

    return {"tps": tps, "avg_tick_ms": avg_tick}


# ─────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  OMS 성능 벤치마크")
    print("=" * 60)

    r1 = bench_price_lookup(n=100_000)
    r2 = bench_db_cache(cycles=50)
    r3 = bench_pretrade_pipeline(n=1000)
    r4 = bench_order_monitoring(n_orders=500, duration_sec=1.0)

    print("=" * 60)
    print("  요약")
    print("=" * 60)
    print(f"  가격 조회:    인메모리가 REST 대비 {r1['speedup']:.0f}x 빠름")
    print(f"  DB 쿼리:      캐시로 {r2['query_reduction']:.1f}x 감소")
    print(f"  Pre-trade:    전체 파이프라인 평균 {statistics.mean(r3['total']):.3f}ms")
    print(f"  모니터링:     {r4['tps']:,.0f} ticks/s ({500}개 포지션 동시)")
    print("=" * 60)
