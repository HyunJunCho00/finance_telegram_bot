"""
OMS 핵심 성능 벤치마크 — SQLite 기반, 외부 API 없음

측정 항목:
  1. 주문 DB write 처리량 (단건 vs 배치)
  2. 동시 주문 충돌 처리 — 멱등성 키 효과
  3. 체결 레이턴시 — 주문 생성 → DB 기록 → 상태 업데이트
  4. 동시 주문 경합 — 락 유무 처리량 차이

실행:
  python -X utf8 -m tests.bench_order_processing
"""

import sqlite3
import threading
import time
import statistics
import random
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import List, Dict


# ── SQLite 인메모리 DB (paper_orders 스키마) ──────────────────────────────

DDL = """
CREATE TABLE IF NOT EXISTS paper_orders (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    idempotency_key     TEXT UNIQUE,
    symbol              TEXT NOT NULL,
    mode                TEXT NOT NULL,
    side                TEXT NOT NULL,
    size_usdt           REAL,
    entry_price         REAL,
    sl_price            REAL,
    tp1_price           REAL,
    allocation_pct      REAL,
    status              TEXT NOT NULL DEFAULT 'OPEN'
);

CREATE TABLE IF NOT EXISTS order_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id    INTEGER,
    event       TEXT NOT NULL,
    payload     TEXT,
    ts          TEXT NOT NULL DEFAULT (datetime('now','subsec'))
);
"""

def make_db(path=":memory:") -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(DDL)
    conn.commit()
    return conn


def sample_order(symbol="BTCUSDT", idem_key=None) -> Dict:
    return {
        "idempotency_key": idem_key or str(uuid.uuid4()),
        "symbol": symbol,
        "mode": "swing",
        "side": random.choice(["LONG", "SHORT"]),
        "size_usdt": round(random.uniform(100, 2000), 2),
        "entry_price": round(random.uniform(80000, 90000), 2),
        "sl_price": round(random.uniform(78000, 82000), 2),
        "tp1_price": round(random.uniform(88000, 95000), 2),
        "allocation_pct": round(random.uniform(5, 30), 2),
    }


# ─────────────────────────────────────────────────────────
# 1. 주문 write 처리량 — 단건 vs 배치
# ─────────────────────────────────────────────────────────

def bench_write_throughput(n=1000):
    print("\n[1] 주문 DB write 처리량")
    print(f"    주문 수: {n}건\n")

    # ── 단건 INSERT ──
    conn = make_db()
    orders = [sample_order() for _ in range(n)]

    t0 = time.perf_counter()
    for o in orders:
        conn.execute("""
            INSERT INTO paper_orders
              (idempotency_key, symbol, mode, side, size_usdt, entry_price, sl_price, tp1_price, allocation_pct)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (o["idempotency_key"], o["symbol"], o["mode"], o["side"],
              o["size_usdt"], o["entry_price"], o["sl_price"], o["tp1_price"], o["allocation_pct"]))
        conn.commit()
    single_elapsed = time.perf_counter() - t0
    conn.close()

    # ── 배치 INSERT ──
    conn = make_db()
    orders = [sample_order() for _ in range(n)]

    t0 = time.perf_counter()
    conn.executemany("""
        INSERT INTO paper_orders
          (idempotency_key, symbol, mode, side, size_usdt, entry_price, sl_price, tp1_price, allocation_pct)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, [(o["idempotency_key"], o["symbol"], o["mode"], o["side"],
           o["size_usdt"], o["entry_price"], o["sl_price"], o["tp1_price"], o["allocation_pct"])
          for o in orders])
    conn.commit()
    batch_elapsed = time.perf_counter() - t0
    conn.close()

    print(f"    단건 INSERT: {n/single_elapsed:>8,.0f} orders/s  ({single_elapsed:.3f}s)")
    print(f"    배치 INSERT: {n/batch_elapsed:>8,.0f} orders/s  ({batch_elapsed:.3f}s)")
    print(f"    → 배치가 {single_elapsed/batch_elapsed:.1f}x 빠름\n")

    return {"single_ops": n/single_elapsed, "batch_ops": n/batch_elapsed}


# ─────────────────────────────────────────────────────────
# 2. 멱등성 키 — 중복 주문 차단
# ─────────────────────────────────────────────────────────

def bench_idempotency(n_threads=20):
    print("[2] 동시 중복 주문 차단 — 멱등성 키")
    print(f"    동시 스레드: {n_threads}개 (동일 주문 {n_threads}번)\n")

    conn = make_db()
    lock = threading.Lock()
    IDEM_KEY = "fund_A_20260408_BTC_LONG_001"  # 의도적으로 동일 키

    results = {"success": 0, "duplicate": 0}

    def submit_order():
        o = sample_order(idem_key=IDEM_KEY)
        try:
            with lock:
                conn.execute("""
                    INSERT INTO paper_orders
                      (idempotency_key, symbol, mode, side, size_usdt, entry_price, sl_price, tp1_price, allocation_pct)
                    VALUES (?,?,?,?,?,?,?,?,?)
                """, (o["idempotency_key"], o["symbol"], o["mode"], o["side"],
                      o["size_usdt"], o["entry_price"], o["sl_price"], o["tp1_price"], o["allocation_pct"]))
                conn.commit()
            results["success"] += 1
        except sqlite3.IntegrityError:
            results["duplicate"] += 1

    threads = [threading.Thread(target=submit_order) for _ in range(n_threads)]
    t0 = time.perf_counter()
    for t in threads: t.start()
    for t in threads: t.join()
    elapsed = time.perf_counter() - t0

    print(f"    동시 {n_threads}개 요청 → 체결: {results['success']}건 / 차단: {results['duplicate']}건")
    print(f"    처리 시간: {elapsed*1000:.2f}ms")
    print(f"    → 중복 {results['duplicate']}건 원천 차단 (DB UNIQUE 제약)\n")

    conn.close()
    return results


# ─────────────────────────────────────────────────────────
# 3. 체결 레이턴시 — 주문 생성 → 상태 업데이트 사이클
# ─────────────────────────────────────────────────────────

def bench_execution_latency(n=500):
    print("[3] 체결 레이턴시 — 주문 생성 → DB 기록 → 상태 업데이트")
    print(f"    주문 수: {n}건\n")

    conn = make_db()
    latencies_ms = []

    for _ in range(n):
        o = sample_order()
        t0 = time.perf_counter()

        # Step 1: 주문 생성 (INSERT)
        cur = conn.execute("""
            INSERT INTO paper_orders
              (idempotency_key, symbol, mode, side, size_usdt, entry_price, sl_price, tp1_price, allocation_pct)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (o["idempotency_key"], o["symbol"], o["mode"], o["side"],
              o["size_usdt"], o["entry_price"], o["sl_price"], o["tp1_price"], o["allocation_pct"]))
        order_id = cur.lastrowid

        # Step 2: 이벤트 기록 (ORDER_CREATED)
        conn.execute("""
            INSERT INTO order_events (order_id, event, payload)
            VALUES (?, 'ORDER_CREATED', ?)
        """, (order_id, f'{{"price": {o["entry_price"]}}}'))

        # Step 3: 체결 시뮬레이션 (FILLED 상태 업데이트)
        fill_price = o["entry_price"] * random.uniform(0.999, 1.001)
        conn.execute("""
            UPDATE paper_orders SET status='FILLED' WHERE id=?
        """, (order_id,))
        conn.execute("""
            INSERT INTO order_events (order_id, event, payload)
            VALUES (?, 'ORDER_FILLED', ?)
        """, (order_id, f'{{"fill_price": {fill_price:.2f}}}'))

        conn.commit()
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    avg = statistics.mean(latencies_ms)
    p50 = statistics.median(latencies_ms)
    p95 = sorted(latencies_ms)[int(n * 0.95)]
    p99 = sorted(latencies_ms)[int(n * 0.99)]

    print(f"    {'지표':<8} {'값(ms)':>10}")
    print(f"    {'-'*20}")
    print(f"    {'mean':<8} {avg:>10.3f}")
    print(f"    {'p50':<8} {p50:>10.3f}")
    print(f"    {'p95':<8} {p95:>10.3f}")
    print(f"    {'p99':<8} {p99:>10.3f}")
    print(f"    {'max':<8} {max(latencies_ms):>10.3f}\n")

    conn.close()
    return {"mean_ms": avg, "p95_ms": p95, "p99_ms": p99}


# ─────────────────────────────────────────────────────────
# 4. 동시 주문 경합 — 락 유무 처리량 차이
# ─────────────────────────────────────────────────────────

def bench_concurrent_contention(n_workers=10, orders_per_worker=100):
    print("[4] 동시 주문 경합 — 락 없음 vs RLock")
    total = n_workers * orders_per_worker
    print(f"    워커: {n_workers}개 / 각 {orders_per_worker}건 = 총 {total}건\n")

    # ── 락 없음 (에러 발생 확인) ──
    conn_no_lock = make_db()
    errors_no_lock = []

    def worker_no_lock(wid):
        local_errors = 0
        for _ in range(orders_per_worker):
            o = sample_order()
            try:
                conn_no_lock.execute("""
                    INSERT INTO paper_orders
                      (idempotency_key, symbol, mode, side, size_usdt, entry_price, sl_price, tp1_price, allocation_pct)
                    VALUES (?,?,?,?,?,?,?,?,?)
                """, (o["idempotency_key"], o["symbol"], o["mode"], o["side"],
                      o["size_usdt"], o["entry_price"], o["sl_price"], o["tp1_price"], o["allocation_pct"]))
                conn_no_lock.commit()
            except Exception:
                local_errors += 1
        return local_errors

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(worker_no_lock, i) for i in range(n_workers)]
        total_errors = sum(f.result() for f in as_completed(futures))
    no_lock_elapsed = time.perf_counter() - t0
    conn_no_lock.close()

    # ── RLock 사용 ──
    conn_locked = make_db()
    db_lock = threading.RLock()

    def worker_locked(wid):
        for _ in range(orders_per_worker):
            o = sample_order()
            with db_lock:
                conn_locked.execute("""
                    INSERT INTO paper_orders
                      (idempotency_key, symbol, mode, side, size_usdt, entry_price, sl_price, tp1_price, allocation_pct)
                    VALUES (?,?,?,?,?,?,?,?,?)
                """, (o["idempotency_key"], o["symbol"], o["mode"], o["side"],
                      o["size_usdt"], o["entry_price"], o["sl_price"], o["tp1_price"], o["allocation_pct"]))
                conn_locked.commit()

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(worker_locked, i) for i in range(n_workers)]
        for f in as_completed(futures): f.result()
    locked_elapsed = time.perf_counter() - t0
    conn_locked.close()

    locked_count = conn_locked.execute if False else n_workers * orders_per_worker - total_errors

    print(f"    락 없음: {total_errors}건 에러 / {no_lock_elapsed:.3f}s")
    print(f"    RLock:   0건 에러   / {locked_elapsed:.3f}s")
    print(f"    → RLock: {total/(locked_elapsed):,.0f} orders/s (에러 없음)")
    print(f"    → 데이터 정합성 보장을 위한 최소 비용: {locked_elapsed/no_lock_elapsed:.2f}x 오버헤드\n")

    return {
        "no_lock_errors": total_errors,
        "locked_tps": total / locked_elapsed,
    }


# ─────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  OMS 주문 처리 벤치마크 (SQLite 기반)")
    print("=" * 55)

    r1 = bench_write_throughput(n=1000)
    r2 = bench_idempotency(n_threads=20)
    r3 = bench_execution_latency(n=500)
    r4 = bench_concurrent_contention(n_workers=10, orders_per_worker=100)

    print("=" * 55)
    print("  요약")
    print("=" * 55)
    print(f"  단건 write:     {r1['single_ops']:>8,.0f} orders/s")
    print(f"  배치 write:     {r1['batch_ops']:>8,.0f} orders/s")
    print(f"  중복 주문 차단: {r2['duplicate']:>8}건 / 20개 동시 요청")
    print(f"  체결 레이턴시:  {r3['mean_ms']:>8.3f} ms (mean) / {r3['p99_ms']:.3f} ms (p99)")
    print(f"  동시 처리량:    {r4['locked_tps']:>8,.0f} orders/s (정합성 보장)")
    print("=" * 55)
