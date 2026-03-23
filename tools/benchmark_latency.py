"""Local latency benchmark suite for repeatable before/after comparisons.

Default mode is synthetic/local only:
- no external APIs
- no DB/network dependency
- saves JSON results under data/benchmarks/

Optional live mode can benchmark selected DB-backed paths when runtime
dependencies are available.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import statistics
import sys
import threading
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("USE_SECRET_MANAGER", "false")

import agents.ai_router as ai_router_module  # noqa: E402
import executors.report_hot_path as hot_path_module  # noqa: E402
import executors.orchestrator as orchestrator_module  # noqa: E402
from config.settings import TradingMode  # noqa: E402
from processors.chart_generator import chart_generator  # noqa: E402
from processors.math_engine import math_engine  # noqa: E402

try:
    from config.database import db  # noqa: E402
except Exception:
    db = None


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(float(v) for v in values)
    k = math.ceil((p / 100.0) * len(xs)) - 1
    k = max(0, min(k, len(xs) - 1))
    return round(xs[k], 4)


def summarize(name: str, latencies_ms: list[float], kind: str, notes: str = "") -> dict:
    if not latencies_ms:
        raise ValueError(f"No latency samples collected for {name}")
    return {
        "name": name,
        "kind": kind,
        "notes": notes,
        "runs": len(latencies_ms),
        "min_ms": round(min(latencies_ms), 4),
        "mean_ms": round(statistics.mean(latencies_ms), 4),
        "p50_ms": percentile(latencies_ms, 50),
        "p95_ms": percentile(latencies_ms, 95),
        "p99_ms": percentile(latencies_ms, 99),
        "max_ms": round(max(latencies_ms), 4),
        "samples_ms": [round(v, 4) for v in latencies_ms],
    }


def benchmark(fn: Callable[[], object], runs: int, warmup: int = 2) -> list[float]:
    for _ in range(max(0, warmup)):
        fn()

    samples: list[float] = []
    for _ in range(runs):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1000.0)
    return samples


def make_synthetic_market_df(rows: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    timestamps = pd.date_range("2026-01-01", periods=rows, freq="min", tz="UTC")
    base = 100000.0 + np.cumsum(rng.normal(0.0, 18.0, size=rows))
    close = np.maximum(base, 1000.0)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    spread = np.abs(rng.normal(14.0, 5.0, size=rows)) + 3.0
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = np.abs(rng.normal(120.0, 40.0, size=rows)) + 10.0

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "taker_buy_volume": volume * 0.52,
            "taker_sell_volume": volume * 0.48,
        }
    )


def build_report_state(symbol: str = "BTCUSDT", mode: TradingMode = TradingMode.SWING) -> dict:
    cache_key = orchestrator_module._cache_key(symbol, mode)
    df = pd.DataFrame([{"timestamp": pd.Timestamp("2026-03-09T00:00:00Z"), "close": 100.0}])
    orchestrator_module._df_cache[cache_key] = df
    orchestrator_module._market_data_cache[cache_key] = {
        "current_price": 100.0,
        "confluence_zones": [],
    }
    return {
        "symbol": symbol,
        "mode": mode.value,
        "final_decision": {
            "decision": "LONG",
            "entry_price": 100.0,
            "take_profit": 110.0,
            "stop_loss": 95.0,
            "confidence": 70.0,
            "reasoning": {"final_logic": "benchmark"},
        },
        "blackboard": {},
        "anomalies": [],
        "raw_funding": {},
        "onchain_context": "",
        "onchain_snapshot": {},
        "notification_context": "benchmark",
        "rag_context": "",
        "telegram_news": "",
        "narrative_text": "",
        "feedback_text": "",
        "market_regime": "RANGE_BOUND",
        "regime_context": {},
        "policy_snapshot": {},
        "chart_bytes": None,
    }


def build_hot_path_bundle(symbol: str = "BTCUSDT", mode: TradingMode = TradingMode.SWING) -> dict:
    now = "2026-03-09T00:00:00Z"
    later = "2099-01-01T00:00:00Z"
    return {
        "symbol": symbol,
        "mode": mode.value,
        "agents": {
            "market_snapshot_agent": {
                "payload": {
                    "market_data_compact": "compact",
                    "market_data": {"current_price": 100.0, "confluence_zones": []},
                    "active_orders": [],
                    "open_positions": "No open positions.",
                    "regime_context": {"regime": "RANGE_BOUND"},
                },
                "created_at": now,
                "expires_at": later,
                "status": "fresh",
            },
            "narrative_agent": {
                "payload": {
                    "narrative_text": "cached narrative",
                    "rag_context": "cached rag",
                    "telegram_news": "",
                    "feedback_text": "",
                },
                "created_at": now,
                "expires_at": later,
                "status": "fresh",
            },
            "funding_liq_agent": {
                "payload": {
                    "funding_context": "cached funding",
                    "liquidation_context": "cached liq",
                    "raw_funding": {},
                },
                "created_at": now,
                "expires_at": later,
                "status": "fresh",
            },
            "onchain_agent": {
                "payload": {
                    "onchain_context": "cached onchain",
                    "onchain_snapshot": {},
                    "onchain_gate": {},
                },
                "created_at": now,
                "expires_at": later,
                "status": "fresh",
            },
            "chart_prep_agent": {
                "payload": {
                    "chart_bytes": None,
                    "chart_rules": {},
                    "vlm_geometry": {},
                },
                "created_at": now,
                "expires_at": later,
                "status": "fresh",
            },
        },
    }


def benchmark_report_path_current(runs: int) -> dict:
    state = build_report_state()

    def slow_post_mortem(*args, **kwargs):
        time.sleep(0.15)
        return {"mistake_summary": ""}

    def run_once():
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    orchestrator_module.report_generator,
                    "generate_report",
                    return_value={
                        "report_id": 1,
                        "id": 1,
                        "symbol": state["symbol"],
                        "created_at": "2026-03-09T00:00:00Z",
                    },
                )
            )
            stack.enter_context(
                patch.object(orchestrator_module, "write_post_mortem", side_effect=slow_post_mortem)
            )
            stack.enter_context(
                patch.object(orchestrator_module.db, "upsert_evaluation_prediction", return_value={"id": 10})
            )
            stack.enter_context(
                patch.object(orchestrator_module.db, "batch_upsert_evaluation_component_scores", return_value={})
            )
            stack.enter_context(patch.object(orchestrator_module.report_generator, "notify", return_value=None))
            stack.enter_context(
                patch("executors.metrics_logger.metrics_logger.log_prediction", return_value=None)
            )
            orchestrator_module.node_generate_report(dict(state))

    latencies = benchmark(run_once, runs=runs, warmup=1)
    return summarize(
        "report_path_current_with_sync_post_mortem",
        latencies,
        kind="synthetic_structural",
        notes="Characterization benchmark. Includes a mocked 150ms synchronous post-mortem delay.",
    )


def benchmark_hot_path_fresh_snapshot(runs: int) -> dict:
    bundle = build_hot_path_bundle()
    decision = {"decision": "HOLD", "allocation_pct": 0, "leverage": 1, "reasoning": {"final_logic": "benchmark"}}
    report = {"report_id": 1, "symbol": bundle["symbol"], "created_at": "2026-03-09T00:00:00Z"}

    def run_once():
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(hot_path_module.agent_state_store, "load_bundle", return_value=bundle)
            )
            stack.enter_context(
                patch.object(hot_path_module.judge_agent, "make_decision_from_snapshot", return_value=decision)
            )
            stack.enter_context(
                patch.object(hot_path_module.risk_policy_engine, "apply", return_value=decision)
            )
            stack.enter_context(
                patch.object(hot_path_module, "_apply_execution_bridge", return_value=decision)
            )
            stack.enter_context(
                patch.object(hot_path_module.report_generator, "generate_report_from_snapshot", return_value=report)
            )
            stack.enter_context(
                patch.object(hot_path_module.report_generator, "notify", return_value=None)
            )
            stack.enter_context(
                patch.object(hot_path_module.performance_telemetry, "log_hot_path_run", return_value=None)
            )
            hot_path_module.run_snapshot_analysis_hot_path(
                bundle["symbol"],
                TradingMode(bundle["mode"]),
                wait_budget_s=0.0,
                notification_context="benchmark",
                execute_trades=False,
            )

    latencies = benchmark(run_once, runs=runs, warmup=1)
    return summarize(
        "hot_path_fresh_snapshot_local",
        latencies,
        kind="synthetic_structural",
        notes="Synthetic hot-path run with fresh snapshot and mocked Judge/Policy/Report stages.",
    )


def benchmark_ai_router_serialization(runs: int) -> dict:
    client = ai_router_module.ai_client
    original_gap = client._MIN_GAP
    client._MIN_GAP = 0.0

    def fake_execute(**kwargs):
        time.sleep(0.15)
        return "ok"

    def run_pair():
        start_event = threading.Event()
        outputs: list[str] = []

        def call_router():
            start_event.wait(timeout=1.0)
            value = client.generate_response(
                system_prompt="sys",
                user_message="msg",
                role="chat",
                max_tokens=10,
                temperature=0.0,
            )
            outputs.append(value)

        with patch.object(client, "_execute_routed_call", side_effect=fake_execute):
            t1 = threading.Thread(target=call_router)
            t2 = threading.Thread(target=call_router)
            t1.start()
            t2.start()
            start_event.set()
            t1.join(timeout=2.0)
            t2.join(timeout=2.0)
        if len(outputs) != 2 or any(v != "ok" for v in outputs):
            raise RuntimeError("AI router benchmark threads did not complete cleanly")

    try:
        latencies = benchmark(run_pair, runs=runs, warmup=1)
    finally:
        client._MIN_GAP = original_gap

    return summarize(
        "ai_router_concurrent_pair_current",
        latencies,
        kind="synthetic_structural",
        notes="Two concurrent calls with mocked 150ms provider latency. Current global lock should force near-serial behavior.",
    )


def benchmark_ai_router_cross_backend(runs: int) -> dict:
    """Two concurrent calls to DIFFERENT backends (judge=gemini, meta_regime=cerebras).
    Before fix: global lock serializes them → ~300ms.
    After fix:  per-backend locks allow true parallel → ~150ms.
    """
    client = ai_router_module.ai_client
    original_gap = client._MIN_GAP
    client._MIN_GAP = 0.0

    def fake_execute(**kwargs):
        time.sleep(0.15)
        return "ok"

    def run_cross_pair():
        start_event = threading.Event()
        outputs: list[str] = []

        def call_judge():
            start_event.wait(timeout=1.0)
            value = client.generate_response(
                system_prompt="sys", user_message="msg",
                role="judge", max_tokens=10, temperature=0.0,
            )
            outputs.append(value)

        def call_meta():
            start_event.wait(timeout=1.0)
            value = client.generate_response(
                system_prompt="sys", user_message="msg",
                role="meta_regime", max_tokens=10, temperature=0.0,
            )
            outputs.append(value)

        with patch.object(client, "_execute_routed_call", side_effect=fake_execute):
            t1 = threading.Thread(target=call_judge)
            t2 = threading.Thread(target=call_meta)
            t1.start()
            t2.start()
            start_event.set()
            t1.join(timeout=2.0)
            t2.join(timeout=2.0)
        if len(outputs) != 2 or any(v != "ok" for v in outputs):
            raise RuntimeError("AI router cross-backend benchmark threads did not complete cleanly")

    try:
        latencies = benchmark(run_cross_pair, runs=runs, warmup=1)
    finally:
        client._MIN_GAP = original_gap

    return summarize(
        "ai_router_cross_backend_pair",
        latencies,
        kind="synthetic_structural",
        notes="judge(gemini) + meta_regime(cerebras) concurrent. Before: global lock serializes (~300ms). After: per-backend locks allow parallel (~150ms).",
    )


def benchmark_math_analysis_local(runs: int) -> dict:
    df = make_synthetic_market_df()

    def run_once():
        math_engine.analyze_market(df, TradingMode.SWING)

    latencies = benchmark(run_once, runs=runs, warmup=1)
    return summarize(
        "math_analysis_swing_local",
        latencies,
        kind="synthetic_cpu",
        notes="Local-only technical analysis on synthetic 1m OHLCV data.",
    )


def benchmark_chart_render_local(runs: int) -> dict:
    df = make_synthetic_market_df()
    analysis = math_engine.analyze_market(df, TradingMode.SWING)

    def run_once():
        chart_bytes = chart_generator.generate_chart(
            df,
            analysis,
            "BTCUSDT",
            mode=TradingMode.SWING,
            timeframe="4h",
            prefer_lane=False,
        )
        if not chart_bytes:
            raise RuntimeError("Chart generation returned empty output")

    latencies = benchmark(run_once, runs=runs, warmup=1)
    return summarize(
        "chart_render_swing_local",
        latencies,
        kind="synthetic_cpu",
        notes="Local-only chart render using synthetic data and precomputed analysis.",
    )


def benchmark_live_market_read(runs: int, symbol: str) -> dict:
    if db is None:
        raise RuntimeError("Database client unavailable")

    def run_once():
        frame = db.get_latest_market_data(symbol, limit=1500)
        if frame is None or frame.empty:
            raise RuntimeError(f"No live market data for {symbol}")

    latencies = benchmark(run_once, runs=runs, warmup=1)
    return summarize(
        f"live_market_read_{symbol.lower()}",
        latencies,
        kind="live_io",
        notes="DB-backed market_data fetch. Network and Supabase latency included.",
    )


def run_suite(runs: int, include_live: bool, symbol: str) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()
    results = [
        benchmark_math_analysis_local(runs=max(3, min(runs, 10))),
        benchmark_chart_render_local(runs=max(3, min(runs, 8))),
        benchmark_report_path_current(runs=max(5, min(runs, 20))),
        benchmark_hot_path_fresh_snapshot(runs=max(5, min(runs, 20))),
        benchmark_ai_router_serialization(runs=max(5, min(runs, 20))),
        benchmark_ai_router_cross_backend(runs=max(5, min(runs, 20))),
    ]

    live_errors: list[str] = []
    if include_live:
        try:
            results.append(benchmark_live_market_read(runs=max(3, min(runs, 8)), symbol=symbol))
        except Exception as e:
            live_errors.append(str(e))

    return {
        "suite": "latency_baseline_v1",
        "mode": "local+live" if include_live else "local",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "benchmarks": results,
        "live_errors": live_errors,
    }


def save_results(result: dict, output_path: Path | None = None) -> Path:
    bench_dir = ROOT / "data" / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    final_path = output_path or (bench_dir / f"latency_baseline_{ts}.json")
    final_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_path = bench_dir / "latest_baseline.json"
    latest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if result.get("mode") == "local":
        local_latest_path = bench_dir / "latest_local_baseline.json"
        local_latest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return final_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local latency benchmarks and save a JSON baseline.")
    parser.add_argument("--runs", type=int, default=7, help="Benchmark repetitions per test case.")
    parser.add_argument("--live", action="store_true", help="Also run selected DB-backed live benchmarks.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol used for optional live benchmarks.")
    parser.add_argument("--output", default="", help="Optional output JSON path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output).resolve() if args.output else None
    result = run_suite(runs=max(1, args.runs), include_live=bool(args.live), symbol=str(args.symbol))
    saved_path = save_results(result, output_path=output_path)
    print(f"Saved benchmark baseline to: {saved_path}")
    for item in result["benchmarks"]:
        print(
            f"- {item['name']}: mean={item['mean_ms']}ms "
            f"p50={item['p50_ms']}ms p95={item['p95_ms']}ms p99={item['p99_ms']}ms"
        )
    if result.get("live_errors"):
        print("Live benchmark errors:")
        for err in result["live_errors"]:
            print(f"- {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
