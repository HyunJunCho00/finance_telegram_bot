"""System integrity audit for multi-agent trading workflow.

Modes:
- default (static): source-level integrity checks (CI-friendly, no external dependency required)
- --live: runtime checks against configured runtime dependencies (DB/env required)

Why both?
- Static checks catch wiring regressions quickly and safely.
- Live checks validate that data truly flows at runtime (not just mock-style pattern checks).
"""

from __future__ import annotations

from pathlib import Path
import argparse
import os
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def assert_contains(path: str, needles: list[str]) -> list[str]:
    src = text(path)
    return [n for n in needles if n not in src]


def run_static_checks() -> tuple[list[str], list[str]]:
    failures: list[str] = []
    passes: list[str] = []

    # 1) Flow and role separation (Expert Swarm + Decision Core)
    miss = assert_contains(
        "executors/orchestrator.py",
        [
            "node_liquidity_expert",
            "node_microstructure_expert",
            "node_macro_options_expert",
            "node_vlm_geometric_expert",
            "node_blackboard_synthesis",
            "node_meta_agent",
            "node_judge_agent",
            "node_risk_manager",
            "node_portfolio_leverage_guard",
        ],
    )
    (passes if not miss else failures).append(
        "multi-agent orchestration chain" if not miss else f"orchestration missing: {miss}"
    )

    # 2) Decision guardrails in active agents
    for file, needles in [
        ("agents/judge_agent.py", ["DEBATE_APPENDIX", "Falsifiability Analysis", "counter_scenario", "choose HOLD"]),
        ("agents/meta_agent.py", ["trust_directive", "risk_budget_pct", "VOLATILITY_PANIC"]),
        ("agents/risk_manager_agent.py", ["cro_veto_applied", "UPBIT", "BINANCE"]),
    ]:
        miss = assert_contains(file, needles)
        (passes if not miss else failures).append(
            f"guardrails in {file}" if not miss else f"{file} missing: {miss}"
        )

    # 3) Indicators coverage
    miss = assert_contains(
        "processors/math_engine.py",
        [
            "calculate_fibonacci_levels",
            "calculate_diagonal_support",
            "calculate_diagonal_resistance",
            "detect_divergences",
            "calculate_volume_profile",
            "ta.macd",
            "ta.rsi",
            "ta.vwap",
            "ta.kc",
        ],
    )
    (passes if not miss else failures).append(
        "technical indicator pipeline" if not miss else f"math_engine missing: {miss}"
    )

    # 4) Chart overlay implementation
    miss = assert_contains(
        "processors/chart_generator.py",
        [
            "_draw_swing_structure",
            "_draw_diagonal_line",
            "argrelextrema",
            "ax.axhline",
            "ax.scatter",
        ],
    )
    (passes if not miss else failures).append(
        "chart structure overlays" if not miss else f"chart_generator missing: {miss}"
    )

    return passes, failures


def run_live_checks(symbol: str) -> tuple[list[str], list[str]]:
    passes: list[str] = []
    failures: list[str] = []

    try:
        from config.database import db
        from config.settings import TradingMode
        from processors.math_engine import math_engine
        from processors.chart_generator import chart_generator
    except Exception as e:
        return passes, [f"live dependency import failed: {e}"]

    # Real DB read
    try:
        df = db.get_latest_market_data(symbol, limit=1500)
        if df is None or df.empty:
            failures.append(f"no market data returned for {symbol}")
            return passes, failures
        passes.append(f"db market data fetch ({symbol}, rows={len(df)})")
    except Exception as e:
        failures.append(f"db market data fetch failed: {e}")
        return passes, failures

    # Real indicator calculation
    try:
        analysis = math_engine.analyze_market(df, TradingMode.SWING)
        if not analysis or "timeframes" not in analysis:
            failures.append("market analysis output malformed")
            return passes, failures
        passes.append("math_engine swing analysis")
    except Exception as e:
        failures.append(f"math_engine analysis failed: {e}")
        return passes, failures

    # Real chart render
    try:
        chart_bytes = chart_generator.generate_chart(df, analysis, symbol, TradingMode.SWING)
        if not chart_bytes or len(chart_bytes) < 1000:
            failures.append("chart generation returned empty/too small output")
        else:
            passes.append(f"chart generation ({len(chart_bytes)} bytes)")
    except Exception as e:
        failures.append(f"chart generation failed: {e}")

    # Real supporting context reads
    try:
        _ = db.get_cvd_data(symbol, limit=240)
        passes.append("db CVD fetch")
    except Exception as e:
        failures.append(f"db CVD fetch failed: {e}")

    try:
        _ = db.get_latest_microstructure(symbol)
        passes.append("db microstructure fetch")
    except Exception as e:
        failures.append(f"db microstructure fetch failed: {e}")

    try:
        _ = db.get_latest_macro_data()
        passes.append("db macro fetch")
    except Exception as e:
        failures.append(f"db macro fetch failed: {e}")

    return passes, failures



def run_preflight_checks() -> tuple[list[str], list[str]]:
    """Deployment readiness checks (dependencies + required runtime config)."""
    passes: list[str] = []
    failures: list[str] = []

    required_modules = [
        "supabase",
        "pandas",
        "numpy",
        "scipy",
        "pandas_ta",
        "matplotlib",
        "mplfinance",
        "PIL",
        "telegram",
        "google.genai",
    ]

    for mod in required_modules:
        try:
            __import__(mod)
            passes.append(f"python module import: {mod}")
        except Exception as e:
            failures.append(f"missing/invalid module {mod}: {e}")

    use_secret_manager = os.getenv("USE_SECRET_MANAGER", "false").lower() == "true"
    if use_secret_manager:
        if os.getenv("PROJECT_ID"):
            passes.append("env PROJECT_ID (for Secret Manager)")
        else:
            failures.append("missing env PROJECT_ID while USE_SECRET_MANAGER=true")
    else:
        # minimal direct envs for core runtime
        required_env = [
            "SUPABASE_URL",
            "SUPABASE_KEY",
            "TELEGRAM_BOT_TOKEN",
            "TELEGRAM_CHAT_ID",
        ]
        for k in required_env:
            if os.getenv(k):
                passes.append(f"env {k}")
            else:
                failures.append(f"missing env {k}")

    return passes, failures



def run_pipeline_live_checks(symbol: str) -> tuple[list[str], list[str]]:
    """Run real ingestion pipeline steps and verify DB freshness (deployment validation)."""
    passes: list[str] = []
    failures: list[str] = []

    try:
        from collectors.price_collector import collector
        from collectors.funding_collector import funding_collector
        from collectors.microstructure_collector import microstructure_collector
        from config.database import db
    except Exception as e:
        return passes, [f"pipeline dependency import failed: {e}"]

    # Execute real collectors (same as scheduler 1m path)
    try:
        collector.run()
        passes.append("collector.run() executed")
    except Exception as e:
        failures.append(f"collector.run() failed: {e}")

    try:
        funding_collector.run()
        passes.append("funding_collector.run() executed")
    except Exception as e:
        failures.append(f"funding_collector.run() failed: {e}")

    try:
        microstructure_collector.run()
        passes.append("microstructure_collector.run() executed")
    except Exception as e:
        failures.append(f"microstructure_collector.run() failed: {e}")

    # Verify DB actually has fresh rows
    try:
        df = db.get_latest_market_data(symbol, limit=3)
        if df is None or df.empty:
            failures.append(f"market_data empty for {symbol} after ingestion")
        else:
            latest_ts = df.iloc[-1]["timestamp"]
            age_min = (datetime.now(timezone.utc) - latest_ts.to_pydatetime()).total_seconds() / 60.0
            if age_min <= 10:
                passes.append(f"market_data fresh ({symbol}, age={age_min:.1f}m)")
            else:
                failures.append(f"market_data stale ({symbol}, age={age_min:.1f}m)")
    except Exception as e:
        failures.append(f"market_data verification failed: {e}")

    try:
        cvd_df = db.get_cvd_data(symbol, limit=5)
        if cvd_df is None or cvd_df.empty:
            failures.append(f"cvd_data empty for {symbol} after ingestion")
        else:
            passes.append(f"cvd_data available ({symbol}, rows={len(cvd_df)})")
    except Exception as e:
        failures.append(f"cvd_data verification failed: {e}")

    try:
        funding_df = db.get_funding_history(symbol, limit=3)
        if funding_df is None or funding_df.empty:
            failures.append(f"funding_data empty for {symbol} after ingestion")
        else:
            latest_ts = funding_df.iloc[-1]["timestamp"]
            age_min = (datetime.now(timezone.utc) - latest_ts.to_pydatetime()).total_seconds() / 60.0
            if age_min <= 30:
                passes.append(f"funding_data fresh ({symbol}, age={age_min:.1f}m)")
            else:
                failures.append(f"funding_data stale ({symbol}, age={age_min:.1f}m)")
    except Exception as e:
        failures.append(f"funding_data verification failed: {e}")

    return passes, failures


def main() -> int:
    parser = argparse.ArgumentParser(description="System integrity audit")
    parser.add_argument("--preflight", action="store_true", help="Run deployment readiness checks (deps/env)")
    parser.add_argument("--live", action="store_true", help="Run runtime live checks (DB/env required)")
    parser.add_argument("--pipeline-live", action="store_true", help="Run real ingestion pipeline + DB freshness verification")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol for live checks")
    args = parser.parse_args()

    static_passes, static_failures = run_static_checks()

    print("SYSTEM INTEGRITY AUDIT")
    print("MODE: STATIC")
    print(f"PASS={len(static_passes)} FAIL={len(static_failures)}")
    for p in static_passes:
        print(f"[PASS] {p}")
    for f in static_failures:
        print(f"[FAIL] {f}")

    preflight_passes: list[str] = []
    preflight_failures: list[str] = []
    if args.preflight:
        preflight_passes, preflight_failures = run_preflight_checks()
        print("\nMODE: PREFLIGHT")
        print(f"PASS={len(preflight_passes)} FAIL={len(preflight_failures)}")
        for p in preflight_passes:
            print(f"[PASS] {p}")
        for f in preflight_failures:
            print(f"[FAIL] {f}")

    live_passes: list[str] = []
    live_failures: list[str] = []
    if args.live:
        live_passes, live_failures = run_live_checks(args.symbol)
        print("\nMODE: LIVE")
        print(f"PASS={len(live_passes)} FAIL={len(live_failures)}")
        for p in live_passes:
            print(f"[PASS] {p}")
        for f in live_failures:
            print(f"[FAIL] {f}")

    pipeline_passes: list[str] = []
    pipeline_failures: list[str] = []
    if args.pipeline_live:
        pipeline_passes, pipeline_failures = run_pipeline_live_checks(args.symbol)
        print("\nMODE: PIPELINE_LIVE")
        print(f"PASS={len(pipeline_passes)} FAIL={len(pipeline_failures)}")
        for p in pipeline_passes:
            print(f"[PASS] {p}")
        for f in pipeline_failures:
            print(f"[FAIL] {f}")

    has_failure = bool(static_failures) or bool(preflight_failures) or bool(live_failures) or bool(pipeline_failures)
    return 1 if has_failure else 0


if __name__ == "__main__":
    sys.exit(main())
