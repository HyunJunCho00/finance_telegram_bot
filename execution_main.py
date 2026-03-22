"""
Execution Process — Lean Entry Point
=====================================
포지션 실행/관리 전용 프로세스. 분석 스택(LangGraph, LLM SDK, LightRAG 등)을
일절 import하지 않아 메모리 ~80MB 수준을 유지한다.

기존 scheduler.py(단일 프로세스)와 완전히 독립적으로 실행 가능:

  # 단일 프로세스 (기존 방식, 유지됨)
  python scheduler.py

  # 분리 방식 (분석 + 실행 독립)
  python scheduler.py          ← 분석/수집/평가 (execution job 제외)
  python execution_main.py     ← 실행 전용 (이 파일)

  단, scheduler.py의 EXECUTION_PROCESS_SEPARATE=False(기본값)일 때는
  scheduler.py가 execution job도 함께 돌린다 → 중복 없음.

Prometheus metrics: http://0.0.0.0:9091/metrics
"""

import sys
import time
import threading
from loguru import logger

# ── 무거운 분석 스택 절대 import 금지 ──────────────────────────────────────
# executors.orchestrator, agents.*, processors.*, collectors.*,
# langchain, langgraph, anthropic, google-generativeai,
# neo4j, pymilvus, lightgbm, xgboost → 여기선 없음
# ───────────────────────────────────────────────────────────────────────────

from apscheduler.triggers.cron import CronTrigger
from config import scheduler_config
from config.settings import settings
from executors.order_manager import execution_desk
from executors.paper_exchange import paper_engine
from executors.outbox_dispatcher import outbox_dispatcher
from executors.execution_repository import execution_repository

# ── Prometheus 메트릭 정의 ──────────────────────────────────────────────────
try:
    from prometheus_client import (
        Counter, Gauge, Histogram,
        start_http_server as _prom_start,
    )

    EXECUTION_TICK_DURATION = Histogram(
        "execution_tick_duration_seconds",
        "1분 execution tick 처리 시간",
        buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
    )
    OPEN_POSITIONS = Gauge(
        "open_positions_total",
        "현재 오픈 포지션 수",
        ["exchange"],
    )
    ORDER_RESULTS = Counter(
        "order_results_total",
        "주문 처리 결과",
        ["result"],   # success / skip / error
    )
    OUTBOX_PUBLISHED = Counter(
        "outbox_published_total",
        "Outbox 발송 메시지 수",
        ["status"],   # published / failed
    )
    FUNDING_FEE_APPLIED = Counter(
        "funding_fee_applied_total",
        "펀딩피 적용 횟수",
    )
    OUTBOX_LAG_SECONDS = Gauge(
        "outbox_oldest_pending_age_seconds",
        "가장 오래된 PENDING outbox 이벤트 경과 시간 (초)",
    )
    POSITION_UNREALIZED_PNL = Gauge(
        "position_unrealized_pnl_usd",
        "포지션 미실현 PnL (USD)",
        ["symbol", "side", "exchange"],
    )
    POSITION_NOTIONAL = Gauge(
        "position_notional_usd",
        "포지션 노셔널 사이즈 (USD)",
        ["symbol", "exchange"],
    )
    POSITION_SIDE = Gauge(
        "position_side",
        "포지션 방향 (1=LONG, -1=SHORT, 0=없음)",
        ["symbol"],
    )
    _PROM_AVAILABLE = True
except ImportError:
    logger.warning("prometheus_client not installed — metrics disabled. pip install prometheus-client")
    _PROM_AVAILABLE = False


def _update_position_gauge(prices: dict[str, float] | None = None) -> None:
    if not _PROM_AVAILABLE:
        return
    try:
        positions = paper_engine.get_open_positions()

        # open_positions_total (exchange별)
        counts: dict[str, int] = {}
        # position_side 초기화 (포지션 없는 심볼 0으로)
        seen_symbols: set[str] = set()

        for p in positions:
            ex = str(p.get("exchange", "unknown")).lower()
            counts[ex] = counts.get(ex, 0) + 1

            symbol = str(p.get("symbol", ""))
            side = str(p.get("side", ""))
            entry = float(p.get("entry_price", 0) or 0)
            size = float(p.get("size", 0) or 0)

            # position_side
            side_val = 1 if side == "LONG" else (-1 if side == "SHORT" else 0)
            POSITION_SIDE.labels(symbol=symbol).set(side_val)
            seen_symbols.add(symbol)

            # position_notional_usd (entry 기준)
            notional = size * entry
            POSITION_NOTIONAL.labels(symbol=symbol, exchange=ex).set(notional)

            # position_unrealized_pnl_usd (현재가 있을 때만)
            if prices and symbol in prices:
                price = prices[symbol]
                pnl = (price - entry) * size if side == "LONG" else (entry - price) * size
                POSITION_UNREALIZED_PNL.labels(symbol=symbol, side=side, exchange=ex).set(pnl)

        for ex, cnt in counts.items():
            OPEN_POSITIONS.labels(exchange=ex).set(cnt)

        # 포지션 없는 심볼은 side=0
        for symbol in settings.trading_symbols:
            if symbol not in seen_symbols:
                POSITION_SIDE.labels(symbol=symbol).set(0)

    except Exception:
        pass


# ── Jobs ────────────────────────────────────────────────────────────────────

def job_1min_execution() -> None:
    """포지션 실행 + Paper TP/SL/청산 체크."""
    start = time.perf_counter()
    try:
        result = execution_desk.process_intents()
        if _PROM_AVAILABLE:
            if isinstance(result, dict):
                ORDER_RESULTS.labels(result="success").inc(result.get("processed", 0))
                ORDER_RESULTS.labels(result="skip").inc(result.get("skipped", 0))
            else:
                ORDER_RESULTS.labels(result="success").inc(1)

        if settings.PAPER_TRADING_MODE:
            from executors.trade_executor import trade_executor
            prices: dict[str, float] = {}
            for symbol in settings.trading_symbols_slash:
                try:
                    t = trade_executor.binance.fetch_ticker(symbol)
                    canonical = symbol.replace("/", "")
                    prices[canonical] = float(t["last"])
                except Exception:
                    pass
            paper_engine.check_liquidations(prices)
            paper_engine.check_tp_sl(prices)

        _update_position_gauge(prices if settings.PAPER_TRADING_MODE else None)
    except Exception as e:
        logger.error(f"[execution_main] 1min execution error: {e}")
        if _PROM_AVAILABLE:
            ORDER_RESULTS.labels(result="error").inc()
    finally:
        if _PROM_AVAILABLE:
            EXECUTION_TICK_DURATION.observe(time.perf_counter() - start)


def job_8hour_funding_fee() -> None:
    """Paper 펀딩피 시뮬레이션 (00:00 / 08:00 / 16:00 UTC)."""
    if not settings.PAPER_TRADING_MODE:
        return
    try:
        from executors.trade_executor import trade_executor
        rates: dict[str, float] = {}
        prices: dict[str, float] = {}
        positions = paper_engine.get_open_positions()
        if not positions:
            return
        for pos in positions:
            symbol = pos["symbol"]
            if symbol not in rates:
                try:
                    f_info = trade_executor.binance.fetch_funding_rate(symbol)
                    rates[symbol] = float(f_info["fundingRate"])
                    t = trade_executor.binance.fetch_ticker(symbol)
                    prices[symbol] = float(t["last"])
                except Exception as e:
                    logger.warning(f"[execution_main] funding rate fetch failed for {symbol}: {e}")
        paper_engine.apply_funding_fees(rates, prices)
        if _PROM_AVAILABLE:
            FUNDING_FEE_APPLIED.inc()
    except Exception as e:
        logger.error(f"[execution_main] 8h funding fee error: {e}")


def job_outbox_drain() -> None:
    """발송 실패한 Outbox 메시지 재시도."""
    try:
        result = outbox_dispatcher.publish_pending(limit=50)
        if _PROM_AVAILABLE and isinstance(result, dict):
            OUTBOX_PUBLISHED.labels(status="published").inc(result.get("published", 0))
            OUTBOX_PUBLISHED.labels(status="failed").inc(result.get("failed", 0))
        if _PROM_AVAILABLE:
            lag = execution_repository.get_oldest_pending_outbox_age_seconds()
            OUTBOX_LAG_SECONDS.set(lag)
    except Exception as e:
        logger.error(f"[execution_main] outbox drain error: {e}")


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
    logger.add(
        "logs/execution_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="7 days",
        level="INFO",
    )

    logger.info("=" * 60)
    logger.info("Execution Process starting")
    logger.info(f"  Symbols  : {', '.join(settings.trading_symbols)}")
    logger.info(f"  Paper    : {settings.PAPER_TRADING_MODE}")
    logger.info(f"  Metrics  : http://0.0.0.0:9091/metrics" if _PROM_AVAILABLE else "  Metrics  : disabled")
    logger.info("=" * 60)

    # Prometheus HTTP 서버 (별도 스레드)
    if _PROM_AVAILABLE:
        try:
            _prom_start(9091)
            logger.info("Prometheus metrics server started on :9091")
        except Exception as e:
            logger.warning(f"Prometheus server failed to start: {e}")

    # 1분 execution tick
    scheduler_config.scheduler.add_job(
        job_1min_execution,
        "interval",
        minutes=1,
        id="exec_1min",
        max_instances=1,
    )

    # 8시간 펀딩피
    scheduler_config.scheduler.add_job(
        job_8hour_funding_fee,
        CronTrigger(hour="0,8,16", minute=0),
        id="exec_8h_funding",
        max_instances=1,
    )

    # 5분 outbox drain
    scheduler_config.scheduler.add_job(
        job_outbox_drain,
        "interval",
        minutes=5,
        id="exec_outbox_drain",
        max_instances=1,
    )

    scheduler_config.scheduler.start()
    logger.info("Execution scheduler started.")

    try:
        while True:
            time.sleep(30)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Execution process shutting down.")
        scheduler_config.scheduler.shutdown(wait=False)


if __name__ == "__main__":
    main()
