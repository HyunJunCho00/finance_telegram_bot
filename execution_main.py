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

# process_intents()의 동시 실행 방지 (Redis 신호 + 1분 fallback이 겹칠 때)
_intent_lock = threading.Lock()


def _run_process_intents(trigger: str = "scheduler") -> None:
    """process_intents()를 단일 실행 보장하며 호출하는 내부 헬퍼."""
    if not _intent_lock.acquire(blocking=False):
        logger.debug(f"[execution_main] process_intents 실행 중 — {trigger} 스킵")
        return
    try:
        execution_desk.process_intents()
    finally:
        _intent_lock.release()


def job_1min_execution() -> None:
    """포지션 실행 + Paper TP/SL/청산 체크.

    Redis 리스너가 살아있을 때는 신호를 이미 즉시 처리했을 가능성이 높지만,
    Redis 장애 / 프로세스 재시작 직후의 누락 intent 처리를 위한 안전망 역할.
    """
    start = time.perf_counter()
    try:
        _run_process_intents(trigger="1min_fallback")

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
        else:
            from scripts.monitor_stop_to_be import check_and_update_stop_to_be
            check_and_update_stop_to_be()

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


# ── Redis Intent Listener ───────────────────────────────────────────────────

def _start_redis_intent_listener() -> bool:
    """Redis exec:new_intent 채널을 구독하는 백그라운드 데몬 스레드를 시작한다.

    신호가 들어오면 _run_process_intents()를 즉시(~1ms) 호출한다.
    Redis 연결이 없으면 False를 반환하고 1분 스케줄러만 작동한다.
    """
    try:
        from utils.redis_client import redis_client
        if not redis_client.enabled:
            logger.warning("[execution_main] Redis 비활성화 — 1분 fallback 스케줄러만 사용")
            return False
    except Exception as e:
        logger.warning(f"[execution_main] Redis 클라이언트 로드 실패: {e}")
        return False

    def _listener_loop() -> None:
        import json as _json
        backoff = 2.0
        while True:
            try:
                # publish/subscribe 전용 연결 (get/set 연결과 분리)
                pubsub = redis_client.client.pubsub(ignore_subscribe_messages=True)
                pubsub.subscribe("exec:new_intent")
                logger.info("[RedisListener] exec:new_intent 채널 구독 시작")
                backoff = 2.0  # 연결 성공 시 백오프 리셋

                for message in pubsub.listen():
                    if message is None:
                        continue
                    if message.get("type") != "message":
                        continue

                    try:
                        data = _json.loads(message["data"])
                        intent_ids = data.get("intent_ids", [])
                        logger.info(
                            f"[RedisListener] 신규 intent 신호 수신 "
                            f"({len(intent_ids)}건) → process_intents() 즉시 실행"
                        )
                    except Exception:
                        logger.info("[RedisListener] intent 신호 수신 → process_intents() 즉시 실행")

                    # 별도 스레드로 실행 — listener 루프를 블로킹하지 않는다
                    threading.Thread(
                        target=_run_process_intents,
                        args=("redis_signal",),
                        daemon=True,
                    ).start()

            except Exception as exc:
                logger.warning(
                    f"[RedisListener] 연결 끊김 ({exc}), "
                    f"{backoff:.0f}s 후 재연결 (1분 fallback 스케줄러 활성 중)"
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)  # 최대 60초 백오프

    t = threading.Thread(target=_listener_loop, name="redis-intent-listener", daemon=True)
    t.start()
    logger.info("[execution_main] Redis intent 리스너 시작 (채널: exec:new_intent)")
    return True


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
            _prom_start(9091, addr="127.0.0.1")
            logger.info("Prometheus metrics server started on 127.0.0.1:9091")
        except Exception as e:
            logger.warning(f"Prometheus server failed to start: {e}")

    # ── Redis Intent 리스너 시작 (실시간 주문 트리거) ──────────────────────
    # 신호가 오면 즉시(~1ms) process_intents() 실행.
    # Redis 비활성/장애 시 아래 1분 스케줄러(fallback)가 자동 대응.
    redis_listener_active = _start_redis_intent_listener()

    # 1분 execution tick (fallback 안전망)
    # Redis 리스너 활성일 때: 누락 intent 복구 + TP/SL 체크 역할
    # Redis 비활성일 때: 기존 방식 그대로 동작
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
    mode_label = "Redis Pub/Sub 실시간 + 1분 안전망" if redis_listener_active else "1분 스케줄러 fallback 전용"
    logger.info(f"Execution scheduler started. 주문 실행 모드: {mode_label}")

    try:
        while True:
            time.sleep(30)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Execution process shutting down.")
        scheduler_config.scheduler.shutdown(wait=False)


if __name__ == "__main__":
    main()
