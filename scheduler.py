# -*- coding: utf-8 -*-
# [PERF] uvloop replaces CPython asyncio event loop with a C-based implementation.
# Improves async I/O throughput 2-4x. Linux/macOS only — degrades silently on Windows.
try:
    import uvloop
    uvloop.install()
except ImportError:
    pass  # Windows dev environment — no-op

from apscheduler.triggers.cron import CronTrigger
from collectors.price_collector import collector
from collectors.funding_collector import funding_collector
from collectors.volatility_monitor import volatility_monitor
from collectors.dune_collector import dune_collector
from collectors.microstructure_collector import microstructure_collector
from collectors.macro_collector import macro_collector
from collectors.deribit_collector import deribit_collector
from collectors.fear_greed_collector import fear_greed_collector
from collectors.coinmetrics_collector import coinmetrics_collector
from executors.orchestrator import orchestrator
from processors.light_rag import light_rag
from processors.gcs_archive import gcs_archive_exporter
from processors.math_engine import math_engine
from processors.gcs_parquet import gcs_parquet_store
from tools.evaluate_market_status_pressure import run_evaluation as run_pressure_signal_evaluation
from config import scheduler_config
from config.settings import settings, TradingMode
from config.database import db
from executors.order_manager import execution_desk
from executors.paper_exchange import paper_engine
from executors.cascade_warning_engine import cascade_warning_engine
from loguru import logger
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import time as _time
from datetime import datetime, timezone, timedelta
from utils.text_sanitizer import looks_english_dominant
from utils.math_utils import project_trendline_price, pct_change, distance_pct
from utils.json_utils import extract_json_object

# ── Prometheus 메트릭 ──────────────────────────────────────────────────────────
try:
    from prometheus_client import Counter, Histogram, start_http_server as _prom_start
    SCHED_JOB_DURATION = Histogram(
        "scheduler_job_duration_seconds",
        "스케줄러 Job 실행 시간",
        ["job"],
        buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1800],
    )
    SCHED_JOB_RESULTS = Counter(
        "scheduler_job_results_total",
        "스케줄러 Job 실행 결과",
        ["job", "result"],  # result: success / error
    )
    _SCHED_PROM = True
except Exception:
    _SCHED_PROM = False

_daily_precision_prepare_lock = threading.Lock()
_daily_precision_prepare_bucket = ""

# Persistent thread pool for 1-minute data collection tick.
# Avoids thread creation/teardown overhead on every firing.
_TICK_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tick")

# ── 프로세스 분리 모드 플래그 ────────────────────────────────────────────────
# True  → execution_main.py를 별도 프로세스로 실행 중
#          scheduler.py는 execution job(1min_execution, 8h_funding)을 등록하지 않음
# False → 기존 단일 프로세스 방식 (기본값, 변경 불필요)
EXECUTION_PROCESS_SEPARATE: bool = True


def _daily_precision_schedule_hours_utc() -> list[int]:
    raw = str(getattr(settings, "DAILY_PRECISION_HOURS_UTC", "") or "").strip()
    parsed: list[int] = []
    if raw:
        for chunk in raw.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                hour = int(chunk)
            except Exception:
                continue
            if 0 <= hour <= 23:
                parsed.append(hour)
    if parsed:
        return sorted(set(parsed))
    return [int(getattr(settings, "DAILY_PRECISION_HOUR_UTC", 0))]


def _daily_precision_schedule_hours_expr_utc() -> str:
    return ",".join(str(hour) for hour in _daily_precision_schedule_hours_utc())


def _daily_precision_label_utc() -> str:
    minute = int(getattr(settings, "DAILY_PRECISION_MINUTE_UTC", 30))
    return ", ".join(f"{hour:02d}:{minute:02d}" for hour in _daily_precision_schedule_hours_utc())


def _daily_precision_schedule_slots_utc(offset_minutes: int = 0) -> list[tuple[int, int]]:
    minute = int(getattr(settings, "DAILY_PRECISION_MINUTE_UTC", 30))
    slots: list[tuple[int, int]] = []
    for hour in _daily_precision_schedule_hours_utc():
        total_minutes = ((hour * 60) + minute + int(offset_minutes)) % (24 * 60)
        slots.append((total_minutes // 60, total_minutes % 60))
    return sorted(set(slots))


def _daily_precision_symbol_label_utc(symbol_index: int) -> str:
    gap = max(0, int(getattr(settings, "DAILY_PRECISION_SYMBOL_GAP_MINUTES", 10)))
    slots = _daily_precision_schedule_slots_utc(offset_minutes=symbol_index * gap)
    return ", ".join(f"{hour:02d}:{minute:02d}" for hour, minute in slots)


def _daily_precision_all_labels_utc() -> str:
    labels = []
    for idx, symbol in enumerate(settings.trading_symbols):
        labels.append(f"{symbol}={_daily_precision_symbol_label_utc(idx)}")
    return " | ".join(labels) if labels else _daily_precision_label_utc()


def _daily_precision_protection_window_minutes() -> int:
    return max(0, int(getattr(settings, "DAILY_PRECISION_PROTECTION_MINUTES", 20)))


def _is_daily_precision_protection_window(now_utc: datetime | None = None) -> bool:
    now = now_utc or datetime.now(timezone.utc)
    protection_minutes = _daily_precision_protection_window_minutes()
    base_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    slot_offsets = [
        idx * max(0, int(getattr(settings, "DAILY_PRECISION_SYMBOL_GAP_MINUTES", 10)))
        for idx, _ in enumerate(settings.trading_symbols)
    ] or [0]
    for day_offset in (-1, 0, 1):
        day_anchor = base_day + timedelta(days=day_offset)
        for offset in slot_offsets:
            for hour, minute in _daily_precision_schedule_slots_utc(offset_minutes=offset):
                scheduled = day_anchor.replace(hour=hour, minute=minute)
                window_start = scheduled - timedelta(minutes=protection_minutes // 2)
                window_end = scheduled + timedelta(minutes=protection_minutes)
                if window_start <= now <= window_end:
                    return True
    return False


def _should_defer_heavy_job(job_name: str) -> bool:
    if orchestrator.is_daily_precision_running():
        logger.info(f"{job_name} skipped (daily precision currently running)")
        return True
    if _is_daily_precision_protection_window():
        logger.info(
            f"{job_name} skipped (inside daily precision protection window around {_daily_precision_all_labels_utc()} UTC)"
        )
        return True
    return False


def _refresh_snapshots_for_modes(
    modes: list[TradingMode],
    *,
    allow_web_search: bool,
    label: str,
) -> None:
    from config.local_state import state_manager

    if not state_manager.is_analysis_enabled():
        logger.info(f"{label} skipped (analysis disabled)")
        return

    for symbol in settings.trading_symbols:
        for mode in modes:
            try:
                logger.info(
                    f"{label}: refreshing snapshot for {symbol}/{mode.value} "
                    f"(perplexity={'on' if allow_web_search else 'off'})"
                )
                orchestrator.refresh_snapshot_with_mode(
                    symbol,
                    mode,
                    allow_web_search=allow_web_search,
                    include_meta=False,
                    include_vlm=False,
                )
            except Exception as e:
                logger.error(f"{label} error for {symbol}/{mode.value}: {e}")


def job_snapshot_refresh_fast():
    """Warm hot-path snapshots for Swing lane without slow web narrative calls."""
    if _should_defer_heavy_job("Snapshot fast refresh"):
        return
    _refresh_snapshots_for_modes(
        [TradingMode.SWING],
        allow_web_search=False,
        label="Snapshot fast refresh",
    )


def job_snapshot_refresh_narrative():
    """Refresh narrative/RAG-rich snapshots for Swing lane."""
    if _should_defer_heavy_job("Snapshot narrative refresh"):
        return
    _refresh_snapshots_for_modes(
        [TradingMode.SWING],
        allow_web_search=True,
        label="Snapshot narrative refresh",
    )






def job_1min_tick():
    tasks = {
        "price":          collector.run,
        "funding":        funding_collector.run,
        "microstructure": microstructure_collector.run,
        "volatility":     volatility_monitor.run,
    }
    futures = {_TICK_POOL.submit(fn): name for name, fn in tasks.items()}
    try:
        for future in as_completed(futures, timeout=55):
            name = futures[future]
            try:
                future.result(timeout=1)
            except Exception as e:
                logger.error(f"{name} collection error: {e}")
    except TimeoutError:
        for future, name in futures.items():
            if not future.done():
                logger.error(f"{name} collection timed out after 55s")

def job_1min_execution():
    """V5: Process Orders, V7: Check Margin Calls + TP/SL"""
    try:
        execution_desk.process_intents()
        
        if settings.PAPER_TRADING_MODE:
            prices = {}
            # [FIX HIGH-13] Reuse existing authenticated CCXT instance
            from executors.trade_executor import trade_executor
            ex = trade_executor.binance
            for symbol in settings.trading_symbols_slash:
                try:
                    t = ex.fetch_ticker(symbol)
                    # Map back to canonical format (BTC/USDT -> BTCUSDT)
                    canonical = symbol.replace('/', '')
                    prices[canonical] = float(t['last'])
                except Exception:
                    pass
            paper_engine.check_liquidations(prices)
            # [FIX CRITICAL-2] Actually call check_tp_sl  - was implemented but never invoked
            paper_engine.check_tp_sl(prices)
        else:
            from scripts.monitor_stop_to_be import check_and_update_stop_to_be
            check_and_update_stop_to_be()
            
    except Exception as e:
        logger.error(f"1-minute execution job error: {e}")


def job_liquidation_cascade_monitor():
    """Score liquidation cascade risk and emit Telegram alerts when thresholds are met."""
    try:
        result = cascade_warning_engine.run_all()
        if result:
            logger.info(f"Liquidation cascade monitor alerts: {list(result.keys())}")
    except Exception as e:
        logger.error(f"Liquidation cascade monitor job error: {e}")



def job_1hour_deribit():
    """Collect Deribit options data: DVOL, PCR, IV Term Structure, 25d Skew."""
    try:
        deribit_collector.run()
    except Exception as e:
        logger.error(f"Deribit collection job error: {e}")

def job_8hour_funding_fee():
    """V8: Simulate funding fee deduction every 8 hours."""
    try:
        from executors.paper_exchange import paper_engine
        from executors.trade_executor import trade_executor
        
        rates = {}
        prices = {}
        positions = paper_engine.get_open_positions()
        if not positions: return
            
        for pos in positions:
            symbol = pos["symbol"]
            if symbol not in rates:
                try:
                    f_info = trade_executor.binance.fetch_funding_rate(symbol)
                    rates[symbol] = float(f_info['fundingRate'])
                    t = trade_executor.binance.fetch_ticker(symbol)
                    prices[symbol] = float(t['last'])
                except Exception as e:
                    logger.warning(f"Failed to fetch funding rate for {symbol}: {e}")
                    
        paper_engine.apply_funding_fees(rates, prices)
    except Exception as e:
        logger.error(f"8-hour funding fee job error: {e}")


def job_daily_fear_greed():
    """Collect Crypto Fear & Greed Index (alternative.me, daily)."""
    try:
        fear_greed_collector.run()
    except Exception as e:
        logger.error(f"Fear & Greed collection job error: {e}")


def job_daily_coinmetrics():
    """Collect daily Coin Metrics snapshots used for regime and risk gating."""
    try:
        if not settings.COINMETRICS_ENABLED:
            logger.info("Coin Metrics job skipped (disabled)")
            return
        coinmetrics_collector.run()
    except Exception as e:
        logger.error(f"Coin Metrics daily job error: {e}")



from cloud_jobs.market_status import run_market_status


def job_routine_market_status():
    run_market_status()


def job_quant_screener():
    """Alpha Confluence Engine (Event-Driven Quant Screener)"""
    try:
        from scripts.run_quant_alert import run_multi_market_screener, send_telegram_alert_and_execute
        logger.info("Running job_quant_screener (Alpha Confluence Engine)")
        anomalies = run_multi_market_screener()
        send_telegram_alert_and_execute(anomalies)
    except Exception as e:
        logger.error(f"Quant screener job error: {e}")



def job_outbox_drain():
    """Safety-net: drain any queued outbox messages that weren't published by their originating job."""
    try:
        from executors.outbox_dispatcher import outbox_dispatcher as _dispatcher
        result = _dispatcher.publish_pending(limit=50)
        if result.get("published", 0) or result.get("failed", 0):
            logger.info(f"Outbox drain: published={result.get('published', 0)} failed={result.get('failed', 0)}")
    except Exception as e:
        logger.error(f"Outbox drain job error: {e}")


def job_pressure_signal_evaluation():
    """Backfill forward-return evaluation into market_status_events.technical_snapshot.evaluation."""
    try:
        logger.info("Running realtime pressure signal evaluation")
        result = run_pressure_signal_evaluation(limit=200, hours=72, dry_run=False)
        logger.info(f"Realtime pressure evaluation result: {result}")
    except Exception as e:
        logger.error(f"Realtime pressure evaluation job error: {e}")


def job_daily_refresh_higher_tf_cache():
    """1m 로컬 parquet → 4h/1d/1w 재계산. 매일 02:00 UTC 실행.

    최근 2~6개월치 1m 데이터를 리샘플링해 4h(월별)/1d/1w(연별) 파일을 갱신.
    차트 요청 시 1m 전체를 메모리에 올리지 않아도 되므로 메모리 할당 문제 해결.
    """
    try:
        symbols = list(settings.trading_symbols)   # e.g. ["BTCUSDT", "ETHUSDT"]
        for symbol in symbols:
            logger.info(f"[refresh_higher_tf] start {symbol}")
            gcs_parquet_store.refresh_higher_tf_cache(symbol)
        logger.info("[refresh_higher_tf] all symbols done")
    except Exception as e:
        logger.error(f"job_daily_refresh_higher_tf_cache error: {e}")


def job_daily_archive_to_gcs():
    """Archive expiring rows to GCS Parquet and verify manifests."""
    try:
        logger.info("Running daily GCS archive job")
        archive_result = gcs_archive_exporter.run_daily_archive()
        logger.info(f"GCS Parquet archive: {archive_result}")
    except Exception as e:
        logger.error(f"Daily GCS archive job error: {e}")


def job_daily_safe_cleanup():
    """Delete only rows covered by verified archive manifests."""
    try:
        logger.info("Running daily safe cleanup job")
        result = gcs_archive_exporter.run_safe_cleanup(limit=1000)
        logger.info(f"Safe cleanup result: {result}")

        from executors.agent_state_store import agent_state_store as _ass
        _ass.purge_expired()

        graph_days = settings.RETENTION_GRAPH_DAYS
        if graph_days > 0:
            light_rag.cleanup_old(days=graph_days)
        stats = light_rag.get_stats()
        logger.info(f"LightRAG stats: {stats}")
    except Exception as e:
        logger.error(f"Daily safe cleanup job error: {e}")

    # Time-based Supabase cleanup (GCS archive 여부와 무관하게 항상 실행)
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=settings.RETENTION_TELEGRAM_DAYS)).isoformat()
        r = db.client.table("telegram_messages").delete().lt("created_at", cutoff).execute()
        logger.info(f"Telegram messages cleanup: deleted rows older than {settings.RETENTION_TELEGRAM_DAYS}d")
    except Exception as e:
        logger.warning(f"telegram_messages cleanup failed: {e}")

    try:
        market_status_days = int(getattr(settings, "RETENTION_MARKET_STATUS_DAYS", 30))
        cutoff = (datetime.now(timezone.utc) - timedelta(days=market_status_days)).isoformat()
        db.client.table("market_status_events").delete().lt("created_at", cutoff).execute()
        logger.info(f"market_status_events cleanup: deleted rows older than {market_status_days}d")
    except Exception as e:
        logger.warning(f"market_status_events cleanup failed: {e}")

    # Retention-based cleanup (GCS archive 없이도 항상 실행)
    try:
        result = db.cleanup_old_data()
        logger.info(f"Retention cleanup result: {result}")
    except Exception as e:
        logger.warning(f"Retention cleanup failed: {e}")





def _daily_precision_prep_bucket(now_utc: datetime | None = None) -> str:
    now = now_utc or datetime.now(timezone.utc)
    candidates = []
    base_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    base_minute = int(getattr(settings, "DAILY_PRECISION_MINUTE_UTC", 30))
    for day_offset in (-1, 0, 1):
        day_anchor = base_day + timedelta(days=day_offset)
        for hour in _daily_precision_schedule_hours_utc():
            candidates.append(day_anchor.replace(hour=hour, minute=base_minute))
    latest = max((item for item in candidates if item <= now), default=None)
    if latest is None:
        latest = min(candidates) if candidates else now
    return latest.isoformat(timespec="minutes")


def job_daily_precision_prepare_shared() -> None:
    global _daily_precision_prepare_bucket

    bucket = _daily_precision_prep_bucket()
    with _daily_precision_prepare_lock:
        if bucket == _daily_precision_prepare_bucket:
            logger.info(f"Daily precision shared prep already completed for bucket={bucket}")
            return
        logger.info(f"Running daily precision shared prep for bucket={bucket}")
        job_daily_coinmetrics()
        macro_collector.run()
        _daily_precision_prepare_bucket = bucket


def job_daily_precision_symbol(symbol: str):
    """Daily UTC single-symbol precision runner."""
    import concurrent.futures
    _HARD_TIMEOUT_S = 90 * 60  # 90분 하드 타임아웃

    try:
        from config.local_state import state_manager
        normalized_symbol = str(symbol or "").upper()
        if not state_manager.is_analysis_enabled():
            logger.info(f"Daily precision skipped for {normalized_symbol} (analysis disabled)")
            return
        job_daily_precision_prepare_shared()

        def _run():
            return orchestrator.run_daily_playbook_for_symbol(normalized_symbol, TradingMode.SWING)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run)
            try:
                result = future.result(timeout=_HARD_TIMEOUT_S)
                logger.info(f"Daily precision result for {normalized_symbol}: {result}")
            except concurrent.futures.TimeoutError:
                logger.error(
                    f"Daily precision TIMEOUT for {normalized_symbol} after {_HARD_TIMEOUT_S // 60}min — "
                    f"force-resetting precision lock"
                )
                # _daily_precision_active_count 카운터 리셋
                orchestrator._exit_daily_precision_run()
                # _analysis_locks의 실제 Lock 객체도 제거 (백그라운드 스레드가 계속 점유 중이므로
                # pop으로 새 Lock을 쓰도록 해야 다음 실행이 SKIPPED_LOCK에 걸리지 않음)
                orchestrator.force_release_analysis_lock(normalized_symbol, TradingMode.SWING)
    except Exception as e:
        logger.error(f"Daily precision job error for {symbol}: {e}")


def job_hourly_monitor():
    """Hourly: evaluate all symbol/mode pairs against Daily Playbook.
    Outputs NO_ACTION / WATCH / TRIGGER.
    TRIGGER  - run analysis + allow order execution (capped at 2/day/symbol).
    """
    _t0 = _time.perf_counter()
    try:
        from config.local_state import state_manager
        if not state_manager.is_analysis_enabled():
            logger.info("Hourly monitor skipped (analysis disabled)")
            return
        orchestrator.run_hourly_monitor()
        if _SCHED_PROM:
            SCHED_JOB_RESULTS.labels(job="hourly_monitor", result="success").inc()
    except Exception as e:
        logger.error(f"Hourly monitor job error: {e}")
        if _SCHED_PROM:
            SCHED_JOB_RESULTS.labels(job="hourly_monitor", result="error").inc()
    finally:
        if _SCHED_PROM:
            SCHED_JOB_DURATION.labels(job="hourly_monitor").observe(_time.perf_counter() - _t0)

    # SpotMode.POSITION: 현물 포지션 thesis invalidation 체크
    try:
        from executors.spot_orchestrator import SPOT_MODE_ENABLED, SPOT_MODE, SpotMode as _SpotMode, spot_orchestrator as _so
        if SPOT_MODE_ENABLED and SPOT_MODE == _SpotMode.POSITION:
            for sym in settings.trading_symbols:
                _so.check_position_invalidation(sym)
    except Exception as e:
        logger.error(f"Spot invalidation check error: {e}")


def main():
    mode = settings.trading_mode

    if _SCHED_PROM:
        try:
            _prom_start(9090, addr="0.0.0.0")
            logger.info("Prometheus metrics server started on 127.0.0.1:9090")
        except Exception as e:
            logger.warning(f"Prometheus server failed to start: {e}")

    logger.info(f"Starting Trading System (mode={mode.value})")
    logger.info(
        f"  Primary analysis cadence (UTC): daily_precision={_daily_precision_all_labels_utc()} | "
        "hourly_monitor=hh:15 | market_status=hh:20"
    )
    logger.info(f"  Timeframes: {settings.analysis_timeframes}")
    logger.info(f"  Chart timeframe: {settings.chart_timeframe}")
    logger.info(f"  Candle limit: {settings.candle_limit}")
    logger.info(f"  Data lookback: {settings.data_lookback_hours}h")
    logger.info(f"  Chart for VLM: {settings.should_use_chart}")
    logger.info(f"  Symbols: {', '.join(settings.trading_symbols)}")
    logger.info(f"  AI: Gemini Judge/VLM (Project A/B) + Cerebras (meta/risk) + Groq (news/rag) + OpenRouter (monitor)")
    logger.info(f"  Data: Global OI + OI Divergence + MFI Proxy + Liquidations + Coin Metrics + Gemini Search + LightRAG")
    logger.info(f"  Dune: {'enabled' if dune_collector else 'disabled'}")
    logger.info(f"  LightRAG: Neo4j {'connected' if settings.NEO4J_URI else 'in-memory'} + "
                f"Milvus {'connected' if settings.MILVUS_URI else 'in-memory'}")

    # [PERF] Start real-time price feed — eliminates REST price fetch at order fire time
    try:
        from collectors.ws_price_feed import ws_price_feed
        ws_price_feed.start()
        logger.info("WS price feed started (zero-latency price reads for order execution)")
    except Exception as e:
        logger.warning(f"WS price feed unavailable (REST fallback active): {e}")

    # [PERF] Start User Data Stream — real-time fill confirmation (replaces outbox polling)
    # Futures user stream only valid with production Futures key — skip in testnet mode
    if not settings.BINANCE_USE_TESTNET:
        try:
            from collectors.ws_user_stream import ws_user_stream
            ws_user_stream.start()
            logger.info("WS user stream started (real-time fill confirmation)")
        except Exception as e:
            logger.warning(f"WS user stream unavailable (outbox fallback active): {e}")
    else:
        logger.info("WS user stream skipped (testnet mode)")

    # Start WebSocket collector for liquidation + whale data
    try:
        from collectors.websocket_collector import websocket_collector
        websocket_collector.start_background()
        logger.info("WebSocket collector started (liquidation + whale trades)")
    except Exception as e:
        logger.warning(f"WebSocket collector unavailable: {e}")

    # WebSocket thread health check — every 5 minutes
    # Detects both: (1) thread death, (2) alive but no messages for >10 min (stuck connection)
    def job_5m_ws_health_check():
        import time as _time
        try:
            from collectors.websocket_collector import websocket_collector
            thread = getattr(websocket_collector, '_thread', None)
            last_msg = getattr(websocket_collector, '_last_message_time', 0.0)
            silent_secs = _time.time() - last_msg if last_msg else float('inf')

            thread_dead = thread is None or not thread.is_alive()
            # 연결 후 10분 이상 메시지 없으면 끊긴 것으로 간주
            # last_msg=0은 한 번도 수신 못한 경우 → 시작 후 5분 초과 시 재시작
            if last_msg == 0.0:
                started = getattr(websocket_collector, '_started_at', _time.time())
                silent_secs = _time.time() - started
            silent_too_long = silent_secs > 300  # 5분 이상 메시지 없으면 재시작

            if thread_dead or silent_too_long:
                reason = "thread dead" if thread_dead else f"no messages for {silent_secs/60:.1f}min"
                logger.warning(f"WebSocket collector unhealthy ({reason}) — restarting...")
                websocket_collector.stop()
                websocket_collector.start_background()
        except Exception as e:
            logger.error(f"WS health check error: {e}")

    scheduler_config.scheduler.add_job(
        job_5m_ws_health_check,
        'interval',
        minutes=5,
        id='job_5m_ws_health',
        max_instances=1
    )

    import os
    # [FIX Cold Start] Run Telegram bot in a daemon thread (unless disabled by env var)
    if os.environ.get("DISABLE_TELEGRAM_BOT") != "true":
        def _run_telegram_bot():
            try:
                from bot.telegram_bot import trading_bot
                trading_bot.run()  # blocking within this thread
            except Exception as e:
                logger.error(f"Telegram bot crashed: {e}")
                logger.info("Trading system continues WITHOUT Telegram bot commands.")

        bot_thread = threading.Thread(target=_run_telegram_bot, name="telegram-bot", daemon=True)
        bot_thread.start()
        logger.info("Telegram bot started in background thread.")
    else:
        logger.info("Telegram bot starting skipped (DISABLE_TELEGRAM_BOT=true).")
        # 더미 스레드를 생성하여 is_alive() 체크에서 터지지 않도록 방어
        bot_thread = threading.Thread(target=lambda: None)

    # [V13.1] Persistent Telegram Listener (Real-Time Alpha Ingestion)
    if os.environ.get("DISABLE_TELEGRAM_LISTENER") != "true":
        def _run_telegram_listener():
            try:
                from collectors.telegram_listener import telegram_listener
                import asyncio
                asyncio.run(telegram_listener.start())
            except Exception as e:
                logger.error(f"Telegram listener crashed: {e}")

        listener_thread = threading.Thread(target=_run_telegram_listener, name="telegram-listener", daemon=True)
        listener_thread.start()
        logger.info("Real-time Telegram Listener (Alpha V13.1) started.")
    else:
        logger.info("Telegram Listener starting skipped (DISABLE_TELEGRAM_LISTENER=true).")
        listener_thread = threading.Thread(target=lambda: None)

    role = os.environ.get("SCHEDULER_ROLE", "all").lower()
    logger.info(f"Scheduler starting with ROLE: {role.upper()}")

    # ==========================================
    # DATA JOBS: 데이터 수집 및 인프라 (가격, 온체인, 뉴스)
    # ==========================================
    if role in ("all", "data"):
        # 1-minute tick: price, funding, microstructure, volatility
        scheduler_config.scheduler.add_job(
            job_1min_tick,
            'interval',
            minutes=1,
            id='job_1min_tick',
            max_instances=1
        )
        
        scheduler_config.scheduler.add_job(
            job_1hour_deribit,
            CronTrigger(minute=0),
            id='job_1hour_deribit',
            max_instances=1
        )

        scheduler_config.scheduler.add_job(
            job_daily_fear_greed,
            CronTrigger(hour=0, minute=15),
            id='job_daily_fear_greed',
            max_instances=1
        )

        scheduler_config.scheduler.add_job(
            job_daily_coinmetrics,
            CronTrigger(hour=3, minute=12),
            id='job_daily_coinmetrics',
            max_instances=1
        )

        scheduler_config.scheduler.add_job(
            job_daily_refresh_higher_tf_cache,
            CronTrigger(hour=2, minute=0),
            id='job_daily_refresh_higher_tf_cache',
            max_instances=1
        )

        scheduler_config.scheduler.add_job(
            job_daily_archive_to_gcs,
            CronTrigger(hour=1, minute=0),
            id='job_daily_archive_to_gcs',
            max_instances=1
        )

        scheduler_config.scheduler.add_job(
            job_daily_safe_cleanup,
            CronTrigger(hour=1, minute=20),
            id='job_daily_safe_cleanup',
            max_instances=1
        )


    # ==========================================
    # BRAIN JOBS: AI 판단, 주문 실행, 텔레그램 리포트 관리
    # ==========================================
    if role in ("all", "brain"):
        if not EXECUTION_PROCESS_SEPARATE:
            scheduler_config.scheduler.add_job(
                job_1min_execution,
                'interval',
                minutes=1,
                id='job_1min_execution',
                max_instances=1
            )

        scheduler_config.scheduler.add_job(
            job_liquidation_cascade_monitor,
            'interval',
            minutes=1,
            id='job_liquidation_cascade_monitor',
            max_instances=1
        )

        if settings.PAPER_TRADING_MODE and not EXECUTION_PROCESS_SEPARATE:
            scheduler_config.scheduler.add_job(
                job_8hour_funding_fee,
                CronTrigger(hour='0,8,16', minute=0),
                id='job_8hour_funding_fee',
                max_instances=1
            )

        scheduler_config.scheduler.add_job(
            job_outbox_drain,
            'interval',
            minutes=5,
            id='job_outbox_drain',
            max_instances=1,
        )

        scheduler_config.scheduler.add_job(
            job_pressure_signal_evaluation,
            CronTrigger(minute='*/15'),
            id='job_pressure_signal_evaluation',
            max_instances=1,
        )

        scheduler_config.scheduler.add_job(
            job_snapshot_refresh_fast,
            CronTrigger(minute='*/15'),
            id='job_snapshot_refresh_fast',
            max_instances=1,
        )

        scheduler_config.scheduler.add_job(
            job_routine_market_status,
            CronTrigger(minute=20),
            id='job_routine_market_status',
            max_instances=1,
        )

        scheduler_config.scheduler.add_job(
            job_quant_screener,
            CronTrigger(hour='*/4', minute=5), # Run every 4 hours at xx:05
            id='job_quant_screener',
            max_instances=1,
        )

        try:
            from executors.spot_orchestrator import (
                SPOT_MODE_ENABLED as _SPOT_ENABLED,
                SPOT_MODE as _SPOT_MODE,
                SPOT_POSITION_ANALYSIS_HOUR_UTC as _SPOT_HOUR,
                SpotMode as _SpotMode,
            )
            if _SPOT_ENABLED and _SPOT_MODE == _SpotMode.POSITION:
                def job_spot_position_daily():
                    try:
                        from executors.spot_orchestrator import spot_orchestrator
                        for sym in settings.trading_symbols:
                            spot_orchestrator.run_position_analysis(sym)
                    except Exception as e:
                        logger.error(f"Spot position daily analysis error: {e}")

                scheduler_config.scheduler.add_job(
                    job_spot_position_daily,
                    CronTrigger(hour=_SPOT_HOUR, minute=30),
                    id='job_spot_position_daily',
                    max_instances=1,
                )
                logger.info(f"Spot position daily job registered at {_SPOT_HOUR:02d}:30 UTC")
        except Exception as _spot_sched_err:
            logger.warning(f"Spot scheduler setup skipped: {_spot_sched_err}")

    scheduler_config.scheduler.start()
    logger.info("Scheduler started.")
    logger.info(
        f"Cadence(UTC): daily_precision={_daily_precision_all_labels_utc()} | snapshot_fast=hh:*/15 | "
        "snapshot_narrative=hh:02,32 | telegram_batch=hh:05 | crypto_news=hh:10 | hourly_monitor=hh:15 | market_status=hh:20 | daily_rollup=00:40 | "
        "hourly_eval=hh:45 | gcs_archive=01:00 | safe_cleanup=01:20 | higher_tf_refresh=02:00"
    )

    logger.info(f"Running startup bootstrap (ROLE={role})...")
    def _startup_telegram_catchup():
        """Handles initial RAG synthesis for the last 24h.
        Raw collection is handled by the telegram_listener thread's backfill task.
        """
        import time
        # Increased delay to 30s to let the Listener finish its metadata-only backfill
        time.sleep(30)
        
        from processors.telegram_batcher import telegram_batcher
        # Reduced to 6h for synthesized catch-up to avoid heavy LLM load on cold start
        telegram_batcher.process_and_ingest(lookback_hours=6)

    def _bootstrap_missing_parquet_cache():
        """스타트업 시 로컬 캐시에 없는 심볼의 parquet 파일을 GCS에서 다운로드."""
        try:
            from processors.gcs_parquet import gcs_parquet_store
            if not gcs_parquet_store.enabled:
                logger.info("GCS disabled — parquet cache bootstrap skipped")
                return
            for symbol in settings.trading_symbols:
                for tf, months_back in [("4h", settings.SWING_HISTORY_MONTHS), ("1d", settings.SWING_HISTORY_MONTHS), ("1w", settings.SWING_HISTORY_MONTHS)]:
                    paths = gcs_parquet_store._build_ohlcv_paths(tf, symbol, months_back)
                    missing = [p for p in paths if not gcs_parquet_store._local_cache_path(p).exists()]
                    if missing:
                        logger.info(f"Bootstrapping {len(missing)} missing {tf} parquet files for {symbol}...")
                        for p in missing:
                            gcs_parquet_store._download_parquet(p)
        except Exception as e:
            logger.warning(f"Parquet cache bootstrap failed (non-fatal): {e}")

    # Phase 1: data collection bootstrap — data role only
    if role in ("all", "data"):
        _parallel_collectors = [
            ("Price + Funding + Microstructure", lambda: (collector.run(), funding_collector.run(), microstructure_collector.run())),
            ("Volatility", lambda: volatility_monitor.run()),
            ("Deribit", lambda: deribit_collector.run()),
            ("Fear & Greed", lambda: fear_greed_collector.run()),
            ("Coin Metrics", lambda: job_daily_coinmetrics()),
        ]
        with ThreadPoolExecutor(max_workers=len(_parallel_collectors), thread_name_prefix="boot") as _boot_pool:
            _boot_futures = {_boot_pool.submit(fn): name for name, fn in _parallel_collectors}
            for _fut in as_completed(_boot_futures, timeout=120):
                _name = _boot_futures[_fut]
                try:
                    _fut.result(timeout=1)
                    logger.info(f"  [OK] {_name} collected")
                except Exception as e:
                    logger.warning(f"  [WARN] {_name} collection failed (non-fatal): {e}")
    else:
        logger.info("Phase 1 data collection skipped (ROLE=brain)")

    # Phase 2: brain bootstrap — all roles need parquet cache, brain jobs only for brain role
    _phase2_jobs = [("Parquet cache bootstrap", _bootstrap_missing_parquet_cache)]
    if role in ("all", "brain"):
        _phase2_jobs += [
            ("Snapshot prewarm", lambda: job_snapshot_refresh_fast()),
            ("Pressure signal evaluation", lambda: job_pressure_signal_evaluation()),
        ]
    for name, fn in _phase2_jobs:
        try:
            fn()
            logger.info(f"  [OK] {name}")
        except Exception as e:
            logger.warning(f"  [WARN] {name} failed (non-fatal): {e}")

    # Phase 3: Telegram catch-up — data role only
    if role in ("all", "data"):
        threading.Thread(
            target=_startup_telegram_catchup,
            name="startup-telegram-catchup",
            daemon=True,
        ).start()
        logger.info("  [OK] Telegram catch-up started in background")
    else:
        logger.info("Phase 3 Telegram catch-up skipped (ROLE=brain)")

    # Main thread: keep alive + graceful shutdown
    try:
        import time
        while True:
            time.sleep(30)
            # Optional: restart bot thread if it dies (only if not disabled)
            if os.environ.get("DISABLE_TELEGRAM_BOT") != "true":
                if not bot_thread.is_alive():
                    logger.warning("Telegram bot thread died — restarting in 5s...")
                    time.sleep(5)
                    bot_thread = threading.Thread(target=_run_telegram_bot, name="telegram-bot", daemon=True)
                    bot_thread.start()

            if os.environ.get("DISABLE_TELEGRAM_LISTENER") != "true":
                if not listener_thread.is_alive():
                    logger.warning("Telegram listener thread died — restarting in 5s...")
                    time.sleep(5)
                    listener_thread = threading.Thread(target=_run_telegram_listener, name="telegram-listener", daemon=True)
                    listener_thread.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down...")
        # Upload Telegram session to Secret Manager before exit
        try:
            from collectors.telegram_listener import upload_session_to_cloud
            upload_session_to_cloud()
        except Exception:
            pass
        try:
            from collectors.websocket_collector import websocket_collector
            websocket_collector.stop()
        except Exception:
            pass
        scheduler_config.scheduler.shutdown()
        sys.exit(0)


if __name__ == "__main__":
    main()







