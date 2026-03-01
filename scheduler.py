from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from collectors.price_collector import collector
from collectors.funding_collector import funding_collector
from collectors.volatility_monitor import volatility_monitor
from collectors.dune_collector import dune_collector
from collectors.microstructure_collector import microstructure_collector
from collectors.macro_collector import macro_collector
from collectors.deribit_collector import deribit_collector
from collectors.fear_greed_collector import fear_greed_collector
from executors.orchestrator import orchestrator
from evalutors.feedback_generator import feedback_generator
from processors.light_rag import light_rag
from processors.gcs_archive import gcs_archive_exporter
from config.settings import settings, TradingMode
from config.database import db
from executors.order_manager import execution_desk
from executors.evaluator_daemon import EvaluatorDaemon
from executors.paper_exchange import paper_engine
from loguru import logger
import sys
import threading


def job_1min_tick():
    # 1. Price Collection (Critical)
    try:
        collector.run()
    except Exception as e:
        logger.error(f"Price collection error: {e}")

    # 2. Funding Rate (8h signal, 1m cadence)
    try:
        funding_collector.run()
    except Exception as e:
        logger.error(f"Funding collection error: {e}")

    # 3. Microstructure (Ephemeral)
    try:
        microstructure_collector.run()
    except Exception as e:
        logger.error(f"Microstructure collection error: {e}")

    # 4. Volatility Monitor
    try:
        volatility_monitor.run()
    except Exception as e:
        logger.error(f"Volatility monitor error: {e}")

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
            # [FIX CRITICAL-2] Actually call check_tp_sl — was implemented but never invoked
            paper_engine.check_tp_sl(prices)
            
    except Exception as e:
        logger.error(f"1-minute execution job error: {e}")


def job_15min_dune():
    """Collect cadence-aware Dune snapshots and persist to DB."""
    if dune_collector is None:
        return
    try:
        dune_collector.run_due_queries(limit=200, offset=0)
    except Exception as e:
        logger.error(f"15-minute Dune job error: {e}")


def job_1hour_deribit():
    """Collect Deribit options data: DVOL, PCR, IV Term Structure, 25d Skew."""
    try:
        deribit_collector.run()
    except Exception as e:
        logger.error(f"Deribit collection job error: {e}")


def job_daily_fear_greed():
    """Collect Crypto Fear & Greed Index (alternative.me, daily)."""
    try:
        fear_greed_collector.run()
    except Exception as e:
        logger.error(f"Fear & Greed collection job error: {e}")


def job_analysis():
    """Mode-aware analysis job. Interval depends on TRADING_MODE."""
    try:
        mode = settings.trading_mode
        logger.info(f"Running analysis job (mode={mode.value}, "
                     f"interval={settings.analysis_interval_hours}h)")
        macro_collector.run()
        orchestrator.run_scheduled_analysis()
    except Exception as e:
        logger.error(f"Analysis job error: {e}")


def job_24hour_evaluation():
    try:
        logger.info("Running 24-hour evaluation job")
        feedback_generator.run_feedback_cycle()
    except Exception as e:
        logger.error(f"24-hour evaluation job error: {e}")

def job_1hour_evaluation():
    """V6: Self-Healing RAG evaluation of completed trades."""
    try:
        logger.info("Running 1-hour episodic memory evaluation")
        daemon = EvaluatorDaemon()
        daemon.evaluate_recent_trades()
    except Exception as e:
        logger.error(f"1-hour evaluation job error: {e}")

def job_1hour_telegram():
    """Fetch and batch Telegram messages into LightRAG every 1 hour."""
    try:
        logger.info("Running 1-hour Telegram batching job")
        # 1. Fetch raw messages directly via the collector
        from collectors.telegram_collector import telegram_collector
        telegram_collector.run(hours=1)
        
        # 2. Synthesize and ingest
        from processors.telegram_batcher import telegram_batcher
        telegram_batcher.process_and_ingest(lookback_hours=1)
        
        # 3. Truth Engine: Triangulate corroborated claims via Perplexity (Web)
        # Process 10 candidates per hour to manage API costs
        light_rag.run_triangulation_worker(limit=10)
    except Exception as e:
        logger.error(f"1-hour Telegram job error: {e}")


def job_daily_cleanup():
    """Cleanup old data + archive to GCS Parquet."""
    try:
        logger.info("Running daily data cleanup")

        # 1) Archive expiring rows to GCS Parquet
        archive_result = gcs_archive_exporter.run_daily_archive()
        logger.info(f"GCS Parquet archive: {archive_result}")

        # 2) Cleanup hot operational DB
        result = db.cleanup_old_data()
        logger.info(f"DB cleanup result: {result}")

        # 3) Cleanup LightRAG in-memory data
        graph_days = settings.RETENTION_GRAPH_DAYS
        if graph_days > 0:
            light_rag.cleanup_old(days=graph_days)
        stats = light_rag.get_stats()
        logger.info(f"LightRAG stats: {stats}")
    except Exception as e:
        logger.error(f"Daily cleanup error: {e}")


def _build_analysis_trigger(mode: TradingMode):
    """Build APScheduler trigger based on trading mode.
    - DAY_TRADING: every 1 hour
    - SWING: every 4 hours
    - POSITION: once daily at 00:00 UTC (= 09:00 KST, Upbit daily open)
    """
    interval = settings.analysis_interval_hours

    if mode == TradingMode.POSITION:
        # Once daily at 00:00 UTC = 09:00 KST (Upbit new day)
        return CronTrigger(hour=0, minute=0)
    else:
        # SWING: every 8 hours (00:00, 08:00, 16:00 UTC) to save costs
        return CronTrigger(hour='0,8,16', minute=0)


def main():
    mode = settings.trading_mode
    interval = settings.analysis_interval_hours

    logger.info(f"Starting Trading System (mode={mode.value})")
    logger.info(f"  Analysis interval: {interval}h")
    logger.info(f"  Timeframes: {settings.analysis_timeframes}")
    logger.info(f"  Chart timeframe: {settings.chart_timeframe}")
    logger.info(f"  Candle limit: {settings.candle_limit}")
    logger.info(f"  Data lookback: {settings.data_lookback_hours}h")
    logger.info(f"  Chart for VLM: {settings.should_use_chart}")
    logger.info(f"  Symbols: {', '.join(settings.trading_symbols)}")
    logger.info(f"  AI: Gemini Flash (agents) + Claude Opus 4.6 (judge)")
    logger.info(f"  Data: Global OI + CVD + Whale CVD + Liquidations + Perplexity + LightRAG")
    logger.info(f"  Dune: {'enabled' if dune_collector else 'disabled'}")
    logger.info(f"  LightRAG: Neo4j {'connected' if settings.NEO4J_URI else 'in-memory'} + "
                f"Milvus {'connected' if settings.MILVUS_URI else 'in-memory'}")

    # Start WebSocket collector for liquidation + whale data
    try:
        from collectors.websocket_collector import websocket_collector
        websocket_collector.start_background()
        logger.info("WebSocket collector started (liquidation + whale trades)")
    except Exception as e:
        logger.warning(f"WebSocket collector unavailable: {e}")

    # [FIX MEDIUM-18] WebSocket thread health check
    def job_5m_ws_health_check():
        try:
            from collectors.websocket_collector import websocket_collector
            if hasattr(websocket_collector, '_thread'):
                if not websocket_collector._thread.is_alive():
                    logger.warning("WebSocket thread died — restarting...")
                    websocket_collector.start_background()
        except Exception as e:
            logger.error(f"WS health check error: {e}")

    # Use BackgroundScheduler so we can also run the Telegram bot
    scheduler = BackgroundScheduler()

    # 1-minute tick: price, funding, microstructure, volatility
    scheduler.add_job(
        job_1min_tick,
        'interval',
        minutes=1,
        id='job_1min_tick',
        max_instances=1
    )
    
    # 1-minute execution tick: ExecutionDesk and Paper Engine Liquidations
    scheduler.add_job(
        job_1min_execution,
        'interval',
        minutes=1,
        id='job_1min_execution',
        max_instances=1
    )

    # 15-minute Dune
    scheduler.add_job(
        job_15min_dune,
        'interval',
        minutes=15,
        id='job_15min_dune',
        max_instances=1
    )

    # Deribit options data: DVOL, PCR, IV Term Structure, 25d Skew — every 1 hour
    scheduler.add_job(
        job_1hour_deribit,
        'interval',
        hours=1,
        id='job_1hour_deribit',
        max_instances=1
    )

    # Fear & Greed Index — daily at 00:15 UTC (after data refreshes)
    scheduler.add_job(
        job_daily_fear_greed,
        CronTrigger(hour=0, minute=15),
        id='job_daily_fear_greed',
        max_instances=1
    )

    # 1-Hour Telegram Batching & Ingestion
    scheduler.add_job(
        job_1hour_telegram,
        'interval',
        hours=1,
        id='job_1hour_telegram',
        max_instances=1
    )

    # Mode-aware analysis (1h / 4h / 24h depending on mode)
    scheduler.add_job(
        job_analysis,
        _build_analysis_trigger(mode),
        id='job_analysis',
        max_instances=1
    )

    # Daily evaluation at 00:30 UTC = 09:30 KST
    scheduler.add_job(
        job_24hour_evaluation,
        CronTrigger(hour=0, minute=30),
        id='job_24hour_evaluation',
        max_instances=1
    )
    
    # 1-Hour RAG Episodic Memory Evaluation (V6)
    scheduler.add_job(
        job_1hour_evaluation,
        'interval',
        hours=1,
        id='job_1hour_evaluation',
        max_instances=1
    )

    # Daily cleanup at 01:00 UTC = 10:00 KST
    scheduler.add_job(
        job_daily_cleanup,
        CronTrigger(hour=1, minute=0),
        id='job_daily_cleanup',
        max_instances=1
    )

    # [FIX MEDIUM-18] WebSocket thread health check — every 5 minutes
    scheduler.add_job(
        job_5m_ws_health_check,
        'interval',
        minutes=5,
        id='job_5m_ws_health',
        max_instances=1
    )

    scheduler.start()
    logger.info("Scheduler started.")

    # [FIX Cold Start] Run initial data collection immediately so first analysis has data
    logger.info("Running initial data collection (cold start bootstrap)...")
    _initial_collectors = [
        ("Price + Funding + Microstructure", lambda: (collector.run(), funding_collector.run(), microstructure_collector.run())),
        ("Volatility", lambda: volatility_monitor.run()),
        ("Deribit", lambda: deribit_collector.run()),
        ("Fear & Greed", lambda: fear_greed_collector.run()),
    ]
    for name, fn in _initial_collectors:
        try:
            fn()
            logger.info(f"  ✅ {name} collected")
        except Exception as e:
            logger.warning(f"  ⚠️ {name} collection failed (non-fatal): {e}")

    # [FIX Cold Start] Run Telegram bot in a daemon thread — if it crashes,
    # scheduler keeps running. This prevents a Telegram session error from
    # killing the entire trading system.
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

    # Main thread: keep alive + graceful shutdown
    try:
        import time
        while True:
            time.sleep(60)
            # Optional: restart bot thread if it dies
            if not bot_thread.is_alive():
                logger.warning("Telegram bot thread died — restarting in 30s...")
                time.sleep(30)
                bot_thread = threading.Thread(target=_run_telegram_bot, name="telegram-bot", daemon=True)
                bot_thread.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down...")
        # Upload Telegram session to Secret Manager before exit
        try:
            from collectors.telegram_collector import upload_session_to_secret_manager
            upload_session_to_secret_manager()
        except Exception:
            pass
        try:
            from collectors.websocket_collector import websocket_collector
            websocket_collector.stop()
        except Exception:
            pass
        scheduler.shutdown()
        sys.exit(0)


if __name__ == "__main__":
    main()
