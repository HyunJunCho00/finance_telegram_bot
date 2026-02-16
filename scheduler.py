from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from collectors.price_collector import collector
from collectors.funding_collector import funding_collector
from collectors.volatility_monitor import volatility_monitor
from collectors.dune_collector import dune_collector
from collectors.microstructure_collector import microstructure_collector
from collectors.macro_collector import macro_collector
from executors.orchestrator import orchestrator
from evalutors.feedback_generator import feedback_generator
from processors.light_rag import light_rag
from processors.gcs_archive import gcs_archive_exporter
from config.settings import settings, TradingMode
from config.database import db
from loguru import logger
import sys
import threading


def job_1min_tick():
    try:
        collector.run()
        funding_collector.run()
        microstructure_collector.run()
        volatility_monitor.run()
    except Exception as e:
        logger.error(f"1-minute tick job error: {e}")


def job_15min_dune():
    """Collect cadence-aware Dune snapshots and persist to DB."""
    if dune_collector is None:
        return
    try:
        dune_collector.run_due_queries(limit=200, offset=0)
    except Exception as e:
        logger.error(f"15-minute Dune job error: {e}")


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
    elif mode == TradingMode.DAY_TRADING:
        # Every hour
        return CronTrigger(minute=0)  # At :00 of every hour
    else:
        # SWING: every 4 hours
        return CronTrigger(hour='*/4', minute=0)


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
    logger.info(f"  Symbols: BTCUSDT, ETHUSDT")
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

    # 15-minute Dune
    scheduler.add_job(
        job_15min_dune,
        'interval',
        minutes=15,
        id='job_15min_dune',
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

    # Daily cleanup at 01:00 UTC = 10:00 KST
    scheduler.add_job(
        job_daily_cleanup,
        CronTrigger(hour=1, minute=0),
        id='job_daily_cleanup',
        max_instances=1
    )

    scheduler.start()
    logger.info("Scheduler started. Now starting Telegram bot...")

    # Run Telegram bot in the main thread (blocking)
    try:
        from bot.telegram_bot import trading_bot
        trading_bot.run()  # This blocks (polling loop)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down...")
        try:
            from collectors.websocket_collector import websocket_collector
            websocket_collector.stop()
        except Exception:
            pass
        scheduler.shutdown()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Telegram bot error: {e}")
        logger.info("Bot failed but scheduler continues. Press Ctrl+C to stop.")
        try:
            import time
            while True:
                time.sleep(60)
        except (KeyboardInterrupt, SystemExit):
            logger.info("Shutting down...")
            try:
                from collectors.websocket_collector import websocket_collector
                websocket_collector.stop()
            except Exception:
                pass
            scheduler.shutdown()
            sys.exit(0)


if __name__ == "__main__":
    main()
