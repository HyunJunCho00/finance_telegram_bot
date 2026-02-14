from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from collectors.price_collector import collector
from collectors.funding_collector import funding_collector
from collectors.volatility_monitor import volatility_monitor
from executors.orchestrator import orchestrator
from evalutors.feedback_generator import feedback_generator
from processors.light_rag import light_rag
from config.settings import settings
from config.database import db
from loguru import logger
import sys
import threading


def job_1min_tick():
    try:
        collector.run()
        funding_collector.run()
        volatility_monitor.run()
    except Exception as e:
        logger.error(f"1-minute tick job error: {e}")


def job_4hour_analysis():
    try:
        mode = settings.trading_mode
        logger.info(f"Running 4-hour analysis job (mode={mode.value})")
        orchestrator.run_scheduled_analysis()
    except Exception as e:
        logger.error(f"4-hour analysis job error: {e}")


def job_24hour_evaluation():
    try:
        logger.info("Running 24-hour evaluation job")
        feedback_generator.run_feedback_cycle()
    except Exception as e:
        logger.error(f"24-hour evaluation job error: {e}")


def job_daily_cleanup():
    """Cleanup old data.
    - PostgreSQL(Supabase): timeseries cleanup (30 days)
    - Neo4j/Milvus: graph data preserved permanently (RETENTION_GRAPH_DAYS=0)
    - LightRAG in-memory: cleanup based on telegram retention"""
    try:
        logger.info("Running daily data cleanup")
        result = db.cleanup_old_data()
        logger.info(f"DB cleanup result: {result}")

        # Cleanup LightRAG in-memory data (not Neo4j/Milvus which are permanent)
        graph_days = settings.RETENTION_GRAPH_DAYS
        if graph_days > 0:
            light_rag.cleanup_old(days=graph_days)
        stats = light_rag.get_stats()
        logger.info(f"LightRAG stats: {stats}")
    except Exception as e:
        logger.error(f"Daily cleanup error: {e}")


def main():
    mode = settings.trading_mode
    logger.info(f"Starting Trading System (mode={mode.value})")
    logger.info(f"  Candle limit: {settings.candle_limit}")
    logger.info(f"  Chart for VLM: {settings.should_use_chart}")
    logger.info(f"  Symbols: BTCUSDT, ETHUSDT")
    logger.info(f"  AI: Gemini Flash (agents) + Claude Opus 4.6 (judge)")
    logger.info(f"  Data: Global OI (3 exchanges) + CVD + Perplexity + LightRAG")
    logger.info(f"  LightRAG: Neo4j {'connected' if settings.NEO4J_URI else 'in-memory'} + "
                f"Milvus {'connected' if settings.MILVUS_URI else 'in-memory'}")
    logger.info(f"  Pipeline: LangGraph StateGraph")
    logger.info(f"  Self-correction: feedback to ALL agents")

    # Use BackgroundScheduler so we can also run the Telegram bot
    scheduler = BackgroundScheduler()

    scheduler.add_job(
        job_1min_tick,
        'interval',
        minutes=1,
        id='job_1min_tick',
        max_instances=1
    )

    scheduler.add_job(
        job_4hour_analysis,
        CronTrigger(hour='*/4'),
        id='job_4hour_analysis',
        max_instances=1
    )

    scheduler.add_job(
        job_24hour_evaluation,
        CronTrigger(hour=0, minute=30),
        id='job_24hour_evaluation',
        max_instances=1
    )

    scheduler.add_job(
        job_daily_cleanup,
        CronTrigger(hour=1, minute=0),  # Run at 01:00 UTC daily
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
        scheduler.shutdown()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Telegram bot error: {e}")
        # Even if bot fails, keep scheduler running
        logger.info("Bot failed but scheduler continues. Press Ctrl+C to stop.")
        try:
            import time
            while True:
                time.sleep(60)
        except (KeyboardInterrupt, SystemExit):
            logger.info("Shutting down...")
            scheduler.shutdown()
            sys.exit(0)


if __name__ == "__main__":
    main()
