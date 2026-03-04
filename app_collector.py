"""Data Collector Application. Runs 24/7."""
import sys
import time
from loguru import logger
from apscheduler.triggers.cron import CronTrigger

from config import scheduler_config
from config.settings import settings

from collectors.price_collector import collector
from collectors.funding_collector import funding_collector
from collectors.volatility_monitor import volatility_monitor
from collectors.dune_collector import dune_collector
from collectors.microstructure_collector import microstructure_collector
from collectors.deribit_collector import deribit_collector
from collectors.fear_greed_collector import fear_greed_collector
from collectors.crypto_news_collector import collector as news_collector

from processors.gcs_archive import gcs_archive_exporter
from config.database import db

def job_1min_tick():
    try: collector.run()
    except Exception as e: logger.error(f"Price collection error: {e}")
    try: funding_collector.run()
    except Exception as e: logger.error(f"Funding collection error: {e}")
    try: microstructure_collector.run()
    except Exception as e: logger.error(f"Microstructure collection error: {e}")
    try: volatility_monitor.run()
    except Exception as e: logger.error(f"Volatility monitor error: {e}")

def job_15min_dune():
    if dune_collector is None: return
    try: dune_collector.run_due_queries(limit=200, offset=0)
    except Exception as e: logger.error(f"15-minute Dune job error: {e}")

def job_1hour_deribit():
    try: deribit_collector.run()
    except Exception as e: logger.error(f"Deribit collection error: {e}")

def job_daily_fear_greed():
    try: fear_greed_collector.run()
    except Exception as e: logger.error(f"Fear & Greed collection error: {e}")

def job_1hour_crypto_news():
    try: news_collector.fetch_and_ingest()
    except Exception as e: logger.error(f"1-hour Crypto News API error: {e}")

def job_daily_cleanup():
    try:
        logger.info("Running daily data cleanup")
        archive_result = gcs_archive_exporter.run_daily_archive()
        logger.info(f"GCS Parquet archive: {archive_result}")
        result = db.cleanup_old_data()
        logger.info(f"DB cleanup result: {result}")
        
        graph_days = settings.RETENTION_GRAPH_DAYS
        from processors.light_rag import light_rag
        if graph_days > 0:
            light_rag.cleanup_old(days=graph_days)
        stats = light_rag.get_stats()
        logger.info(f"LightRAG stats: {stats}")
    except Exception as e:
        logger.error(f"Daily cleanup error: {e}")

def job_5m_ws_health_check():
    try:
        from collectors.websocket_collector import websocket_collector
        if hasattr(websocket_collector, '_thread') and not websocket_collector._thread.is_alive():
            logger.warning("WebSocket thread died — restarting...")
            websocket_collector.start_background()
    except Exception as e:
        logger.error(f"WS health check error: {e}")

def main():
    logger.info("Starting Data Collector Application (app_collector.py)")
    
    try:
        from collectors.websocket_collector import websocket_collector
        websocket_collector.start_background()
        logger.info("WebSocket collector started")
    except Exception as e:
        logger.warning(f"WebSocket collector unavailable: {e}")

    sched = scheduler_config.scheduler
    sched.add_job(job_5m_ws_health_check, 'interval', minutes=5, id='job_5m_ws_health', max_instances=1)
    sched.add_job(job_1min_tick, 'interval', minutes=1, id='job_1min_tick', max_instances=1)
    sched.add_job(job_15min_dune, 'interval', minutes=15, id='job_15min_dune', max_instances=1)
    sched.add_job(job_1hour_deribit, 'interval', hours=1, id='job_1hour_deribit', max_instances=1)
    sched.add_job(job_daily_fear_greed, CronTrigger(hour=0, minute=15), id='job_daily_fear_greed', max_instances=1)
    sched.add_job(job_1hour_crypto_news, 'interval', hours=1, id='job_1hour_crypto_news', max_instances=1)
    sched.add_job(job_daily_cleanup, CronTrigger(hour=1, minute=0), id='job_daily_cleanup', max_instances=1)
    sched.start()
    logger.info("Collector Scheduler started.")

    import threading
    def _run_telegram_listener():
        try:
            from collectors.telegram_listener import telegram_listener
            import asyncio
            asyncio.run(telegram_listener.start())
        except Exception as e:
            logger.error(f"Telegram listener crashed: {e}")

    listener_thread = threading.Thread(target=_run_telegram_listener, name="telegram-listener", daemon=True)
    listener_thread.start()
    logger.info("Real-time Telegram Listener started.")

    try:
        while True:
            time.sleep(60)
            if not listener_thread.is_alive():
                logger.warning("Telegram listener thread died — restarting in 30s...")
                time.sleep(30)
                listener_thread = threading.Thread(target=_run_telegram_listener, name="telegram-listener", daemon=True)
                listener_thread.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down Collector...")
        try:
            from collectors.telegram_listener import upload_session_to_cloud
            upload_session_to_cloud()
        except Exception: pass
        try:
            from collectors.websocket_collector import websocket_collector
            websocket_collector.stop()
        except Exception: pass
        sched.shutdown()
        sys.exit(0)

if __name__ == "__main__":
    main()
