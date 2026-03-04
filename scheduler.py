from apscheduler.triggers.cron import CronTrigger
from collectors.price_collector import collector
from collectors.funding_collector import funding_collector
from collectors.volatility_monitor import volatility_monitor
from collectors.dune_collector import dune_collector
from collectors.microstructure_collector import microstructure_collector
from collectors.macro_collector import macro_collector
from collectors.deribit_collector import deribit_collector
from collectors.fear_greed_collector import fear_greed_collector
from collectors.crypto_news_collector import collector as news_collector
from executors.orchestrator import orchestrator
from evaluators.feedback_generator import feedback_generator
from processors.light_rag import light_rag
from processors.gcs_archive import gcs_archive_exporter
from agents.market_monitor_agent import market_monitor_agent
from config import scheduler_config
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
            # [FIX CRITICAL-2] Actually call check_tp_sl ??was implemented but never invoked
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


def job_analysis():
    """Mode-aware analysis job. Interval depends on TRADING_MODE."""
    try:
        mode = settings.trading_mode
        logger.info(f"Running analysis job (mode={mode.value}, "
                     f"interval={settings.analysis_interval_hours}h)")
        from config.local_state import state_manager
        if not state_manager.is_analysis_enabled():
            logger.info("Analysis job skipped (disabled by user)")
            return
            
        macro_collector.run()
        orchestrator.run_scheduled_analysis()
    except Exception as e:
        logger.error(f"Analysis job error: {e}")

def job_routine_market_status():
    """V13.3: Routine Market Status check (Free-First) with Multi-Coin & Telegram Intel."""
    try:
        logger.info("Running routine market status check (Free-First)")

        indicators = {}
        for symbol in settings.trading_symbols:
            indicators[symbol] = {}
            # Price
            try:
                df = db.get_latest_market_data(symbol, limit=1)
                if not df.empty and "close" in df.columns:
                    indicators[symbol]["price"] = float(df["close"].iloc[-1])
            except Exception:
                pass

            # Funding Rate
            try:
                f_df = db.get_funding_history(symbol, limit=1)
                if not f_df.empty and "funding_rate" in f_df.columns:
                    indicators[symbol]["funding_rate"] = float(f_df["funding_rate"].iloc[-1])
            except Exception:
                pass

            # Volatility
            indicators[symbol]["volatility"] = volatility_monitor.calculate_price_change(symbol)
                
        # Telegram Intel (Last 1 hour)
        telegram_intel = "No recent significant news."
        try:
            from mcp_server.tools import mcp_tools
            news_res = mcp_tools.get_telegram_summary(hours=1)
            if "summary" in news_res and news_res["summary"]:
                telegram_intel = news_res["summary"]
        except Exception as e:
            logger.warning(f"Failed to fetch Telegram intel for routine update: {e}")
            
        indicators["TELEGRAM_INTEL"] = telegram_intel

        summary = market_monitor_agent.summarize_current_status(indicators)
        logger.success(f"Market Summary Generated:\n{summary}")

        # Optionally send to Telegram
        from bot.telegram_bot import trading_bot
        if trading_bot:
            import asyncio
            import base64
            from mcp_server.tools import mcp_tools
            
            # 1. Send Text Summary
            asyncio.run(trading_bot.send_message(settings.TELEGRAM_CHAT_ID, f"?뱤 *Routine Market Update*\n\n{summary}"))
            
            # 2. Send fixed SWING lane charts for all symbols
            for symbol in settings.trading_symbols:
                try:
                    chart_res = mcp_tools.get_chart_image(symbol, lane="swing")
                    if "chart_base64" in chart_res:
                        photo_bytes = base64.b64decode(chart_res["chart_base64"])
                        asyncio.run(trading_bot.send_photo(
                            settings.TELEGRAM_CHAT_ID, 
                            photo_bytes, 
                            caption=f"?뱢 <b>{symbol} SWING Trend</b>"
                        ))
                except Exception as chart_err:
                    logger.warning(f"Failed to send routine chart for {symbol}: {chart_err}")

    except Exception as e:
        logger.error(f"Routine market status job error: {e}")


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
    """Batch stored Telegram messages into LightRAG every 1 hour (Real-time listener handles collection)."""
    try:
        logger.info("Running 1-hour Telegram batching job")

        # 1. Synthesize and ingest to LightRAG
        from processors.telegram_batcher import telegram_batcher
        telegram_batcher.process_and_ingest(lookback_hours=1)

        # 3. Truth Engine: Triangulate corroborated claims via Perplexity (Web)
        from config.local_state import state_manager
        if state_manager.is_analysis_enabled():
            light_rag.run_triangulation_worker(limit=3)
        else:
            logger.info("Triangulation worker skipped (AI analysis disabled)")
    except Exception as e:
        logger.error(f"1-hour Telegram job error: {e}")

def job_1hour_crypto_news():
    """Fetch Free Crypto News API and ingest to LightRAG every 1 hour."""
    try:
        logger.info("Running 1-hour Crypto News API fetch job")
        news_collector.fetch_and_ingest()
    except Exception as e:
        logger.error(f"1-hour Crypto News API job error: {e}")


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





def job_daily_precision():
    """00:00 UTC serial: BTC POSITION ??ETH POSITION ??BTC SWING ??ETH SWING.
    Runs full analysis and refreshes Daily Playbooks.
    Schedule:
      00:00 BTC POSITION
      00:03 ETH POSITION
      00:06 BTC SWING
      00:09 ETH SWING
    """
    try:
        from config.local_state import state_manager
        if not state_manager.is_analysis_enabled():
            logger.info("Daily precision skipped (analysis disabled)")
            return
        macro_collector.run()
        orchestrator.run_daily_playbook()
    except Exception as e:
        logger.error(f"Daily precision job error: {e}")


def job_hourly_monitor():
    """Hourly: evaluate all symbol/mode pairs against Daily Playbook.
    Outputs NO_ACTION / WATCH / TRIGGER.
    TRIGGER ??run analysis + allow order execution (capped at 2/day/symbol).
    """
    try:
        from config.local_state import state_manager
        if not state_manager.is_analysis_enabled():
            logger.info("Hourly monitor skipped (analysis disabled)")
            return
        orchestrator.run_hourly_monitor()
    except Exception as e:
        logger.error(f"Hourly monitor job error: {e}")


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
    logger.info(f"  AI: Gemini Judge/VLM (Project A/B) + Cerebras (meta/risk) + Groq (news/rag) + OpenRouter (monitor)")
    logger.info(f"  Data: Global OI + OI Divergence + MFI Proxy + Liquidations + Perplexity + LightRAG")
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

    # [FIX] WebSocket thread health check ??every 5 minutes
    def job_5m_ws_health_check():
        try:
            from collectors.websocket_collector import websocket_collector
            if hasattr(websocket_collector, '_thread'):
                if not websocket_collector._thread.is_alive():
                    logger.warning("WebSocket thread died ??restarting...")
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

    # [FIX Cold Start] Run Telegram bot in a daemon thread
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

    # [V13.1] Persistent Telegram Listener (Real-Time Alpha Ingestion)
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

    # 1-minute tick: price, funding, microstructure, volatility
    scheduler_config.scheduler.add_job(
        job_1min_tick,
        'interval',
        minutes=1,
        id='job_1min_tick',
        max_instances=1
    )
    
    # 1-minute execution tick: ExecutionDesk and Paper Engine Liquidations
    scheduler_config.scheduler.add_job(
        job_1min_execution,
        'interval',
        minutes=1,
        id='job_1min_execution',
        max_instances=1
    )

    # 15-minute Dune
    scheduler_config.scheduler.add_job(
        job_15min_dune,
        'interval',
        minutes=15,
        id='job_15min_dune',
        max_instances=1
    )

    # Deribit options data: DVOL, PCR, IV Term Structure, 25d Skew ??every 1 hour
    scheduler_config.scheduler.add_job(
        job_1hour_deribit,
        'interval',
        hours=1,
        id='job_1hour_deribit',
        max_instances=1
    )

    # Funding Fee Simulation - every 8 hours (00:00, 08:00, 16:00 UTC)
    if settings.PAPER_TRADING_MODE:
        scheduler_config.scheduler.add_job(
            job_8hour_funding_fee,
            CronTrigger(hour='0,8,16', minute=0),
            id='job_8hour_funding_fee',
            max_instances=1
        )

    # Fear & Greed Index ??daily at 00:15 UTC (after data refreshes)
    scheduler_config.scheduler.add_job(
        job_daily_fear_greed,
        CronTrigger(hour=0, minute=15),
        id='job_daily_fear_greed',
        max_instances=1
    )

    # 1-Hour Telegram Batching & Ingestion
    scheduler_config.scheduler.add_job(
        job_1hour_telegram,
        'interval',
        hours=1,
        id='job_1hour_telegram',
        max_instances=1
    )

    # 1-Hour Crypto News API Fetch & Ingestion
    scheduler_config.scheduler.add_job(
        job_1hour_crypto_news,
        'interval',
        hours=1,
        id='job_1hour_crypto_news',
        max_instances=1
    )

    # ?? Daily Precision (00:00 UTC) ??BTC/ETH 횞 SWING/POSITION Playbook generation ??
    scheduler_config.scheduler.add_job(
        job_daily_precision,
        CronTrigger(hour=0, minute=0),
        id='job_daily_precision',
        max_instances=1,
    )

    # ?? Hourly Monitor ??NO_ACTION / WATCH / TRIGGER against Daily Playbook ??
    scheduler_config.scheduler.add_job(
        job_hourly_monitor,
        'interval',
        hours=1,
        id='job_hourly_monitor',
        max_instances=1,
    )

    # Legacy mode-aware analysis (kept for emergency/manual trigger via bot)
    scheduler_config.scheduler.add_job(
        job_analysis,
        scheduler_config._build_analysis_trigger(mode),
        id='job_analysis',
        max_instances=1,
    )

    # Routine Market Status (Free-First) ??kept for passive hourly Telegram update
    scheduler_config.scheduler.add_job(
        job_routine_market_status,
        'interval',
        hours=1,
        id='job_market_status',
        max_instances=1,
    )

    # Daily evaluation at 00:30 UTC = 09:30 KST
    scheduler_config.scheduler.add_job(
        job_24hour_evaluation,
        CronTrigger(hour=0, minute=30),
        id='job_24hour_evaluation',
        max_instances=1
    )
    
    # 1-Hour RAG Episodic Memory Evaluation (V6)
    scheduler_config.scheduler.add_job(
        job_1hour_evaluation,
        'interval',
        hours=1,
        id='job_1hour_evaluation',
        max_instances=1
    )

    # Daily cleanup at 01:00 UTC = 10:00 KST
    scheduler_config.scheduler.add_job(
        job_daily_cleanup,
        CronTrigger(hour=1, minute=0),
        id='job_daily_cleanup',
        max_instances=1
    )



    scheduler_config.scheduler.start()
    logger.info("Scheduler started.")

    # [FIX Cold Start] Run initial data collection immediately so first analysis has data
    logger.info("Running initial data collection (cold start bootstrap)...")
    def _startup_telegram_catchup():
        """Handles initial RAG synthesis for the last 24h.
        Raw collection is handled by the telegram_listener thread's backfill task.
        """
        import time
        # Small delay to let the listener start fetching backfill messages before we batch them
        time.sleep(10)
        
        from processors.telegram_batcher import telegram_batcher
        # Ingest recent backlog so restart does not leave an analysis blind window
        telegram_batcher.process_and_ingest(lookback_hours=24)

    _initial_collectors = [
        ("Price + Funding + Microstructure", lambda: (collector.run(), funding_collector.run(), microstructure_collector.run())),
        ("Volatility", lambda: volatility_monitor.run()),
        ("Deribit", lambda: deribit_collector.run()),
        ("Fear & Greed", lambda: fear_greed_collector.run()),
        ("Telegram catch-up (24h)", _startup_telegram_catchup),
    ]
    for name, fn in _initial_collectors:
        try:
            fn()
            logger.info(f"  ??{name} collected")
        except Exception as e:
            logger.warning(f"  ?좑툘 {name} collection failed (non-fatal): {e}")

    # Main thread: keep alive + graceful shutdown
    try:
        import time
        while True:
            time.sleep(60)
            # Optional: restart bot thread if it dies
            if not bot_thread.is_alive():
                logger.warning("Telegram bot thread died ??restarting in 30s...")
                time.sleep(30)
                bot_thread = threading.Thread(target=_run_telegram_bot, name="telegram-bot", daemon=True)
                bot_thread.start()

            if not listener_thread.is_alive():
                logger.warning("Telegram listener thread died ??restarting in 30s...")
                time.sleep(30)
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
