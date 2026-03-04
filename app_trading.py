"""Trading, Analysis, and Reporting Application."""
import sys
import time
import os
from loguru import logger
from apscheduler.triggers.cron import CronTrigger

from config import scheduler_config
from config.settings import settings

from executors.orchestrator import orchestrator
from evaluators.feedback_generator import feedback_generator
from processors.light_rag import light_rag
from agents.market_monitor_agent import market_monitor_agent
from config.database import db
from executors.order_manager import execution_desk
from executors.evaluator_daemon import EvaluatorDaemon
from executors.paper_exchange import paper_engine
from collectors.macro_collector import macro_collector
from collectors.volatility_monitor import volatility_monitor
from mcp_server.tools import mcp_tools

def job_1min_execution():
    try:
        execution_desk.process_intents()
        if settings.PAPER_TRADING_MODE:
            prices = {}
            from executors.trade_executor import trade_executor
            ex = trade_executor.binance
            for symbol in settings.trading_symbols_slash:
                try:
                    t = ex.fetch_ticker(symbol)
                    canonical = symbol.replace('/', '')
                    prices[canonical] = float(t['last'])
                except Exception:
                    pass
            paper_engine.check_liquidations(prices)
            paper_engine.check_tp_sl(prices)
    except Exception as e:
        logger.error(f"1-minute execution job error: {e}")

def job_8hour_funding_fee():
    try:
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

def job_analysis():
    try:
        mode = settings.trading_mode
        logger.info(f"Running analysis job (mode={mode.value}, interval={settings.analysis_interval_hours}h)")
        from config.local_state import state_manager
        if not state_manager.is_analysis_enabled():
            logger.info("Analysis job skipped (disabled by user)")
            return
        macro_collector.run()
        orchestrator.run_scheduled_analysis()
    except Exception as e:
        logger.error(f"Analysis job error: {e}")

def job_routine_market_status():
    try:
        logger.info("Running routine market status check")
        indicators = {}
        for symbol in settings.trading_symbols:
            indicators[symbol] = {}
            try:
                df = db.get_latest_market_data(symbol, limit=1)
                if not df.empty and "close" in df.columns:
                    indicators[symbol]["price"] = float(df["close"].iloc[-1])
            except Exception: pass
            try:
                f_df = db.get_funding_history(symbol, limit=1)
                if not f_df.empty and "funding_rate" in f_df.columns:
                    indicators[symbol]["funding_rate"] = float(f_df["funding_rate"].iloc[-1])
            except Exception: pass
            indicators[symbol]["volatility"] = volatility_monitor.calculate_price_change(symbol)
            
        telegram_intel = "No recent significant news."
        try:
            news_res = mcp_tools.get_telegram_summary(hours=1)
            if "summary" in news_res and news_res["summary"]:
                telegram_intel = news_res["summary"]
        except Exception as e:
            logger.warning(f"Failed to fetch Telegram intel: {e}")
        indicators["TELEGRAM_INTEL"] = telegram_intel

        summary = market_monitor_agent.summarize_current_status(indicators)
        logger.success(f"Market Summary Generated:\n{summary}")

        from bot.telegram_bot import trading_bot
        # Note: trading_bot here might be uninitialized in app_trading if it hooks differently.
        # It's safer to use a pure HTTP poster or direct asyncio bot.
        # But for now we rely on the bot instance if it gets created elsewhere or we explicitly initialize it.
        # Since Telegram is now decouple, we must instantiate a bot just to SEND messages.
        if trading_bot:
            import asyncio
            import base64
            asyncio.run(trading_bot.send_message(settings.TELEGRAM_CHAT_ID, f"📊 *Routine Market Update*\n\n{summary}"))
            for symbol in settings.trading_symbols:
                try:
                    chart_res = mcp_tools.get_chart_image(symbol, timeframe="4h")
                    if "chart_base64" in chart_res:
                        photo_bytes = base64.b64decode(chart_res["chart_base64"])
                        asyncio.run(trading_bot.send_photo(
                            settings.TELEGRAM_CHAT_ID, photo_bytes, caption=f"📈 <b>{symbol} 4h Trend</b>"
                        ))
                except Exception as chart_err:
                    logger.warning(f"Failed to send chart for {symbol}: {chart_err}")
    except Exception as e:
        logger.error(f"Routine market status error: {e}")

def job_24hour_evaluation():
    try:
        logger.info("Running 24-hour evaluation job")
        feedback_generator.run_feedback_cycle()
    except Exception as e:
        logger.error(f"24-hour evaluation job error: {e}")

def job_1hour_evaluation():
    try:
        logger.info("Running 1-hour episodic memory evaluation")
        daemon = EvaluatorDaemon()
        daemon.evaluate_recent_trades()
    except Exception as e:
        logger.error(f"1-hour evaluation job error: {e}")

def job_1hour_telegram():
    try:
        logger.info("Running Triangulation worker")
        from config.local_state import state_manager
        if state_manager.is_analysis_enabled():
            light_rag.run_triangulation_worker(limit=3)
    except Exception as e:
        logger.error(f"Triangulation worker error: {e}")

def job_check_mode_change():
    """Poll DB to see if trading_mode or analysis_enabled changed externally."""
    try:
        from config.local_state import state_manager
        current_db_mode = state_manager.get_system_config("trading_mode", "swing")
        if current_db_mode != settings.trading_mode.value:
            logger.info(f"External mode change detected: {settings.trading_mode.value} -> {current_db_mode}")
            os.environ["TRADING_MODE"] = current_db_mode
            settings.__setattr__('TRADING_MODE', current_db_mode)
            scheduler_config.reschedule_analysis_job(current_db_mode)
    except Exception as e:
        logger.error(f"Mode check error: {e}")

def main():
    logger.info("Starting Trading & Analysis Application (app_trading.py)")
    mode = settings.trading_mode
    sched = scheduler_config.scheduler

    sched.add_job(job_check_mode_change, 'interval', minutes=1, id='job_check_mode_change', max_instances=1)
    sched.add_job(job_1min_execution, 'interval', minutes=1, id='job_1min_execution', max_instances=1)
    if settings.PAPER_TRADING_MODE:
        sched.add_job(job_8hour_funding_fee, CronTrigger(hour='0,8,16', minute=0), id='job_8hour_funding_fee', max_instances=1)
    
    sched.add_job(job_1hour_telegram, 'interval', hours=1, id='job_1hour_telegram', max_instances=1)
    sched.add_job(job_analysis, scheduler_config._build_analysis_trigger(mode), id='job_analysis', max_instances=1)
    sched.add_job(job_routine_market_status, 'interval', hours=1, id='job_market_status', max_instances=1)
    sched.add_job(job_24hour_evaluation, CronTrigger(hour=0, minute=30), id='job_24hour_evaluation', max_instances=1)
    sched.add_job(job_1hour_evaluation, 'interval', hours=1, id='job_1hour_evaluation', max_instances=1)

    sched.start()
    logger.info("Trading Scheduler started.")

    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down Trading app...")
        sched.shutdown()
        sys.exit(0)

if __name__ == "__main__":
    main()
