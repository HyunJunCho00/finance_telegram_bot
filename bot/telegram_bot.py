"""Telegram Bot interactive command handler.

Commands:
/start   - Welcome message
/status  - Current position and latest decision
/analyze - Run immediate analysis (BTC or ETH)
/mode    - Show or switch trading mode
/report  - Resend latest report
/help    - List commands
"""

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from config.settings import settings, TradingMode
from config.database import db
from loguru import logger
import json
import os


class TradingBot:
    def __init__(self):
        self.bot_token = settings.TELEGRAM_BOT_TOKEN
        self.chat_id = settings.TELEGRAM_CHAT_ID

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        mode = settings.trading_mode.value.upper()
        await update.message.reply_text(
            f"Crypto Trading Bot Active\n"
            f"Mode: {mode}\n"
            f"Symbols: BTC, ETH\n\n"
            f"Commands:\n"
            f"/status - Current position\n"
            f"/analyze BTC - Immediate analysis\n"
            f"/mode swing|position - Switch mode\n"
            f"/report - Latest report\n"
            f"/help - Show commands"
        )

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "/status - Current positions & latest decision\n"
            "/analyze BTC|ETH - Run immediate analysis\n"
            "/mode - Show current mode\n"
            "/mode swing - Switch to swing (multi-day)\n"
            "/mode position - Switch to position (long-term)\n"
            "/report - Resend latest report\n"
            "/help - Show this message"
        )

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            mode = settings.trading_mode.value.upper()
            is_paper = settings.PAPER_TRADING_MODE

            lines = [
                f"🛡️ <b>Trading Status</b>",
                f"Mode: <code>{mode}</code>",
                f"Type: <code>{'PAPER' if is_paper else 'LIVE'}</code>",
                ""
            ]

            # 1. Active Mock Positions (If applicable)
            if is_paper:
                from executors.paper_exchange import paper_engine
                positions = paper_engine.get_open_positions()
                
                lines.append("📌 <b>Active Positions</b>")
                if not positions:
                    lines.append("<i>- No open positions</i>")
                else:
                    for pos in positions:
                        # Calculate unrealized PnL briefly if possible (needs current price)
                        # For now, show entry and target
                        lines.append(
                            f"• <b>{pos['exchange'].upper()}</b> | {pos['symbol']}: {pos['side']} "
                            f"(Entry: {pos['entry_price']:.2f}, Lev: {pos['leverage']}x)"
                        )
                
                # Wallet Balances
                lines.append("\n💰 <b>Mock Balances</b>")
                for ex in ['binance', 'upbit']:
                    bal = paper_engine.get_wallet_balance(ex)
                    unit = 'USD' if ex == 'binance' else 'KRW'
                    lines.append(f"• {ex.upper()}: {bal:,.2f} {unit}")
                lines.append("")

            # 2. Strategy Analysis (Latest Reports)
            lines.append("🤖 <b>Latest Strategy</b>")
            for symbol in settings.trading_symbols:
                report = db.get_latest_report(symbol=symbol)
                if report:
                    fd = report.get('final_decision')
                    if isinstance(fd, str):
                        fd = json.loads(fd)

                    decision = fd.get('decision', 'N/A') if fd else 'N/A'
                    confidence = fd.get('confidence', 0) if fd else 0
                    ts = report.get('timestamp', 'N/A')[:16].replace('T', ' ')

                    lines.append(
                        f"• {symbol}: <b>{decision}</b> ({confidence}%) "
                        f"<pre>{ts}</pre>"
                    )
                else:
                    lines.append(f"• {symbol}: No report yet")

            await update.message.reply_text('\n'.join(lines), parse_mode='HTML')
        except Exception as e:
            logger.error(f"Status command error: {e}")
            await update.message.reply_text(f"Error: {e}")

    async def cmd_analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        args = context.args
        valid_bases = settings.trading_symbols_base
        if not args or args[0].upper() not in valid_bases:
            await update.message.reply_text(f"Usage: /analyze {'|'.join(valid_bases)}")
            return

        symbol = args[0].upper() + 'USDT'
        await update.message.reply_text(f"Running {settings.trading_mode.value.upper()} analysis for {symbol}...")

        try:
            from executors.orchestrator import orchestrator
            result = orchestrator.run_analysis(symbol, is_emergency=False)
            if result:
                decision = result.get('decision', 'N/A')
                confidence = result.get('confidence', 0)
                await update.message.reply_text(
                    f"Analysis complete: {decision} ({confidence}%)\n"
                    f"Full report sent to chat."
                )
            else:
                await update.message.reply_text("Analysis failed. Check logs.")
        except Exception as e:
            logger.error(f"Analyze command error: {e}")
            await update.message.reply_text(f"Error: {e}")

    async def cmd_mode(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        args = context.args

        if not args:
            mode = settings.trading_mode.value
            await update.message.reply_text(
                f"Current mode: {mode.upper()}\n"
                f"Candle limit: {settings.candle_limit}\n"
                f"Chart for AI: {settings.should_use_chart}\n\n"
                f"Switch: /mode swing | position"
            )
            return

        new_mode = args[0].lower().strip()
        if new_mode not in ('swing', 'position'):
            await update.message.reply_text("Invalid. Use: /mode swing | position")
            return

        os.environ['TRADING_MODE'] = new_mode
        from config.settings import get_settings
        get_settings.cache_clear()

        await update.message.reply_text(
            f"Mode switched to {new_mode.upper()}\n"
            f"Next analysis will use {new_mode} timeframes."
        )

    async def cmd_report(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            for symbol in settings.trading_symbols:
                report = db.get_latest_report(symbol=symbol)
                if report:
                    fd = report.get('final_decision')
                    if isinstance(fd, str):
                        fd = json.loads(fd)

                    text = (
                        f"Latest {symbol} Report\n"
                        f"Time: {report.get('timestamp', 'N/A')[:19]}\n"
                        f"Decision: {fd.get('decision', 'N/A')}\n"
                        f"Confidence: {fd.get('confidence', 0)}%\n"
                        f"Entry: {fd.get('entry_price', 'N/A')}\n"
                        f"SL: {fd.get('stop_loss', 'N/A')}\n"
                        f"TP: {fd.get('take_profit', 'N/A')}\n"
                        f"Hold: {fd.get('hold_duration', 'N/A')}\n\n"
                        f"Reasoning: {fd.get('reasoning', 'N/A')[:500]}"
                    )
                    await update.message.reply_text(text)
                else:
                    await update.message.reply_text(f"No report for {symbol}")
        except Exception as e:
            logger.error(f"Report command error: {e}")
            await update.message.reply_text(f"Error: {e}")

    def run(self):
        """Start the bot (non-blocking, runs in background thread)."""
        app = Application.builder().token(self.bot_token).build()

        app.add_handler(CommandHandler("start", self.cmd_start))
        app.add_handler(CommandHandler("help", self.cmd_help))
        app.add_handler(CommandHandler("status", self.cmd_status))
        app.add_handler(CommandHandler("analyze", self.cmd_analyze))
        app.add_handler(CommandHandler("mode", self.cmd_mode))
        app.add_handler(CommandHandler("report", self.cmd_report))

        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        logger.info("Telegram bot started (polling)")
        app.run_polling(drop_pending_updates=True, stop_signals=None)


trading_bot = TradingBot()
