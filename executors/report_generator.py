from typing import Dict, Optional
import os
from datetime import datetime, timezone
from config.database import db
from config.settings import settings, TradingMode
from telegram import Bot
from io import BytesIO
import asyncio
import threading
from loguru import logger
import json
import requests
from utils.retry import api_retry

class ReportGenerator:
    def __init__(self):
        self.bot_token = settings.TELEGRAM_BOT_TOKEN
        self.chat_id = settings.TELEGRAM_CHAT_ID

    def generate_report(
        self,
        symbol: str,
        market_data: Dict,
        bull_opinion: str,
        bear_opinion: str,
        risk_assessment: str,
        final_decision: Dict,
        funding_data: Dict = None,
        mode: TradingMode = TradingMode.SWING,
    ) -> Dict:
        report = {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "market_data": json.dumps(market_data, default=str),
            "bull_opinion": bull_opinion,
            "bear_opinion": bear_opinion,
            "risk_assessment": risk_assessment,
            "final_decision": json.dumps(final_decision),
            "created_at": datetime.now(timezone.utc).isoformat()
        }

        try:
            report_id = db.insert_ai_report(report)
            logger.info(f"Report generated for {symbol} ({mode.value}) with ID: {report_id}")
            report['report_id'] = report_id
            return report
        except Exception as e:
            logger.error(f"Report generation error: {e}")
            return {}

    def format_telegram_message(self, report: Dict, mode: TradingMode = TradingMode.SWING) -> str:
        decision = json.loads(report['final_decision']) if isinstance(report['final_decision'], str) else report['final_decision']
        market_data = json.loads(report['market_data']) if isinstance(report['market_data'], str) else report['market_data']

        d = decision.get('decision', 'N/A')
        icon = {'LONG': '\U0001F7E2', 'SHORT': '\U0001F534', 'HOLD': '\U0001F7E1'}.get(d, '\u2B55')
        mode_icon_map = {
            TradingMode.DAY_TRADING: '\u26A1',
            TradingMode.SWING: '\U0001F4C8',
            TradingMode.POSITION: '\U0001F3D4\uFE0F',
        }
        mode_label_map = {
            TradingMode.DAY_TRADING: 'DAY_TRADING',
            TradingMode.SWING: 'SWING',
            TradingMode.POSITION: 'POSITION',
        }
        mode_icon = mode_icon_map.get(mode, '\u2B55')
        mode_label = mode_label_map.get(mode, mode.value.upper())

        key_factors = decision.get('key_factors', [])
        factors_text = '\n'.join([f'  \u2022 {f}' for f in key_factors[:5]]) if key_factors else '  N/A'

        # Get relevant timeframe data based on mode
        if mode == TradingMode.DAY_TRADING:
            tf_primary = market_data.get('timeframes', {}).get('5m', {})
            tf_secondary = market_data.get('timeframes', {}).get('15m', {})
            tf_labels = ('5M', '15M')
        elif mode == TradingMode.POSITION:
            tf_primary = market_data.get('timeframes', {}).get('1d', {})
            tf_secondary = market_data.get('timeframes', {}).get('1w', {})
            tf_labels = ('1D', '1W')
        else:
            tf_primary = market_data.get('timeframes', {}).get('4h', {})
            tf_secondary = market_data.get('timeframes', {}).get('1d', {})
            tf_labels = ('4H', '1D')

        # Fibonacci info (swing/position)
        fib_text = ""
        fib = market_data.get('fibonacci', {}).get('4h')
        if fib and mode in (TradingMode.SWING, TradingMode.POSITION):
            fib_text = f"\n<b>Fibonacci ({fib.get('trend', '?')}):</b>\n  38.2%: {fib.get('fib_382', '?')} | 50%: {fib.get('fib_500', '?')} | 61.8%: {fib.get('fib_618', '?')}\n  Nearest: {fib.get('nearest_fib', '?')}"

        hold_duration = decision.get('hold_duration', 'N/A')

        # CRO Risk Note
        veto_flag = "🚨 <b>CRO VETO APPLIED!</b>\n" if decision.get('cro_veto_applied') else ""
        cro_note = f"\n{veto_flag}<b>CRO Risk Note:</b>\n{decision.get('risk_manager_note', 'No overlay applied.')}\n" if 'risk_manager_note' in decision else ""

        # Execution Receipt
        receipt_text = ""
        receipt = decision.get("execution_receipt")
        if receipt and receipt.get("success"):
            strat = receipt.get('strategy_applied', 'UNKNOWN')
            notional = receipt.get('total_notional', 0)
            
            # Formulate receipts block
            orders = "\n".join([
                f"    - [{r.get('exchange', '?')}] {r.get('side')} ${r.get('notional', 0):.0f} | ID: {r.get('order_id', '?')[:8]} | {r.get('note', '')}" 
                for r in receipt.get('receipts', [])
            ])
            
            # Formulate Active Pending Orders block from the State
            active_list = report.get('active_orders', [])
            active_str = ""
            if active_list:
                active_str = "\n<b>⏳ ACTIVE INTENTS (V5):</b>\n" + "\n".join([
                    f"    - {o.get('execution_style')} | {o.get('side', o.get('direction', ''))} | Rem: ${o.get('remaining_amount', 0):.0f}"
                    for o in active_list
                ]) + "\n"
                
            mode_prefix = "[PAPER]" if settings.PAPER_TRADING_MODE else "[LIVE]"
            receipt_text = f"<b>{mode_prefix} EXECUTION DESK ({strat})</b>\n  Total Notional: ${notional:.0f}\n{orders}\n{active_str}"
        elif recipe := decision.get("execution_receipt"):
            receipt_text = f"<b>🚨 EXECUTION FAILED:</b> {recipe.get('error', recipe.get('note', 'Unknown'))}\n"

        message = f"""{icon} <b>{mode_label} REPORT</b> {mode_icon}

<b>Symbol:</b> {report['symbol']}
<b>Time:</b> {report['timestamp'][:19]} UTC

<b>DECISION: {d}</b>
<b>Allocation:</b> {decision.get('allocation_pct', 0)}%
<b>Leverage:</b> {decision.get('leverage', 1)}x
<b>Confidence:</b> {decision.get('confidence', 0)}%
<b>Hold:</b> {hold_duration}

<b>Entry:</b> {decision.get('entry_price', 'N/A')}
<b>Stop Loss:</b> {decision.get('stop_loss', 'N/A')}
<b>Take Profit:</b> {decision.get('take_profit', 'N/A')}

{receipt_text}
<b>Key Factors (PM):</b>
{factors_text}
{cro_note}
<b>{tf_labels[0]}:</b> RSI={tf_primary.get('rsi', '?')} MACD={tf_primary.get('macd_histogram', '?')} ADX={tf_primary.get('adx', '?')}
<b>{tf_labels[1]}:</b> RSI={tf_secondary.get('rsi', '?')} MACD={tf_secondary.get('macd_histogram', '?')} ADX={tf_secondary.get('adx', '?')}{fib_text}

<b>PM Reasoning:</b>
{decision.get('reasoning', 'N/A')[:700]}"""

        return message

    @api_retry(max_attempts=3, delay_seconds=10)
    async def send_telegram_notification(self, report: Dict, chart_bytes: Optional[bytes] = None,
                                          mode: TradingMode = TradingMode.SWING) -> None:
        try:
            bot = Bot(token=self.bot_token)
            message = self.format_telegram_message(report, mode)

            if chart_bytes:
                photo = BytesIO(chart_bytes)
                photo.name = f"{report['symbol']}_chart.png"
                caption = message[:1024]
                await bot.send_photo(
                    chat_id=self.chat_id,
                    photo=photo,
                    caption=caption,
                    parse_mode='HTML'
                )
                if len(message) > 1024:
                    await bot.send_message(
                        chat_id=self.chat_id,
                        text=message,
                        parse_mode='HTML'
                    )
            else:
                await bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode='HTML'
                )

            logger.info("Telegram notification sent")
        except Exception as e:
            logger.error(f"Telegram notification error: {e}")

    def notify(self, report: Dict, chart_bytes: Optional[bytes] = None,
               mode: TradingMode = TradingMode.SWING) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.send_telegram_notification(report, chart_bytes, mode))
            return

        # If called from an async context (e.g. Telegram command handler),
        # run notification in a separate thread with its own event loop.
        threading.Thread(
            target=lambda: asyncio.run(self.send_telegram_notification(report, chart_bytes, mode)),
            daemon=True,
        ).start()


report_generator = ReportGenerator()
