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

        except Exception as e:
            logger.error(f"Error formatting advanced indicators: {e}")

    def format_telegram_message(self, report: Dict, mode: TradingMode = TradingMode.SWING) -> str:
        """Premium formatted Telegram message with rich layout and emojis."""
        decision = json.loads(report['final_decision']) if isinstance(report['final_decision'], str) else report['final_decision']
        market_data = json.loads(report['market_data']) if isinstance(report['market_data'], str) else report['market_data']

        d = decision.get('decision', 'N/A')
        confidence = decision.get('confidence', 0)
        
        # Color coding for decision
        if d == 'LONG':
            header_icon = "🟢"
            status_text = "✨ BULLISH SETUP DETECTED"
        elif d == 'SHORT':
            header_icon = "🔴"
            status_text = "📉 BEARISH SETUP DETECTED"
        else:
            header_icon = "🟡"
            status_text = "⚖️ NEUTRAL / MONITORING"

        mode_label = "SWING" if mode == TradingMode.SWING else "POSITION"
        mode_icon = "📈" if mode == TradingMode.SWING else "🏔️"
        
        # Summary block (High level)
        summary_lines = [
            f"{header_icon} <b>{mode_label} REPORT | {report['symbol']}</b> {mode_icon}",
            f"<code>{report['timestamp'][:16].replace('T', ' ')} UTC</code>",
            "",
            f"🎯 <b>DECISION: {d}</b>",
            f"📊 <b>Confidence: {confidence}%</b> | <b>Alloc: {decision.get('allocation_pct', 0)}%</b>",
            f"⏲️ <b>Exp. Hold: {decision.get('hold_duration', 'N/A')}</b>",
            "",
        ]

        # Targets block
        def fmt_price(val):
            if val is None or val == 0 or val == '0': return '---'
            try: return f"{float(val):,.2f}"
            except: return str(val)

        summary_lines += [
            "🏁 <b>TRADING TARGETS</b>",
            f"  • Entry: <code>{fmt_price(decision.get('entry_price'))}</code>",
            f"  • Stop : <code>{fmt_price(decision.get('stop_loss'))}</code>",
            f"  • TP   : <code>{fmt_price(decision.get('take_profit'))}</code>",
            "",
        ]

        # Execution Status
        receipt_text = ""
        receipt = decision.get("execution_receipt")
        if receipt and receipt.get("success"):
            orders_count = len(receipt.get('receipts', []))
            receipt_text = f"✅ <b>EXECUTION ACTIVE</b> ({orders_count} orders)\n"
        elif recipe := decision.get("execution_receipt"):
            if recipe.get("note") == "No valid trade direction":
                receipt_text = "⏸️ <b>NO TRADE TRIGGERED</b> (Directionless)\n"
            else:
                receipt_text = f"🚨 <b>EXECUTION ERROR:</b> {recipe.get('error', 'Check Logs')}\n"
        
        if receipt_text:
            summary_lines.append(receipt_text)

        # Key Factors (Bulleted)
        key_factors = decision.get('key_factors', [])
        if key_factors:
            summary_lines.append("🔍 <b>KEY RATIONALE</b>")
            for f in key_factors[:5]:
                summary_lines.append(f"  ▫️ {f}")
            summary_lines.append("")

        # Post-Mortem / Reasoning (Structured or Flat)
        reasoning = decision.get('reasoning', 'N/A')
        
        summary_lines.append("📝 <b>DECISION RATIONALE</b>")
        
        if isinstance(reasoning, dict):
            # Format structured reasoning
            mapping = {
                "technical": "📏 TA",
                "derivatives": "⛓️ DERIV",
                "experts": "🧠 EXP",
                "narrative": "🌐 NARR",
                "final_logic": "💡 FIN"
            }
            for key, emoji in mapping.items():
                val = reasoning.get(key)
                if val:
                    # Escape HTML for each part
                    safe_val = str(val).replace('<', '&lt;').replace('>', '&gt;')
                    summary_lines.append(f"<b>{emoji}:</b> <i>{safe_val}</i>")
        else:
            # Fallback for flat string reasoning
            safe_reasoning = str(reasoning).replace('<', '&lt;').replace('>', '&gt;')
            if len(safe_reasoning) > 600:
                safe_reasoning = safe_reasoning[:600] + "..."
            summary_lines.append(f"<i>{safe_reasoning}</i>")

        return "\n".join(summary_lines)

    @api_retry(max_attempts=3, delay_seconds=10)
    async def send_telegram_notification(self, report: Dict, chart_bytes: Optional[bytes] = None,
                                          mode: TradingMode = TradingMode.SWING) -> None:
        try:
            bot = Bot(token=self.bot_token)
            message = self.format_telegram_message(report, mode)

            if chart_bytes:
                logger.info(f"Sending Telegram photo for {report['symbol']} ({len(chart_bytes)} bytes)")
                photo = BytesIO(chart_bytes)
                photo.name = f"{report['symbol']}_chart.png"
                
                # If message is short, send as caption. Otherwise send separately.
                if len(message) <= 1024:
                    await bot.send_photo(chat_id=self.chat_id, photo=photo, caption=message, parse_mode='HTML')
                else:
                    # Send photo first with a brief header
                    header = f"<b>📈 {report['symbol']} Chart</b>"
                    await bot.send_photo(chat_id=self.chat_id, photo=photo, caption=header, parse_mode='HTML')
                    # Follow up with full report message
                    await bot.send_message(chat_id=self.chat_id, text=message, parse_mode='HTML')
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
