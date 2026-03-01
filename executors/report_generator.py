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

    def format_summary_message(self, report: Dict, mode: TradingMode = TradingMode.SWING) -> str:
        """Concise summary message with key targets and final logic."""
        decision = json.loads(report['final_decision']) if isinstance(report['final_decision'], str) else report['final_decision']
        
        d = decision.get('decision', 'N/A')
        confidence = decision.get('confidence', 0)
        
        header_icon = "🟢" if d == "LONG" else "🔴" if d == "SHORT" else "🟡"
        mode_label = "SWING" if mode == TradingMode.SWING else "POSITION"
        
        # Format prices
        def fmt_price(val):
            if val is None or val == 0 or val == '0': return '---'
            try: return f"{float(val):,.2f}"
            except: return str(val)

        summary_lines = [
            f"{header_icon} <b>{mode_label} REPORT | {report['symbol']}</b>",
            f"<code>{report['timestamp'][:16].replace('T', ' ')} UTC</code>",
            "",
            f"🎯 <b>DECISION: {d}</b> ({confidence}%)",
            f"🏁 Entry: <code>{fmt_price(decision.get('entry_price'))}</code> | SL: <code>{fmt_price(decision.get('stop_loss'))}</code>",
            f"🎯 TP: <code>{fmt_price(decision.get('take_profit'))}</code>",
            "",
        ]

        # Execution Status
        receipt = decision.get("execution_receipt")
        if receipt and receipt.get("success"):
            summary_lines.append(f"✅ <b>EXECUTION ACTIVE</b> ({len(receipt.get('receipts', []))} orders)")
        elif receipt:
            summary_lines.append(f"⏸️ <b>STATUS:</b> {receipt.get('note', 'PENDING')}")

        # Final Logic (Summary)
        reasoning = decision.get('reasoning', {})
        final_logic = reasoning.get('final_logic', 'No summary available.') if isinstance(reasoning, dict) else str(reasoning)[:200]
        
        summary_lines.append(f"\n💡 <b>SUMMARY:</b> <i>{final_logic}</i>")
        
        return "\n".join(summary_lines)

    def format_detail_message(self, report: Dict, mode: TradingMode = TradingMode.SWING) -> str:
        """Full reasoning detail from the report."""
        decision = json.loads(report['final_decision']) if isinstance(report['final_decision'], str) else report['final_decision']
        reasoning = decision.get('reasoning', 'N/A')
        
        lines = [f"🔍 <b>DEEP ANALYSIS | {report['symbol']}</b>\n"]
        
        if isinstance(reasoning, dict):
            mapping = {
                "technical": "📏 TECHNICALS",
                "derivatives": "⛓️ DERIVATIVES",
                "experts": "🧠 EXPERT SWARM",
                "narrative": "🌐 NARRATIVE/RAG",
                "counter_scenario": "⚠️ RISK/FALSIFIABILITY"
            }
            for key, title in mapping.items():
                val = reasoning.get(key)
                if val:
                    safe_val = str(val).replace('<', '&lt;').replace('>', '&gt;')
                    if len(safe_val) > 800: safe_val = safe_val[:797] + "..."
                    lines.append(f"<b>{title}:</b>\n<i>{safe_val}</i>\n")
        else:
            lines.append(f"<i>{str(reasoning).replace('<', '&lt;').replace('>', '&gt;')}</i>")

        final_msg = "\n".join(lines)
        return final_msg[:4000]

    @api_retry(max_attempts=3, delay_seconds=10)
    async def send_telegram_notification(self, report: Dict, chart_bytes: Optional[bytes] = None,
                                          mode: TradingMode = TradingMode.SWING) -> None:
        try:
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup
            bot = Bot(token=self.bot_token)
            
            # 1. Summary message
            summary_text = self.format_summary_message(report, mode)
            
            # 2. Add Button
            report_id = report.get('report_id') or report.get('id')
            keyboard = [[InlineKeyboardButton("🔍 View Deep Analysis", callback_data=f"detail_{report_id}")]]
            reply_markup = InlineKeyboardMarkup(keyboard)

            if chart_bytes:
                photo = BytesIO(chart_bytes)
                photo.name = f"{report['symbol']}_chart.png"
                
                # Send photo with summary as caption (if short) or separately
                if len(summary_text) <= 1024:
                    await bot.send_photo(
                        chat_id=self.chat_id, 
                        photo=photo, 
                        caption=summary_text, 
                        parse_mode='HTML',
                        reply_markup=reply_markup
                    )
                else:
                    await bot.send_photo(chat_id=self.chat_id, photo=photo, caption=f"📈 {report['symbol']} Analysis", parse_mode='HTML')
                    await bot.send_message(chat_id=self.chat_id, text=summary_text, parse_mode='HTML', reply_markup=reply_markup)
            else:
                await bot.send_message(chat_id=self.chat_id, text=summary_text, parse_mode='HTML', reply_markup=reply_markup)

            logger.info(f"Telegram summary notification sent (ID: {report_id})")
        except Exception as e:
            logger.error(f"Telegram notification error: {e}")

    def notify(self, report: Dict, chart_bytes: Optional[bytes] = None,
               mode: TradingMode = TradingMode.SWING) -> None:
        # Always run in a background thread to avoid blocking the caller
        # (APScheduler threads have no running asyncio loop, causing synchronous
        # execution and up to 6-minute blocking on Telegram API timeouts)
        threading.Thread(
            target=lambda: asyncio.run(self.send_telegram_notification(report, chart_bytes, mode)),
            daemon=True,
        ).start()

report_generator = ReportGenerator()
