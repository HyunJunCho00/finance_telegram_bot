from typing import Dict, Optional
from datetime import datetime, timezone
from config.database import db
from config.settings import settings, TradingMode
from telegram import Bot
from io import BytesIO
import asyncio
import threading
from loguru import logger
import json
from utils.retry import api_retry


class ReportGenerator:
    def __init__(self):
        self.bot_token = settings.TELEGRAM_BOT_TOKEN
        self.chat_id = settings.TELEGRAM_CHAT_ID

    @staticmethod
    def _load_json_field(value, default):
        if value is None:
            return default
        if isinstance(value, (dict, list)):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return default
        return default

    @staticmethod
    def _escape_html(value) -> str:
        return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    @staticmethod
    def _fmt_price(value) -> str:
        if value in (None, 0, "0"):
            return "---"
        try:
            return f"{float(value):,.2f}"
        except Exception:
            return str(value)

    @staticmethod
    def _fmt_num(value, decimals: int = 3, suffix: str = "") -> str:
        try:
            return f"{float(value):,.{decimals}f}{suffix}"
        except Exception:
            return "N/A"

    def _build_onchain_lines(
        self,
        symbol: str,
        snapshot: Optional[Dict] = None,
        onchain_context: str = "",
        include_header: bool = True,
    ) -> list[str]:
        snapshot = snapshot or {}
        raw = snapshot.get("raw_metrics", {}) if isinstance(snapshot, dict) else {}
        derived = snapshot.get("derived_features", {}) if isinstance(snapshot, dict) else {}
        flags = snapshot.get("regime_flags", {}) if isinstance(snapshot, dict) else {}

        lines: list[str] = []
        if include_header:
            lines.append(f"<b>ON-CHAIN SNAPSHOT | {self._escape_html(symbol)}</b>")

        if snapshot:
            freshness = "stale" if snapshot.get("is_stale") else "fresh"
            age_hours = snapshot.get("age_hours")
            freshness_text = freshness
            if age_hours is not None:
                freshness_text = f"{freshness} / {self._fmt_num(age_hours, 1)}h"

            lines.extend([
                f"As of: <code>{self._escape_html(snapshot.get('as_of_date', 'N/A'))}</code> ({freshness_text})",
                (
                    f"Bias: <b>{self._escape_html(snapshot.get('risk_bias', 'NEUTRAL'))}</b> | "
                    f"Score: <code>{self._fmt_num(snapshot.get('bias_score'), 2)}</code> | "
                    f"Quality: <code>{self._escape_html(snapshot.get('data_quality', 'unknown'))}</code>"
                ),
                (
                    f"MVRV: <code>{self._fmt_num(raw.get('mvrv'), 4)}</code> | "
                    f"Valuation: <code>{self._escape_html(flags.get('valuation_state', 'unknown'))}</code>"
                ),
                (
                    f"Price vs Realized: <code>{self._fmt_num(derived.get('price_vs_realized_pct'), 2, '%')}</code> | "
                    f"Exchange Supply 30d: <code>{self._fmt_num(derived.get('exchange_supply_30d_delta_pct'), 2, '%')}</code>"
                ),
                (
                    f"Flow: <code>{self._escape_html(flags.get('flow_state', 'unknown'))}</code> | "
                    f"Tx Ratio 7d/30d: <code>{self._fmt_num(derived.get('tx_count_ratio_7d_30d'), 3)}</code> | "
                    f"Activity: <code>{self._escape_html(flags.get('activity_state', 'unknown'))}</code>"
                ),
                (
                    f"Gate: long=<code>{self._escape_html(flags.get('allow_long', 'N/A'))}</code> "
                    f"short=<code>{self._escape_html(flags.get('allow_short', 'N/A'))}</code> "
                    f"chase_long_blocked=<code>{self._escape_html(flags.get('chase_long_blocked', 'N/A'))}</code>"
                ),
                (
                    f"Size Mult: long=<code>{self._fmt_num(flags.get('long_size_multiplier'), 2)}</code>x "
                    f"short=<code>{self._fmt_num(flags.get('short_size_multiplier'), 2)}</code>x"
                ),
            ])

            if snapshot.get("asset") == "eth":
                lines.append(
                    f"ETH Struct: <code>{self._fmt_num(derived.get('tx_eip1559_ratio_7d_30d'), 3)}</code> | "
                    f"State: <code>{self._escape_html(flags.get('eth_state', 'unknown'))}</code>"
                )
        else:
            lines.append("<i>No stored on-chain snapshot.</i>")

        if onchain_context:
            lines.extend([
                "",
                "<b>LLM INPUT (ON-CHAIN CONTEXT)</b>",
                f"<code>{self._escape_html(onchain_context)}</code>",
            ])

        return lines

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
        onchain_context: str = "",
        onchain_snapshot: Optional[Dict] = None,
    ) -> Dict:
        report = {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "market_data": json.dumps(market_data, default=str),
            "bull_opinion": bull_opinion,
            "bear_opinion": bear_opinion,
            "risk_assessment": risk_assessment,
            "final_decision": json.dumps(final_decision),
            "onchain_context": onchain_context,
            "onchain_snapshot": onchain_snapshot or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            report_id = db.insert_ai_report(report)
            logger.info(f"Report generated for {symbol} ({mode.value}) with ID: {report_id}")
            report["report_id"] = report_id
            return report
        except Exception as e:
            logger.error(f"Report generation error: {e}")
            return {}

    def format_summary_message(self, report: Dict, mode: TradingMode = TradingMode.SWING) -> str:
        decision = self._load_json_field(report.get("final_decision"), {})

        direction = decision.get("decision", "N/A")
        confidence = decision.get("confidence", 0)

        if direction == "LONG":
            header_icon, color_theme = "🚀", "🟢 강세 (LONG)"
        elif direction == "SHORT":
            header_icon, color_theme = "🔻", "🔴 약세 (SHORT)"
        else:
            header_icon, color_theme = "⚖️", "⚪ 관망 (HOLD)"

        mode_label = "SWING (스윙)" if mode == TradingMode.SWING else "POSITION (포지션)"
        reasoning = decision.get("reasoning", {})
        final_logic = reasoning.get("final_logic", "No summary available.") if isinstance(reasoning, dict) else str(reasoning)[:200]

        summary_lines = [
            f"{header_icon} <b>[ {mode_label} ] AI 분석 리포트 | {report['symbol']}</b>",
            f"🕒 <code>{report['timestamp'][:16].replace('T', ' ')} UTC</code>\n",
            f"🎯 <b>최종 결정: {color_theme}</b> (확신도: {confidence}%)",
            "<blockquote>",
            f"🔵 진입가: <code>{self._fmt_price(decision.get('entry_price'))}</code>",
            f"🛑 손절가: <code>{self._fmt_price(decision.get('stop_loss'))}</code>",
            f"🏁 목표가: <code>{self._fmt_price(decision.get('take_profit'))}</code>",
            "</blockquote>\n",
            "📝 <b>Summary:</b>",
            f"<i>{final_logic}</i>\n",
        ]

        receipt = decision.get("execution_receipt")
        if receipt and receipt.get("success"):
            summary_lines.append(f"✅ <b>자동매매 실행 완료</b> ({len(receipt.get('receipts', []))} orders)")
        elif receipt:
            summary_lines.append(f"⚠️ <b>실행 보류:</b> {receipt.get('note', 'PENDING')}")

        return "\n".join(summary_lines)

    def format_detail_message(self, report: Dict, mode: TradingMode = TradingMode.SWING) -> str:
        decision = self._load_json_field(report.get("final_decision"), {})
        reasoning = decision.get("reasoning", "N/A")
        onchain_snapshot = self._load_json_field(report.get("onchain_snapshot"), {})
        onchain_context = str(report.get("onchain_context", "") or "").strip()

        lines = [f"🔍 <b>DEEP ANALYSIS | {report['symbol']}</b>\n"]

        if isinstance(reasoning, dict):
            mapping = {
                "technical": "📊 기술적 분석 (Technical)",
                "derivatives": "⛓️ 파생상품 심리 (Derivatives)",
                "experts": "🤖 전문가 스웜 (Expert Swarm)",
                "narrative": "🌐 시장 내러티브 (Narrative/News)",
                "counter_scenario": "🚨 최악의 시나리오 (Risk)",
            }
            for key, title in mapping.items():
                value = reasoning.get(key)
                if value:
                    safe_value = self._escape_html(value)
                    if len(safe_value) > 1500:
                        safe_value = safe_value[:1497] + "..."
                    safe_value = safe_value.replace(". ", ".\n\n")
                    lines.append(f"<b>{title}:</b>\n<i>{safe_value}</i>\n")
        else:
            lines.append(f"<i>{self._escape_html(reasoning)}</i>")

        if onchain_snapshot or onchain_context:
            lines.append("")
            lines.extend(
                self._build_onchain_lines(
                    symbol=report.get("symbol", "UNKNOWN"),
                    snapshot=onchain_snapshot,
                    onchain_context=onchain_context,
                    include_header=True,
                )
            )

        final_msg = "\n".join(lines)
        return final_msg[:4000]

    def format_onchain_message(self, symbol: str, snapshot: Optional[Dict]) -> str:
        if not snapshot:
            return f"<b>ON-CHAIN SNAPSHOT | {self._escape_html(symbol)}</b>\n<i>No snapshot available.</i>"
        return "\n".join(self._build_onchain_lines(symbol=symbol, snapshot=snapshot, onchain_context="", include_header=True))[:4000]

    @api_retry(max_attempts=3, delay_seconds=10)
    async def send_telegram_notification(
        self,
        report: Dict,
        chart_bytes: Optional[bytes] = None,
        mode: TradingMode = TradingMode.SWING,
    ) -> None:
        try:
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup

            bot = Bot(token=self.bot_token)
            summary_text = self.format_summary_message(report, mode)

            report_id = report.get("report_id") or report.get("id")
            keyboard = [[InlineKeyboardButton("🔍 View Deep Analysis", callback_data=f"detail_{report_id}")]]
            reply_markup = InlineKeyboardMarkup(keyboard)

            if chart_bytes:
                photo = BytesIO(chart_bytes)
                photo.name = f"{report['symbol']}_chart.png"

                if len(summary_text) <= 1024:
                    await bot.send_photo(
                        chat_id=self.chat_id,
                        photo=photo,
                        caption=summary_text,
                        parse_mode="HTML",
                        reply_markup=reply_markup,
                    )
                else:
                    await bot.send_photo(
                        chat_id=self.chat_id,
                        photo=photo,
                        caption=f"📊 {report['symbol']} Analysis",
                        parse_mode="HTML",
                    )
                    await bot.send_message(
                        chat_id=self.chat_id,
                        text=summary_text,
                        parse_mode="HTML",
                        reply_markup=reply_markup,
                    )
            else:
                await bot.send_message(
                    chat_id=self.chat_id,
                    text=summary_text,
                    parse_mode="HTML",
                    reply_markup=reply_markup,
                )

            logger.info(f"Telegram summary notification sent (ID: {report_id})")
        except Exception as e:
            logger.error(f"Telegram notification error: {e}")

    def notify(
        self,
        report: Dict,
        chart_bytes: Optional[bytes] = None,
        mode: TradingMode = TradingMode.SWING,
    ) -> None:
        threading.Thread(
            target=lambda: asyncio.run(self.send_telegram_notification(report, chart_bytes, mode)),
            daemon=True,
        ).start()


report_generator = ReportGenerator()
