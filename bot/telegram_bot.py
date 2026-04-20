"""Telegram Bot interactive command handler.

Commands:
/start     - Welcome message & status overview
/status    - Current position and latest decision
/analyze   - Run immediate analysis (BTC or ETH)
/evaluate  - Show recent realtime-pressure evaluation summary
/mode      - Show fixed dual-mode policy
/report    - Resend latest analysis report
/daily_status - Show latest scheduled daily result per symbol
/onchain   - Show latest on-chain snapshot
/chart     - Generate premium technical chart
/report_on - Resume automated AI analysis
/report_off - Pause automated AI analysis (Save cost)
/help     - List all commands
"""
from typing import List, Optional, Dict, Any, Callable
import re
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from config.settings import settings, TradingMode
from config.database import db
from processors.gcs_parquet import gcs_parquet_store
from loguru import logger
import json
from io import BytesIO
import base64


def _precision_schedule_labels_utc() -> str:
    raw = str(getattr(settings, "DAILY_PRECISION_HOURS_UTC", "") or "").strip()
    minute = int(getattr(settings, "DAILY_PRECISION_MINUTE_UTC", 30))
    hours = []
    if raw:
        for chunk in raw.split(","):
            chunk = chunk.strip()
            if chunk.isdigit():
                hours.append(int(chunk))
    if not hours:
        hours = [int(getattr(settings, "DAILY_PRECISION_HOUR_UTC", 0))]
    gap = max(0, int(getattr(settings, "DAILY_PRECISION_SYMBOL_GAP_MINUTES", 10)))
    labels = []
    for idx, symbol in enumerate(settings.trading_symbols):
        slots = []
        for hour in sorted(set(hours)):
            total_minutes = ((hour * 60) + minute + (idx * gap)) % (24 * 60)
            slots.append(f"{total_minutes // 60:02d}:{total_minutes % 60:02d}")
        labels.append(f"{symbol}={', '.join(slots)}")
    return " | ".join(labels)

# --- 자연어 채팅 핸들러용 상수 ---

_CHAT_SYSTEM = """당신 독립형 트레이딩봇의 AI 어시스턴트입니다. 
사용자의 자연어 질문에 답하기 위해 적절한 툴을 선택해 데이터를 조회하고, 결과를 한국어로 간결하게 해석해서 답하세요.

현 시스템 전략 안내:
1. SWING 레인: 바이낸스 선물 헤 모드 (Long/Short 양방향), 4시간봉 기반 추세 추종.
2. POSITION 레인: 바이낸스 현물 및 업비트 적립식 (Long Only), 일봉/주봉 기반 거시 사이클 투자.
두 레인을 동시에 활용해 하락장 리스크는 헤징하고 상승장 수익 극화하는 Dual-Lane 전략을 사용 중입니다.

침:
- BTC/비트코인 언급 시 BTCUSDT, ETH/이더리 언급 시 ETHUSDT 조회
- 숫자는 읽기 쉽게 적절히 반올림 (예: $98,432.12 -> $98,432)
- 질문에 필요한 툴만 사용하고 과도한 호출 피할 것
- 모든 답과 분석 반드시 **한국어**로 작성할 것"""

_CHAT_TOOLS = [
    {
        "name": "analyze_market",
        "description": "BTC/ETH 티임프레임 기술적 분석 (RSI, MACD, 볼린밴드 등)",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string", "description": "예: BTCUSDT"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "get_funding_info",
        "description": "딩비·OI·롱숏비율 조회 및 과열/공포 분석",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "get_global_oi",
        "description": "로벌 미결제약정(OI) 총합 (Binance+Bybit+OKX 합산)",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "get_cvd",
        "description": "CVD(누적 체결 델) 및 매수/매도 압력 추세",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "minutes": {"type": "integer", "description": "조회 기간(분), 기본 240"},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_indicator_summary",
        "description": "기술적 표 요약 (compact 포맷)",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "get_news_summary",
        "description": "최근 N시간 텔레그램 채널 뉴스 요약",
        "input_schema": {
            "type": "object",
            "properties": {"hours": {"type": "integer", "description": "기간(시간), 기본 4"}},
            "required": [],
        },
    },
    {
        "name": "get_latest_trading_report",
        "description": "장 최근 AI 트레이딩 결정 리포트",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_current_position",
        "description": "현재 포션 상태 (진입, 방향, PnL 등)",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "get_trading_mode",
        "description": "현재 트레이딩 모드 및 설정 조회",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "toggle_ai_analysis",
        "description": "AI 분석 및 리포트 생성을 켜거나 끕니다 (비용 절약)",
        "input_schema": {
            "type": "object",
            "properties": {"enabled": {"type": "boolean", "description": "True: 켜기, False: 끄기"}},
            "required": ["enabled"],
        },
    },
    {
        "name": "get_feedback_history",
        "description": "과거 트레이딩 실수 및 교훈 기록",
        "input_schema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "description": "조회 개수, 기본 5"}},
            "required": [],
        },
    },
    {
        "name": "query_knowledge_graph",
        "description": (
            "LightRAG 식 그래프 조회. 1시간마다 14개 텔레그램 채널에서 수집된 "
            "시장 내러티브·고래 동향·거시적 이벤트/뉴스 등 계 맥락 색. "
            "'~를 알려줘' 질문 시 필수."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "색어 (예: 'BTC whale accumulation', 'ETH ETF')"},
                "mode": {"type": "string", "description": "'local'(단기), 'global'(장기), 'hybrid'(모두). 기본 hybrid"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_narrative",
        "description": (
            "Perplexity API로 실시간 색. BTC/ETH 격 직임의 최신 이유·매크로 이벤트/뉴스 흐름 파악. "
            "RAG로 족할 때 사용. API 쿼터(200회) 주의."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string", "description": "예: BTCUSDT"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "get_chart_image",
        "description": "기술적 차트 이미를 생성해 텔레그램으로 전송. lane swing 또는 position만 허용",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "예: BTCUSDT"},
                "lane": {"type": "string", "description": "옵션: swing 또는 position. 기본 swing"}
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "search_web",
        "description": "Tavily API(비용)를 사용한 일반 웹 색. 최신 뉴스 및 일반적인 정보 조사용",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "색어"}},
            "required": ["query"],
        },
    },
]

_TYPE_CHECKERS: Dict[str, Callable[[Any], bool]] = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
}


def _coerce_tool_args(schema: Dict[str, Any], raw_args: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and minimally coerce function-call args from model output."""
    args = dict(raw_args or {})
    props = schema.get("properties", {}) or {}
    required = schema.get("required", []) or []

    for key in required:
        if key not in args:
            raise ValueError(f"Missing required arg: {key}")

    for key, spec in props.items():
        if key not in args:
            continue
        expected = spec.get("type", "string")
        value = args[key]

        if expected == "integer" and isinstance(value, str) and value.strip().isdigit():
            args[key] = int(value.strip())
            value = args[key]
        elif expected == "number" and isinstance(value, str):
            try:
                args[key] = float(value.strip())
                value = args[key]
            except Exception:
                pass
        elif expected == "boolean" and isinstance(value, str):
            low = value.strip().lower()
            if low in ("true", "1", "yes", "on"):
                args[key] = True
                value = True
            elif low in ("false", "0", "no", "off"):
                args[key] = False
                value = False

        checker = _TYPE_CHECKERS.get(expected)
        if checker and not checker(value):
            raise ValueError(f"Invalid arg type for '{key}': expected {expected}")

    return args


class TradingBot:
    def __init__(self):
        self.bot_token = settings.TELEGRAM_BOT_TOKEN
        self.chat_id = settings.TELEGRAM_CHAT_ID

    def _is_authorized(self, update: Update) -> bool:
        """Reject commands from any chat that isn't the configured owner chat."""
        if not self.chat_id:
            return True  # No restriction configured — allow (backwards-compatible)
        sender_id = str(getattr(update.effective_chat, "id", "") or "")
        return sender_id == str(self.chat_id)

    async def _reject_unauthorized(self, update: Update) -> None:
        sender_id = getattr(update.effective_chat, "id", "unknown")
        logger.warning(f"Unauthorized command attempt from chat_id={sender_id}")
        await update.message.reply_text("Unauthorized.")

    async def _send_chat_text(self, chat_id: str, text: str):
        """Send a chunked message to the given chat, preserving Telegram-safe formatting."""
        await self.send_message(chat_id, text)

    async def send_message(self, chat_id: str, text: str):
        """Send a message via the bot API (stateless)."""
        from telegram import Bot
        bot = Bot(token=self.bot_token)
        safe_text = self._sanitize_html_for_telegram(text)
        try:
            for chunk in self._chunk_text_for_telegram(safe_text):
                await bot.send_message(chat_id=chat_id, text=chunk, parse_mode='HTML')
        except Exception as e:
            # LLM output may contain unsupported HTML; retry as plain text instead of dropping the update.
            logger.warning(f"Telegram HTML send failed, retrying plain text: {e}")
            plain_text = re.sub(r"<[^>]+>", "", safe_text)
            for chunk in self._chunk_text_for_telegram(plain_text):
                await bot.send_message(chat_id=chat_id, text=chunk)

    async def send_photo(self, chat_id: str, photo_bytes: bytes, caption: str = ""):
        """Send a photo via the bot API (stateless)."""
        from telegram import Bot
        bot = Bot(token=self.bot_token)
        safe_caption = self._sanitize_html_for_telegram(caption)
        try:
            await bot.send_photo(chat_id=chat_id, photo=photo_bytes, caption=safe_caption, parse_mode='HTML')
        except Exception as e:
            logger.warning(f"Telegram photo HTML caption failed, retrying plain text caption: {e}")
            plain_caption = re.sub(r"<[^>]+>", "", safe_caption)
            caption_head = plain_caption[:1024]
            caption_tail = plain_caption[1024:]
            await bot.send_photo(chat_id=chat_id, photo=photo_bytes, caption=caption_head)
            for chunk in self._chunk_text_for_telegram(caption_tail):
                await bot.send_message(chat_id=chat_id, text=chunk)

    @staticmethod
    def _sanitize_html_for_telegram(text: str) -> str:
        """Normalize common unsupported HTML list tags into plain bullets."""
        if not text:
            return text

        sanitized = text
        # Convert common markdown-ish patterns into Telegram-safe HTML/plain text.
        sanitized = re.sub(r"^\s*#{1,6}\s*(.+)$", r"\n<b>\1</b>", sanitized, flags=re.MULTILINE)
        sanitized = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", sanitized)
        sanitized = re.sub(r"^\s*\*\s+", "- ", sanitized, flags=re.MULTILINE)
        sanitized = re.sub(r"</?ul[^>]*>", "", sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r"<li[^>]*>\s*", "\n- ", sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r"</li>", "", sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
        return sanitized.strip()

    @staticmethod
    def _chunk_text_for_telegram(text: str, limit: int = 3800) -> list[str]:
        """Split long Telegram messages on line boundaries to stay below message limits."""
        if not text:
            return [text]
        if len(text) <= limit:
            return [text]

        chunks = []
        current = ""

        for line in text.splitlines():
            candidate = f"{current}\n{line}" if current else line
            if len(candidate) <= limit:
                current = candidate
                continue

            if current:
                chunks.append(current)
                current = ""

            while len(line) > limit:
                chunks.append(line[:limit])
                line = line[limit:]
            current = line

        if current:
            chunks.append(current)

        return chunks or [text[:limit]]

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            await self._reject_unauthorized(update)
            return
        await update.message.reply_text(
            f"Crypto Trading Bot Active\n"
            f"Mode: DUAL (SWING + POSITION)\n"
            f"Symbols: BTC, ETH\n\n"
            f"Commands:\n"
            f"/status  - Current position & equity\n"
            f"/analyze - Live analysis (BTC/ETH)\n"
            f"/evaluate - Pressure signal scorecard\n"
            f"/onchain - Latest on-chain snapshot\n"
            f"/chart   - Generate HD technical chart\n"
            f"/mode    - Show fixed policy\n"
            f"/report  - Resend last scheduled report\n"
            f"/daily_status - Latest scheduled daily status\n"
            f"/report_off - Pause AI (Save $)\n"
            f"/report_on  - Resume AI\n"
            f"/help    - Show all commands"
        )

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            await self._reject_unauthorized(update)
            return
        await update.message.reply_text(
            "📚 <b>Available Commands</b>\n\n"
            "/status - Real-time positions & PnL\n"
            "/analyze [BTC|ETH] - Trigger instant AI analysis\n"
            "/evaluate [BTC|ETH] [hours] - Realtime pressure hit-rate summary\n"
            "  Example: /evaluate BTC\n"
            "  Example: /evaluate ETH 72\n"
            "/onchain [BTC|ETH] - Show latest on-chain snapshot and MVRV\n"
            "/chart [BTC|ETH] [swing|position] - Generate premium HD chart\n"
            "/report - Resend last scheduled report\n"
            "/daily_status - Latest scheduled daily status for BTC/ETH\n"
            "/mode - Show fixed dual-mode policy (change disabled)\n"
            "/report_off - PAUSE AI automation (Save cost)\n"
            "/report_on  - RESUME AI automation\n"
            "/help - Show this message\n\n"
            "💡 <i>Tip: You can also talk to me in natural language!</i>",
            parse_mode='HTML'
        )

    @staticmethod
    def _normalize_symbol_arg(raw_symbol: str) -> Optional[str]:
        token = str(raw_symbol or "").strip().upper()
        if not token:
            return None
        if token in settings.trading_symbols:
            return token
        if token in settings.trading_symbols_base:
            return f"{token}USDT"
        alias_map = {
            "BITCOIN": "BTCUSDT",
            "ETHEREUM": "ETHUSDT",
        }
        return alias_map.get(token)

    @staticmethod
    def _extract_pressure_eval(snapshot: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        if not isinstance(snapshot, dict):
            return {}, {}
        pressure = (
            snapshot.get("realtime_pressure")
            or (snapshot.get("swing") or {}).get("realtime_pressure")
            or (snapshot.get("position") or {}).get("realtime_pressure")
            or {}
        )
        evaluation = snapshot.get("evaluation") if isinstance(snapshot.get("evaluation"), dict) else {}
        return pressure if isinstance(pressure, dict) else {}, evaluation

    def _build_pressure_evaluation_summary(self, symbol: str, hours: int = 168, limit: int = 300) -> str:
        rows = db.get_market_status_events(symbol=symbol, limit=limit, hours=hours)
        if not rows:
            return f"<b>{symbol} Pressure Evaluation</b>\n- 최근 {hours}시간 이벤트 없습니다."

        horizons = (5, 15, 30)
        overall = {m: {"correct": 0, "total": 0, "returns": []} for m in horizons}
        groups: Dict[str, Dict[str, Any]] = {}
        evaluated_rows = 0

        for row in rows:
            snapshot = row.get("technical_snapshot") if isinstance(row.get("technical_snapshot"), dict) else {}
            pressure, evaluation = self._extract_pressure_eval(snapshot)
            signal = evaluation.get("signal") or pressure.get("signal")
            summary = str(evaluation.get("summary") or pressure.get("summary") or "unknown")
            if signal not in ("bullish", "bearish"):
                continue

            group = groups.setdefault(
                summary,
                {"count": 0, "signal": signal, "returns": {m: [] for m in horizons}, "correct": {m: 0 for m in horizons}, "total": {m: 0 for m in horizons}},
            )
            group["count"] += 1
            evaluated_rows += 1

            for minutes in horizons:
                ret_key = f"forward_{minutes}m_return_pct"
                outcome_key = f"outcome_{minutes}m"
                ret = evaluation.get(ret_key)
                outcome = evaluation.get(outcome_key)
                if not isinstance(ret, (int, float)):
                    continue
                group["returns"][minutes].append(float(ret))
                group["total"][minutes] += 1
                overall[minutes]["returns"].append(float(ret))
                overall[minutes]["total"] += 1
                if outcome == "correct":
                    group["correct"][minutes] += 1
                    overall[minutes]["correct"] += 1

        if not groups:
            return f"<b>{symbol} Pressure Evaluation</b>\n- 최근 {hours}시간 내 평 완료된 신호 없습니다."

        lines = [
            f"<b>{symbol} Pressure Evaluation</b>",
            f"- Window: 최근 {hours}시간",
            f"- Evaluated signals: {evaluated_rows}",
            "<b>Overall</b>",
        ]

        for minutes in horizons:
            total = overall[minutes]["total"]
            correct = overall[minutes]["correct"]
            avg_ret = sum(overall[minutes]["returns"]) / len(overall[minutes]["returns"]) if overall[minutes]["returns"] else None
            rate = (correct / total * 100.0) if total else None
            lines.append(
                f"- {minutes}m: {correct}/{total} "
                f"({rate:.1f}% ) | avg {avg_ret:+.3f}%"
                if total and avg_ret is not None
                else f"- {minutes}m: N/A"
            )

        lines.append("<b>By Summary</b>")
        for summary, data in sorted(groups.items(), key=lambda item: item[1]["count"], reverse=True)[:6]:
            line = f"- {summary} ({data['signal']}) n={data['count']}"
            details = []
            for minutes in horizons:
                total = data["total"][minutes]
                if not total:
                    continue
                correct = data["correct"][minutes]
                avg_ret = sum(data["returns"][minutes]) / len(data["returns"][minutes]) if data["returns"][minutes] else 0.0
                details.append(f"{minutes}m {correct}/{total} {correct/total*100:.0f}% avg {avg_ret:+.3f}%")
            if details:
                line += " | " + " | ".join(details)
            lines.append(line)

        return "\n".join(lines)

    @staticmethod
    def _load_daily_precision_summary() -> Dict[str, Any]:
        from config.local_state import state_manager

        raw = state_manager.get_system_config("daily_precision_last_summary", "")
        default = {"updated_at": "", "symbols": {}}
        if not raw:
            return dict(default)
        try:
            payload = json.loads(raw)
        except Exception:
            return dict(default)
        if not isinstance(payload, dict):
            return dict(default)
        if isinstance(payload.get("symbols"), dict):
            return payload
        if isinstance(payload.get("results"), list):
            return {
                "updated_at": str(payload.get("finished_at", "") or ""),
                "symbols": {
                    str(item.get("symbol")): item
                    for item in payload.get("results", [])
                    if isinstance(item, dict) and item.get("symbol")
                },
            }
        return dict(default)

    @staticmethod
    def _format_daily_precision_status(summary: Dict[str, Any]) -> str:
        symbols = (summary.get("symbols", {}) if isinstance(summary, dict) else {}) or {}
        updated_at = str((summary or {}).get("updated_at", "") or "")
        lines = [
            "<b>Daily Precision Status</b>",
            f"Schedule (UTC): <code>{_precision_schedule_labels_utc()}</code>",
        ]
        if updated_at:
            lines.append(f"Updated: <code>{updated_at[:16].replace('T', ' ')} UTC</code>")

        for symbol in settings.trading_symbols:
            item = symbols.get(symbol)
            if not isinstance(item, dict):
                lines.append(f"- <b>{symbol}</b> | status=<code>NO_RECORD</code>")
                continue
            status = str(item.get("status", "UNKNOWN") or "UNKNOWN").upper()
            decision = str(item.get("decision", "") or "-")
            report_id = str(item.get("report_id", "") or "-")
            finished_at = str(item.get("finished_at", "") or "")
            line = (
                f"- <b>{symbol}</b> | status=<code>{status}</code> "
                f"| decision=<code>{decision}</code> | report_id=<code>{report_id}</code>"
            )
            if finished_at:
                line += f" | at <code>{finished_at[:16].replace('T', ' ')}</code>"
            lines.append(line)
            error = str(item.get("error", "") or "").strip()
            if error and status != "SUCCESS":
                safe_error = error[:180].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                lines.append(f"  error: <code>{safe_error}</code>")

        return "\n".join(lines)

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            await self._reject_unauthorized(update)
            return
        try:
            is_paper = settings.PAPER_TRADING_MODE

            lines = [
                f"📊 <b>Trading Status</b>",
                f"Mode: <code>DUAL (SWING + POSITION)</code>",
                f"Type: <code>{'PAPER' if is_paper else 'LIVE'}</code>",
                ""
            ]

            # 1. Active Mock Positions (If applicable)
            if is_paper:
                from executors.paper_exchange import paper_engine
                from executors.trade_executor import trade_executor
                positions = paper_engine.get_open_positions()
                
                binance_unrealized = 0.0
                binance_margin = 0.0
                upbit_unrealized = 0.0
                upbit_margin = 0.0

                lines.append("💼 <b>Active Positions</b>")
                if not positions:
                    lines.append("<i>- No open positions</i>")
                else:
                    for pos in positions:
                        symbol = pos['symbol']
                        side = pos['side']
                        entry = pos['entry_price']
                        size = pos['size']
                        leverage = pos['leverage']
                        exchange = pos['exchange']

                        # Get current price for PnL
                        current_price = trade_executor._get_reference_price(symbol)
                        pnl = 0.0
                        if current_price > 0:
                            pnl = (current_price - entry) * size if side == "LONG" else (entry - current_price) * size

                        initial_margin = (size * entry) / leverage
                        if exchange == 'binance':
                            binance_unrealized += pnl
                            binance_margin += initial_margin
                        else:
                            upbit_unrealized += pnl
                            upbit_margin += initial_margin

                        pnl_icon = "📈" if pnl >= 0 else "📉"
                        lines.append(
                            f"➡️ {exchange.upper()} | <b>{symbol}</b>: {side} {leverage}x\n"
                            f"  Entry: {entry:,.2f} | Now: {current_price:,.2f}\n"
                            f"  PnL: {pnl_icon} <b>${pnl:+.2f}</b>"
                        )

                # Wallet Balances & Equity (거래소별 독립 계산)
                lines.append("\n💰 <b>Mock Balances</b>")
                for ex in ['binance', 'upbit']:
                    bal = paper_engine.get_wallet_balance(ex)
                    ex_margin = binance_margin if ex == 'binance' else upbit_margin
                    ex_unrealized = binance_unrealized if ex == 'binance' else upbit_unrealized
                    equity = bal + ex_margin + ex_unrealized
                    lines.append(f"🏦 {ex.upper()} Balance: <code>${bal:,.2f}</code>")
                    lines.append(f"💰 {ex.upper()} <b>Total Equity: ${equity:,.2f}</b>")
                    lines.append(f"  (Margin: ${ex_margin:.2f} | UnPnL: ${ex_unrealized:+.2f})")
                lines.append("")

            # 2. Strategy Analysis (Latest Reports)
            lines.append("🧠 <b>Latest Strategy</b>")
            for symbol in settings.trading_symbols:
                report = db.get_latest_report(symbol=symbol)
                if report:
                    fd = report.get('final_decision')
                    if isinstance(fd, str):
                        fd = json.loads(fd)

                    decision = fd.get('decision', 'N/A') if fd else 'N/A'
                    confidence = (
                        fd.get('confidence', fd.get('win_probability_pct', 0))
                        if fd else 0
                    )
                    ts = report.get('timestamp', 'N/A')[:16].replace('T', ' ')

                    lines.append(
                        f"➡️ {symbol}: <b>{decision}</b> ({confidence}%) "
                        f"<pre>{ts}</pre>"
                    )
                else:
                    lines.append(f"➡️ {symbol}: No report yet")

            from telegram import InlineKeyboardButton, InlineKeyboardMarkup
            keyboard = []
            if is_paper and 'positions' in locals() and positions:
                for pos in positions:
                    pos_id = pos.get('position_id')
                    sym = pos.get('symbol')
                    if pos_id:
                        keyboard.append([
                            InlineKeyboardButton(f"🛡️ SL 본절", callback_data=f"pos_sl_{pos_id}"),
                            InlineKeyboardButton(f"✂️ {sym} 50%", callback_data=f"pos_half_{pos_id}"),
                            InlineKeyboardButton(f"❌ {sym} 100%", callback_data=f"pos_close_{pos_id}")
                        ])
            reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None
            await update.message.reply_text('\n'.join(lines), parse_mode='HTML', reply_markup=reply_markup)
        except Exception as e:
            logger.error(f"Status command error: {e}")
            await update.message.reply_text(f"Error: {e}")

    async def cmd_analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            await self._reject_unauthorized(update)
            return
        args = context.args
        valid_bases = settings.trading_symbols_base
        if not args or args[0].upper() not in valid_bases:
            await update.message.reply_text(f"Usage: /analyze {'|'.join(valid_bases)}")
            return

        symbol = args[0].upper() + 'USDT'
        await update.message.reply_text(
            f"Running precision analysis for {symbol} "
            f"(single POSITION lane; dual playbook will be refreshed)..."
        )

        try:
            from executors.orchestrator import orchestrator
            result = await asyncio.to_thread(
                orchestrator.run_analysis_with_mode,
                symbol,
                TradingMode.SWING,
                False,
                True,
                True,
            )

            if result:
                decision = result.get('decision', 'N/A')
                confidence = result.get('confidence', result.get('win_probability_pct', 0))
                await update.message.reply_text(
                    f"Analysis complete for {symbol}\n"
                    f"SWING: {decision} ({confidence}%)\n"
                    f"Report sent to chat."
                )
            else:
                await update.message.reply_text("Analysis failed. Check logs.")
        except Exception as e:
            logger.error(f"Analyze command error: {e}")
            await update.message.reply_text(f"Error: {e}")

    async def cmd_evaluate(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            await self._reject_unauthorized(update)
            return
        args = context.args or []
        if not args:
            await update.message.reply_text(f"Usage: /evaluate {'|'.join(settings.trading_symbols_base)} [hours]")
            return

        symbol = self._normalize_symbol_arg(args[0])
        if not symbol:
            await update.message.reply_text(f"Usage: /evaluate {'|'.join(settings.trading_symbols_base)} [hours]")
            return

        hours = 168
        if len(args) > 1:
            try:
                hours = max(1, min(24 * 30, int(args[1])))
            except Exception:
                await update.message.reply_text("Hours must be an integer. Example: /evaluate BTC 168")
                return

        try:
            text = await asyncio.to_thread(self._build_pressure_evaluation_summary, symbol, hours, 300)
            await update.message.reply_text(text, parse_mode='HTML')
        except Exception as e:
            logger.error(f"Evaluate command error: {e}")
            await update.message.reply_text(f"Error: {e}")

    async def cmd_mode(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            await self._reject_unauthorized(update)
            return
        await update.message.reply_text(
            "Mode switching is disabled.\n"
            "Policy is fixed:\n"
            "- SWING: futures hedge lane (LONG/SHORT)\n"
            "- POSITION: spot accumulation lane (LONG only)\n"
            "Both lanes run together every cycle."
        )

    async def cmd_daily_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            await self._reject_unauthorized(update)
            return
        try:
            summary = await asyncio.to_thread(self._load_daily_precision_summary)
            await update.message.reply_text(
                self._format_daily_precision_status(summary),
                parse_mode='HTML',
            )
        except Exception as e:
            logger.error(f"Daily status command error: {e}")
            await update.message.reply_text(f"Error: {e}")

    async def cmd_report(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            await self._reject_unauthorized(update)
            return
        try:
            from executors.report_generator import report_generator

            chat_id = getattr(update.effective_chat, "id", None)
            if chat_id is None:
                await update.message.reply_text("Unable to detect current chat.")
                return

            replayed = await report_generator.replay_last_scheduled_report(str(chat_id))
            if not replayed:
                await update.message.reply_text("장된 마막 정기 리포트 없습니다.")
        except Exception as e:
            logger.error(f"Report command error: {e}")
            await update.message.reply_text(f"Error: {e}")

    async def cmd_onchain(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            await self._reject_unauthorized(update)
            return
        args = context.args or []
        symbols = settings.trading_symbols

        if args:
            normalized = self._normalize_symbol_arg(args[0])
            if not normalized:
                await update.message.reply_text(
                    f"Usage: /onchain [{'|'.join(settings.trading_symbols_base)}]"
                )
                return
            symbols = [normalized]

        try:
            from executors.report_generator import report_generator

            found = False
            chat_id = getattr(update.effective_chat, "id", None)
            if chat_id is None:
                await update.message.reply_text("Unable to detect current chat.")
                return
            for symbol in symbols:
                snapshot = gcs_parquet_store.get_latest_row("onchain", symbol)
                if not snapshot:
                    await update.message.reply_text(f"No on-chain snapshot for {symbol}")
                    continue
                found = True
                text = report_generator.format_onchain_message(symbol, snapshot)
                await self._send_chat_text(str(chat_id), text)

            if not found and not args:
                await update.message.reply_text("No on-chain snapshots available.")
        except Exception as e:
            logger.error(f"On-chain command error: {e}")
            await update.message.reply_text(f"Error: {e}")

    async def cmd_report_on(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            await self._reject_unauthorized(update)
            return
        from config.local_state import state_manager
        state_manager.set_analysis_enabled(True)
        chat_id = getattr(update.effective_chat, "id", None)
        if chat_id is not None:
            state_manager.set_telegram_chat_id(str(chat_id))
        await update.message.reply_text(
            "정기 AI 리포트를 다시 켰습니다.\n"
            "이 채팅방을 정기 발송 상으로 장했습니다.",
            parse_mode='HTML',
        )

    async def cmd_report_off(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            await self._reject_unauthorized(update)
            return
        from config.local_state import state_manager
        state_manager.set_analysis_enabled(False)
        await update.message.reply_text(
            "정기 AI 리포트를 껐습니다.\n"
            "데이터 수집 계속되만 자동 분석/발송 중단됩니다.",
            parse_mode='HTML',
        )

    async def cmd_gate(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Confluence Gate 현황 조회 및 수동 조정.
        /gate          → 현재 threshold + 최근 통계
        /gate 2.5      → threshold 수동 설정 (1.0 ~ 5.0)
        /gate tune     → 즉시 auto-tune 실행
        """
        if not self._is_authorized(update):
            await self._reject_unauthorized(update)
            return

        from config.local_state import state_manager
        from executors.gate_tuner import analyze as _gate_analyze, run as _gate_run

        args = context.args or []

        # ── 수동 설정 ──────────────────────────────────────────────
        if args and args[0].replace(".", "").isdigit():
            try:
                new_val = float(args[0])
                state_manager.set_confluence_gate_threshold(new_val, reason="manual override via /gate")
                await update.message.reply_text(
                    f"✅ Confluence Gate threshold → <b>{state_manager.get_confluence_gate_threshold()}</b>",
                    parse_mode="HTML",
                )
            except ValueError:
                await update.message.reply_text("숫자를 입력하세요. 예: /gate 2.5")
            return

        # ── 즉시 tune ─────────────────────────────────────────────
        if args and args[0].lower() == "tune":
            results = []
            for sym in settings.trading_symbols:
                r = await asyncio.to_thread(_gate_run, sym)
                results.append(f"<b>{sym}</b>: {r.get('reason', '-')}")
            await update.message.reply_text(
                "🔧 <b>Gate Auto-Tune 실행</b>\n" + "\n".join(results),
                parse_mode="HTML",
            )
            return

        # ── 현황 조회 ──────────────────────────────────────────────
        current = state_manager.get_confluence_gate_threshold()
        log = state_manager.get_confluence_gate_tuner_log()

        lines = [
            f"⚙️ <b>Confluence Gate</b>",
            f"현재 threshold: <b>{current}</b> / 범위 1.0~5.0",
            "",
        ]
        for sym in settings.trading_symbols:
            stats = await asyncio.to_thread(_gate_analyze, sym)
            wr = f"{stats['win_rate']:.0%}" if stats['win_rate'] is not None else "N/A"
            hr = f"{stats['hold_rate']:.0%}" if stats['hold_rate'] is not None else "N/A"
            lines.append(
                f"<b>{sym}</b> (최근 14일) | "
                f"진입 {stats['entries']}회 | "
                f"W{stats['wins']}/L{stats['losses']} | "
                f"승률 {wr} | HOLD {hr}"
            )

        if log:
            lines += ["", "<b>조정 이력 (최근 3건)</b>"]
            for entry in log[-3:]:
                ts = entry.get("ts", "")[:16].replace("T", " ")
                lines.append(f"• {ts} → {entry.get('threshold')} ({entry.get('reason', '')})")

        lines += ["", "명령어: /gate 2.5 (설정) | /gate tune (자동조정)"]
        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    async def cmd_chart(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Direct command to get a chart image: /chart [symbol] [swing|position]"""
        if not self._is_authorized(update):
            await self._reject_unauthorized(update)
            return
        args = context.args
        if not args:
            await update.message.reply_text("Usage: /chart BTC|ETH [swing|position]")
            return

        base = args[0].upper()
        if base not in settings.trading_symbols_base:
            await update.message.reply_text(f"Invalid symbol. Use: {'|'.join(settings.trading_symbols_base)}")
            return

        symbol = base + "USDT"
        lane = args[1].lower() if len(args) > 1 else "swing"
        if lane not in ("swing", "position"):
            await update.message.reply_text("Invalid lane. Use: /chart BTC|ETH [swing|position]")
            return

        await update.message.reply_text(f"Generating premium {lane.upper()} chart for {symbol}...")

        try:
            from mcp_server.tools import mcp_tools
            result = mcp_tools.get_chart_images(symbol, lane=lane)

            if "charts" in result:
                charts = result.get("charts", [])
                total = len(charts)
                for idx, chart in enumerate(charts, start=1):
                    chart_bytes = base64.b64decode(chart["chart_base64"])
                    tf = str(chart.get("timeframe", "-")).upper()
                    buf = BytesIO(chart_bytes)
                    buf.name = f"{symbol}_{lane}_{tf}.png"

                    caption = (
                        f"<b>{symbol} {result.get('mode', '').upper()} CHART</b>\n"
                        f"Lane: <code>{result.get('lane', lane)}</code>\n"
                        f"Timeframe: <code>{tf}</code>\n"
                        f"Panel: <code>{idx}/{total}</code>\n"
                        f"Quality: <code>Premium HD</code>"
                    )
                    await update.message.reply_photo(photo=buf, caption=caption, parse_mode='HTML')
            elif "chart_base64" in result:
                chart_bytes = base64.b64decode(result["chart_base64"])
                buf = BytesIO(chart_bytes)
                buf.name = f"{symbol}_{lane}.png"
                
                # Send photo
                caption = f"📈 <b>{symbol} {result.get('mode', '').upper()} CHART</b>\n"
                caption += f"Lane: <code>{result.get('lane', lane)}</code>\n"
                caption += f"Timeframe: <code>{result.get('timeframe', '-')}</code> (Fixed)\n"
                caption += f"Quality: <code>Premium HD</code>"
                
                await update.message.reply_photo(photo=buf, caption=caption, parse_mode='HTML')
            else:
                await update.message.reply_text(f"Failed to generate chart: {result.get('error', 'Unknown error')}")
        except Exception as e:
            logger.error(f"Chart command error: {e}")
            await update.message.reply_text(f"Error: {e}")

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button clicks for deep analysis."""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        if data.startswith("pos_"):
            try:
                parts = data.split("_")
                action_type = parts[1]
                pos_id = "_".join(parts[2:])
                
                from executors.paper_exchange import paper_engine
                from executors.trade_executor import trade_executor
                pos = paper_engine.get_open_position(pos_id)

                if not pos:
                    await query.edit_message_text(f"{query.message.text}\n\n⚠️ Position not found or closed.")
                    return

                symbol = pos["symbol"]
                current_price = trade_executor._get_reference_price(symbol)
                
                if action_type == "sl":
                    res = paper_engine.update_sl_to_breakeven(pos_id)
                elif action_type == "half":
                    res = paper_engine.close_position_partial(pos_id, current_price, 0.5)
                elif action_type == "close":
                    res = paper_engine.close_position_partial(pos_id, current_price, 1.0)
                else:
                    res = {"success": False, "error": "Unknown action"}
                    
                if res.get("success"):
                    msg = f"✅ Action {action_type.upper()} processed successfully."
                    await query.edit_message_text(f"{query.message.text}\n\n{msg}", parse_mode='HTML')
                else:
                    await query.message.reply_text(f"Failed: {res.get('error')}")
            except Exception as e:
                logger.error(f"Position control error: {e}")
                await query.message.reply_text(f"Error: {e}")
        elif data.startswith("detail_"):
            try:
                report_id = data.split("_")[1]
                # If report_id is not integer (UUID), we might need to adjust lookup
                # Depending on how db.insert_ai_report returns the ID
                
                # Fetch report from DB
                from config.database import db
                report = None
                
                # Try lookup by created_at or ID if we have it
                # For now, let's assume we can get it by newest for the symbol
                # or better yet, search in the ai_reports table if report_id is the primary key.
                response = db.client.table("ai_reports").select("*").eq("id", report_id).execute()
                if response.data:
                    report = response.data[0]
                
                if report:
                    from executors.report_generator import report_generator
                    # Determine mode - default to current settings if not found
                    mode = TradingMode.SWING

                    detail_text = report_generator.format_detail_message(report, mode)
                    chat_id = getattr(query.message.chat, "id", None)
                    if chat_id is None:
                        await query.message.reply_text("Unable to detect current chat.")
                    else:
                        await self._send_chat_text(str(chat_id), detail_text)
                else:
                    await query.message.reply_text("Report no longer available in database.")
            except Exception as e:
                logger.error(f"Callback error: {e}")
                await query.message.reply_text(f"Error fetching detail: {e}")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """자연어 메시를 Gemini Flash function calling으로 처리.
        분석 리포트(Judge)는 Claude를 유하고, 화형 채팅 비용 Gemini Flash 사용.
        """
        if not self._is_authorized(update):
            await self._reject_unauthorized(update)
            return
        user_text = update.message.text
        if not user_text or user_text.startswith('/'):
            return

        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing",
        )

        try:
            import asyncio
            from agents.ai_router import ai_client

            _mcp_tools = None
            def _get_mcp_tools():
                nonlocal _mcp_tools
                if _mcp_tools is None:
                    from mcp_server.tools import mcp_tools as _loaded_mcp_tools
                    _mcp_tools = _loaded_mcp_tools
                return _mcp_tools

            tool_fn_map = {
                "analyze_market":            lambda a: _get_mcp_tools().get_market_analysis(a["symbol"]),
                "get_funding_info":          lambda a: _get_mcp_tools().get_funding_data(a["symbol"]),
                "get_global_oi":             lambda a: _get_mcp_tools().get_global_oi(a["symbol"]),
                "get_cvd":                   lambda a: _get_mcp_tools().get_cvd_data(a["symbol"], limit=a.get("minutes", 240)),
                "get_indicator_summary":     lambda a: _get_mcp_tools().get_indicator_summary(a["symbol"]),
                "get_news_summary":          lambda a: _get_mcp_tools().get_telegram_summary(a.get("hours", 4)),
                "get_latest_trading_report": lambda a: _get_mcp_tools().get_latest_report(),
                "get_current_position":      lambda a: _get_mcp_tools().get_position_status(a["symbol"]),
                "get_trading_mode":          lambda a: {
                    "mode": "dual",
                    "swing": {"venue": "binance_futures", "direction": "long_short"},
                    "position": {"venue": "binance_spot_upbit", "direction": "long_only"},
                    "chart_enabled": settings.should_use_chart,
                    "primary_scheduler_utc": {
                        "job_daily_precision": _precision_schedule_labels_utc(),
                        "job_hourly_monitor": "hh:15",
                        "job_routine_market_status": "hh:20",
                    },
                },
                "toggle_ai_analysis":        lambda a: (__import__('config.local_state', fromlist=['state_manager']).state_manager.set_analysis_enabled(a["enabled"]), {"status": f"AI 분석 {'활성화' if a['enabled'] else '비활성화'} 완료"})[1],
                "get_feedback_history":      lambda a: _get_mcp_tools().get_feedback_history(a.get("limit", 5)),
                "query_knowledge_graph":     lambda a: _get_mcp_tools().query_rag(a["query"], mode=a.get("mode", "hybrid")),
                "search_narrative":          lambda a: _get_mcp_tools().search_market_narrative(a["symbol"]),
                "get_chart_image":           lambda a: _get_mcp_tools().get_chart_image(a["symbol"], lane=a.get("lane", "swing")),
                "search_web":                lambda a: _get_mcp_tools().search_web(a["query"]),
            }

            tool_aliases = {
                "get_news_brief": "get_news_summary",
                "query_kg": "query_knowledge_graph",
            }

            tool_schemas = {t["name"]: t["input_schema"] for t in _CHAT_TOOLS}
            tool_registry = {
                name: {
                    "schema": tool_schemas.get(name, {"type": "object", "properties": {}, "required": []}),
                    "handler": handler,
                }
                for name, handler in tool_fn_map.items()
            }

            declared = {t["name"] for t in _CHAT_TOOLS}
            executable = set(tool_registry.keys())
            if declared - executable:
                logger.warning(f"Tool declarations without handlers: {sorted(declared - executable)}")
            if executable - declared:
                logger.warning(f"Tool handlers missing declarations: {sorted(executable - declared)}")

            def _extract_first_json(text_val: str) -> Dict[str, Any]:
                if not text_val:
                    return {}
                start_idx = text_val.find("{")
                if start_idx < 0:
                    return {}
                depth = 0
                in_string = False
                escape = False
                for i in range(start_idx, len(text_val)):
                    ch = text_val[i]
                    if escape:
                        escape = False
                        continue
                    if ch == "\\":
                        escape = True
                        continue
                    if ch == "\"":
                        in_string = not in_string
                        continue
                    if in_string:
                        continue
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            raw = text_val[start_idx:i + 1]
                            try:
                                obj = json.loads(raw)
                                return obj if isinstance(obj, dict) else {}
                            except Exception:
                                return {}
                return {}

            def _extract_sources(obj: Any) -> List[str]:
                tags: List[str] = []
                if isinstance(obj, dict):
                    src = obj.get("source")
                    channel = obj.get("channel")
                    url = obj.get("url")
                    if src and url:
                        tags.append(f"[{src} - {url}]")
                    elif src and channel:
                        tags.append(f"[{src} - telegram]")
                    elif src:
                        tags.append(f"[{src}]")
                    for v in obj.values():
                        tags.extend(_extract_sources(v))
                elif isinstance(obj, list):
                    for v in obj:
                        tags.extend(_extract_sources(v))
                return tags

            planner_system = (
                "You are a query planner for a crypto trading assistant. "
                "Return STRICT JSON only. Do not include markdown fences."
            )
            available_tools = ", ".join(sorted(tool_registry.keys()))
            planner_user = (
                "Plan tool usage for this user query. Keep plan short and deterministic.\n"
                f"Allowed tools: {available_tools}\n"
                "Output JSON schema:\n"
                "{\n"
                "  \"intent\": \"market|portfolio|news|knowledge|ops|other\",\n"
                "  \"needs_tools\": true|false,\n"
                "  \"tool_plan\": [{\"tool\": \"name\", \"args\": {}}],\n"
                "  \"answer_style\": \"brief|detailed\",\n"
                "  \"clarify\": \"question if missing critical args, else empty\"\n"
                "}\n"
                "Rules: max 3 tools; use symbols BTCUSDT/ETHUSDT when relevant; if no tools needed set needs_tools=false.\n\n"
                f"USER_QUERY:\n{user_text}"
            )

            planner_raw = await asyncio.to_thread(
                ai_client.generate_response,
                planner_system,
                planner_user,
                800,
                0.1,
                None,
                False,
                "triage",
            )
            plan_obj = _extract_first_json(planner_raw)

            if not plan_obj:
                fallback = await asyncio.to_thread(
                    ai_client.generate_response,
                    _CHAT_SYSTEM,
                    user_text,
                    1500,
                    0.3,
                    None,
                    False,
                    "chat",
                )
                await update.message.reply_text(fallback or "요청을 이해하 못했습니다.")
                return

            tool_calls = plan_obj.get("tool_plan", []) if isinstance(plan_obj.get("tool_plan"), list) else []
            if len(tool_calls) > 3:
                tool_calls = tool_calls[:3]
            clarify_q = str(plan_obj.get("clarify", "")).strip()
            if clarify_q and not tool_calls:
                await update.message.reply_text(clarify_q)
                return

            execution_results: List[Dict[str, Any]] = []
            citation_tags: List[str] = []

            if bool(plan_obj.get("needs_tools", True)):
                for call in tool_calls:
                    if not isinstance(call, dict):
                        continue
                    requested = str(call.get("tool", "")).strip()
                    fn_name = tool_aliases.get(requested, requested)
                    raw_args = call.get("args", {})
                    if not isinstance(raw_args, dict):
                        raw_args = {}

                    reg = tool_registry.get(fn_name)
                    if reg is None:
                        execution_results.append({
                            "tool": requested,
                            "status": "error",
                            "result": {"error": f"Unknown tool: {requested}"},
                        })
                        continue

                    try:
                        safe_args = _coerce_tool_args(reg["schema"], raw_args)
                        result = await asyncio.to_thread(reg["handler"], safe_args)
                    except Exception as tool_err:
                        result = {"error": f"Tool execution failed: {type(tool_err).__name__}: {tool_err}"}
                        safe_args = raw_args

                    if fn_name == "get_chart_image" and isinstance(result, dict) and "chart_base64" in result:
                        try:
                            chart_bytes = base64.b64decode(result["chart_base64"])
                            buf = BytesIO(chart_bytes)
                            buf.name = f"{safe_args.get('symbol', 'chart')}.png"
                            await update.message.reply_photo(photo=buf)
                            result = {
                                "status": "chart_sent",
                                "symbol": safe_args.get("symbol", ""),
                                "size_bytes": result.get("size_bytes", 0),
                            }
                        except Exception as img_err:
                            result = {"error": f"Chart send failed: {img_err}"}

                    execution_results.append({
                        "tool": fn_name,
                        "args": safe_args,
                        "status": "ok" if not (isinstance(result, dict) and result.get("error")) else "error",
                        "result": result,
                    })
                    citation_tags.extend(_extract_sources(result))

            citation_tags = sorted(set(citation_tags))[:20]

            responder_system = (
                "You are a crypto assistant. Answer in Korean with concise, accurate synthesis for medium/long-term investors. "
                "Output format MUST be Telegram-safe HTML only (use <b>, <i>, <code>, plain '-' bullets). "
                "Do NOT use Markdown markers like ###, **, *, or markdown numbered lists. "
                "Structure strictly as: <b>1) News Summary</b>, <b>2) Technical Summary</b>, "
                "<b>3) Swing Plan</b>, <b>4) Position Plan</b>, <b>5) Conclusion</b>. "
                "When claims rely on fetched data, include source tags like [source - url] or [source - telegram]. "
                "Never invent sources. If evidence is weak, say uncertainty explicitly."
            )
            responder_payload = {
                "user_query": user_text,
                "plan": plan_obj,
                "tool_results": execution_results,
                "source_tags": citation_tags,
            }
            responder_user = "Use this JSON as ground truth:\n" + json.dumps(responder_payload, ensure_ascii=False, default=str)

            final_answer = await asyncio.to_thread(
                ai_client.generate_response,
                responder_system,
                responder_user,
                1800,
                0.2,
                None,
                False,
                "chat",
            )
            safe_answer = self._sanitize_html_for_telegram(final_answer or "결과를 생성하 못했습니다.")
            await update.message.reply_text(safe_answer, parse_mode='HTML')

        except Exception as e:
            logger.error(f"handle_message error: {e}")
            await update.message.reply_text(f"오류 발생했습니다: {type(e).__name__}")

    def run(self):
        """Start the bot (non-blocking, runs in background thread)."""
        app = Application.builder().token(self.bot_token).build()

        app.add_handler(CommandHandler("start", self.cmd_start))
        app.add_handler(CommandHandler("help", self.cmd_help))
        app.add_handler(CommandHandler("status", self.cmd_status))
        app.add_handler(CommandHandler("analyze", self.cmd_analyze))
        app.add_handler(CommandHandler("evaluate", self.cmd_evaluate))
        app.add_handler(CommandHandler("mode", self.cmd_mode))
        app.add_handler(CommandHandler("report", self.cmd_report))
        app.add_handler(CommandHandler("daily_status", self.cmd_daily_status))
        app.add_handler(CommandHandler("onchain", self.cmd_onchain))
        app.add_handler(CommandHandler("report_on", self.cmd_report_on))
        app.add_handler(CommandHandler("report_off", self.cmd_report_off))
        app.add_handler(CommandHandler("chart", self.cmd_chart))
        app.add_handler(CommandHandler("gate", self.cmd_gate))
        app.add_handler(CallbackQueryHandler(self.handle_callback))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        logger.info("Telegram bot started (polling)")
        app.run_polling(drop_pending_updates=True, stop_signals=None)


trading_bot = TradingBot()

if __name__ == "__main__":
    trading_bot.run()
