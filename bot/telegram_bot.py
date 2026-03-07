"""Telegram Bot interactive command handler.

Commands:
/start     - Welcome message & status overview
/status    - Current position and latest decision
/analyze   - Run immediate analysis (BTC or ETH)
/evaluate  - Show recent realtime-pressure evaluation summary
/mode      - Show fixed dual-mode policy
/report    - Resend latest analysis report
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
from loguru import logger
import json
from io import BytesIO
import base64

# --- 자연어 채팅 핸들러용 상수 ---

_CHAT_SYSTEM = """당신은 독립형 트레이딩봇의 AI 어시스턴트입니다. 
사용자의 자연어 질문에 답하기 위해 적절한 툴을 선택해 데이터를 조회하고, 결과를 한국어로 간결하게 해석해서 답변하세요.

현 시스템 전략 안내:
1. SWING 레인: 바이낸스 선물 헤지 모드 (Long/Short 양방향), 4시간봉 기반 추세 추종.
2. POSITION 레인: 바이낸스 현물 및 업비트 적립식 (Long Only), 일봉/주봉 기반 거시 사이클 투자.
두 레인을 동시에 활용해 하락장 리스크는 헤징하고 상승장 수익은 극대화하는 Dual-Lane 전략을 사용 중입니다.

지침:
- BTC/비트코인 언급 시 BTCUSDT, ETH/이더리움 언급 시 ETHUSDT 조회
- 숫자는 읽기 쉽게 적절히 반올림 (예: $98,432.12 -> $98,432)
- 질문에 필요한 툴만 사용하고 과도한 호출은 피할 것
- 모든 답변과 분석은 반드시 **한국어**로 작성할 것"""

_CHAT_TOOLS = [
    {
        "name": "analyze_market",
        "description": "BTC/ETH 멀티타임프레임 기술적 분석 (RSI, MACD, 볼린저밴드 등)",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string", "description": "예: BTCUSDT"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "get_funding_info",
        "description": "펀딩비·OI·롱숏비율 조회 및 과열/공포 분석",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "get_global_oi",
        "description": "글로벌 미결제약정(OI) 총합 (Binance+Bybit+OKX 합산)",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "get_cvd",
        "description": "CVD(누적 체결 델타) 및 매수/매도 압력 추세",
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
        "description": "기술적 지표 요약 (compact 포맷)",
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
        "description": "가장 최근 AI 트레이딩 결정 리포트",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_current_position",
        "description": "현재 포지션 상태 (진입가, 방향, PnL 등)",
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
            "LightRAG 지식 그래프 조회. 1시간마다 14개 텔레그램 채널에서 수집된 "
            "시장 내러티브·고래 동향·거시적 이벤트/뉴스 등 관계 맥락 검색. "
            "'~를 알려줘' 질문 시 필수."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "검색어 (예: 'BTC whale accumulation', 'ETH ETF')"},
                "mode": {"type": "string", "description": "'local'(단기), 'global'(장기), 'hybrid'(모두). 기본 hybrid"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_narrative",
        "description": (
            "Perplexity API로 실시간 검색. BTC/ETH 가격 움직임의 최신 이유·매크로 이벤트/뉴스 흐름 파악. "
            "RAG로 부족할 때 사용. API 쿼터(200회) 주의."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string", "description": "예: BTCUSDT"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "get_chart_image",
        "description": "기술적 차트 이미지를 생성해 텔레그램으로 전송. lane은 swing 또는 position만 허용",
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
        "description": "Tavily API(저비용)를 사용한 일반 웹 검색. 최신 뉴스 및 일반적인 정보 조사용",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "검색어"}},
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

    async def send_message(self, chat_id: str, text: str):
        """Send a message via the bot API (stateless)."""
        from telegram import Bot
        bot = Bot(token=self.bot_token)
        safe_text = self._sanitize_html_for_telegram(text)
        try:
            await bot.send_message(chat_id=chat_id, text=safe_text, parse_mode='HTML')
        except Exception as e:
            # LLM output may contain unsupported HTML; retry as plain text instead of dropping the update.
            logger.warning(f"Telegram HTML send failed, retrying plain text: {e}")
            plain_text = re.sub(r"<[^>]+>", "", safe_text)
            await bot.send_message(chat_id=chat_id, text=plain_text)

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
            await bot.send_photo(chat_id=chat_id, photo=photo_bytes, caption=plain_caption)

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

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
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
            f"/report  - Resend latest report\n"
            f"/report_off - Pause AI (Save $)\n"
            f"/report_on  - Resume AI\n"
            f"/help    - Show all commands"
        )

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "📚 <b>Available Commands</b>\n\n"
            "/status - Real-time positions & PnL\n"
            "/analyze [BTC|ETH] - Trigger instant AI analysis\n"
            "/evaluate [BTC|ETH] [hours] - Realtime pressure hit-rate summary\n"
            "  Example: /evaluate BTC\n"
            "  Example: /evaluate ETH 72\n"
            "/onchain [BTC|ETH] - Show latest on-chain snapshot and MVRV\n"
            "/chart [BTC|ETH] [swing|position] - Generate premium HD chart\n"
            "/report - Resend latest analysis report\n"
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
            return f"<b>{symbol} Pressure Evaluation</b>\n- 최근 {hours}시간 이벤트가 없습니다."

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
            return f"<b>{symbol} Pressure Evaluation</b>\n- 최근 {hours}시간 내 평가 완료된 신호가 없습니다."

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

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
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
                TradingMode.POSITION,
                False,
                True,
            )

            if result:
                decision = result.get('decision', 'N/A')
                confidence = result.get('confidence', result.get('win_probability_pct', 0))
                await update.message.reply_text(
                    f"Analysis complete for {symbol}\n"
                    f"POSITION: {decision} ({confidence}%)\n"
                    f"Report sent to chat."
                )
            else:
                await update.message.reply_text("Analysis failed. Check logs.")
        except Exception as e:
            logger.error(f"Analyze command error: {e}")
            await update.message.reply_text(f"Error: {e}")

    async def cmd_evaluate(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
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
        await update.message.reply_text(
            "Mode switching is disabled.\n"
            "Policy is fixed:\n"
            "- SWING: futures hedge lane (LONG/SHORT)\n"
            "- POSITION: spot accumulation lane (LONG only)\n"
            "Both lanes run together every cycle."
        )

    async def cmd_report(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            for symbol in settings.trading_symbols:
                report = db.get_latest_report(symbol=symbol)
                if report:
                    fd = report.get('final_decision')
                    if isinstance(fd, str):
                        fd = json.loads(fd)
                    reasoning_value = fd.get('reasoning', 'N/A')
                    if isinstance(reasoning_value, dict):
                        reasoning_text = reasoning_value.get('final_logic') or json.dumps(reasoning_value, ensure_ascii=False)
                    else:
                        reasoning_text = str(reasoning_value)

                    text = (
                        f"Latest {symbol} Report\n"
                        f"Time: {report.get('timestamp', 'N/A')[:19]}\n"
                        f"Decision: {fd.get('decision', 'N/A')}\n"
                        f"Confidence: {fd.get('confidence', fd.get('win_probability_pct', 0))}%\n"
                        f"Entry: {fd.get('entry_price', 'N/A')}\n"
                        f"SL: {fd.get('stop_loss', 'N/A')}\n"
                        f"TP: {fd.get('take_profit', 'N/A')}\n"
                        f"Hold: {fd.get('hold_duration', 'N/A')}\n\n"
                        f"Reasoning: {reasoning_text[:500]}"
                    )
                    if report.get("onchain_context"):
                        text += "\n\nOn-chain context saved: Yes (/onchain or Deep Analysis)"
                    await update.message.reply_text(text)
                else:
                    await update.message.reply_text(f"No report for {symbol}")
        except Exception as e:
            logger.error(f"Report command error: {e}")
            await update.message.reply_text(f"Error: {e}")

    async def cmd_onchain(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
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
            for symbol in symbols:
                snapshot = db.get_latest_onchain_snapshot(symbol, max_age_hours=48)
                if not snapshot:
                    await update.message.reply_text(f"No on-chain snapshot for {symbol}")
                    continue
                found = True
                text = report_generator.format_onchain_message(symbol, snapshot)
                await update.message.reply_text(text, parse_mode='HTML')

            if not found and not args:
                await update.message.reply_text("No on-chain snapshots available.")
        except Exception as e:
            logger.error(f"On-chain command error: {e}")
            await update.message.reply_text(f"Error: {e}")

    async def cmd_report_on(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        from config.local_state import state_manager
        state_manager.set_analysis_enabled(True)
        await update.message.reply_text("🚀 <b>AI 리포트 및 분석이 활성화되었습니다.</b>\n이제 정기적으로 시장을 분석하고 정보를 검증합니다.", parse_mode='HTML')

    async def cmd_report_off(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        from config.local_state import state_manager
        state_manager.set_analysis_enabled(False)
        await update.message.reply_text("🛑 <b>AI 리포트 및 분석이 중단되었습니다.</b>\n데이터 수집은 계속되지만, 고비용 AI 호출은 발생하지 않습니다.", parse_mode='HTML')

    async def cmd_chart(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Direct command to get a chart image: /chart [symbol] [swing|position]"""
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
            result = mcp_tools.get_chart_image(symbol, lane=lane)

            if "chart_base64" in result:
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
                cursor = paper_engine._conn.cursor()
                cursor.execute("SELECT symbol FROM paper_positions WHERE position_id = ?", (pos_id,))
                pos = cursor.fetchone()
                
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
                    mode = settings.trading_mode 
                    if "POSITION" in report.get('final_decision', ''):
                        mode = TradingMode.POSITION

                    detail_text = report_generator.format_detail_message(report, mode)
                    await query.message.reply_text(detail_text, parse_mode='HTML')
                else:
                    await query.message.reply_text("Report no longer available in database.")
            except Exception as e:
                logger.error(f"Callback error: {e}")
                await query.message.reply_text(f"Error fetching detail: {e}")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """자연어 메시지를 Gemini Flash function calling으로 처리.
        분석 리포트(Judge)는 Claude를 유지하고, 대화형 채팅은 저비용 Gemini Flash 사용.
        """
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
                        "job_daily_precision": "00:00",
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
                await update.message.reply_text(fallback or "요청을 이해하지 못했습니다.")
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
            safe_answer = self._sanitize_html_for_telegram(final_answer or "결과를 생성하지 못했습니다.")
            await update.message.reply_text(safe_answer, parse_mode='HTML')

        except Exception as e:
            logger.error(f"handle_message error: {e}")
            await update.message.reply_text(f"오류가 발생했습니다: {type(e).__name__}")

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
        app.add_handler(CommandHandler("onchain", self.cmd_onchain))
        app.add_handler(CommandHandler("report_on", self.cmd_report_on))
        app.add_handler(CommandHandler("report_off", self.cmd_report_off))
        app.add_handler(CommandHandler("chart", self.cmd_chart))
        app.add_handler(CallbackQueryHandler(self.handle_callback))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        logger.info("Telegram bot started (polling)")
        app.run_polling(drop_pending_updates=True, stop_signals=None)


trading_bot = TradingBot()
