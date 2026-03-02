"""Telegram Bot interactive command handler.

Commands:
/start     - Welcome message & status overview
/status    - Current position and latest decision
/analyze   - Run immediate analysis (BTC or ETH)
/mode      - Show or switch trading mode (swing/position)
/report    - Resend latest analysis report
/chart     - Generate premium technical chart
/report_on - Resume automated AI analysis
/report_off - Pause automated AI analysis (Save cost)
/help     - List all commands
"""
from typing import List, Optional, Dict
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from config.settings import settings, TradingMode
from config.database import db
from loguru import logger
import json
import os
from io import BytesIO
import base64

# ── 자연어 채팅 핸들러용 상수 ───────────────────────────────────────────────

_CHAT_SYSTEM = """당신은 크립토 트레이딩봇의 AI 어시스턴트입니다.
사용자의 자연어 질문에 답하기 위해 적절한 툴을 선택해 데이터를 조회하고, 결과를 한국어로 간결하게 해석해서 답변하세요.
- BTC/비트코인 언급 → BTCUSDT, ETH/이더리움 언급 → ETHUSDT
- 숫자는 읽기 쉽게 적절히 반올림
- 질문에 필요한 툴만 사용하고 과도한 호출은 피할 것"""

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
        "description": "글로벌 미결제약정(OI) — Binance+Bybit+OKX 합산",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "get_cvd",
        "description": "CVD(누적 체결 델타) — 매수/매도 압력 추세",
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
        "description": "AI 분석 및 리포트 생성을 켜거나 끕니다 (비용 절약용)",
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
            "시장 내러티브·고래 동향·온체인 이벤트·규제 뉴스 등 관계 맥락 검색. "
            "'왜 오르나/떨어지나' 류 질문에 필수."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "검색어 (예: 'BTC whale accumulation', 'ETH ETF')"},
                "mode": {"type": "string", "description": "'local'(엔티티), 'global'(테마), 'hybrid'(둘 다). 기본 hybrid"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_narrative",
        "description": (
            "Perplexity API로 실시간 웹 검색. BTC/ETH 가격 움직임의 최신 이유·매크로 이벤트·뉴스 흐름 파악. "
            "RAG로 답 안될 때 사용. API 쿼터(200콜/일) 주의."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string", "description": "예: BTCUSDT"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "get_chart_image",
        "description": "기술적 차트 이미지를 생성해 텔레그램으로 전송. 특정 타임프레임(1d, 4h 등) 요청 가능.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "예: BTCUSDT"},
                "timeframe": {"type": "string", "description": "옵션: 1d(long-term), 4h(swing). 기본은 현재 모드"}
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "search_web",
        "description": "Tavily API(저비용)를 사용한 일반 웹 검색. 최신 뉴스나 일반적인 정보 조색용.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "검색어"}},
            "required": ["query"],
        },
    },
]


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
            f"/status  - Current position & equity\n"
            f"/analyze - Live analysis (BTC/ETH)\n"
            f"/chart   - Generate HD technical chart\n"
            f"/mode    - Switch Swing/Position\n"
            f"/report  - Resend latest report\n"
            f"/report_off - Pause AI (Save $)\n"
            f"/report_on  - Resume AI\n"
            f"/help    - Show all commands"
        )

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🔍 <b>Available Commands</b>\n\n"
            "/status - Real-time positions & PnL\n"
            "/analyze [BTC|ETH] - Trigger instant AI analysis\n"
            "/chart [BTC|ETH] [4h|1d] - Generate premium HD chart\n"
            "/report - Resend latest analysis report\n"
            "/mode [swing|position] - Switch system timeframe\n"
            "/report_off - PAUSE AI automation (Save cost)\n"
            "/report_on  - RESUME AI automation\n"
            "/help - Show this message\n\n"
            "💡 <i>Tip: You can also talk to me in natural language!</i>",
            parse_mode='HTML'
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
                from executors.trade_executor import trade_executor
                positions = paper_engine.get_open_positions()
                
                # [BUG-FIX] 거래소별 마진/PnL 분리 추적 (이전: 모두 합산해 Binance Equity에 표시)
                binance_unrealized = 0.0
                binance_margin = 0.0
                upbit_unrealized = 0.0
                upbit_margin = 0.0

                lines.append("📌 <b>Active Positions</b>")
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

                        pnl_icon = "🟢" if pnl >= 0 else "🔴"
                        lines.append(
                            f"• {exchange.upper()} | <b>{symbol}</b>: {side} {leverage}x\n"
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
                    lines.append(f"• {ex.upper()} Balance: <code>${bal:,.2f}</code>")
                    lines.append(f"• {ex.upper()} <b>Total Equity: ${equity:,.2f}</b>")
                    lines.append(f"  (Margin: ${ex_margin:.2f} | UnPnL: ${ex_unrealized:+.2f})")
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

            from telegram import InlineKeyboardButton, InlineKeyboardMarkup
            keyboard = []
            if is_paper and 'positions' in locals() and positions:
                for pos in positions:
                    pos_id = pos.get('position_id')
                    sym = pos.get('symbol')
                    if pos_id:
                        keyboard.append([
                            InlineKeyboardButton(f"🔒 SL 본절", callback_data=f"pos_sl_{pos_id}"),
                            InlineKeyboardButton(f"✂️ {sym} 50%", callback_data=f"pos_half_{pos_id}"),
                            InlineKeyboardButton(f"🛑 {sym} 100%", callback_data=f"pos_close_{pos_id}")
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

    async def cmd_report_on(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        from config.local_state import state_manager
        state_manager.set_analysis_enabled(True)
        await update.message.reply_text("✅ <b>AI 리포트 및 분석이 활성화되었습니다.</b>\n이제 정기적으로 시장을 분석하고 속보를 검증합니다.", parse_mode='HTML')

    async def cmd_report_off(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        from config.local_state import state_manager
        state_manager.set_analysis_enabled(False)
        await update.message.reply_text("🛑 <b>AI 리포트 및 분석이 중단되었습니다.</b>\n데이터 수집은 계속되지만, 고비용 AI 호출은 발생하지 않습니다.", parse_mode='HTML')

    async def cmd_chart(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Direct command to get a chart image: /chart [symbol] [timeframe]"""
        args = context.args
        if not args:
            await update.message.reply_text("Usage: /chart BTC|ETH [1d|4h]")
            return

        base = args[0].upper()
        if base not in settings.trading_symbols_base:
            await update.message.reply_text(f"Invalid symbol. Use: {'|'.join(settings.trading_symbols_base)}")
            return

        symbol = base + "USDT"
        timeframe = args[1].lower() if len(args) > 1 else None

        await update.message.reply_text(f"Generating premium {timeframe or 'auto'} chart for {symbol}...")

        try:
            from mcp_server.tools import mcp_tools
            result = mcp_tools.get_chart_image(symbol, timeframe=timeframe)

            if "chart_base64" in result:
                chart_bytes = base64.b64decode(result["chart_base64"])
                buf = BytesIO(chart_bytes)
                buf.name = f"{symbol}_{timeframe or 'current'}.png"
                
                # Send photo
                caption = f"📊 <b>{symbol} {result.get('mode', '').upper()} CHART</b>\n"
                if timeframe:
                    caption += f"Timeframe: <code>{timeframe}</code> (Requested)\n"
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
                    await query.edit_message_text(f"{query.message.text}\n\n❌ Position not found or closed.")
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
                    # Determine mode - default to swing if not found (or store in callback_data)
                    mode = TradingMode.SWING 
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
            from agents.claude_client import claude_client as ai_client
            from google.genai import types as gtypes
            from mcp_server.tools import mcp_tools

            gemini = ai_client._gemini_client

            tool_fn_map = {
                "analyze_market":            lambda a: mcp_tools.get_market_analysis(a["symbol"]),
                "get_funding_info":          lambda a: mcp_tools.get_funding_data(a["symbol"]),
                "get_global_oi":             lambda a: mcp_tools.get_global_oi(a["symbol"]),
                "get_cvd":                   lambda a: mcp_tools.get_cvd_data(a["symbol"], limit=a.get("minutes", 240)),
                "get_indicator_summary":     lambda a: mcp_tools.get_indicator_summary(a["symbol"]),
                "get_news_summary":          lambda a: mcp_tools.get_telegram_summary(a.get("hours", 4)),
                "get_latest_trading_report": lambda a: mcp_tools.get_latest_report(),
                "get_current_position":      lambda a: mcp_tools.get_position_status(a["symbol"]),
                "get_trading_mode":          lambda a: {
                    "mode": settings.trading_mode.value,
                    "candle_limit": settings.candle_limit,
                    "chart_enabled": settings.should_use_chart,
                    "analysis_interval_hours": settings.analysis_interval_hours,
                },
                "switch_trading_mode":       lambda a: mcp_tools.switch_mode(a["mode"]),
                "toggle_ai_analysis":        lambda a: (state_manager.set_analysis_enabled(a["enabled"]), {"status": f"AI 분석 {'활성화' if a['enabled'] else '비활성화'} 완료"})[1],
                "get_feedback_history":      lambda a: mcp_tools.get_feedback_history(a.get("limit", 5)),
                "query_knowledge_graph":     lambda a: mcp_tools.query_rag(a["query"], mode=a.get("mode", "hybrid")),
                "search_narrative":          lambda a: mcp_tools.search_market_narrative(a["symbol"]),
                "get_chart_image":           lambda a: mcp_tools.get_chart_image(a["symbol"], timeframe=a.get("timeframe")),
                "search_web":                lambda a: mcp_tools.search_web(a["query"]),
            }

            # _CHAT_TOOLS (Anthropic 포맷) → Gemini FunctionDeclaration 변환
            _type_map = {
                "object": "OBJECT", "string": "STRING",
                "integer": "INTEGER", "number": "NUMBER", "boolean": "BOOLEAN",
            }

            def _to_gemini_schema(js: dict):
                props = {
                    k: gtypes.Schema(
                        type=_type_map.get(v.get("type", "string"), "STRING"),
                        description=v.get("description", ""),
                    )
                    for k, v in js.get("properties", {}).items()
                }
                return gtypes.Schema(
                    type=_type_map.get(js.get("type", "object"), "OBJECT"),
                    properties=props or None,
                    required=js.get("required") or None,
                )

            fn_decls = [
                gtypes.FunctionDeclaration(
                    name=t["name"],
                    description=t["description"],
                    parameters=_to_gemini_schema(t["input_schema"])
                    if t["input_schema"].get("properties")
                    else None,
                )
                for t in _CHAT_TOOLS
            ]

            config = gtypes.GenerateContentConfig(
                system_instruction=_CHAT_SYSTEM,
                tools=[gtypes.Tool(function_declarations=fn_decls)],
                automatic_function_calling=gtypes.AutomaticFunctionCallingConfig(disable=True),
                max_output_tokens=2000,
                thinking_config=gtypes.ThinkingConfig(thinking_level="LOW"),
            )

            contents = [gtypes.Content(role="user", parts=[gtypes.Part.from_text(text=user_text)])]

            for _ in range(5):  # 최대 5회 agentic 루프
                response = await asyncio.to_thread(
                    gemini.models.generate_content,
                    model=settings.MODEL_CHAT,
                    contents=contents,
                    config=config,
                )

                if not response.candidates:
                    break

                model_parts = response.candidates[0].content.parts
                fn_call_parts = [p for p in model_parts if p.function_call is not None]

                if not fn_call_parts:
                    # 최종 텍스트 답변
                    text = "".join(p.text for p in model_parts if p.text)
                    await update.message.reply_text(text or "결과가 없습니다.")
                    return

                # 모델 응답 히스토리에 추가
                contents.append(gtypes.Content(role="model", parts=model_parts))

                # 툴 실행 후 function_response 수집
                fn_response_parts = []
                for part in fn_call_parts:
                    fn_name = part.function_call.name
                    fn_args = dict(part.function_call.args)
                    fn = tool_fn_map.get(fn_name)
                    if fn is None:
                        result = {"error": f"Unknown tool: {fn_name}"}
                    else:
                        result = await asyncio.to_thread(fn, fn_args)

                    # 차트 이미지 특수 처리: base64 → reply_photo()
                    if fn_name == "get_chart_image" and "chart_base64" in result:
                        try:
                            chart_bytes = base64.b64decode(result["chart_base64"])
                            buf = BytesIO(chart_bytes)
                            buf.name = f"{fn_args.get('symbol', 'chart')}.png"
                            await update.message.reply_photo(photo=buf)
                            result = {
                                "status": "차트 전송 완료",
                                "symbol": fn_args.get("symbol", ""),
                                "size_bytes": result.get("size_bytes", 0),
                            }
                        except Exception as img_err:
                            logger.error(f"Chart send error: {img_err}")
                            result = {"error": f"차트 전송 실패: {img_err}"}

                    result_str = json.dumps(result, ensure_ascii=False, default=str)[:3000]
                    fn_response_parts.append(
                        gtypes.Part(
                            function_response=gtypes.FunctionResponse(
                                name=fn_name,
                                response={"result": result_str},
                            )
                        )
                    )

                contents.append(gtypes.Content(role="user", parts=fn_response_parts))

            await update.message.reply_text("응답을 처리하지 못했습니다.")

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
        app.add_handler(CommandHandler("mode", self.cmd_mode))
        app.add_handler(CommandHandler("report", self.cmd_report))
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
