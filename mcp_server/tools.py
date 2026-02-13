from typing import Dict
from config.database import db
from config.settings import settings, TradingMode
from processors.math_engine import math_engine
from processors.chart_generator import chart_generator
from loguru import logger
import os


class MCPTools:

    def get_market_analysis(self, symbol: str) -> Dict:
        """Get mode-specific market analysis. Raw data only."""
        try:
            df = db.get_latest_market_data(symbol, limit=settings.candle_limit)
            if df.empty:
                return {"error": "No market data available"}

            mode = settings.trading_mode
            analysis = math_engine.analyze_market(df, mode)
            compact = math_engine.format_compact(analysis)
            return {
                "symbol": symbol,
                "mode": mode.value,
                "compact_analysis": compact,
                "data_points": len(df)
            }
        except Exception as e:
            logger.error(f"Market analysis error: {e}")
            return {"error": str(e)}

    def get_telegram_summary(self, hours: int = 4) -> Dict:
        try:
            messages = db.get_recent_telegram_messages(hours=hours)
            if not messages:
                return {"summary": "No recent news", "message_count": 0}

            summary_text = "\n".join([
                f"[{msg['channel']}] {msg['text'][:200]}"
                for msg in messages[:15]
            ])
            return {"summary": summary_text, "message_count": len(messages)}
        except Exception as e:
            logger.error(f"Telegram summary error: {e}")
            return {"error": str(e)}

    def get_funding_data(self, symbol: str) -> Dict:
        try:
            response = db.client.table("funding_data")\
                .select("*")\
                .eq("symbol", symbol)\
                .order("timestamp", desc=True)\
                .limit(1)\
                .execute()

            if response.data:
                raw = response.data[0]
                context = math_engine.analyze_funding_context(raw)
                return {"raw": raw, "context": context}
            return {"error": "No funding data available"}
        except Exception as e:
            logger.error(f"Funding data error: {e}")
            return {"error": str(e)}

    def get_latest_report(self) -> Dict:
        try:
            report = db.get_latest_report()
            return report if report else {"message": "No reports available"}
        except Exception as e:
            logger.error(f"Report retrieval error: {e}")
            return {"error": str(e)}

    def get_position_status(self, symbol: str) -> Dict:
        try:
            position = db.get_position_status(symbol)
            return position if position else {"message": "No active position"}
        except Exception as e:
            logger.error(f"Position status error: {e}")
            return {"error": str(e)}

    def get_chart_image(self, symbol: str) -> Dict:
        try:
            df = db.get_latest_market_data(symbol, limit=settings.candle_limit)
            if df.empty:
                return {"error": "No market data available"}

            mode = settings.trading_mode
            analysis = math_engine.analyze_market(df, mode)
            chart_bytes = chart_generator.generate_chart(df, analysis, symbol, mode)

            if chart_bytes:
                b64 = chart_generator.chart_to_base64(chart_bytes)
                return {
                    "symbol": symbol,
                    "mode": mode.value,
                    "chart_base64": b64,
                    "size_bytes": len(chart_bytes)
                }
            return {"error": "Chart generation failed"}
        except Exception as e:
            logger.error(f"Chart image error: {e}")
            return {"error": str(e)}

    def get_indicator_summary(self, symbol: str) -> Dict:
        try:
            df = db.get_latest_market_data(symbol, limit=settings.candle_limit)
            if df.empty:
                return {"error": "No market data available"}

            mode = settings.trading_mode
            analysis = math_engine.analyze_market(df, mode)
            compact = math_engine.format_compact(analysis)
            return {"symbol": symbol, "mode": mode.value, "summary": compact}
        except Exception as e:
            logger.error(f"Indicator summary error: {e}")
            return {"error": str(e)}

    def switch_mode(self, mode: str) -> Dict:
        """Switch trading mode at runtime by updating env var."""
        try:
            mode_lower = mode.lower().strip()
            if mode_lower not in ('swing', 'scalp'):
                return {"error": f"Invalid mode '{mode}'. Use 'swing' or 'scalp'."}

            # Update environment variable (affects settings on next access)
            os.environ['TRADING_MODE'] = mode_lower
            # Clear settings cache to pick up new mode
            from config.settings import get_settings
            get_settings.cache_clear()

            return {
                "success": True,
                "new_mode": mode_lower,
                "candle_limit": 4320 if mode_lower == 'swing' else 720,
                "chart_for_vlm": mode_lower == 'swing',
                "message": f"Switched to {mode_lower.upper()} mode"
            }
        except Exception as e:
            logger.error(f"Mode switch error: {e}")
            return {"error": str(e)}


mcp_tools = MCPTools()
