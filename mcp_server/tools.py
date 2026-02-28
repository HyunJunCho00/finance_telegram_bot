from typing import Dict
from config.database import db
from config.settings import settings, TradingMode
from processors.math_engine import math_engine
from processors.chart_generator import chart_generator
from loguru import logger
import os
import json


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

    def get_global_oi(self, symbol: str) -> Dict:
        """Get Global Open Interest breakdown from 3 exchanges."""
        try:
            response = db.client.table("funding_data")\
                .select("timestamp,symbol,open_interest_value,oi_binance,oi_bybit,oi_okx")\
                .eq("symbol", symbol)\
                .order("timestamp", desc=True)\
                .limit(1)\
                .execute()

            if response.data:
                raw = response.data[0]
                return {
                    "symbol": symbol,
                    "global_oi_usd": raw.get("open_interest_value", 0),
                    "binance": raw.get("oi_binance", 0),
                    "bybit": raw.get("oi_bybit", 0),
                    "okx": raw.get("oi_okx", 0),
                    "timestamp": raw.get("timestamp", ""),
                }
            return {"error": "No OI data available"}
        except Exception as e:
            logger.error(f"Global OI error: {e}")
            return {"error": str(e)}

    def get_cvd_data(self, symbol: str, limit: int = 240) -> Dict:
        """Get CVD (Cumulative Volume Delta) data."""
        try:
            cvd_df = db.get_cvd_data(symbol, limit=limit)
            if cvd_df.empty:
                return {"error": "No CVD data available"}

            total_delta = float(cvd_df['volume_delta'].sum())
            recent_delta = float(cvd_df.tail(60)['volume_delta'].sum())
            cvd_final = float(cvd_df['cvd'].iloc[-1])

            return {
                "symbol": symbol,
                "period_minutes": limit,
                "total_delta": round(total_delta, 4),
                "total_direction": "BUYING" if total_delta > 0 else "SELLING",
                "recent_1h_delta": round(recent_delta, 4),
                "recent_direction": "BUYING" if recent_delta > 0 else "SELLING",
                "cvd_cumulative": round(cvd_final, 4),
                "data_points": len(cvd_df),
            }
        except Exception as e:
            logger.error(f"CVD data error: {e}")
            return {"error": str(e)}

    def search_market_narrative(self, symbol: str) -> Dict:
        """Search market narrative via Perplexity API."""
        try:
            from collectors.perplexity_collector import perplexity_collector
            coin = "BTC" if "BTC" in symbol.upper() else "ETH"
            narrative = perplexity_collector.search_market_narrative(coin)
            formatted = perplexity_collector.format_for_agents(narrative)
            return {
                "symbol": symbol,
                "narrative": formatted,
                "sentiment": narrative.get("sentiment", "neutral"),
                "raw": {k: v for k, v in narrative.items() if k != "sources"},
            }
        except Exception as e:
            logger.error(f"Perplexity search error: {e}")
            return {"error": str(e)}

    def query_rag(self, query: str, mode: str = "hybrid") -> Dict:
        """Query LightRAG knowledge graph + vector search."""
        try:
            from processors.light_rag import light_rag
            result = light_rag.query(query, mode=mode)
            formatted = light_rag.format_context_for_agents(result, max_length=2000)
            stats = light_rag.get_stats()
            return {
                "query": query,
                "mode": mode,
                "context": formatted,
                "entities_found": result.get("entities_found", []),
                "stats": stats,
            }
        except Exception as e:
            logger.error(f"RAG query error: {e}")
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
        """사용자 질의용: 모든 지표 포함 (PSAR, KC, Aroon, HMA 등).
        분석 파이프라인에는 format_compact() 화이트리스트만 전달됨."""
        try:
            df = db.get_latest_market_data(symbol, limit=settings.candle_limit)
            if df.empty:
                return {"error": "No market data available"}

            mode = settings.trading_mode
            # 모드별 주요 타임프레임 원시 지표 반환 (화이트리스트 미적용)
            primary_tf = '4h' if mode.value == 'swing' else '1d'
            tf_df = math_engine.resample_to_timeframe(df, primary_tf)
            indicators = math_engine.calculate_indicators_for_timeframe(tf_df)

            return {
                "symbol": symbol,
                "timeframe": primary_tf,
                "mode": mode.value,
                "indicators": indicators,  # PSAR, KC, Aroon, HMA 포함 전체
            }
        except Exception as e:
            logger.error(f"Indicator summary error: {e}")
            return {"error": str(e)}

    def switch_mode(self, mode: str) -> Dict:
        """Switch trading mode at runtime by updating env var."""
        try:
            mode_lower = mode.lower().strip()
            if mode_lower not in ('swing', 'position'):
                return {"error": f"Invalid mode '{mode}'. Use 'swing', or 'position'."}

            os.environ['TRADING_MODE'] = mode_lower
            from config.settings import get_settings
            get_settings.cache_clear()

            mode_cfg = {
                "swing": {"candle_limit": settings.SWING_CANDLE_LIMIT, "chart_for_vlm": settings.USE_CHART_IMAGES},
                "position": {"candle_limit": settings.POSITION_CANDLE_LIMIT, "chart_for_vlm": settings.USE_CHART_IMAGES},
            }

            return {
                "success": True,
                "new_mode": mode_lower,
                "candle_limit": mode_cfg[mode_lower]["candle_limit"],
                "chart_for_vlm": mode_cfg[mode_lower]["chart_for_vlm"],
                "message": f"Switched to {mode_lower.upper()} mode"
            }
        except Exception as e:
            logger.error(f"Mode switch error: {e}")
            return {"error": str(e)}

    def get_feedback_history(self, limit: int = 5) -> Dict:
        """Get past trading mistakes and lessons learned."""
        try:
            feedback = db.get_feedback_history(limit=limit)
            if not feedback:
                return {"message": "No feedback history yet", "entries": []}
            return {
                "entries": [
                    {
                        "symbol": f.get("symbol"),
                        "predicted": f.get("predicted_direction"),
                        "actual": f.get("actual_direction"),
                        "change_pct": f.get("actual_change_pct"),
                        "mistake": f.get("mistake_summary", "")[:200],
                        "lesson": f.get("lesson_learned", "")[:200],
                        "date": f.get("created_at", "")[:19],
                    }
                    for f in feedback
                ],
                "count": len(feedback),
            }
        except Exception as e:
            logger.error(f"Feedback history error: {e}")
            return {"error": str(e)}


mcp_tools = MCPTools()
