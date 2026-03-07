from typing import Dict, Optional, List
from config.database import db
from config.settings import get_settings, TradingMode
from processors.math_engine import math_engine
from processors.chart_generator import chart_generator
from processors.gcs_parquet import gcs_parquet_store
from processors.cvd_normalizer import build_price_timeline, merge_cvd_sources, normalize_cvd_to_usd
from loguru import logger
import json
import pandas as pd
from math import nan # for serialization safety
from agents.market_monitor_agent import market_monitor_agent


class MCPTools:
    def _load_chart_context(self, symbol: str, mode: TradingMode) -> Optional[Dict]:
        settings = get_settings()
        limit = settings.SWING_CANDLE_LIMIT if mode == TradingMode.SWING else settings.POSITION_CANDLE_LIMIT

        df = db.get_latest_market_data(symbol, limit=limit)
        if df.empty:
            return None

        df_4h, df_1d, df_1w = None, None, None
        cvd_df, funding_df, liq_df = None, None, None

        if gcs_parquet_store.enabled:
            try:
                m_back = settings.history_lookback_months_for_mode(mode)
                if mode == TradingMode.SWING:
                    df_4h = gcs_parquet_store.load_ohlcv("4h", symbol, months_back=m_back)
                df_1d = gcs_parquet_store.load_ohlcv("1d", symbol, months_back=m_back)
                if mode == TradingMode.POSITION:
                    df_1w = gcs_parquet_store.load_ohlcv("1w", symbol, months_back=m_back)

                m_back_timeseries = m_back
                db_limit = 45000
                now_utc = pd.Timestamp.now(tz='UTC')

                cvd_hist = gcs_parquet_store.load_timeseries("cvd", symbol, months_back=m_back_timeseries)
                if not cvd_hist.empty:
                    cvd_hist['timestamp'] = pd.to_datetime(cvd_hist['timestamp'], utc=True, errors='coerce')
                    month_start = now_utc.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                    cvd_hist = cvd_hist[cvd_hist['timestamp'] < month_start].copy()

                since_cvd = cvd_hist['timestamp'].max() if not cvd_hist.empty else None
                bridge_cvd = db.get_cvd_data(symbol, limit=db_limit, since=since_cvd) if since_cvd is not None else pd.DataFrame()
                cvd_recent = db.get_cvd_data(symbol, limit=db_limit)
                merged_cvd = merge_cvd_sources(cvd_hist, bridge_cvd, cvd_recent)

                fallback_px = float(df['close'].iloc[-1]) if not df.empty else 60000.0
                price_timeline = build_price_timeline(df_1m=df, df_1d=df_1d, fallback_price=fallback_px)
                cvd_df = normalize_cvd_to_usd(merged_cvd, price_timeline, fallback_price=fallback_px)

                fnd_hist_dfs = []
                for m in range(1, m_back + 1):
                    month_str = (now_utc - pd.DateOffset(months=m)).strftime("%Y-%m")
                    path = f"funding/{symbol}/{month_str}.parquet"
                    df_part = gcs_parquet_store._download_parquet(path)
                    if df_part is not None:
                        fnd_hist_dfs.append(df_part)

                fnd_hist = pd.concat(fnd_hist_dfs, ignore_index=True) if fnd_hist_dfs else pd.DataFrame()
                if not fnd_hist.empty:
                    fnd_hist.rename(columns={'open_interest_value': 'open_interest'}, inplace=True)
                    fnd_hist = fnd_hist.loc[:, ~fnd_hist.columns.duplicated()].reset_index(drop=True)
                    fnd_hist['timestamp'] = pd.to_datetime(fnd_hist['timestamp'], utc=True)

                    fnd_recent = db.get_funding_history(symbol, limit=db_limit)
                    if fnd_recent is not None and not fnd_recent.empty:
                        fnd_recent.rename(columns={'open_interest_value': 'open_interest'}, inplace=True)
                        fnd_recent = fnd_recent.loc[:, ~fnd_recent.columns.duplicated()].reset_index(drop=True)
                        fnd_recent['timestamp'] = pd.to_datetime(fnd_recent['timestamp'], utc=True)

                        funding_df = pd.concat([fnd_hist, fnd_recent], ignore_index=True).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                        funding_df = funding_df.reset_index(drop=True)
                    else:
                        funding_df = fnd_hist
                else:
                    funding_df = db.get_funding_history(symbol, limit=db_limit)
                    if funding_df is not None and not funding_df.empty:
                        funding_df.rename(columns={'open_interest_value': 'open_interest'}, inplace=True)
                        funding_df['timestamp'] = pd.to_datetime(funding_df['timestamp'], utc=True)

                liq_df = db.get_liquidation_data(symbol, limit=db_limit)
            except Exception as e:
                import traceback
                logger.warning(f"GCS load for chart tool skipped: {e}\n{traceback.format_exc()}")

        fixed_timeframe = "4h" if mode == TradingMode.SWING else "1d"
        analysis = math_engine.analyze_market(df, mode, df_4h=df_4h, df_1d=df_1d, df_1w=df_1w, timeframe=fixed_timeframe)
        return {
            "df": df,
            "analysis": analysis,
            "df_4h": df_4h,
            "df_1d": df_1d,
            "df_1w": df_1w,
            "cvd_df": cvd_df,
            "funding_df": funding_df,
            "liq_df": liq_df,
            "fixed_timeframe": fixed_timeframe,
        }

    def get_market_analysis(self, symbol: str) -> Dict:
        """Get mode-specific market analysis. Raw data only."""
        try:
            settings = get_settings()
            df = db.get_latest_market_data(symbol, limit=settings.candle_limit)
            if df.empty:
                return {"error": "No market data available"}

            mode = settings.trading_mode
            
            # Load higher timeframe data from GCS for deeper indicator history
            df_4h, df_1d, df_1w = None, None, None
            cvd_df, funding_df, liq_df = None, None, None

            if gcs_parquet_store.enabled:
                try:
                    m_back = settings.history_lookback_months_for_mode(mode)
                    if mode == TradingMode.SWING:
                        df_4h = gcs_parquet_store.load_ohlcv("4h", symbol, months_back=m_back)
                    df_1d = gcs_parquet_store.load_ohlcv("1d", symbol, months_back=m_back)
                    if mode == TradingMode.POSITION:
                        df_1w = gcs_parquet_store.load_ohlcv("1w", symbol, months_back=m_back)

                    # Also load CVD/Funding history for a complete analysis
                    cvd_hist = gcs_parquet_store.load_timeseries("cvd", symbol, months_back=m_back)
                    fnd_hist = gcs_parquet_store.load_timeseries("funding", symbol, months_back=m_back)

                    # Merge with recent DB data
                    cvd_recent = db.get_cvd_data(symbol, limit=45000)
                    if not cvd_hist.empty:
                        cvd_df = pd.concat([cvd_hist, cvd_recent]).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                    else:
                        cvd_df = cvd_recent

                    fnd_recent = db.get_funding_history(symbol, limit=45000)
                    if not fnd_hist.empty:
                        funding_df = pd.concat([fnd_hist, fnd_recent]).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                    else:
                        funding_df = fnd_recent
                except Exception as e:
                    logger.warning(f"GCS load for analysis tool skipped: {e}")

            analysis = math_engine.analyze_market(df, mode, df_4h=df_4h, df_1d=df_1d, df_1w=df_1w)
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
        """Search market narrative via Perplexity API. [Report Grade]"""
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

    def search_web(self, query: str) -> Dict:
        """Search the web for general information using Tavily (Low Cost)."""
        try:
            from collectors.tavily_collector import tavily_collector
            result = tavily_collector.search(query, search_depth="basic", max_results=5)
            if result.get("status") == "ok":
                # Concise formatting for conversational AI
                formatted = "\n".join([
                    f"[{i+1}] {r.get('title', 'N/A')}: {r.get('content', '')[:200]}... ({r.get('url')})"
                    for i, r in enumerate(result.get("results", []))
                ])
                return {
                    "query": query,
                    "results": formatted,
                    "answer": result.get("answer", ""),
                    "status": "ok"
                }
            return {"error": "Search failed"}
        except Exception as e:
            logger.error(f"Tavily web search error: {e}")
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

    def get_chart_image(self, symbol: str, lane: Optional[str] = None) -> Dict:
        try:
            lane_norm = (lane or "swing").lower().strip()
            if lane_norm not in ("swing", "position"):
                return {"error": "Invalid lane. Use 'swing' or 'position'."}

            mode = TradingMode.SWING if lane_norm == "swing" else TradingMode.POSITION
            ctx = self._load_chart_context(symbol, mode)
            if not ctx:
                return {"error": "No market data available"}
            chart_bytes = chart_generator.generate_chart(
                ctx["df"], ctx["analysis"], symbol, mode,
                df_4h=ctx["df_4h"],
                df_1d=ctx["df_1d"], df_1w=ctx["df_1w"],
                cvd_df=ctx["cvd_df"],
                funding_df=ctx["funding_df"],
                liquidation_df=ctx["liq_df"],
                timeframe=ctx["fixed_timeframe"],
            )

            if chart_bytes:
                b64 = chart_generator.chart_to_base64(chart_bytes)
                return {
                    "symbol": symbol,
                    "mode": mode.value,
                    "lane": lane_norm,
                    "timeframe": ctx["fixed_timeframe"],
                    "chart_base64": b64,
                    "size_bytes": len(chart_bytes)
                }
            return {"error": "Chart generation failed"}
        except Exception as e:
            logger.error(f"Chart image error: {e}")
            return {"error": str(e)}

    def get_chart_images(self, symbol: str, lane: Optional[str] = None) -> Dict:
        try:
            lane_norm = (lane or "swing").lower().strip()
            if lane_norm not in ("swing", "position"):
                return {"error": "Invalid lane. Use 'swing' or 'position'."}

            mode = TradingMode.SWING if lane_norm == "swing" else TradingMode.POSITION
            ctx = self._load_chart_context(symbol, mode)
            if not ctx:
                return {"error": "No market data available"}

            timeframes = ["1d", "4h"] if mode == TradingMode.SWING else ["1w", "1d"]
            charts = []
            for timeframe in timeframes:
                chart_bytes = chart_generator.generate_chart(
                    ctx["df"], ctx["analysis"], symbol, mode,
                    df_4h=ctx["df_4h"],
                    df_1d=ctx["df_1d"], df_1w=ctx["df_1w"],
                    cvd_df=ctx["cvd_df"],
                    funding_df=ctx["funding_df"],
                    liquidation_df=ctx["liq_df"],
                    timeframe=timeframe,
                    prefer_lane=False,
                )
                if not chart_bytes:
                    continue
                charts.append({
                    "timeframe": timeframe,
                    "chart_base64": chart_generator.chart_to_base64(chart_bytes),
                    "size_bytes": len(chart_bytes),
                })

            if not charts:
                return {"error": "Chart generation failed"}

            return {
                "symbol": symbol,
                "mode": mode.value,
                "lane": lane_norm,
                "charts": charts,
            }
        except Exception as e:
            logger.error(f"Chart images error: {e}")
            return {"error": str(e)}

    def get_indicator_summary(self, symbol: str) -> Dict:
        """Return compact indicator bundle (PSAR, KC, Aroon, HMA, etc.)."""
        try:
            settings = get_settings()
            df = db.get_latest_market_data(symbol, limit=settings.candle_limit)
            if df.empty:
                return {"error": "No market data available"}

            mode = settings.trading_mode
            # Primary timeframe aligned with mode policy.
            primary_tf = '4h' if mode.value == 'swing' else '1d'
            tf_df = math_engine.resample_to_timeframe(df, primary_tf)
            indicators = math_engine.calculate_indicators_for_timeframe(tf_df)

            return {
                "symbol": symbol,
                "timeframe": primary_tf,
                "mode": mode.value,
                "indicators": indicators,
            }
        except Exception as e:
            logger.error(f"Indicator summary error: {e}")
            return {"error": str(e)}

    def switch_mode(self, mode: str) -> Dict:
        """Mode switching is intentionally disabled (dual-mode policy)."""
        return {
            "success": False,
            "disabled": True,
            "message": (
                "Mode switching is disabled. Policy is fixed: "
                "SWING (futures long/short) + POSITION (spot long-only) run together."
            ),
        }

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
    def get_market_summary(self, symbol: str) -> Dict:
        """Get a concise market summary (Sentiment, Anomalies, Outlook) via Free-First AI."""
        try:
            # 1. Fetch raw data similar to scheduler job
            from collectors.price_collector import collector
            from collectors.funding_collector import funding_collector
            from collectors.volatility_monitor import volatility_monitor
            
            indicators = {
                "btc_price": collector.get_last_price(symbol),
                "funding_rate": funding_collector.get_last_rate(symbol),
                "volatility": volatility_monitor.get_current_volatility()
            }
            
            # 2. Use MarketMonitorAgent (Free-First Routing)
            summary = market_monitor_agent.summarize_current_status(indicators)
            
            return {
                "symbol": symbol,
                "summary": summary,
                "indicators_snapshot": indicators,
                "provider": "Free-First AI (Groq/WorkersAI)"
            }
        except Exception as e:
            logger.error(f"Market summary tool error: {e}")
            return {"error": str(e)}

mcp_tools = MCPTools()
