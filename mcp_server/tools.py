from typing import Dict, Optional, List
from config.database import db
from config.settings import get_settings, TradingMode
from processors.math_engine import math_engine
from processors.chart_generator import chart_generator
from processors.gcs_parquet import gcs_parquet_store
from loguru import logger
import os
import json
import pandas as pd
from math import nan # for serialization safety
from agents.market_monitor_agent import market_monitor_agent


class MCPTools:

    def get_market_analysis(self, symbol: str) -> Dict:
        """Get mode-specific market analysis. Raw data only."""
        try:
            settings = get_settings()
            df = db.get_latest_market_data(symbol, limit=settings.candle_limit)
            if df.empty:
                return {"error": "No market data available"}

            mode = settings.trading_mode
            
            # Load higher timeframe data from GCS for deeper indicator history
            df_1d, df_1w = None, None
            cvd_df, funding_df, liq_df = None, None, None

            if gcs_parquet_store.enabled:
                try:
                    # Fixed lookback: SWING=12 months, POSITION=60 months (5 years)
                    m_back = 60 if mode == TradingMode.POSITION else 12
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

            analysis = math_engine.analyze_market(df, mode, df_1d=df_1d, df_1w=df_1w)
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

    def get_chart_image(self, symbol: str, timeframe: Optional[str] = None) -> Dict:
        try:
            # [FIX] Fetch latest settings dynamically to reflect mode changes immediately
            settings = get_settings()
            # Map common timeframe aliases to TradingMode
            mode = settings.trading_mode
            limit = settings.candle_limit
            
            # Data Volume is determined entirely by the global `mode` parameter.
            # Display TF (timeframe) only controls resampling and visual structure, not the mode.
            if mode == TradingMode.POSITION:
                limit = 45000  # Bridge GCS gap for macro analysis
            else:
                limit = 10000  # Default ~7-10 days for swing

            df = db.get_latest_market_data(symbol, limit=limit)
            if df.empty:
                return {"error": "No market data available"}

            # Load higher timeframe data from GCS for deeper indicator history
            df_1d, df_1w = None, None
            cvd_df, funding_df, liq_df = None, None, None
            
            if gcs_parquet_store.enabled:
                try:
                    # Fixed lookback: SWING=12 months, POSITION=60 months (5 years)
                    m_back = 60 if mode == TradingMode.POSITION else 12
                    df_1d = gcs_parquet_store.load_ohlcv("1d", symbol, months_back=m_back)
                    if timeframe and timeframe.lower() in ('1w', 'w') or mode == TradingMode.POSITION:
                        df_1w = gcs_parquet_store.load_ohlcv("1w", symbol, months_back=m_back)

                    m_back_timeseries = m_back

                    # Bridge GCS gap: 45,000 rows covers ~31 days of 1m data
                    db_limit = 45000

                    # ── 1. CVD ──
                    # Load historical months from GCS cache (past months only, always skips current)
                    cvd_hist_dfs = []
                    now_utc = pd.Timestamp.now(tz='UTC')
                    current_month_str = now_utc.strftime("%Y-%m")
                    for m in range(1, m_back_timeseries + 1):  # start from 1 to SKIP current month
                        month_str = (now_utc - pd.DateOffset(months=m)).strftime("%Y-%m")
                        path = f"cvd/{symbol}/{month_str}.parquet"
                        df_part = gcs_parquet_store._download_parquet(path)
                        if df_part is not None:
                            cvd_hist_dfs.append(df_part)
                    
                    cvd_hist = pd.concat(cvd_hist_dfs, ignore_index=True) if cvd_hist_dfs else pd.DataFrame()
                    
                    if not cvd_hist.empty:
                        # [V14.6] Robustness: Remove duplicate columns and reset index before processing
                        cvd_hist = cvd_hist.loc[:, ~cvd_hist.columns.duplicated()].reset_index(drop=True)
                        cvd_hist.rename(columns={
                            'taker_buy_volume': 'whale_buy_vol',
                            'taker_sell_volume': 'whale_sell_vol'
                        }, inplace=True)
                        
                        # Build daily price map once — shared for BOTH hist and recent scaling
                        price_map = None
                        if df_1d is not None and not df_1d.empty:
                            _pm = df_1d.copy()
                            _pm['timestamp'] = pd.to_datetime(_pm['timestamp'], utc=True).dt.floor('D')
                            _pm = _pm.drop_duplicates(subset='timestamp').set_index('timestamp')['close']
                            price_map = _pm[~_pm.index.duplicated(keep='last')]

                        # Convert historical coin volume → USD using per-day price
                        _fallback_px = float(df['close'].iloc[-1]) if not df.empty else 60000.0
                        if price_map is not None:
                            ts_col = pd.to_datetime(cvd_hist['timestamp'], utc=True).dt.floor('D')
                            # Use .values to avoid Series index mismatch during multiplication
                            daily_prices = ts_col.map(price_map).ffill().bfill().fillna(_fallback_px).values
                        else:
                            daily_prices = _fallback_px

                        cvd_hist['whale_buy_vol'] = cvd_hist['whale_buy_vol'] * daily_prices
                        cvd_hist['whale_sell_vol'] = cvd_hist['whale_sell_vol'] * daily_prices
                        cvd_hist['timestamp'] = pd.to_datetime(cvd_hist['timestamp'], utc=True)
                    else:
                        price_map = None         # No historical data → price_map unavailable
                        cvd_hist = pd.DataFrame()  # Ensure empty DF if concat failed
                    
                    # Merge with recent DB data
                    cvd_recent = db.get_cvd_data(symbol, limit=db_limit)
                    if not cvd_recent.empty:
                        cvd_recent = cvd_recent.loc[:, ~cvd_recent.columns.duplicated()].reset_index(drop=True)
                        cvd_recent['timestamp'] = pd.to_datetime(cvd_recent['timestamp'], utc=True)
                        
                        # Convert recent coin volume → USD using per-MINUTE prices from df (1m OHLCV).
                        # df already holds ~31 days of 1m candles (limit=45000), so per-minute
                        # price lookup is exact — no intraday approximation error unlike daily close.
                        _fallback_px = float(df['close'].iloc[-1]) if not df.empty else 60000.0

                        if not df.empty:
                            # Build per-minute price map: timestamp floored to minute → 1m close
                            _df_min = df.copy()
                            _df_min['timestamp'] = pd.to_datetime(_df_min['timestamp'], utc=True).dt.floor('min')
                            _df_min = (_df_min
                                       .drop_duplicates(subset='timestamp')
                                       .set_index('timestamp')['close']
                                       .sort_index())

                            ts_col_r = pd.to_datetime(cvd_recent['timestamp'], utc=True).dt.floor('min')
                            recent_prices = ts_col_r.map(_df_min)

                            # Fill any gaps (bot restarts / sparse periods) with daily price_map
                            if recent_prices.isna().any() and price_map is not None:
                                daily_fill = ts_col_r.dt.floor('D').map(price_map)
                                recent_prices = recent_prices.fillna(daily_fill)

                            recent_prices = recent_prices.fillna(_fallback_px).values
                        elif price_map is not None:
                            # df unavailable → fall back to daily price_map
                            ts_col_r = pd.to_datetime(cvd_recent['timestamp'], utc=True).dt.floor('D')
                            recent_prices = ts_col_r.map(price_map).ffill().bfill().fillna(_fallback_px).values
                        else:
                            recent_prices = _fallback_px

                        if 'whale_buy_vol' in cvd_recent.columns:
                            cvd_recent['whale_buy_vol'] = cvd_recent['whale_buy_vol'] * recent_prices
                        if 'whale_sell_vol' in cvd_recent.columns:
                            cvd_recent['whale_sell_vol'] = cvd_recent['whale_sell_vol'] * recent_prices
                            
                        cvd_df = pd.concat([cvd_hist, cvd_recent], ignore_index=True) if not cvd_hist.empty else cvd_recent
                        cvd_df = cvd_df.loc[:, ~cvd_df.columns.duplicated()]\
                                       .drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                        cvd_df = cvd_df.reset_index(drop=True)
                    else:
                        cvd_df = cvd_hist
                    
                    earliest_hist = cvd_df['timestamp'].min() if not cvd_df.empty else None
                    
                    # ── 2. Funding / OI ──
                    fnd_hist_dfs = []
                    for m in range(1, m_back_timeseries + 1):  # start from 1 to SKIP current month
                        month_str = (now_utc - pd.DateOffset(months=m)).strftime("%Y-%m")
                        path = f"funding/{symbol}/{month_str}.parquet"
                        df_part = gcs_parquet_store._download_parquet(path)
                        if df_part is not None:
                            fnd_hist_dfs.append(df_part)
                    
                    fnd_hist = pd.concat(fnd_hist_dfs, ignore_index=True) if fnd_hist_dfs else pd.DataFrame()
                    
                    if not fnd_hist.empty:
                        # [V14.5 Fix] Standardize OI to USD value
                        fnd_hist.rename(columns={'open_interest_value': 'open_interest'}, inplace=True)
                        # [V14.6] Dedup columns — GCS parquet may already have 'open_interest',
                        # causing duplicate columns after rename which breaks pd.concat
                        fnd_hist = fnd_hist.loc[:, ~fnd_hist.columns.duplicated()].reset_index(drop=True)
                        fnd_hist['timestamp'] = pd.to_datetime(fnd_hist['timestamp'], utc=True)

                        fnd_recent = db.get_funding_history(symbol, limit=db_limit)
                        if fnd_recent is not None and not fnd_recent.empty:
                            fnd_recent.rename(columns={'open_interest_value': 'open_interest'}, inplace=True)
                            fnd_recent = fnd_recent.loc[:, ~fnd_recent.columns.duplicated()].reset_index(drop=True)
                            fnd_recent['timestamp'] = pd.to_datetime(fnd_recent['timestamp'], utc=True)

                            funding_df = pd.concat([fnd_hist, fnd_recent], ignore_index=True)\
                                           .drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                            funding_df = funding_df.reset_index(drop=True)
                        else:
                            funding_df = fnd_hist
                    else:
                        funding_df = db.get_funding_history(symbol, limit=db_limit)
                        if not funding_df.empty:
                            funding_df.rename(columns={'open_interest_value': 'open_interest'}, inplace=True)
                            funding_df['timestamp'] = pd.to_datetime(funding_df['timestamp'], utc=True)
                    
                    # 3. Liquidations (Increase limit to cover gaps)
                    liq_df = db.get_liquidation_data(symbol, limit=db_limit)
                except Exception as e:
                    import traceback
                    logger.warning(f"GCS load for chart tool skipped: {e}\n{traceback.format_exc()}")

            # Get analysis and generate chart
            analysis = math_engine.analyze_market(df, mode, df_1d=df_1d, df_1w=df_1w, 
                                                   timeframe=timeframe)
            chart_bytes = chart_generator.generate_chart(df, analysis, symbol, mode, 
                                                          df_1d=df_1d, df_1w=df_1w,
                                                          cvd_df=cvd_df, 
                                                          funding_df=funding_df,
                                                          liquidation_df=liq_df,
                                                          timeframe=timeframe)

            if chart_bytes:
                b64 = chart_generator.chart_to_base64(chart_bytes)
                return {
                    "symbol": symbol,
                    "mode": mode.value,
                    "timeframe_requested": timeframe or mode.value,
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
            settings = get_settings()
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
        """Switch trading mode at runtime by updating env var and persisting to SQLite."""
        try:
            mode_lower = mode.lower().strip()
            if mode_lower not in ('swing', 'position'):
                return {"error": f"Invalid mode '{mode}'. Use 'swing', or 'position'."}

            # [FIX] SQLite DB + 환경변수 + 필드 싱글톤 모두 업데이트
            from config.local_state import state_manager
            state_manager.set_trading_mode(mode_lower)
            os.environ['TRADING_MODE'] = mode_lower
            
            # [NEW] 메모리 싱글톤 즉시 업데이트
            settings = get_settings()
            settings.TRADING_MODE = mode_lower

            # [NEW] 스케줄러 주기 즉시 재설정
            try:
                from config import scheduler_config
                scheduler_config.reschedule_analysis_job(mode_lower)
            except Exception as e:
                logger.error(f"Failed to reschedule analysis via tool: {e}")

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
