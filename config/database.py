from supabase import create_client, Client
from config.settings import settings
from typing import Dict, List, Optional
from datetime import datetime, timedelta, timezone
import pandas as pd
from loguru import logger


class DatabaseClient:
    def __init__(self):
        self.client: Client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_KEY
        )

    # ─────────────── Market Data ───────────────

    def insert_market_data(self, data: Dict) -> Dict:
        return self.client.table("market_data").insert(data).execute()

    def batch_insert_market_data(self, data_list: List[Dict]) -> Dict:
        """Upsert to avoid duplicate errors on (timestamp, symbol, exchange)."""
        return self.client.table("market_data").upsert(
            data_list, on_conflict="timestamp,symbol,exchange"
        ).execute()

    def get_latest_market_data(self, symbol: str, limit: int = 1000, exchange: str = "binance") -> pd.DataFrame:
        response = self.client.table("market_data")\
            .select("*")\
            .eq("symbol", symbol)\
            .eq("exchange", exchange)\
            .order("timestamp", desc=True)\
            .limit(limit)\
            .execute()

        if response.data:
            df = pd.DataFrame(response.data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            return df.sort_values('timestamp').reset_index(drop=True)
        return pd.DataFrame()

    # ─────────────── CVD Data ───────────────

    def batch_upsert_cvd_data(self, data_list: List[Dict]) -> Dict:
        """Upsert CVD (volume delta) data."""
        return self.client.table("cvd_data").upsert(
            data_list, on_conflict="timestamp,symbol"
        ).execute()

    def get_cvd_data(self, symbol: str, limit: int = 240) -> pd.DataFrame:
        """Get recent CVD data for a symbol. Default 240 = 4 hours of 1m data."""
        response = self.client.table("cvd_data")\
            .select("*")\
            .eq("symbol", symbol)\
            .order("timestamp", desc=True)\
            .limit(limit)\
            .execute()

        if response.data:
            df = pd.DataFrame(response.data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df = df.sort_values('timestamp').reset_index(drop=True)
            # Calculate cumulative volume delta
            df['cvd'] = df['volume_delta'].cumsum()
            # Calculate whale CVD if columns exist
            if 'whale_buy_vol' in df.columns and 'whale_sell_vol' in df.columns:
                df['whale_buy_vol'] = df['whale_buy_vol'].fillna(0)
                df['whale_sell_vol'] = df['whale_sell_vol'].fillna(0)
                df['whale_delta'] = df['whale_buy_vol'] - df['whale_sell_vol']
                df['whale_cvd'] = df['whale_delta'].cumsum()
            return df
        return pd.DataFrame()

    # ─────────────── Whale Data (from WebSocket aggTrade) ───────────────

    def batch_upsert_whale_data(self, data_list: List[Dict]) -> Dict:
        """Upsert whale trade data into cvd_data table (whale columns)."""
        for attempt in range(2):
            try:
                return self.client.table("cvd_data").upsert(
                    data_list, on_conflict="timestamp,symbol"
                ).execute()
            except Exception as e:
                if attempt == 0:
                    logger.warning(f"Whale upsert failed, reconnecting: {e}")
                    self.client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
                else:
                    logger.error(f"Whale upsert failed after reconnect: {e}")
                    raise

    # ─────────────── Liquidation Data ───────────────

    def batch_upsert_liquidations(self, data_list: List[Dict]) -> Dict:
        """Upsert liquidation data."""
        return self.client.table("liquidations").upsert(
            data_list, on_conflict="timestamp,symbol"
        ).execute()

    def get_liquidation_data(self, symbol: str, limit: int = 240) -> pd.DataFrame:
        """Get recent liquidation data. Default 240 = 4 hours of 1m data."""
        response = self.client.table("liquidations")\
            .select("*")\
            .eq("symbol", symbol)\
            .order("timestamp", desc=True)\
            .limit(limit)\
            .execute()

        if response.data:
            df = pd.DataFrame(response.data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            return df.sort_values('timestamp').reset_index(drop=True)
        return pd.DataFrame()

    # ─────────────── Telegram Messages ───────────────

    def upsert_telegram_message(self, data: Dict) -> Dict:
        """Upsert to avoid duplicate errors on (channel, message_id)."""
        return self.client.table("telegram_messages").upsert(
            data, on_conflict="channel,message_id"
        ).execute()

    def insert_telegram_message(self, data: Dict) -> Dict:
        """Backward compatible - now uses upsert."""
        return self.upsert_telegram_message(data)

    def get_recent_telegram_messages(self, hours: int = 24) -> List[Dict]:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        response = self.client.table("telegram_messages")\
            .select("*")\
            .gte("created_at", cutoff)\
            .order("created_at", desc=True)\
            .execute()

        return response.data if response.data else []

    def get_telegram_messages_for_rag(self, days: int = 7) -> List[Dict]:
        """Get telegram messages for LightRAG ingestion."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        response = self.client.table("telegram_messages")\
            .select("*")\
            .gte("created_at", cutoff)\
            .order("created_at", desc=True)\
            .limit(1000)\
            .execute()

        return response.data if response.data else []

    # ─────────────── Funding Data ───────────────

    def upsert_funding_data(self, data: Dict) -> Dict:
        """Upsert funding data. Uses truncated timestamp to avoid duplication."""
        return self.client.table("funding_data").upsert(
            data, on_conflict="timestamp,symbol"
        ).execute()



    # ─────────────── Microstructure Data ───────────────

    def batch_upsert_microstructure_data(self, data_list: List[Dict]) -> Dict:
        return self.client.table("microstructure_data").upsert(
            data_list, on_conflict="timestamp,symbol,exchange"
        ).execute()

    def get_latest_microstructure(self, symbol: str) -> Optional[Dict]:
        response = self.client.table("microstructure_data")\
            .select("*")\
            .eq("symbol", symbol)\
            .order("timestamp", desc=True)\
            .limit(1)\
            .execute()
        return response.data[0] if response.data else None

    # ─────────────── Macro Data ───────────────

    def upsert_macro_data(self, data: Dict) -> Dict:
        return self.client.table("macro_data").upsert(
            data, on_conflict="timestamp,source"
        ).execute()

    def get_latest_macro_data(self) -> Optional[Dict]:
        response = self.client.table("macro_data")\
            .select("*")\
            .order("timestamp", desc=True)\
            .limit(1)\
            .execute()
        return response.data[0] if response.data else None

    # ─────────────── Narrative Data ───────────────

    def upsert_narrative_data(self, data: Dict) -> Dict:
        """Upsert Perplexity/market narrative data for auditability."""
        return self.client.table("narrative_data").upsert(
            data, on_conflict="timestamp,symbol,source"
        ).execute()

    def get_latest_narrative_data(self, symbol: str, source: str = "perplexity") -> Optional[Dict]:
        response = self.client.table("narrative_data")\
            .select("*")\
            .eq("symbol", symbol)\
            .eq("source", source)\
            .order("timestamp", desc=True)\
            .limit(1)\
            .execute()

        return response.data[0] if response.data else None


    # ─────────────── Dune Query Data ───────────────

    def upsert_dune_query_result(self, data: Dict) -> Dict:
        """Upsert Dune query snapshot by (query_id, collected_at)."""
        return self.client.table("dune_query_results").upsert(
            data, on_conflict="query_id,collected_at"
        ).execute()

    def get_latest_dune_query_result(self, query_id: int) -> Optional[Dict]:
        response = self.client.table("dune_query_results")\
            .select("*")\
            .eq("query_id", query_id)\
            .order("collected_at", desc=True)\
            .limit(1)\
            .execute()

        return response.data[0] if response.data else None

    # ─────────────── Deribit Options Data ───────────────

    def upsert_deribit_data(self, data: Dict) -> Dict:
        """Upsert Deribit options snapshot (DVOL, PCR, IV Term Structure, Skew)."""
        return self.client.table("deribit_data").upsert(
            data, on_conflict="symbol,timestamp"
        ).execute()

    def get_latest_deribit_data(self, symbol: str) -> Optional[Dict]:
        response = self.client.table("deribit_data")\
            .select("*")\
            .eq("symbol", symbol)\
            .order("timestamp", desc=True)\
            .limit(1)\
            .execute()
        return response.data[0] if response.data else None

    # ─────────────── Fear & Greed Data ───────────────

    def upsert_fear_greed(self, data: Dict) -> Dict:
        """Upsert daily Crypto Fear & Greed Index snapshot."""
        return self.client.table("fear_greed_data").upsert(
            data, on_conflict="timestamp"
        ).execute()

    def get_latest_fear_greed(self) -> Optional[Dict]:
        response = self.client.table("fear_greed_data")\
            .select("*")\
            .order("timestamp", desc=True)\
            .limit(1)\
            .execute()
        return response.data[0] if response.data else None

    # ─────────────── AI Reports ───────────────

    def insert_ai_report(self, data: Dict) -> Dict:
        return self.client.table("ai_reports").insert(data).execute()

    def get_latest_report(self, symbol: str = None) -> Optional[Dict]:
        query = self.client.table("ai_reports")\
            .select("*")\
            .order("created_at", desc=True)

        if symbol:
            query = query.eq("symbol", symbol)

        response = query.limit(1).execute()
        return response.data[0] if response.data else None

    # ─────────────── Feedback ───────────────

    def insert_feedback(self, data: Dict) -> Dict:
        return self.client.table("feedback_logs").insert(data).execute()

    def get_feedback_history(self, limit: int = 10) -> List[Dict]:
        response = self.client.table("feedback_logs")\
            .select("*")\
            .order("created_at", desc=True)\
            .limit(limit)\
            .execute()

        return response.data if response.data else []

    # ─────────────── Trade Executions ───────────────

    def insert_trade_execution(self, data: Dict) -> Dict:
        return self.client.table("trade_executions").insert(data).execute()

    def get_position_status(self, symbol: str) -> Optional[Dict]:
        response = self.client.table("trade_executions")\
            .select("*")\
            .eq("symbol", symbol)\
            .order("created_at", desc=True)\
            .limit(1)\
            .execute()

        return response.data[0] if response.data else None

    # ─────────────── Funding History ───────────────

    def get_funding_history(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        response = self.client.table("funding_data")\
            .select("*")\
            .eq("symbol", symbol)\
            .order("timestamp", desc=True)\
            .limit(limit)\
            .execute()

        if response.data:
            df = pd.DataFrame(response.data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            return df.sort_values('timestamp').reset_index(drop=True)
        return pd.DataFrame()

    # ─────────────── Data Cleanup (500MB free tier) ───────────────

    def cleanup_old_data(self) -> Dict:
        """Delete data older than configured retention days.
        Uses settings for retention periods. Called daily by the scheduler."""
        results = {}

        try:
            # Market data
            cutoff = (datetime.now(timezone.utc) - timedelta(
                days=settings.RETENTION_MARKET_DATA_DAYS
            )).isoformat()
            r = self.client.table("market_data")\
                .delete()\
                .lt("timestamp", cutoff)\
                .execute()
            results['market_data_deleted'] = len(r.data) if r.data else 0

            # CVD data
            cutoff_cvd = (datetime.now(timezone.utc) - timedelta(
                days=settings.RETENTION_CVD_DAYS
            )).isoformat()
            r = self.client.table("cvd_data")\
                .delete()\
                .lt("timestamp", cutoff_cvd)\
                .execute()
            results['cvd_data_deleted'] = len(r.data) if r.data else 0

            # Funding data
            r = self.client.table("funding_data")\
                .delete()\
                .lt("timestamp", cutoff)\
                .execute()
            results['funding_data_deleted'] = len(r.data) if r.data else 0

            # Telegram messages (30 days for LightRAG)
            cutoff_telegram = (datetime.now(timezone.utc) - timedelta(
                days=settings.RETENTION_TELEGRAM_DAYS
            )).isoformat()
            r = self.client.table("telegram_messages")\
                .delete()\
                .lt("created_at", cutoff_telegram)\
                .execute()
            results['telegram_deleted'] = len(r.data) if r.data else 0

            # Microstructure data
            cutoff_micro = (datetime.now(timezone.utc) - timedelta(
                days=settings.RETENTION_MARKET_DATA_DAYS
            )).isoformat()
            r = self.client.table("microstructure_data")\
                .delete()\
                .lt("timestamp", cutoff_micro)\
                .execute()
            results['microstructure_deleted'] = len(r.data) if r.data else 0

            # Macro data
            cutoff_macro = (datetime.now(timezone.utc) - timedelta(
                days=settings.RETENTION_REPORTS_DAYS
            )).isoformat()
            r = self.client.table("macro_data")\
                .delete()\
                .lt("timestamp", cutoff_macro)\
                .execute()
            results['macro_deleted'] = len(r.data) if r.data else 0

            # Narrative data
            cutoff_narrative = (datetime.now(timezone.utc) - timedelta(
                days=settings.RETENTION_REPORTS_DAYS
            )).isoformat()
            r = self.client.table("narrative_data")\
                .delete()\
                .lt("timestamp", cutoff_narrative)\
                .execute()
            results['narrative_deleted'] = len(r.data) if r.data else 0

            # AI reports (90 days)
            cutoff_reports = (datetime.now(timezone.utc) - timedelta(
                days=settings.RETENTION_REPORTS_DAYS
            )).isoformat()
            r = self.client.table("ai_reports")\
                .delete()\
                .lt("created_at", cutoff_reports)\
                .execute()
            results['reports_deleted'] = len(r.data) if r.data else 0

            # Dune snapshots
            r = self.client.table("dune_query_results")\
                .delete()\
                .lt("collected_at", cutoff_reports)\
                .execute()
            results['dune_deleted'] = len(r.data) if r.data else 0

            # Liquidations (same retention as market data)
            try:
                r = self.client.table("liquidations")\
                    .delete()\
                    .lt("timestamp", cutoff)\
                    .execute()
                results['liquidations_deleted'] = len(r.data) if r.data else 0
            except Exception:
                results['liquidations_deleted'] = 0  # Table may not exist yet

            # Deribit options data (same retention as market data)
            try:
                r = self.client.table("deribit_data")\
                    .delete()\
                    .lt("timestamp", cutoff)\
                    .execute()
                results['deribit_deleted'] = len(r.data) if r.data else 0
            except Exception:
                results['deribit_deleted'] = 0

            # Fear & Greed data (keep 90 days — same as reports)
            cutoff_fg = (datetime.now(timezone.utc) - timedelta(
                days=settings.RETENTION_REPORTS_DAYS
            )).isoformat()
            try:
                r = self.client.table("fear_greed_data")\
                    .delete()\
                    .lt("timestamp", cutoff_fg)\
                    .execute()
                results['fear_greed_deleted'] = len(r.data) if r.data else 0
            except Exception:
                results['fear_greed_deleted'] = 0

            logger.info(f"Data cleanup completed: {results}")
            return results

        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            return {"error": str(e)}


db = DatabaseClient()
