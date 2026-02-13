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

    # ─────────────── Funding Data ───────────────

    def upsert_funding_data(self, data: Dict) -> Dict:
        """Upsert funding data. Uses truncated timestamp to avoid duplication."""
        return self.client.table("funding_data").upsert(
            data, on_conflict="timestamp,symbol"
        ).execute()

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

    def cleanup_old_data(self, days: int = 30) -> Dict:
        """Delete data older than N days to stay within Supabase 500MB free tier.
        Called daily by the scheduler."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        results = {}

        try:
            # Market data (biggest table - 1min candles)
            r = self.client.table("market_data")\
                .delete()\
                .lt("timestamp", cutoff)\
                .execute()
            results['market_data_deleted'] = len(r.data) if r.data else 0

            # Funding data
            r = self.client.table("funding_data")\
                .delete()\
                .lt("timestamp", cutoff)\
                .execute()
            results['funding_data_deleted'] = len(r.data) if r.data else 0

            # Telegram messages (keep 7 days only)
            cutoff_7d = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            r = self.client.table("telegram_messages")\
                .delete()\
                .lt("created_at", cutoff_7d)\
                .execute()
            results['telegram_deleted'] = len(r.data) if r.data else 0

            # AI reports (keep 90 days)
            cutoff_90d = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
            r = self.client.table("ai_reports")\
                .delete()\
                .lt("created_at", cutoff_90d)\
                .execute()
            results['reports_deleted'] = len(r.data) if r.data else 0

            logger.info(f"Data cleanup completed: {results}")
            return results

        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            return {"error": str(e)}


db = DatabaseClient()
