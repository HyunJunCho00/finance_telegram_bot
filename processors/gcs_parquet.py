"""GCS Parquet storage module for long-term time-series data.

Dual purpose:
1. Archive: Daily export of expiring PostgreSQL rows as Parquet (replaces jsonl.gz)
2. Reader:  Load historical data from GCS for analysis (higher timeframes)

Storage layout:
  gs://bucket/
    ohlcv/{timeframe}/{symbol}/{YYYY-MM}.parquet   (1m: monthly, others: yearly)
    funding/{symbol}/{YYYY-MM}.parquet
    cvd/{symbol}/{YYYY-MM}.parquet
    liquidations/{symbol}/{YYYY-MM}.parquet

Parquet advantages over JSONL:
  - 5x better compression for numeric columns (OHLCV)
  - Direct pandas read/write
  - BigQuery external table compatible
  - 5 years of 1m BTC data ≈ 88MB
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from config.database import db
from config.settings import settings


class GCSParquetStore:
    """Read/write Parquet files to GCS for long-term time-series storage."""

    def __init__(self):
        self.bucket_name = settings.GCS_ARCHIVE_BUCKET
        self.enabled = bool(settings.ENABLE_GCS_ARCHIVE and self.bucket_name)
        self._client = None

    @property
    def storage_client(self):
        if self._client is None:
            from google.cloud import storage
            self._client = storage.Client(project=settings.PROJECT_ID or None)
        return self._client

    @property
    def bucket(self):
        return self.storage_client.bucket(self.bucket_name)

    # ─────────────── Write Operations ───────────────

    def _upload_parquet(self, object_path: str, df: pd.DataFrame) -> int:
        """Upload DataFrame as Parquet to GCS. Returns byte size."""
        buf = BytesIO()
        df.to_parquet(buf, engine="pyarrow", index=False, compression="snappy")
        payload = buf.getvalue()
        blob = self.bucket.blob(object_path)
        blob.upload_from_string(payload, content_type="application/octet-stream")
        return len(payload)

    def _merge_upload_parquet(self, object_path: str, new_df: pd.DataFrame,
                              dedup_cols: List[str]) -> int:
        """Download existing Parquet, merge with new data, dedup, re-upload."""
        existing_df = self._download_parquet(object_path)
        if existing_df is not None and not existing_df.empty:
            merged = pd.concat([existing_df, new_df], ignore_index=True)
            merged = merged.drop_duplicates(subset=dedup_cols, keep="last")
            merged = merged.sort_values(dedup_cols[0]).reset_index(drop=True)
        else:
            merged = new_df.sort_values(dedup_cols[0]).reset_index(drop=True)
        return self._upload_parquet(object_path, merged)

    def archive_ohlcv(self, table: str = "market_data", retention_days: int = None) -> Dict:
        """Archive expiring OHLCV rows to Parquet, partitioned by month."""
        if not self.enabled:
            return {"enabled": False}

        retention = retention_days or settings.RETENTION_MARKET_DATA_DAYS
        cutoff = (datetime.now(timezone.utc) - timedelta(days=retention)).isoformat()

        try:
            rows = self._fetch_all_rows(table, "timestamp", cutoff)
            if not rows:
                return {"table": table, "rows": 0}

            df = pd.DataFrame(rows)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df["month"] = df["timestamp"].dt.strftime("%Y-%m")

            total_bytes = 0
            for (symbol, month), group in df.groupby(["symbol", "month"]):
                path = f"ohlcv/1m/{symbol}/{month}.parquet"
                size = self._merge_upload_parquet(path, group.drop(columns=["month"]),
                                                  ["timestamp", "symbol", "exchange"])
                total_bytes += size
                logger.info(f"Parquet archive: {path} ({len(group)} rows, {size:,} bytes)")

            return {"table": table, "rows": len(df), "bytes": total_bytes}
        except Exception as e:
            logger.error(f"OHLCV archive error: {e}")
            return {"table": table, "error": str(e)}

    def archive_timeseries(self, table: str, time_col: str, retention_days: int,
                           dedup_cols: List[str], prefix: str) -> Dict:
        """Generic archiver for any time-series table."""
        if not self.enabled:
            return {"enabled": False}

        cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()

        try:
            rows = self._fetch_all_rows(table, time_col, cutoff)
            if not rows:
                return {"table": table, "rows": 0}

            df = pd.DataFrame(rows)
            df[time_col] = pd.to_datetime(df[time_col], utc=True)
            df["month"] = df[time_col].dt.strftime("%Y-%m")

            # Group by symbol if present, otherwise by month only
            total_bytes = 0
            if "symbol" in df.columns:
                for (symbol, month), group in df.groupby(["symbol", "month"]):
                    path = f"{prefix}/{symbol}/{month}.parquet"
                    size = self._merge_upload_parquet(path, group.drop(columns=["month"]), dedup_cols)
                    total_bytes += size
            else:
                for month, group in df.groupby("month"):
                    path = f"{prefix}/{month}.parquet"
                    size = self._merge_upload_parquet(path, group.drop(columns=["month"]), dedup_cols)
                    total_bytes += size

            return {"table": table, "rows": len(df), "bytes": total_bytes}
        except Exception as e:
            logger.error(f"Archive error for {table}: {e}")
            return {"table": table, "error": str(e)}

    def archive_resampled_ohlcv(self, timeframe: str, symbol: str,
                                 df: pd.DataFrame) -> Dict:
        """Save pre-resampled OHLCV (1h/4h/1d/1w) to GCS.
        Called by incremental append jobs."""
        if not self.enabled:
            return {"enabled": False}

        try:
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

            # For higher timeframes, use yearly partitions (small data)
            if timeframe in ("1d", "1w"):
                year = df["timestamp"].dt.strftime("%Y").iloc[0]
                path = f"ohlcv/{timeframe}/{symbol}/{year}.parquet"
            else:
                month = df["timestamp"].dt.strftime("%Y-%m").iloc[0]
                path = f"ohlcv/{timeframe}/{symbol}/{month}.parquet"

            size = self._merge_upload_parquet(path, df, ["timestamp"])
            return {"path": path, "rows": len(df), "bytes": size}
        except Exception as e:
            logger.error(f"Resampled OHLCV archive error: {e}")
            return {"error": str(e)}

    # ─────────────── Read Operations ───────────────

    def _download_parquet(self, object_path: str) -> Optional[pd.DataFrame]:
        """Download a Parquet file from GCS. Returns None if not found."""
        try:
            blob = self.bucket.blob(object_path)
            if not blob.exists():
                return None
            buf = BytesIO(blob.download_as_bytes())
            return pd.read_parquet(buf, engine="pyarrow")
        except Exception as e:
            logger.debug(f"Parquet download failed ({object_path}): {e}")
            return None

    def load_ohlcv(self, timeframe: str, symbol: str,
                   months_back: int = 3) -> pd.DataFrame:
        """Load OHLCV data from GCS for a given timeframe and symbol.
        Loads recent N months of Parquet files and concatenates."""
        if not self.enabled:
            return pd.DataFrame()

        dfs = []
        now = datetime.now(timezone.utc)

        if timeframe in ("1d", "1w"):
            # Yearly partition: load current year + previous year
            for year_offset in range(2):
                year = (now - timedelta(days=365 * year_offset)).strftime("%Y")
                path = f"ohlcv/{timeframe}/{symbol}/{year}.parquet"
                df = self._download_parquet(path)
                if df is not None:
                    dfs.append(df)
        else:
            # Monthly partition
            for m in range(months_back):
                month = (now - timedelta(days=30 * m)).strftime("%Y-%m")
                path = f"ohlcv/{timeframe}/{symbol}/{month}.parquet"
                df = self._download_parquet(path)
                if df is not None:
                    dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        result = pd.concat(dfs, ignore_index=True)
        # Handle numeric timestamps (ms) if they somehow got stored that way, otherwise ISO strings
        if pd.api.types.is_numeric_dtype(result["timestamp"]):
            result["timestamp"] = pd.to_datetime(result["timestamp"], unit='ms', utc=True)
        else:
            result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
        result = result.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return result

    def load_timeseries(self, prefix: str, symbol: str,
                        months_back: int = 3) -> pd.DataFrame:
        """Generic loader for funding/cvd/liquidations."""
        if not self.enabled:
            return pd.DataFrame()

        dfs = []
        now = datetime.now(timezone.utc)
        for m in range(months_back):
            month = (now - timedelta(days=30 * m)).strftime("%Y-%m")
            path = f"{prefix}/{symbol}/{month}.parquet"
            df = self._download_parquet(path)
            if df is not None:
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        result = pd.concat(dfs, ignore_index=True)
        # Handle numeric timestamps (ms) if they somehow got stored that way
        time_col = result.columns[0] # Usually 'timestamp'
        if pd.api.types.is_numeric_dtype(result[time_col]):
            result[time_col] = pd.to_datetime(result[time_col], unit='ms', utc=True)
        else:
            result[time_col] = pd.to_datetime(result[time_col], utc=True)
            
        return result.sort_values(time_col).reset_index(drop=True)

    # ─────────────── Helpers ───────────────

    def _fetch_all_rows(self, table: str, time_col: str, cutoff: str,
                        page_size: int = 1000) -> List[Dict]:
        """Paginated fetch of rows older than cutoff."""
        all_rows: List[Dict] = []
        page = 0
        while True:
            start = page * page_size
            end = start + page_size - 1
            response = (
                db.client.table(table)
                .select("*")
                .lt(time_col, cutoff)
                .order(time_col, desc=False)
                .range(start, end)
                .execute()
            )
            rows = response.data if response.data else []
            if not rows:
                break
            all_rows.extend(rows)
            if len(rows) < page_size:
                break
            page += 1
        return all_rows

    # ─────────────── Daily Archive Job ───────────────

    def run_daily_archive(self) -> Dict:
        """Archive all expiring time-series data as Parquet.
        Called daily by scheduler before cleanup."""
        if not self.enabled:
            return {"enabled": False, "reason": "GCS archive disabled"}

        summary = {
            "enabled": True,
            "bucket": self.bucket_name,
            "archived_at": datetime.now(timezone.utc).isoformat(),
            "tables": {},
        }

        # OHLCV (market_data)
        summary["tables"]["market_data"] = self.archive_ohlcv()

        # CVD
        summary["tables"]["cvd_data"] = self.archive_timeseries(
            "cvd_data", "timestamp", settings.RETENTION_CVD_DAYS,
            ["timestamp", "symbol"], "cvd"
        )

        # Funding
        summary["tables"]["funding_data"] = self.archive_timeseries(
            "funding_data", "timestamp", settings.RETENTION_MARKET_DATA_DAYS,
            ["timestamp", "symbol"], "funding"
        )

        # Liquidations
        try:
            summary["tables"]["liquidations"] = self.archive_timeseries(
                "liquidations", "timestamp", settings.RETENTION_MARKET_DATA_DAYS,
                ["timestamp", "symbol"], "liquidations"
            )
        except Exception:
            summary["tables"]["liquidations"] = {"rows": 0, "note": "table may not exist"}

        logger.info(f"GCS Parquet archive complete: {summary}")
        return summary


# Singleton
gcs_parquet_store = GCSParquetStore()
