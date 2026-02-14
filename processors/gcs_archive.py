"""Daily archival of expiring PostgreSQL rows into GCS as compressed JSONL.

Purpose:
- Keep operational DB retention small and cheap.
- Preserve long-term history for backtesting/audits in low-cost object storage.
"""

from __future__ import annotations

import gzip
import json
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Dict, List, Tuple

from loguru import logger

from config.database import db
from config.settings import settings


class GCSArchiveExporter:
    """Exports rows that are about to be deleted by retention cleanup."""

    def __init__(self):
        self.bucket_name = settings.GCS_ARCHIVE_BUCKET
        self.enabled = bool(settings.ENABLE_GCS_ARCHIVE and self.bucket_name)

    def _retention_plan(self) -> List[Tuple[str, str, int]]:
        """(table, time_column, retention_days)."""
        return [
            ("market_data", "timestamp", settings.RETENTION_MARKET_DATA_DAYS),
            ("cvd_data", "timestamp", settings.RETENTION_CVD_DAYS),
            ("funding_data", "timestamp", settings.RETENTION_MARKET_DATA_DAYS),
            ("telegram_messages", "created_at", settings.RETENTION_TELEGRAM_DAYS),
            ("narrative_data", "timestamp", settings.RETENTION_REPORTS_DAYS),
            ("ai_reports", "created_at", settings.RETENTION_REPORTS_DAYS),
        ]

    def _fetch_expiring_rows(self, table: str, time_col: str, retention_days: int, page_size: int = 1000) -> List[Dict]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
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

    def _upload_jsonl_gz(self, object_path: str, rows: List[Dict]) -> int:
        from google.cloud import storage

        storage_client = storage.Client(project=settings.PROJECT_ID or None)
        bucket = storage_client.bucket(self.bucket_name)
        blob = bucket.blob(object_path)

        bio = BytesIO()
        with gzip.GzipFile(fileobj=bio, mode="wb") as gz:
            for row in rows:
                line = json.dumps(row, default=str, ensure_ascii=False)
                gz.write((line + "\n").encode("utf-8"))

        payload = bio.getvalue()
        blob.upload_from_string(payload, content_type="application/gzip")
        return len(payload)

    def run_daily_archive(self) -> Dict:
        if not self.enabled:
            return {"enabled": False, "reason": "ENABLE_GCS_ARCHIVE disabled or GCS_ARCHIVE_BUCKET missing"}

        summary = {
            "enabled": True,
            "bucket": self.bucket_name,
            "tables": {},
            "archived_at": datetime.now(timezone.utc).isoformat(),
        }

        for table, time_col, days in self._retention_plan():
            try:
                rows = self._fetch_expiring_rows(table, time_col, days)
                count = len(rows)
                if count == 0:
                    summary["tables"][table] = {"rows": 0, "bytes": 0, "object": None}
                    continue

                day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                object_path = f"archive/{table}/dt={day}/{table}.jsonl.gz"
                size_bytes = self._upload_jsonl_gz(object_path, rows)
                summary["tables"][table] = {
                    "rows": count,
                    "bytes": size_bytes,
                    "object": object_path,
                }
                logger.info(f"GCS archive {table}: {count} rows -> gs://{self.bucket_name}/{object_path}")
            except Exception as e:
                logger.error(f"GCS archive error for {table}: {e}")
                summary["tables"][table] = {"error": str(e)}

        return summary


gcs_archive_exporter = GCSArchiveExporter()
