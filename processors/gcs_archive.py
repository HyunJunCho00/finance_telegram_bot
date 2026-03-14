"""GCS Archive  now delegates to GCS Parquet store.

Backward-compatible wrapper. The scheduler calls gcs_archive_exporter.run_daily_archive()
which now uses Parquet format instead of compressed JSONL.

Old JSONL.gz files in GCS are preserved but no longer written.
"""

from __future__ import annotations

from typing import Dict
from loguru import logger
from processors.gcs_parquet import gcs_parquet_store


class GCSArchiveExporter:
    """Backward-compatible wrapper over GCSParquetStore."""

    def __init__(self):
        self._store = gcs_parquet_store

    @property
    def enabled(self) -> bool:
        return self._store.enabled

    def run_daily_archive(self) -> Dict:
        """Archive expiring rows as Parquet (replaces old JSONL.gz method)."""
        return self._store.run_daily_archive()

    def run_safe_cleanup(self, limit: int = 500) -> Dict:
        """Delete only rows covered by verified archive manifests."""
        return self._store.run_safe_cleanup(limit=limit)


gcs_archive_exporter = GCSArchiveExporter()
