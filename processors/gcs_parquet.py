"""GCS Parquet storage for verified cold archival and historical reads."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from hashlib import md5
from io import BytesIO
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from config.database import db
from config.settings import settings


ARCHIVE_SCHEMA_VERSION = "archive_v2"


@dataclass(frozen=True)
class ArchiveSpec:
    table_name: str
    time_column: str
    retention_days: int
    gcs_prefix: str
    pk_column: str = "id"
    symbol_column: Optional[str] = None
    time_kind: str = "timestamp"
    cleanup_enabled: bool = True
    schema_version: str = ARCHIVE_SCHEMA_VERSION


class GCSParquetStore:
    def __init__(self):
        self.bucket_name = settings.GCS_ARCHIVE_BUCKET
        self.enabled = bool(settings.ENABLE_GCS_ARCHIVE and self.bucket_name)
        self._client = None
        self.cache_dir = Path("cache/gcs_parquet")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._archive_specs = self._build_archive_specs()
        self._spec_by_table = {spec.table_name: spec for spec in self._archive_specs}

    @property
    def storage_client(self):
        if self._client is None:
            from google.cloud import storage
            self._client = storage.Client(project=settings.PROJECT_ID or None)
        return self._client

    @property
    def bucket(self):
        return self.storage_client.bucket(self.bucket_name)

    @property
    def archive_specs(self) -> List[ArchiveSpec]:
        return list(self._archive_specs)

    def _build_archive_specs(self) -> List[ArchiveSpec]:
        report_days = int(getattr(settings, "RETENTION_REPORTS_DAYS", 365))
        market_days = int(getattr(settings, "RETENTION_MARKET_DATA_DAYS", 30))
        cvd_days = int(getattr(settings, "RETENTION_CVD_DAYS", 30))
        telegram_days = int(getattr(settings, "RETENTION_TELEGRAM_DAYS", 90))
        execution_days = int(getattr(settings, "RETENTION_EXECUTIONS_DAYS", report_days))
        evaluation_days = int(getattr(settings, "RETENTION_EVALUATION_DAYS", report_days))
        evaluation_component_days = int(getattr(settings, "RETENTION_EVALUATION_COMPONENT_DAYS", report_days))
        rollup_days = int(getattr(settings, "RETENTION_ROLLUPS_DAYS", report_days * 10))

        return [
            ArchiveSpec("market_data", "timestamp", market_days, "ohlcv/1m", symbol_column="symbol"),
            ArchiveSpec("cvd_data", "timestamp", cvd_days, "cvd", symbol_column="symbol"),
            ArchiveSpec("funding_data", "timestamp", market_days, "funding", symbol_column="symbol"),
            ArchiveSpec("telegram_messages", "created_at", telegram_days, "archive/telegram_messages"),
            ArchiveSpec("narrative_data", "timestamp", report_days, "archive/narrative_data", symbol_column="symbol"),
            ArchiveSpec("ai_reports", "created_at", report_days, "archive/ai_reports", symbol_column="symbol"),
            ArchiveSpec("feedback_logs", "created_at", report_days, "archive/feedback_logs", symbol_column="symbol"),
            ArchiveSpec("trade_executions", "created_at", execution_days, "archive/trade_executions", symbol_column="symbol"),
            ArchiveSpec("dune_query_results", "collected_at", report_days, "archive/dune_query_results"),
            ArchiveSpec("microstructure_data", "timestamp", market_days, "archive/microstructure_data", symbol_column="symbol"),
            ArchiveSpec("macro_data", "timestamp", report_days, "archive/macro_data"),
            ArchiveSpec("deribit_data", "timestamp", market_days, "archive/deribit_data", symbol_column="symbol"),
            ArchiveSpec("fear_greed_data", "timestamp", report_days, "archive/fear_greed_data"),
            ArchiveSpec("liquidations", "timestamp", market_days, "liquidations", symbol_column="symbol"),
            ArchiveSpec("daily_playbooks", "created_at", report_days, "archive/daily_playbooks", symbol_column="symbol"),
            ArchiveSpec("monitor_logs", "created_at", report_days, "archive/monitor_logs", symbol_column="symbol"),
            ArchiveSpec("market_status_events", "created_at", report_days, "archive/market_status_events", symbol_column="symbol"),
            ArchiveSpec("onchain_daily_snapshots", "as_of_date", report_days, "archive/onchain_daily_snapshots", symbol_column="symbol", time_kind="date"),
            ArchiveSpec("evaluation_predictions", "prediction_time", evaluation_days, "archive/evaluation_predictions", symbol_column="symbol"),
            ArchiveSpec("evaluation_outcomes", "evaluated_at", evaluation_days, "archive/evaluation_outcomes"),
            ArchiveSpec("evaluation_component_scores", "created_at", evaluation_component_days, "archive/evaluation_component_scores"),
            ArchiveSpec("evaluation_rollups_daily", "rollup_date", rollup_days, "archive/evaluation_rollups_daily", symbol_column="symbol", time_kind="date"),
        ]

    def _build_parquet_payload(self, df: pd.DataFrame) -> tuple[bytes, str]:
        buf = BytesIO()
        df.to_parquet(buf, engine="pyarrow", index=False, compression="snappy")
        payload = buf.getvalue()
        return payload, md5(payload).hexdigest()

    def _upload_parquet_payload(self, object_path: str, payload: bytes) -> int:
        blob = self.bucket.blob(object_path)
        blob.upload_from_string(payload, content_type="application/octet-stream")
        return len(payload)

    def _merge_upload_parquet(self, object_path: str, new_df: pd.DataFrame, dedup_cols: List[str]) -> int:
        existing_df = self._download_parquet(object_path)
        if existing_df is not None and not existing_df.empty:
            merged = pd.concat([existing_df, new_df], ignore_index=True)
            merged = merged.drop_duplicates(subset=dedup_cols, keep="last")
            merged = merged.sort_values(dedup_cols[0]).reset_index(drop=True)
        else:
            merged = new_df.sort_values(dedup_cols[0]).reset_index(drop=True)
        payload, _ = self._build_parquet_payload(merged)
        return self._upload_parquet_payload(object_path, payload)

    def archive_resampled_ohlcv(self, timeframe: str, symbol: str, df: pd.DataFrame) -> Dict:
        if not self.enabled:
            return {"enabled": False}
        try:
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
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

    def _normalize_for_parquet(self, df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
        out = df.copy()
        json_cols: List[str] = []
        for col in out.columns:
            if col.startswith("_partition_"):
                continue
            needs_json = False
            for value in out[col].head(50).tolist():
                if isinstance(value, (dict, list)):
                    needs_json = True
                    break
            if not needs_json:
                continue
            json_cols.append(col)
            out[col] = out[col].apply(
                lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list))
                else value
            )
        return out, json_cols

    def _prepare_rows_for_archive(self, rows: List[Dict], spec: ArchiveSpec) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        if spec.time_kind == "date":
            df[spec.time_column] = pd.to_datetime(df[spec.time_column].astype(str), errors="coerce").dt.date
        else:
            df[spec.time_column] = pd.to_datetime(df[spec.time_column], utc=True, errors="coerce")
        df["_partition_date"] = (
            pd.to_datetime(df[spec.time_column].astype(str), errors="coerce").dt.date
            if spec.time_kind == "date"
            else pd.to_datetime(df[spec.time_column], utc=True, errors="coerce").dt.date
        )
        if spec.symbol_column and spec.symbol_column in df.columns:
            df["_partition_symbol"] = df[spec.symbol_column].fillna("").astype(str)
        else:
            df["_partition_symbol"] = ""
        return df.dropna(subset=["_partition_date"]).reset_index(drop=True)

    def _retention_cutoff(self, spec: ArchiveSpec):
        now = datetime.now(timezone.utc)
        if spec.time_kind == "date":
            return (now - timedelta(days=spec.retention_days)).date()
        return now - timedelta(days=spec.retention_days)

    def _fetch_all_rows(self, table: str, time_col: str, cutoff_value, page_size: int = 1000) -> List[Dict]:
        all_rows: List[Dict] = []
        page = 0
        cutoff_str = cutoff_value.isoformat() if hasattr(cutoff_value, "isoformat") else str(cutoff_value)
        while True:
            start = page * page_size
            end = start + page_size - 1
            response = (
                db.client.table(table)
                .select("*")
                .lt(time_col, cutoff_str)
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

    def _fetch_partition_rows(self, spec: ArchiveSpec, partition_date: date, symbol: str = "", page_size: int = 1000) -> List[Dict]:
        rows: List[Dict] = []
        page = 0
        start_str = partition_date.isoformat()
        end_str = (partition_date + timedelta(days=1)).isoformat()
        while True:
            start = page * page_size
            end = start + page_size - 1
            query = db.client.table(spec.table_name).select("*").gte(spec.time_column, start_str).lt(spec.time_column, end_str)
            if spec.symbol_column:
                query = query.eq(spec.symbol_column, symbol)
            response = query.order(spec.time_column, desc=False).range(start, end).execute()
            chunk = response.data if response.data else []
            if not chunk:
                break
            rows.extend(chunk)
            if len(chunk) < page_size:
                break
            page += 1
        return rows

    def _delete_partition_rows(self, spec: ArchiveSpec, partition_date: date, symbol: str = "") -> Dict:
        start_str = partition_date.isoformat()
        end_str = (partition_date + timedelta(days=1)).isoformat()
        query = db.client.table(spec.table_name).delete().gte(spec.time_column, start_str).lt(spec.time_column, end_str)
        if spec.symbol_column:
            query = query.eq(spec.symbol_column, symbol)
        return query.execute()

    def _download_parquet(self, object_path: str) -> Optional[pd.DataFrame]:
        try:
            now_utc = datetime.now(timezone.utc)
            current_month_dt = now_utc.strftime("%Y-%m")
            prev_month_dt = (now_utc - timedelta(days=32)).strftime("%Y-%m")
            is_mutable = (current_month_dt in object_path) or (prev_month_dt in object_path)
            safe_name = object_path.replace("/", "_").replace("\\", "_")
            cache_path = self.cache_dir / safe_name
            if cache_path.exists() and not is_mutable:
                return pd.read_parquet(cache_path, engine="pyarrow")
            blob = self.bucket.blob(object_path)
            if not blob.exists():
                return None
            payload = blob.download_as_bytes()
            df = pd.read_parquet(BytesIO(payload), engine="pyarrow")
            if not is_mutable and not df.empty:
                try:
                    cache_path.write_bytes(payload)
                except Exception as e:
                    logger.debug(f"Failed to write GCS cache to disk: {e}")
            return df
        except Exception as e:
            logger.debug(f"Parquet load failed ({object_path}): {e}")
            return None

    def _object_path(self, spec: ArchiveSpec, partition_date: date, symbol: str = "") -> str:
        day_str = partition_date.isoformat()
        if symbol:
            return f"{spec.gcs_prefix}/{symbol}/{day_str}.parquet"
        return f"{spec.gcs_prefix}/{day_str}.parquet"

    def _partition_key(self, partition_date: date, symbol: str = "") -> str:
        return f"{partition_date.isoformat()}|{symbol}" if symbol else partition_date.isoformat()

    def _verify_partition(self, original_df: pd.DataFrame, archived_df: Optional[pd.DataFrame], spec: ArchiveSpec) -> tuple[bool, str]:
        if archived_df is None or archived_df.empty:
            return False, "archived parquet missing or empty"
        if len(original_df) != len(archived_df):
            return False, f"row_count mismatch db={len(original_df)} parquet={len(archived_df)}"
        if spec.pk_column in original_df.columns and spec.pk_column in archived_df.columns:
            if int(original_df[spec.pk_column].min()) != int(archived_df[spec.pk_column].min()):
                return False, "min_pk mismatch"
            if int(original_df[spec.pk_column].max()) != int(archived_df[spec.pk_column].max()):
                return False, "max_pk mismatch"
        return True, ""

    def _manifest_base_payload(self, spec: ArchiveSpec, partition_date: date, symbol: str, object_path: str, cutoff_value, parquet_df: pd.DataFrame, json_cols: List[str]) -> Dict:
        partition_end = partition_date + timedelta(days=1)
        return {
            "table_name": spec.table_name,
            "partition_key": self._partition_key(partition_date, symbol),
            "gcs_path": object_path,
            "status": "pending",
            "schema_version": spec.schema_version,
            "time_column": spec.time_column,
            "pk_column": spec.pk_column,
            "symbol_column": spec.symbol_column,
            "partition_start": datetime.combine(partition_date, datetime.min.time(), tzinfo=timezone.utc).isoformat() if spec.time_kind != "date" else None,
            "partition_end": datetime.combine(partition_end, datetime.min.time(), tzinfo=timezone.utc).isoformat() if spec.time_kind != "date" else None,
            "partition_start_date": partition_date.isoformat() if spec.time_kind == "date" else None,
            "partition_end_date": partition_end.isoformat() if spec.time_kind == "date" else None,
            "partition_symbol": symbol if spec.symbol_column else None,
            "archive_started_at": datetime.now(timezone.utc).isoformat(),
            "archive_completed_at": None,
            "verified_at": None,
            "cleanup_completed_at": None,
            "retention_cutoff": cutoff_value.isoformat() if spec.time_kind != "date" else None,
            "retention_cutoff_date": cutoff_value.isoformat() if spec.time_kind == "date" else None,
            "row_count_db": int(len(parquet_df)),
            "row_count_parquet": 0,
            "min_pk": int(parquet_df[spec.pk_column].min()) if spec.pk_column in parquet_df.columns else None,
            "max_pk": int(parquet_df[spec.pk_column].max()) if spec.pk_column in parquet_df.columns else None,
            "min_time": (
                pd.to_datetime(parquet_df[spec.time_column], utc=True, errors="coerce").min().isoformat()
                if spec.time_kind != "date" and spec.time_column in parquet_df.columns and not parquet_df.empty
                else None
            ),
            "max_time": (
                pd.to_datetime(parquet_df[spec.time_column], utc=True, errors="coerce").max().isoformat()
                if spec.time_kind != "date" and spec.time_column in parquet_df.columns and not parquet_df.empty
                else None
            ),
            "min_date": min(parquet_df[spec.time_column]).isoformat() if spec.time_kind == "date" and not parquet_df.empty else None,
            "max_date": max(parquet_df[spec.time_column]).isoformat() if spec.time_kind == "date" and not parquet_df.empty else None,
            "file_size_bytes": None,
            "md5_hash": None,
            "error_message": None,
            "metadata": {
                "columns": [str(col) for col in parquet_df.columns],
                "json_columns": json_cols,
                "cleanup_enabled": spec.cleanup_enabled,
            },
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _archive_partition(self, spec: ArchiveSpec, partition_df: pd.DataFrame, partition_date: date, symbol: str, cutoff_value) -> Dict:
        partition_key = self._partition_key(partition_date, symbol)
        existing = db.get_archive_manifest(spec.table_name, partition_key)
        if existing and existing.get("status") in ("verified", "cleanup_done"):
            return {"partition_key": partition_key, "status": "skipped_verified", "rows": int(existing.get("row_count_db") or 0)}

        parquet_df, json_cols = self._normalize_for_parquet(
            partition_df.drop(columns=["_partition_date", "_partition_symbol"], errors="ignore")
        )
        object_path = self._object_path(spec, partition_date, symbol)
        manifest_row = db.upsert_archive_manifest(
            self._manifest_base_payload(spec, partition_date, symbol, object_path, cutoff_value, parquet_df, json_cols)
        )

        try:
            payload, payload_md5 = self._build_parquet_payload(parquet_df)
            file_size = self._upload_parquet_payload(object_path, payload)
            archived_df = self._download_parquet(object_path)
            verified, verify_error = self._verify_partition(parquet_df, archived_df, spec)
            update_payload = {
                "status": "verified" if verified else "failed",
                "archive_completed_at": datetime.now(timezone.utc).isoformat(),
                "verified_at": datetime.now(timezone.utc).isoformat() if verified else None,
                "row_count_parquet": int(len(archived_df)) if archived_df is not None else 0,
                "file_size_bytes": int(file_size),
                "md5_hash": payload_md5,
                "error_message": verify_error or None,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            if manifest_row and manifest_row.get("id") is not None:
                db.update_archive_manifest(int(manifest_row["id"]), update_payload)
            return {"partition_key": partition_key, "status": update_payload["status"], "rows": int(len(parquet_df))}
        except Exception as e:
            logger.error(f"Archive partition error for {spec.table_name}/{partition_key}: {e}")
            if manifest_row and manifest_row.get("id") is not None:
                db.update_archive_manifest(
                    int(manifest_row["id"]),
                    {"status": "failed", "error_message": str(e), "updated_at": datetime.now(timezone.utc).isoformat()},
                )
            return {"partition_key": partition_key, "status": "failed", "rows": 0, "error": str(e)}

    def _archive_table(self, spec: ArchiveSpec) -> Dict:
        cutoff_value = self._retention_cutoff(spec)
        rows = self._fetch_all_rows(spec.table_name, spec.time_column, cutoff_value)
        if not rows:
            return {"table": spec.table_name, "rows": 0, "partitions": 0}
        prepared = self._prepare_rows_for_archive(rows, spec)
        if prepared.empty:
            return {"table": spec.table_name, "rows": 0, "partitions": 0}
        results = []
        group_cols = ["_partition_date", "_partition_symbol"] if spec.symbol_column else ["_partition_date"]
        for keys, group in prepared.groupby(group_cols):
            if isinstance(keys, tuple):
                partition_date, symbol = keys[0], str(keys[1] or "")
            else:
                partition_date, symbol = keys, ""
            results.append(self._archive_partition(spec, group.reset_index(drop=True), partition_date, symbol, cutoff_value))
        return {
            "table": spec.table_name,
            "rows": int(len(prepared)),
            "partitions": len(results),
            "verified_partitions": sum(1 for item in results if item.get("status") == "verified"),
            "failed_partitions": sum(1 for item in results if item.get("status") == "failed"),
            "skipped_partitions": sum(1 for item in results if item.get("status") == "skipped_verified"),
        }

    def run_daily_archive(self) -> Dict:
        if not self.enabled:
            return {"enabled": False, "reason": "GCS archive disabled"}
        summary = {
            "enabled": True,
            "bucket": self.bucket_name,
            "archived_at": datetime.now(timezone.utc).isoformat(),
            "tables": {},
        }
        for spec in self.archive_specs:
            try:
                summary["tables"][spec.table_name] = self._archive_table(spec)
            except Exception as e:
                logger.error(f"Archive table error for {spec.table_name}: {e}")
                summary["tables"][spec.table_name] = {"table": spec.table_name, "error": str(e)}
        logger.info(f"GCS Parquet archive complete: {summary}")
        return summary

    def _manifest_partition_date(self, manifest: Dict, spec: ArchiveSpec) -> Optional[date]:
        raw_value = manifest.get("partition_start_date") if spec.time_kind == "date" else manifest.get("partition_start")
        if raw_value is None:
            return None
        return date.fromisoformat(str(raw_value)) if spec.time_kind == "date" else datetime.fromisoformat(str(raw_value).replace("Z", "+00:00")).date()

    def _cleanup_manifest(self, manifest: Dict) -> Dict:
        table_name = str(manifest.get("table_name", ""))
        spec = self._spec_by_table.get(table_name)
        if spec is None or not spec.cleanup_enabled:
            return {"table": table_name, "status": "skipped"}
        partition_date = self._manifest_partition_date(manifest, spec)
        if partition_date is None:
            return {"table": table_name, "status": "cleanup_failed"}
        symbol = str(manifest.get("partition_symbol") or "")
        current_rows = self._fetch_partition_rows(spec, partition_date, symbol)
        expected_rows = int(manifest.get("row_count_db") or 0)
        manifest_id = int(manifest["id"])
        if not current_rows:
            db.update_archive_manifest(
                manifest_id,
                {"status": "cleanup_done", "cleanup_completed_at": datetime.now(timezone.utc).isoformat(), "error_message": None, "updated_at": datetime.now(timezone.utc).isoformat()},
            )
            return {"table": table_name, "status": "cleanup_done", "rows": 0}
        if len(current_rows) != expected_rows:
            db.update_archive_manifest(
                manifest_id,
                {"status": "cleanup_failed", "error_message": f"pre-delete row mismatch current={len(current_rows)} expected={expected_rows}", "updated_at": datetime.now(timezone.utc).isoformat()},
            )
            return {"table": table_name, "status": "cleanup_failed", "rows": len(current_rows)}
        self._delete_partition_rows(spec, partition_date, symbol)
        remaining_rows = self._fetch_partition_rows(spec, partition_date, symbol)
        if remaining_rows:
            db.update_archive_manifest(
                manifest_id,
                {"status": "cleanup_failed", "error_message": f"post-delete remaining_rows={len(remaining_rows)}", "updated_at": datetime.now(timezone.utc).isoformat()},
            )
            return {"table": table_name, "status": "cleanup_failed", "rows": len(remaining_rows)}
        db.update_archive_manifest(
            manifest_id,
            {"status": "cleanup_done", "cleanup_completed_at": datetime.now(timezone.utc).isoformat(), "error_message": None, "updated_at": datetime.now(timezone.utc).isoformat()},
        )
        return {"table": table_name, "status": "cleanup_done", "rows": expected_rows}

    def run_safe_cleanup(self, limit: int = 500) -> Dict:
        manifests = db.get_archive_manifests(statuses=["verified", "cleanup_failed"], cleanup_pending_only=True, limit=limit)
        summary = {
            "enabled": True,
            "cleanup_at": datetime.now(timezone.utc).isoformat(),
            "manifests": len(manifests),
            "tables": {},
        }
        for manifest in manifests:
            result = self._cleanup_manifest(manifest)
            table_name = result.get("table", "unknown")
            bucket = summary["tables"].setdefault(table_name, {"cleanup_done": 0, "cleanup_failed": 0, "skipped": 0, "rows": 0})
            status = result.get("status")
            if status == "cleanup_done":
                bucket["cleanup_done"] += 1
                bucket["rows"] += int(result.get("rows") or 0)
            elif status == "cleanup_failed":
                bucket["cleanup_failed"] += 1
            else:
                bucket["skipped"] += 1
        logger.info(f"GCS Parquet safe cleanup complete: {summary}")
        return summary

    def load_ohlcv(self, timeframe: str, symbol: str, months_back: int = 3) -> pd.DataFrame:
        if not self.enabled:
            return pd.DataFrame()
        dfs = []
        now = datetime.now(timezone.utc)
        if timeframe == "1m":
            for day_offset in range(max(1, months_back * 31)):
                day = (now - timedelta(days=day_offset)).date().isoformat()
                df = self._download_parquet(f"ohlcv/1m/{symbol}/{day}.parquet")
                if df is not None:
                    dfs.append(df)
            if not dfs:
                for m in range(months_back):
                    month = (now - timedelta(days=30 * m)).strftime("%Y-%m")
                    df = self._download_parquet(f"ohlcv/1m/{symbol}/{month}.parquet")
                    if df is not None:
                        dfs.append(df)
        elif timeframe in ("1d", "1w"):
            years_back = max(2, (months_back + 11) // 12)
            for year_offset in range(years_back):
                year = (now - timedelta(days=365 * year_offset)).strftime("%Y")
                df = self._download_parquet(f"ohlcv/{timeframe}/{symbol}/{year}.parquet")
                if df is not None:
                    dfs.append(df)
        else:
            for m in range(months_back):
                month = (now - timedelta(days=30 * m)).strftime("%Y-%m")
                df = self._download_parquet(f"ohlcv/{timeframe}/{symbol}/{month}.parquet")
                if df is not None:
                    dfs.append(df)
        if not dfs:
            return pd.DataFrame()
        result = pd.concat(dfs, ignore_index=True)
        if pd.api.types.is_numeric_dtype(result["timestamp"]):
            result["timestamp"] = pd.to_datetime(result["timestamp"], unit="ms", utc=True)
        else:
            result["timestamp"] = pd.to_datetime(result["timestamp"].astype(str), format="mixed", utc=True, errors="coerce").bfill()
        return result.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    def load_timeseries(self, prefix: str, symbol: str, months_back: int = 3) -> pd.DataFrame:
        if not self.enabled:
            return pd.DataFrame()
        dfs = []
        now = datetime.now(timezone.utc)
        for day_offset in range(max(1, months_back * 31)):
            day = (now - timedelta(days=day_offset)).date().isoformat()
            df = self._download_parquet(f"{prefix}/{symbol}/{day}.parquet")
            if df is not None:
                dfs.append(df)
        if not dfs:
            for m in range(months_back):
                month = (now - timedelta(days=30 * m)).strftime("%Y-%m")
                df = self._download_parquet(f"{prefix}/{symbol}/{month}.parquet")
                if df is not None:
                    dfs.append(df)
        if not dfs:
            return pd.DataFrame()
        result = pd.concat(dfs, ignore_index=True, copy=False)
        time_col = "timestamp" if "timestamp" in result.columns else result.columns[0]
        if pd.api.types.is_numeric_dtype(result[time_col]):
            result[time_col] = pd.to_datetime(result[time_col], unit="ms", utc=True)
        else:
            if not pd.api.types.is_datetime64_any_dtype(result[time_col]):
                result[time_col] = pd.to_datetime(result[time_col], format="mixed", utc=True, errors="coerce")
        return result.sort_values(time_col).reset_index(drop=True)


gcs_parquet_store = GCSParquetStore()
