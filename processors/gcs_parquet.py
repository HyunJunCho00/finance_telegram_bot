"""GCS Parquet storage for verified cold archival and historical reads."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
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
        execution_days = int(getattr(settings, "RETENTION_EXECUTIONS_DAYS", report_days))
        evaluation_days = int(getattr(settings, "RETENTION_EVALUATION_DAYS", report_days))
        evaluation_component_days = int(getattr(settings, "RETENTION_EVALUATION_COMPONENT_DAYS", report_days))
        rollup_days = int(getattr(settings, "RETENTION_ROLLUPS_DAYS", report_days * 10))

        # 로컬 parquet으로 이전된 테이블은 여기서 제외:
        # market_data, cvd_data, funding_data, telegram_messages,
        # macro_data, deribit_data, fear_greed_data, onchain_daily_snapshots
        return [
            ArchiveSpec("narrative_data", "timestamp", report_days, "archive/narrative_data", symbol_column="symbol"),
            ArchiveSpec("ai_reports", "created_at", report_days, "archive/ai_reports", symbol_column="symbol"),
            ArchiveSpec("feedback_logs", "created_at", report_days, "archive/feedback_logs", symbol_column="symbol"),
            ArchiveSpec("trade_executions", "created_at", execution_days, "archive/trade_executions", symbol_column="symbol"),
            ArchiveSpec("dune_query_results", "collected_at", report_days, "archive/dune_query_results"),
            ArchiveSpec("microstructure_data", "timestamp", market_days, "archive/microstructure_data", symbol_column="symbol"),
            ArchiveSpec("liquidations", "timestamp", market_days, "liquidations", symbol_column="symbol"),
            ArchiveSpec("daily_playbooks", "created_at", report_days, "archive/daily_playbooks", symbol_column="symbol"),
            ArchiveSpec("monitor_logs", "created_at", report_days, "archive/monitor_logs", symbol_column="symbol"),
            ArchiveSpec("market_status_events", "created_at", report_days, "archive/market_status_events", symbol_column="symbol"),
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

    def _local_cache_path(self, object_path: str) -> Path:
        safe_name = object_path.replace("/", "_").replace("\\", "_")
        return self.cache_dir / safe_name

    def _is_mutable_path(self, object_path: str) -> bool:
        now_utc = datetime.now(timezone.utc)
        current_month_dt = now_utc.strftime("%Y-%m")
        prev_month_dt = (now_utc - timedelta(days=32)).strftime("%Y-%m")
        return (current_month_dt in object_path) or (prev_month_dt in object_path)

    def _read_local_cache(self, object_path: str) -> Optional[pd.DataFrame]:
        """로컬 캐시만 읽음 (GCS 미호출). GCS 비활성화 시 fallback용."""
        try:
            cache_path = self._local_cache_path(object_path)
            if cache_path.exists() and not self._is_mutable_path(object_path):
                return pd.read_parquet(cache_path, engine="pyarrow")
        except Exception as e:
            logger.debug(f"Local cache read failed ({object_path}): {e}")
        return None

    def _download_parquet(self, object_path: str) -> Optional[pd.DataFrame]:
        """로컬 캐시 우선 → GCS fallback.

        price_collector가 ohlcv/1m 과 cvd 경로에 매분 직접 기록하므로
        mutable 경로라도 로컬 캐시가 있으면 GCS 다운로드를 건너뜁니다.
        → Supabase/GCS egress 0.
        """
        try:
            cache_path = self._local_cache_path(object_path)
            is_mutable = self._is_mutable_path(object_path)

            # 로컬 캐시 우선 (mutable 포함)
            if cache_path.exists():
                return pd.read_parquet(cache_path, engine="pyarrow")

            # 로컬 없으면 GCS fallback
            if not self.enabled:
                return None
            blob = self.bucket.blob(object_path)
            if not blob.exists():
                return None
            payload = blob.download_as_bytes()
            df = pd.read_parquet(BytesIO(payload), engine="pyarrow")
            # GCS에서 받은 데이터도 로컬에 저장 (이후 재요청 시 로컬 hit)
            if not df.empty:
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

    def _build_ohlcv_paths(self, timeframe: str, symbol: str, months_back: int) -> list:
        """타임프레임·기간에 따른 GCS object_path 목록 반환."""
        paths = []
        now = datetime.now(timezone.utc)
        if timeframe == "1m":
            for day_offset in range(max(1, months_back * 31)):
                day = (now - timedelta(days=day_offset)).date().isoformat()
                paths.append(f"ohlcv/1m/{symbol}/{day}.parquet")
            if not paths:
                for m in range(months_back):
                    month = (now - timedelta(days=30 * m)).strftime("%Y-%m")
                    paths.append(f"ohlcv/1m/{symbol}/{month}.parquet")
        elif timeframe in ("1d", "1w"):
            years_back = max(2, (months_back + 11) // 12)
            for year_offset in range(years_back):
                year = (now - timedelta(days=365 * year_offset)).strftime("%Y")
                paths.append(f"ohlcv/{timeframe}/{symbol}/{year}.parquet")
        else:
            for m in range(months_back):
                month = (now - timedelta(days=30 * m)).strftime("%Y-%m")
                paths.append(f"ohlcv/{timeframe}/{symbol}/{month}.parquet")
        return paths

    def load_ohlcv(self, timeframe: str, symbol: str, months_back: int = 3) -> pd.DataFrame:
        paths = self._build_ohlcv_paths(timeframe, symbol, months_back)
        read_fn = self._download_parquet
        workers = min(8, max(1, len(paths)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(read_fn, paths))
        dfs = [df for df in results if df is not None]
        if not dfs:
            return pd.DataFrame()
        result = pd.concat(dfs, ignore_index=True)
        if pd.api.types.is_numeric_dtype(result["timestamp"]):
            result["timestamp"] = pd.to_datetime(result["timestamp"], unit="ms", utc=True)
        else:
            result["timestamp"] = pd.to_datetime(result["timestamp"].astype(str), format="mixed", utc=True, errors="coerce").bfill()
        return result.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    def load_timeseries(self, prefix: str, symbol: str, months_back: int = 3) -> pd.DataFrame:
        dfs = []
        now = datetime.now(timezone.utc)
        paths = []
        for day_offset in range(max(1, months_back * 31)):
            day = (now - timedelta(days=day_offset)).date().isoformat()
            paths.append(f"{prefix}/{symbol}/{day}.parquet")
        if not paths:
            for m in range(months_back):
                month = (now - timedelta(days=30 * m)).strftime("%Y-%m")
                paths.append(f"{prefix}/{symbol}/{month}.parquet")
        for path in paths:
            df = self._download_parquet(path)
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


    # ------------------------------------------------------------------
    # 로컬 직접 쓰기 API (price_collector 전용)
    # Supabase / GCS 없이 VM 로컬 disk에 기록 → load_ohlcv/load_timeseries
    # 가 동일 경로를 읽으므로 분석 파이프라인 변경 불필요.
    # ------------------------------------------------------------------

    def _merge_to_local_cache(self, object_path: str, new_df: pd.DataFrame, dedup_cols: List[str]) -> int:
        """로컬 캐시 parquet에 merge-upsert."""
        try:
            cache_path = self._local_cache_path(object_path)
            if cache_path.exists():
                existing = pd.read_parquet(cache_path, engine="pyarrow")
                merged = pd.concat([existing, new_df], ignore_index=True)
                merged = merged.drop_duplicates(subset=dedup_cols, keep="last")
                merged = merged.sort_values(dedup_cols[0]).reset_index(drop=True)
            else:
                merged = new_df.sort_values(dedup_cols[0]).reset_index(drop=True)
            payload, _ = self._build_parquet_payload(merged)
            cache_path.write_bytes(payload)
            return len(merged)
        except Exception as e:
            logger.error(f"_merge_to_local_cache failed ({object_path}): {e}")
            return 0

    def write_ohlcv_to_local(self, symbol: str, df: pd.DataFrame) -> int:
        """OHLCV 1m 데이터를 오늘 날짜 로컬 캐시 parquet에 merge 기록.

        경로: cache/gcs_parquet/ohlcv_1m_{symbol}_{today}.parquet
        → load_ohlcv("1m", symbol) 가 동일 파일을 읽음.
        """
        if df is None or df.empty:
            return 0
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        today = datetime.now(timezone.utc).date().isoformat()
        object_path = f"ohlcv/1m/{symbol}/{today}.parquet"
        return self._merge_to_local_cache(object_path, df, ["timestamp"])

    def write_timeseries_to_local(
        self, prefix: str, symbol: str, df: pd.DataFrame, dedup_cols: List[str]
    ) -> int:
        """CVD / funding 등 시계열을 오늘 날짜 로컬 캐시 parquet에 merge 기록.

        경로: cache/gcs_parquet/{prefix}_{symbol}_{today}.parquet
        → load_timeseries(prefix, symbol) 가 동일 파일을 읽음.
        """
        if df is None or df.empty:
            return 0
        df = df.copy()
        today = datetime.now(timezone.utc).date().isoformat()
        object_path = f"{prefix}/{symbol}/{today}.parquet"
        return self._merge_to_local_cache(object_path, df, dedup_cols)

    # ------------------------------------------------------------------
    # Supabase 대체 읽기 API
    # QUANT 시계열 + telegram → 로컬 parquet에서 직접 읽기.
    # ------------------------------------------------------------------

    def get_latest_row(self, prefix: str, symbol: str = "global") -> Optional[Dict]:
        """prefix/symbol 경로에서 가장 최근 행 1개를 반환.

        오늘 → 어제 → 그제 순으로 탐색해 데이터가 있는 첫 파일의 마지막 행을 반환.
        fear_greed, macro, deribit, onchain 등 'latest snapshot' 패턴에 사용.
        """
        now = datetime.now(timezone.utc)
        for delta in range(3):
            day = (now - timedelta(days=delta)).date().isoformat()
            df = self._download_parquet(f"{prefix}/{symbol}/{day}.parquet")
            if df is not None and not df.empty:
                return df.iloc[-1].to_dict()
        return None

    def get_funding_history_parquet(self, symbol: str, limit: int = 100, since=None) -> pd.DataFrame:
        """funding/{symbol} 로컬 parquet에서 최근 N행 반환.

        db.get_funding_history() 대체용. since(datetime)로 시작 시점 필터 가능.
        """
        df = self.load_timeseries("funding", symbol, months_back=2)
        if df.empty:
            return df
        if since is not None:
            since_ts = pd.Timestamp(since)
            if since_ts.tzinfo is None:
                since_ts = since_ts.tz_localize("UTC")
            if "timestamp" in df.columns:
                df = df[df["timestamp"] >= since_ts]
        if limit > 0:
            return df.tail(limit).reset_index(drop=True)
        return df.reset_index(drop=True)

    def write_telegram_to_local(self, data: Dict) -> None:
        """telegram 메시지 1건을 오늘 날짜 로컬 parquet에 append."""
        try:
            df = pd.DataFrame([data])
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            self.write_timeseries_to_local("telegram", "global", df, ["message_id"])
        except Exception as e:
            logger.debug(f"Telegram local write failed: {e}")

    def get_max_telegram_message_id(self, channel: str) -> int:
        """채널별 마지막으로 처리된 message_id 반환 (중복 방지용).

        db.client.table('telegram_messages') 쿼리 대체.
        최근 3일치 parquet을 스캔해 해당 채널의 최대 message_id를 반환.
        """
        now = datetime.now(timezone.utc)
        max_id = 0
        for delta in range(3):
            day = (now - timedelta(days=delta)).date().isoformat()
            df = self._download_parquet(f"telegram/global/{day}.parquet")
            if df is None or df.empty:
                continue
            if "message_id" not in df.columns or "channel" not in df.columns:
                continue
            ch_df = df[df["channel"] == channel]
            if not ch_df.empty:
                max_id = max(max_id, int(ch_df["message_id"].max()))
        return max_id

    def get_recent_telegram_messages_parquet(self, hours: int = 24) -> List[Dict]:
        """최근 N시간 telegram 메시지 반환.

        db.get_recent_telegram_messages() 대체용.
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=hours)
        days_back = max(1, hours // 24 + 1)
        dfs = []
        for delta in range(days_back + 1):
            day = (now - timedelta(days=delta)).date().isoformat()
            df = self._download_parquet(f"telegram/global/{day}.parquet")
            if df is not None and not df.empty:
                dfs.append(df)
        if not dfs:
            return []
        result = pd.concat(dfs, ignore_index=True)
        if "timestamp" in result.columns:
            result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
            result = result[result["timestamp"] >= cutoff]
        return result.to_dict("records")

    def get_telegram_messages_for_rag_parquet(self, days: int = 7, limit: int = 1000) -> List[Dict]:
        """RAG 인덱싱용 최근 N일 telegram 메시지 반환.

        db.get_telegram_messages_for_rag() 대체용.
        """
        now = datetime.now(timezone.utc)
        dfs = []
        for delta in range(days + 1):
            day = (now - timedelta(days=delta)).date().isoformat()
            df = self._download_parquet(f"telegram/global/{day}.parquet")
            if df is not None and not df.empty:
                dfs.append(df)
        if not dfs:
            return []
        result = pd.concat(dfs, ignore_index=True)
        if "timestamp" in result.columns:
            result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
            result = result.sort_values("timestamp")
        return result.tail(limit).to_dict("records")


    def refresh_higher_tf_cache(self, symbol: str) -> None:
        """1m 로컬 캐시에서 4h/1d/1w를 재계산해 로컬 parquet 갱신.

        스케줄러에서 하루 1회 호출. 최근 2달치 4h gap을 자동으로 채움.
        - 4h: monthly 파일  (ohlcv/4h/{symbol}/{YYYY-MM}.parquet)
        - 1d: yearly 파일   (ohlcv/1d/{symbol}/{YYYY}.parquet)
        - 1w: yearly 파일   (ohlcv/1w/{symbol}/{YYYY}.parquet)
        """
        # 6개월치 1m 데이터 로드 (1d/1w 연간 파일도 최근 연도가 바뀔 수 있음)
        df_1m = self.load_ohlcv("1m", symbol, months_back=6)
        if df_1m is None or df_1m.empty:
            logger.warning(f"refresh_higher_tf_cache: no 1m data for {symbol}")
            return

        df_1m = df_1m.copy()
        df_1m["timestamp"] = pd.to_datetime(df_1m["timestamp"], utc=True, errors="coerce")
        df_1m = df_1m.dropna(subset=["timestamp"]).set_index("timestamp")
        ohlcv_cols = [c for c in ("open", "high", "low", "close", "volume") if c in df_1m.columns]
        df_1m = df_1m[ohlcv_cols].astype(float)

        agg = {c: ("first" if c == "open" else "max" if c == "high" else "min" if c == "low" else "last" if c == "close" else "sum") for c in ohlcv_cols}

        tf_configs = [
            ("4h",  "4h",  "%Y-%m"),   # monthly partitions
            ("1d",  "1D",  "%Y"),      # yearly partitions
            ("1w",  "1W-MON", "%Y"),   # yearly partitions (week anchored Monday)
        ]

        for tf, rule, period_fmt in tf_configs:
            try:
                resampled = df_1m.resample(rule, label="left", closed="left").agg(agg).dropna().reset_index()
                resampled["symbol"] = symbol

                resampled["_period"] = resampled["timestamp"].dt.strftime(period_fmt)
                for period, group in resampled.groupby("_period"):
                    group = group.drop(columns=["_period"]).reset_index(drop=True)
                    object_path = f"ohlcv/{tf}/{symbol}/{period}.parquet"
                    written = self._merge_to_local_cache(object_path, group, ["timestamp"])
                    logger.debug(f"[refresh_higher_tf] {object_path}: {written} rows")

                logger.info(f"[refresh_higher_tf] {symbol} {tf}: {len(resampled)} candles refreshed")
            except Exception as e:
                logger.error(f"refresh_higher_tf_cache failed for {symbol} {tf}: {e}")


gcs_parquet_store = GCSParquetStore()
