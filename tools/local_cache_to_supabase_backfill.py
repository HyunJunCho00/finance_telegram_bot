"""Local Parquet Cache → Supabase Backfill Script.

Usage:
    python tools/local_cache_to_supabase_backfill.py [--dry-run] [--tables TABLE ...] [--since YYYY-MM-DD]

Examples:
    # dry-run (아무것도 쓰지 않음, 파일 목록만 출력)
    python tools/local_cache_to_supabase_backfill.py --dry-run

    # 특정 테이블만
    python tools/local_cache_to_supabase_backfill.py --tables ohlcv funding

    # 특정 날짜 이후만
    python tools/local_cache_to_supabase_backfill.py --since 2026-03-19

    # 전체 업로드
    python tools/local_cache_to_supabase_backfill.py

Supported table prefixes:
    ohlcv          → market_data          (batch_insert_market_data)
    cvd            → cvd_data             (batch_upsert_cvd_data)
    funding        → funding_data         (upsert_funding_data per row)
    liquidations   → liquidations         (batch_upsert_liquidations)
    microstructure → microstructure_data  (batch_upsert_microstructure_data)
    deribit        → deribit_data         (upsert_deribit_data per row)

Local cache path pattern:
    cache/gcs_parquet/{prefix}_{symbol}_{YYYY-MM-DD}.parquet
"""

from __future__ import annotations

import argparse
import sys
import os

# project root → sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from config.database import db


CACHE_DIR = Path("cache/gcs_parquet")

# prefix → handler config
# handler: "batch" uses a single list call; "per_row" calls upsert once per row
TABLE_CONFIG = {
    "ohlcv": {
        "table": "market_data",
        "handler": "batch",
        "batch_fn": lambda rows: db.batch_insert_market_data(rows),
        "batch_size": 500,
        "required_cols": ["timestamp", "symbol", "open", "high", "low", "close", "volume"],
    },
    "cvd": {
        "table": "cvd_data",
        "handler": "batch",
        "batch_fn": lambda rows: db.batch_upsert_cvd_data(rows),
        "batch_size": 500,
        "required_cols": ["timestamp", "symbol"],
    },
    "funding": {
        "table": "funding_data",
        "handler": "per_row",
        "row_fn": lambda row: db.upsert_funding_data(row),
        "required_cols": ["timestamp", "symbol"],
    },
    "liquidations": {
        "table": "liquidations",
        "handler": "batch",
        "batch_fn": lambda rows: db.batch_upsert_liquidations(rows),
        "batch_size": 500,
        "required_cols": ["timestamp", "symbol"],
    },
    "microstructure": {
        "table": "microstructure_data",
        "handler": "batch",
        "batch_fn": lambda rows: db.batch_upsert_microstructure_data(rows),
        "batch_size": 500,
        "required_cols": ["timestamp", "symbol"],
    },
    "deribit": {
        "table": "deribit_data",
        "handler": "per_row",
        "row_fn": lambda row: db.upsert_deribit_data(row),
        "required_cols": ["timestamp"],
    },
}

# local_cache_path translates "/" → "_", so:
#   ohlcv/1m/{sym}/{date}.parquet  → ohlcv_1m_{sym}_{date}.parquet
# We match by checking which prefix the filename starts with.
PREFIX_FILE_PREFIXES = {
    "ohlcv":          "ohlcv_1m_",   # price_collector writes ohlcv/1m/{sym}/{date}
    "cvd":            "cvd_",
    "funding":        "funding_",
    "liquidations":   "liquidations_",
    "microstructure": "microstructure_",
    "deribit":        "deribit_",
}


def _find_cache_files(table_prefixes: List[str], since: Optional[date]) -> Dict[str, List[Path]]:
    """Scan cache dir and return {prefix: [parquet_path, ...]} filtered by date."""
    result: Dict[str, List[Path]] = {p: [] for p in table_prefixes}

    if not CACHE_DIR.exists():
        logger.warning(f"Cache dir not found: {CACHE_DIR}")
        return result

    for f in sorted(CACHE_DIR.glob("*.parquet")):
        for prefix in table_prefixes:
            file_prefix = PREFIX_FILE_PREFIXES[prefix]
            if not f.name.startswith(file_prefix):
                continue
            # extract date from filename: last segment before .parquet
            # e.g.  ohlcv_1m_BTCUSDT_2026-03-19.parquet → 2026-03-19
            parts = f.stem.split("_")
            date_str = None
            for i in range(len(parts) - 1, -1, -1):
                candidate = parts[i]
                # date looks like YYYY-MM-DD when split on "_" → "2026-03-19" but stem splits on "_"
                # Actually YYYY-MM-DD split by "_" would be ["2026-03-19"], hmm.
                # The full stem is e.g. "ohlcv_1m_BTCUSDT_2026-03-19"
                # so parts[-1] = "2026-03-19" — hyphen, not underscore
                break
            last_part = parts[-1]  # "2026-03-19"
            try:
                file_date = date.fromisoformat(last_part)
            except ValueError:
                continue
            if since and file_date < since:
                continue
            result[prefix].append(f)
            break

    return result


def _rows_to_upload(df: pd.DataFrame, required_cols: List[str]) -> List[Dict]:
    """Convert DataFrame to list of dicts, drop rows missing required cols."""
    # ensure timestamp is iso string (Supabase expects string)
    if "timestamp" in df.columns:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.warning(f"Required columns missing from parquet: {missing}")
        return []

    # drop rows where required cols are null
    df = df.dropna(subset=required_cols)
    # drop internal partition columns if present
    drop_cols = [c for c in df.columns if c.startswith("_partition_")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df.to_dict(orient="records")


def _upload_batch(rows: List[Dict], cfg: Dict, dry_run: bool) -> int:
    """Upload a batch of rows. Returns number of rows uploaded."""
    if not rows:
        return 0
    if dry_run:
        return len(rows)

    batch_size = cfg.get("batch_size", 500)
    handler = cfg["handler"]
    uploaded = 0

    if handler == "batch":
        for i in range(0, len(rows), batch_size):
            chunk = rows[i : i + batch_size]
            try:
                cfg["batch_fn"](chunk)
                uploaded += len(chunk)
            except Exception as e:
                logger.error(f"Batch upload error (chunk {i}): {e}")
    elif handler == "per_row":
        for row in rows:
            try:
                cfg["row_fn"](row)
                uploaded += 1
            except Exception as e:
                logger.error(f"Per-row upload error: {e}")

    return uploaded


def run_backfill(
    table_prefixes: List[str],
    since: Optional[date] = None,
    dry_run: bool = False,
) -> Dict:
    label = "[DRY-RUN] " if dry_run else ""
    logger.info(f"{label}Starting local cache → Supabase backfill")
    logger.info(f"  tables : {table_prefixes}")
    logger.info(f"  since  : {since or 'all'}")

    file_map = _find_cache_files(table_prefixes, since)

    summary: Dict = {}
    for prefix, files in file_map.items():
        cfg = TABLE_CONFIG[prefix]
        table = cfg["table"]
        total_files = len(files)
        total_rows = 0
        uploaded_rows = 0
        skipped_files = 0

        logger.info(f"\n--- {prefix} → {table}: {total_files} file(s) ---")

        for f in files:
            try:
                df = pd.read_parquet(f, engine="pyarrow")
                if df.empty:
                    logger.debug(f"  skip (empty): {f.name}")
                    skipped_files += 1
                    continue

                rows = _rows_to_upload(df, cfg["required_cols"])
                if not rows:
                    logger.debug(f"  skip (no valid rows): {f.name}")
                    skipped_files += 1
                    continue

                n = _upload_batch(rows, cfg, dry_run)
                uploaded_rows += n
                total_rows += len(rows)
                logger.info(f"  {label}{f.name}: {n}/{len(rows)} rows → {table}")

            except Exception as e:
                logger.error(f"  Error reading {f.name}: {e}")
                skipped_files += 1

        summary[prefix] = {
            "table": table,
            "files": total_files,
            "skipped_files": skipped_files,
            "total_rows": total_rows,
            "uploaded_rows": uploaded_rows,
        }
        logger.info(
            f"  {label}Done: {uploaded_rows}/{total_rows} rows, "
            f"{total_files - skipped_files}/{total_files} files OK"
        )

    logger.info(f"\n{label}Backfill complete: {summary}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Upload local parquet cache → Supabase")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan files and count rows but do not write to Supabase",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        choices=list(TABLE_CONFIG.keys()),
        default=list(TABLE_CONFIG.keys()),
        help=f"Tables to backfill (default: all). Choices: {list(TABLE_CONFIG.keys())}",
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="Only upload files dated on or after YYYY-MM-DD",
    )
    args = parser.parse_args()

    since_date: Optional[date] = None
    if args.since:
        try:
            since_date = date.fromisoformat(args.since)
        except ValueError:
            logger.error(f"Invalid --since date: {args.since}. Use YYYY-MM-DD.")
            sys.exit(1)

    run_backfill(
        table_prefixes=args.tables,
        since=since_date,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
