# -*- coding: utf-8 -*-
import json
import time
from types import SimpleNamespace
from typing import Dict, List, Optional
from datetime import datetime, timedelta, timezone

import psycopg2
import psycopg2.extras
import psycopg2.pool
import psycopg2.extensions
from loguru import logger
import pandas as pd

from config.settings import settings


# ---------------------------------------------------------------------------
# JSON adapter: Python dict/list → psycopg2 Json wrapper
# ---------------------------------------------------------------------------
psycopg2.extensions.register_adapter(dict, psycopg2.extras.Json)
psycopg2.extensions.register_adapter(list, psycopg2.extras.Json)


# ---------------------------------------------------------------------------
# Minimal Supabase-SDK-compatible query builder over psycopg2
# Supports the subset of the Supabase chained-builder API used in this repo.
# ---------------------------------------------------------------------------
class _QueryBuilder:
    def __init__(self, pool: psycopg2.pool.ThreadedConnectionPool, table: str):
        self._pool = pool
        self._table = table
        self._select_cols = "*"
        self._where_parts: List[str] = []
        self._params: List = []
        self._order_col: Optional[str] = None
        self._order_desc: bool = False
        self._limit_val: Optional[int] = None
        self._offset_val: Optional[int] = None
        self._op = "select"
        self._mutation_data = None
        self._on_conflict: Optional[str] = None
        self._update_data: Optional[Dict] = None

    # ── Filters ──────────────────────────────────────────────────────────────
    def eq(self, col: str, val):
        self._where_parts.append(f'"{col}" = %s')
        self._params.append(val)
        return self

    def lt(self, col: str, val):
        self._where_parts.append(f'"{col}" < %s')
        self._params.append(val)
        return self

    def lte(self, col: str, val):
        self._where_parts.append(f'"{col}" <= %s')
        self._params.append(val)
        return self

    def gte(self, col: str, val):
        self._where_parts.append(f'"{col}" >= %s')
        self._params.append(val)
        return self

    def in_(self, col: str, vals):
        self._where_parts.append(f'"{col}" = ANY(%s)')
        self._params.append(list(vals))
        return self

    def is_(self, col: str, val):
        if str(val).lower() == "null":
            self._where_parts.append(f'"{col}" IS NULL')
        else:
            self._where_parts.append(f'"{col}" IS NOT NULL')
        return self

    # ── Projection / ordering / paging ───────────────────────────────────────
    def select(self, cols: str = "*"):
        self._select_cols = cols
        return self

    def order(self, col: str, desc: bool = False):
        self._order_col = col
        self._order_desc = desc
        return self

    def limit(self, n: int):
        self._limit_val = n
        return self

    def range(self, start: int, end: int):
        self._offset_val = start
        self._limit_val = end - start + 1
        return self

    # ── Mutations ─────────────────────────────────────────────────────────────
    def insert(self, data):
        self._op = "insert"
        self._mutation_data = data
        return self

    def upsert(self, data, on_conflict: Optional[str] = None):
        self._op = "upsert"
        self._mutation_data = data
        self._on_conflict = on_conflict
        return self

    def update(self, data: Dict):
        self._op = "update"
        self._update_data = data
        return self

    def delete(self):
        self._op = "delete"
        return self

    # ── Execution ─────────────────────────────────────────────────────────────
    def execute(self) -> SimpleNamespace:
        return self._run(attempt=0)

    def _run(self, attempt: int) -> SimpleNamespace:
        conn = None
        try:
            conn = self._pool.getconn()
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                rows = self._dispatch(cur)
            conn.commit()
            self._pool.putconn(conn)
            return SimpleNamespace(data=rows)
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as exc:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
                self._pool.putconn(conn, close=True)
            if attempt == 0:
                logger.warning(f"[DB] Connection error on {self._table}.{self._op}, retrying… ({exc})")
                time.sleep(1.5)
                return self._run(attempt=1)
            raise
        except Exception:
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
                self._pool.putconn(conn)
            raise

    def _dispatch(self, cur):
        if self._op == "select":
            return self._do_select(cur)
        if self._op == "insert":
            return self._do_insert(cur)
        if self._op == "upsert":
            return self._do_upsert(cur)
        if self._op == "update":
            return self._do_update(cur)
        if self._op == "delete":
            return self._do_delete(cur)
        raise ValueError(f"Unknown op: {self._op}")

    def _where_clause(self):
        if not self._where_parts:
            return "", []
        return "WHERE " + " AND ".join(self._where_parts), list(self._params)

    def _do_select(self, cur) -> List[Dict]:
        where_sql, params = self._where_clause()
        order_sql = ""
        if self._order_col:
            order_sql = f'ORDER BY "{self._order_col}" {"DESC" if self._order_desc else "ASC"}'
        limit_sql = f"LIMIT {self._limit_val}" if self._limit_val is not None else ""
        offset_sql = f"OFFSET {self._offset_val}" if self._offset_val else ""
        sql = " ".join(filter(None, [
            f"SELECT {self._select_cols} FROM {self._table}",
            where_sql, order_sql, limit_sql, offset_sql
        ]))
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]

    def _do_insert(self, cur) -> List[Dict]:
        rows = self._mutation_data if isinstance(self._mutation_data, list) else [self._mutation_data]
        if not rows:
            return []
        cols = list(rows[0].keys())
        col_sql = ", ".join(f'"{c}"' for c in cols)
        placeholders = "(" + ", ".join(["%s"] * len(cols)) + ")"
        sql = f"INSERT INTO {self._table} ({col_sql}) VALUES {placeholders} RETURNING *"
        if len(rows) == 1:
            cur.execute(sql, [rows[0][c] for c in cols])
            return [dict(r) for r in cur.fetchall()]
        psycopg2.extras.execute_batch(
            cur,
            f"INSERT INTO {self._table} ({col_sql}) VALUES {placeholders}",
            [[r[c] for c in cols] for r in rows],
        )
        return []

    def _do_upsert(self, cur) -> List[Dict]:
        rows = self._mutation_data if isinstance(self._mutation_data, list) else [self._mutation_data]
        if not rows:
            return []
        cols = list(rows[0].keys())
        col_sql = ", ".join(f'"{c}"' for c in cols)
        placeholders = "(" + ", ".join(["%s"] * len(cols)) + ")"
        update_cols = [c for c in cols if c not in ("id", "created_at")]
        update_sql = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in update_cols)

        if self._on_conflict:
            conflict_cols = [c.strip() for c in self._on_conflict.split(",")]
            conflict_sql = "(" + ", ".join(f'"{c}"' for c in conflict_cols) + ")"
            do_clause = f"ON CONFLICT {conflict_sql} DO UPDATE SET {update_sql}"
        else:
            do_clause = "ON CONFLICT DO NOTHING"

        base_sql = f"INSERT INTO {self._table} ({col_sql}) VALUES {placeholders} {do_clause}"
        vals_list = [[r[c] for c in cols] for r in rows]

        if len(rows) == 1:
            cur.execute(base_sql + " RETURNING *", vals_list[0])
            return [dict(r) for r in cur.fetchall()]
        psycopg2.extras.execute_batch(cur, base_sql, vals_list)
        return []

    def _do_update(self, cur) -> List[Dict]:
        set_parts = [f'"{k}" = %s' for k in self._update_data]
        set_params = list(self._update_data.values())
        where_sql, where_params = self._where_clause()
        sql = f"UPDATE {self._table} SET {', '.join(set_parts)} {where_sql} RETURNING *"
        cur.execute(sql, set_params + where_params)
        return [dict(r) for r in cur.fetchall()]

    def _do_delete(self, cur) -> List[Dict]:
        where_sql, params = self._where_clause()
        sql = f"DELETE FROM {self._table} {where_sql}"
        cur.execute(sql, params)
        return []


# ---------------------------------------------------------------------------
# Table router: db.client.table("x") → _QueryBuilder
# ---------------------------------------------------------------------------
class _TableRouter:
    def __init__(self, pool: psycopg2.pool.ThreadedConnectionPool):
        self._pool = pool

    def table(self, table_name: str) -> _QueryBuilder:
        return _QueryBuilder(self._pool, table_name)


# ===========================================================================
# DatabaseClient
# ===========================================================================
class DatabaseClient:
    # Keep these sets for any external code that may reference them
    QUANT_TABLES = {
        "market_data", "cvd_data", "funding_data", "liquidations",
        "microstructure_data", "liquidation_cascade_features",
        "liquidation_cascade_predictions", "macro_data", "deribit_data",
        "fear_greed_data", "onchain_daily_snapshots", "archive_manifests",
    }
    TEXT_TABLES = {
        "telegram_messages", "narrative_data", "ai_reports", "feedback_logs",
        "trade_executions", "dune_query_results", "daily_playbooks",
        "monitor_logs", "market_status_events", "evaluation_predictions",
        "evaluation_outcomes", "evaluation_component_scores",
        "evaluation_rollups_daily", "paper_orders",
    }

    MARKET_DATA_OHLCV_COLUMNS = (
        "timestamp,symbol,exchange,open,high,low,close,volume,"
        "taker_buy_volume,taker_sell_volume"
    )

    def __init__(self):
        dsn = settings.DATABASE_URL
        if not dsn:
            raise RuntimeError(
                "DATABASE_URL is not set in .env — "
                "example: postgresql://botuser:pass@localhost:5432/financebot"
            )
        if "connect_timeout" not in dsn:
            dsn += ("&" if "?" in dsn else "?") + "connect_timeout=10"
        self._pool = psycopg2.pool.ThreadedConnectionPool(minconn=2, maxconn=12, dsn=dsn)
        self.client = _TableRouter(self._pool)
        logger.info("[DB] PostgreSQL connection pool initialised")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _qb(self, table: str) -> _QueryBuilder:
        return _QueryBuilder(self._pool, table)

    def _fetch_paginated(
        self,
        table: str,
        limit: int,
        order_col: str,
        since: Optional[datetime] = None,
        columns: Optional[str] = None,
        **eq_filters,
    ) -> List[Dict]:
        all_rows: List[Dict] = []
        fetched = 0
        page_size = 1000
        while fetched < limit:
            fetch_size = min(page_size, limit - fetched)
            qb = self._qb(table).select(columns or "*")
            for k, v in eq_filters.items():
                qb = qb.eq(k, v)
            if since:
                qb = qb.gte("timestamp", since.isoformat())
            qb = qb.order(order_col, desc=True).range(fetched, fetched + fetch_size - 1)
            rows = qb.execute().data or []
            if not rows:
                break
            all_rows.extend(rows)
            fetched += len(rows)
            if len(rows) < fetch_size:
                break
        return all_rows

    # ------------------------------------------------------------------
    # MARKET DATA
    # ------------------------------------------------------------------
    def insert_market_data(self, data: Dict) -> Dict:
        return self._qb("market_data").insert(data).execute()

    def batch_insert_market_data(self, data_list: List[Dict]) -> Dict:
        return self._qb("market_data").upsert(
            data_list, on_conflict="timestamp,symbol,exchange"
        ).execute()

    def get_latest_market_data(
        self,
        symbol: str,
        limit: int = 1000,
        exchange: str = "binance",
        columns: Optional[str] = None,
    ) -> pd.DataFrame:
        effective_columns = columns if columns is not None else self.MARKET_DATA_OHLCV_COLUMNS
        rows = self._fetch_paginated(
            "market_data", limit, "timestamp",
            columns=effective_columns, symbol=symbol, exchange=exchange,
        )
        if rows:
            df = pd.DataFrame(rows)
            df["timestamp"] = pd.to_datetime(
                df["timestamp"].astype(str), format="mixed", utc=True, errors="coerce"
            ).bfill()
            return df.sort_values("timestamp").reset_index(drop=True)
        return pd.DataFrame()

    def get_market_data_since(
        self,
        symbol: str,
        since: datetime,
        limit: int = 120,
        exchange: str = "binance",
        columns: Optional[str] = None,
    ) -> pd.DataFrame:
        rows = (
            self._qb("market_data")
            .select(columns or "*")
            .eq("symbol", symbol)
            .eq("exchange", exchange)
            .gte("timestamp", since.isoformat())
            .order("timestamp")
            .limit(limit)
            .execute()
            .data or []
        )
        if rows:
            df = pd.DataFrame(rows)
            df["timestamp"] = pd.to_datetime(
                df["timestamp"].astype(str), format="mixed", utc=True, errors="coerce"
            ).bfill()
            return df.sort_values("timestamp").reset_index(drop=True)
        return pd.DataFrame()

    def get_market_data_gap(
        self,
        symbol: str,
        since: datetime,
        limit: int = 2880,
        exchange: str = "binance",
        columns: Optional[str] = None,
    ) -> pd.DataFrame:
        effective_columns = columns if columns is not None else self.MARKET_DATA_OHLCV_COLUMNS
        rows = self._fetch_paginated(
            "market_data", limit, "timestamp",
            since=since, columns=effective_columns,
            symbol=symbol, exchange=exchange,
        )
        if rows:
            df = pd.DataFrame(rows)
            df["timestamp"] = pd.to_datetime(
                df["timestamp"].astype(str), format="mixed", utc=True, errors="coerce"
            ).bfill()
            return df.sort_values("timestamp").reset_index(drop=True)
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # CVD DATA
    # ------------------------------------------------------------------
    def batch_upsert_cvd_data(self, data_list: List[Dict]) -> Dict:
        return self._qb("cvd_data").upsert(
            data_list, on_conflict="timestamp,symbol"
        ).execute()

    def get_cvd_data(
        self,
        symbol: str,
        limit: int = 240,
        since: Optional[datetime] = None,
        columns: Optional[str] = None,
    ) -> pd.DataFrame:
        rows = self._fetch_paginated(
            "cvd_data", limit, "timestamp", since=since, columns=columns, symbol=symbol
        )
        if rows:
            df = pd.DataFrame(rows)
            df["timestamp"] = pd.to_datetime(
                df["timestamp"].astype(str), format="mixed", utc=True, errors="coerce"
            ).bfill()
            df = df.sort_values("timestamp").reset_index(drop=True)
            df["cvd"] = df["volume_delta"].cumsum()
            if "whale_buy_vol" in df.columns and "whale_sell_vol" in df.columns:
                df["whale_buy_vol"] = df["whale_buy_vol"].fillna(0)
                df["whale_sell_vol"] = df["whale_sell_vol"].fillna(0)
                df["whale_delta"] = df["whale_buy_vol"] - df["whale_sell_vol"]
                df["whale_cvd"] = df["whale_delta"].cumsum()
            return df
        return pd.DataFrame()

    def batch_upsert_whale_data(self, data_list: List[Dict]) -> Dict:
        return self._qb("cvd_data").upsert(
            data_list, on_conflict="timestamp,symbol"
        ).execute()

    # ------------------------------------------------------------------
    # LIQUIDATION DATA
    # ------------------------------------------------------------------
    def batch_upsert_liquidations(self, data_list: List[Dict]) -> Dict:
        return self._qb("liquidations").upsert(
            data_list, on_conflict="timestamp,symbol"
        ).execute()

    def get_liquidation_data(
        self,
        symbol: str,
        limit: int = 240,
        since: Optional[datetime] = None,
        columns: Optional[str] = None,
    ) -> pd.DataFrame:
        rows = self._fetch_paginated(
            "liquidations", limit, "timestamp", since=since, columns=columns, symbol=symbol
        )
        if rows:
            df = pd.DataFrame(rows)
            df["timestamp"] = pd.to_datetime(
                df["timestamp"].astype(str), format="mixed", utc=True, errors="coerce"
            ).bfill()
            return df.sort_values("timestamp").reset_index(drop=True)
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # FUNDING DATA
    # ------------------------------------------------------------------
    def upsert_funding_data(self, data: Dict) -> Dict:
        return self._qb("funding_data").upsert(
            data, on_conflict="timestamp,symbol"
        ).execute()

    def get_funding_history(
        self,
        symbol: str,
        limit: int = 100,
        since: Optional[datetime] = None,
        columns: Optional[str] = None,
    ) -> pd.DataFrame:
        rows = self._fetch_paginated(
            "funding_data", limit, "timestamp", since=since, columns=columns, symbol=symbol
        )
        if rows:
            df = pd.DataFrame(rows)
            df["timestamp"] = pd.to_datetime(
                df["timestamp"].astype(str), format="mixed", utc=True, errors="coerce"
            ).bfill()
            return df.sort_values("timestamp").reset_index(drop=True)
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # MICROSTRUCTURE DATA
    # ------------------------------------------------------------------
    def batch_upsert_microstructure_data(self, data_list: List[Dict]) -> Dict:
        return self._qb("microstructure_data").upsert(
            data_list, on_conflict="timestamp,symbol,exchange"
        ).execute()

    def get_latest_microstructure(self, symbol: str) -> Optional[Dict]:
        rows = (
            self._qb("microstructure_data")
            .select("*")
            .eq("symbol", symbol)
            .order("timestamp", desc=True)
            .limit(1)
            .execute()
            .data
        )
        return rows[0] if rows else None

    def get_microstructure_history(
        self,
        symbol: str,
        limit: int = 240,
        since: Optional[datetime] = None,
        columns: Optional[str] = None,
    ) -> pd.DataFrame:
        rows = self._fetch_paginated(
            "microstructure_data", limit, "timestamp",
            since=since, columns=columns, symbol=symbol, exchange="binance",
        )
        if rows:
            df = pd.DataFrame(rows)
            df["timestamp"] = pd.to_datetime(
                df["timestamp"].astype(str), format="mixed", utc=True, errors="coerce"
            ).bfill()
            return df.sort_values("timestamp").reset_index(drop=True)
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # LIQUIDATION CASCADE
    # ------------------------------------------------------------------
    def batch_upsert_liquidation_cascade_features(self, data_list: List[Dict]) -> Dict:
        return self._qb("liquidation_cascade_features").upsert(
            data_list, on_conflict="timestamp,symbol,side,feature_version"
        ).execute()

    def insert_liquidation_cascade_prediction(self, data: Dict) -> Dict:
        return self._qb("liquidation_cascade_predictions").insert(data).execute()

    # ------------------------------------------------------------------
    # MACRO DATA
    # ------------------------------------------------------------------
    def upsert_macro_data(self, data: Dict) -> Dict:
        return self._qb("macro_data").upsert(
            data, on_conflict="timestamp,source"
        ).execute()

    def get_latest_macro_data(self) -> Optional[Dict]:
        rows = (
            self._qb("macro_data")
            .select("*")
            .order("timestamp", desc=True)
            .limit(1)
            .execute()
            .data
        )
        return rows[0] if rows else None

    # ------------------------------------------------------------------
    # DERIBIT DATA
    # ------------------------------------------------------------------
    def upsert_deribit_data(self, data: Dict) -> Dict:
        return self._qb("deribit_data").upsert(
            data, on_conflict="symbol,timestamp"
        ).execute()

    def get_latest_deribit_data(self, symbol: str) -> Optional[Dict]:
        rows = (
            self._qb("deribit_data")
            .select("*")
            .eq("symbol", symbol)
            .order("timestamp", desc=True)
            .limit(1)
            .execute()
            .data
        )
        return rows[0] if rows else None

    # ------------------------------------------------------------------
    # FEAR & GREED DATA
    # ------------------------------------------------------------------
    def upsert_fear_greed(self, data: Dict) -> Dict:
        return self._qb("fear_greed_data").upsert(
            data, on_conflict="timestamp"
        ).execute()

    def get_latest_fear_greed(self) -> Optional[Dict]:
        rows = (
            self._qb("fear_greed_data")
            .select("*")
            .order("timestamp", desc=True)
            .limit(1)
            .execute()
            .data
        )
        return rows[0] if rows else None

    # ------------------------------------------------------------------
    # ONCHAIN DAILY SNAPSHOTS
    # ------------------------------------------------------------------
    def upsert_onchain_daily_snapshot(self, data: Dict) -> Dict:
        return self._qb("onchain_daily_snapshots").upsert(
            data, on_conflict="symbol,as_of_date,source"
        ).execute()

    def get_latest_onchain_snapshot(
        self, symbol: str, max_age_hours: Optional[int] = 48
    ) -> Optional[Dict]:
        rows = (
            self._qb("onchain_daily_snapshots")
            .select("*")
            .eq("symbol", symbol)
            .order("as_of_date", desc=True)
            .limit(1)
            .execute()
            .data
        )
        row = rows[0] if rows else None
        if not row or max_age_hours is None:
            return row
        try:
            as_of_date = datetime.fromisoformat(str(row["as_of_date"]))
            age_hours = (
                datetime.now(timezone.utc) - as_of_date.replace(tzinfo=timezone.utc)
            ).total_seconds() / 3600.0
            row = dict(row)
            row["is_stale"] = age_hours > max_age_hours
            row["age_hours"] = round(age_hours, 2)
        except Exception:
            row = dict(row)
            row["is_stale"] = None
        return row

    # ------------------------------------------------------------------
    # ARCHIVE MANIFESTS
    # ------------------------------------------------------------------
    def upsert_archive_manifest(self, data: Dict) -> Optional[Dict]:
        rows = self._qb("archive_manifests").upsert(
            data, on_conflict="table_name,partition_key"
        ).execute().data
        return rows[0] if rows else None

    def get_archive_manifest(self, table_name: str, partition_key: str) -> Optional[Dict]:
        rows = (
            self._qb("archive_manifests")
            .select("*")
            .eq("table_name", table_name)
            .eq("partition_key", partition_key)
            .limit(1)
            .execute()
            .data
        )
        return rows[0] if rows else None

    def update_archive_manifest(self, manifest_id: int, data: Dict) -> Optional[Dict]:
        rows = (
            self._qb("archive_manifests")
            .update(data)
            .eq("id", manifest_id)
            .execute()
            .data
        )
        return rows[0] if rows else None

    def get_archive_manifests(
        self,
        statuses: Optional[List[str]] = None,
        cleanup_pending_only: bool = False,
        limit: int = 1000,
    ) -> List[Dict]:
        all_rows: List[Dict] = []
        fetched = 0
        page_size = 500
        while fetched < limit:
            fetch_size = min(page_size, limit - fetched)
            qb = self._qb("archive_manifests").select("*")
            if statuses:
                qb = qb.in_("status", statuses)
            if cleanup_pending_only:
                qb = qb.is_("cleanup_completed_at", "null")
            qb = qb.order("archive_started_at").range(fetched, fetched + fetch_size - 1)
            chunk = qb.execute().data or []
            if not chunk:
                break
            all_rows.extend(chunk)
            fetched += len(chunk)
            if len(chunk) < fetch_size:
                break
        return all_rows

    # ------------------------------------------------------------------
    # TELEGRAM MESSAGES
    # ------------------------------------------------------------------
    def upsert_telegram_message(self, data: Dict) -> Dict:
        try:
            if data.get("text"):
                data = {**data, "text": data["text"].replace("\x00", "")}
            response = self._qb("telegram_messages").upsert(
                data, on_conflict="channel,message_id"
            ).execute()
            if response.data:
                msg = data.get("text", "")[:30].replace("\n", " ")
                logger.info(
                    f"DB: Saved [{data.get('channel')}] ID:{data.get('message_id')} | {msg}..."
                )
            return response
        except Exception as e:
            logger.error(f"DB: Failed to upsert telegram message: {e}")
            raise

    def insert_telegram_message(self, data: Dict) -> Dict:
        return self.upsert_telegram_message(data)

    def get_recent_telegram_messages(
        self, hours: int = 24, limit: int = 200, columns: Optional[str] = None
    ) -> List[Dict]:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        return (
            self._qb("telegram_messages")
            .select(columns or "channel,message_id,text,created_at,timestamp")
            .gte("created_at", cutoff)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
            .data or []
        )

    def get_telegram_messages_for_rag(
        self, days: int = 7, limit: int = 1000, columns: Optional[str] = None
    ) -> List[Dict]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        return (
            self._qb("telegram_messages")
            .select(columns or "channel,message_id,text,created_at,timestamp")
            .gte("created_at", cutoff)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
            .data or []
        )

    # ------------------------------------------------------------------
    # NARRATIVE DATA
    # ------------------------------------------------------------------
    def upsert_narrative_data(self, data: Dict) -> Dict:
        return self._qb("narrative_data").upsert(
            data, on_conflict="timestamp,symbol,source"
        ).execute()

    def get_latest_narrative_data(
        self, symbol: str, source: str = "perplexity"
    ) -> Optional[Dict]:
        rows = (
            self._qb("narrative_data")
            .select("*")
            .eq("symbol", symbol)
            .eq("source", source)
            .order("timestamp", desc=True)
            .limit(1)
            .execute()
            .data
        )
        return rows[0] if rows else None

    # ------------------------------------------------------------------
    # AI REPORTS
    # ------------------------------------------------------------------
    def insert_ai_report(self, data: Dict) -> Optional[str]:
        try:
            rows = self._qb("ai_reports").insert(data).execute().data
            return rows[0]["id"] if rows else None
        except Exception as e:
            msg = str(e).lower()
            if "onchain_context" in msg or "onchain_snapshot" in msg:
                logger.warning("ai_reports insert fallback: on-chain columns missing")
                legacy = {k: v for k, v in data.items() if k not in {"onchain_context", "onchain_snapshot"}}
                rows = self._qb("ai_reports").insert(legacy).execute().data
                return rows[0]["id"] if rows else None
            raise

    def get_latest_report(self, symbol: str = None) -> Optional[Dict]:
        qb = self._qb("ai_reports").select("*").order("created_at", desc=True)
        if symbol:
            qb = qb.eq("symbol", symbol)
        rows = qb.limit(1).execute().data
        return rows[0] if rows else None

    # ------------------------------------------------------------------
    # DUNE QUERY RESULTS
    # ------------------------------------------------------------------
    def upsert_dune_query_result(self, data: Dict) -> Dict:
        try:
            return self._qb("dune_query_results").upsert(
                data, on_conflict="query_id,collected_at"
            ).execute()
        except Exception as e:
            import json
            from pathlib import Path
            logger.error(f"[Fallback] DB failed for dune_query_results. Saving locally. {e}")
            try:
                path = Path("cache/dune_fallback")
                path.mkdir(parents=True, exist_ok=True)
                fp = path / f"dune_{data.get('query_id')}_{data.get('collected_at','').replace(':','')}.json"
                fp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
            except Exception as ie:
                logger.error(f"Failed to write dune fallback: {ie}")
            return SimpleNamespace(data=[data])

    def get_latest_dune_query_result(
        self, query_id: int, columns: str = "*"
    ) -> Optional[Dict]:
        try:
            rows = (
                self._qb("dune_query_results")
                .select(columns)
                .eq("query_id", query_id)
                .order("collected_at", desc=True)
                .limit(1)
                .execute()
                .data
            )
            if rows:
                return rows[0]
        except Exception as e:
            logger.warning(f"[Fallback] DB read failed for dune query {query_id}: {e}. Checking local cache.")
        try:
            from pathlib import Path
            import json
            path = Path("cache/dune_fallback")
            if path.exists():
                files = sorted(path.glob(f"dune_{query_id}_*.json"), reverse=True)
                if files:
                    return json.loads(files[0].read_text())
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # FEEDBACK LOGS
    # ------------------------------------------------------------------
    def insert_feedback(self, data: Dict) -> Dict:
        return self._qb("feedback_logs").insert(data).execute()

    def get_feedback_history(
        self, limit: int = 10, feedback_type: Optional[str] = None
    ) -> List[Dict]:
        qb = self._qb("feedback_logs").select("*")
        if feedback_type in ("positive", "negative"):
            qb = qb.eq("feedback_type", feedback_type)
        return qb.order("created_at", desc=True).limit(limit).execute().data or []

    # ------------------------------------------------------------------
    # TRADE EXECUTIONS
    # ------------------------------------------------------------------
    def insert_trade_execution(self, data: Dict) -> Dict:
        return self._qb("trade_executions").insert(data).execute()

    def get_trade_execution_by_order_id(self, order_id: str) -> Optional[Dict]:
        if not order_id:
            return None
        rows = (
            self._qb("trade_executions")
            .select("*")
            .eq("order_id", order_id)
            .limit(1)
            .execute()
            .data
        )
        return rows[0] if rows else None

    def update_trade_execution_fill_price(
        self, order_id: str, fill_price: float
    ) -> Optional[Dict]:
        if not order_id or not fill_price:
            return None
        rows = (
            self._qb("trade_executions")
            .update({"filled_price": fill_price})
            .eq("order_id", order_id)
            .execute()
            .data
        )
        return rows[0] if rows else None

    def get_position_status(self, symbol: str) -> Optional[Dict]:
        rows = (
            self._qb("trade_executions")
            .select("*")
            .eq("symbol", symbol)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
            .data
        )
        return rows[0] if rows else None

    # ------------------------------------------------------------------
    # MARKET STATUS EVENTS
    # ------------------------------------------------------------------
    def insert_market_status_event(self, data: Dict) -> Dict:
        return self._qb("market_status_events").insert(data).execute()

    def get_market_status_events(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
        hours: Optional[int] = None,
    ) -> List[Dict]:
        qb = self._qb("market_status_events").select("*").order("created_at", desc=True).limit(limit)
        if symbol:
            qb = qb.eq("symbol", symbol)
        if hours is not None:
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
            qb = qb.gte("created_at", cutoff)
        return qb.execute().data or []

    def get_market_status_events_with_fallback(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
        hours: Optional[int] = None,
    ) -> List[Dict]:
        result = self.get_market_status_events(symbol=symbol, limit=limit, hours=hours)
        if result:
            return result
        try:
            from processors.gcs_parquet import gcs_parquet_store
            months_back = max((hours or 168) / 720.0, 0.1)
            df = gcs_parquet_store.load_timeseries("market_status_events", symbol or "BTCUSDT", months_back=months_back)
            if df is None or df.empty:
                return []
            if "created_at" in df.columns:
                df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
                df = df.sort_values("created_at", ascending=False)
                if hours is not None:
                    cutoff_dt = datetime.now(timezone.utc) - timedelta(hours=hours)
                    df = df[df["created_at"] >= cutoff_dt]
            if symbol and "symbol" in df.columns:
                df = df[df["symbol"] == symbol]
            records = df.head(limit).to_dict("records")
            logger.info(f"[GCS Fallback] market_status_events loaded {len(records)} rows for {symbol}")
            return records
        except Exception as gcs_e:
            logger.warning(f"[GCS Fallback] market_status_events read failed: {gcs_e}")
        return []

    def update_market_status_event_technical_snapshot(
        self, event_id: int, technical_snapshot: Dict
    ) -> Dict:
        return (
            self._qb("market_status_events")
            .update({"technical_snapshot": technical_snapshot})
            .eq("id", event_id)
            .execute()
        )

    # ------------------------------------------------------------------
    # NEWS IMPACT LOG
    # ------------------------------------------------------------------
    def log_news_impact_predictions(self, items: list, predicted_at: datetime) -> None:
        if not items:
            return
        prices: dict = {}
        for sym in settings.trading_symbols:
            try:
                df = self.get_latest_market_data(sym, limit=1)
                if not df.empty and "close" in df.columns:
                    prices[sym] = float(df["close"].iloc[-1])
            except Exception:
                pass
        rows = []
        for item in items:
            rows.append({
                "predicted_at": predicted_at.isoformat(),
                "headline": str(item.get("headline", ""))[:200],
                "claim": str(item.get("claim", ""))[:500],
                "impact_score": int(item.get("impact", 3)),
                "already_priced_in": bool(item.get("already_priced_in", False)),
                "why": str(item.get("why", ""))[:300],
                "btc_price_at_time": prices.get("BTCUSDT"),
                "eth_price_at_time": prices.get("ETHUSDT"),
            })
        self._qb("news_impact_log").insert(rows).execute()

    # ------------------------------------------------------------------
    # EVALUATION — PREDICTIONS
    # ------------------------------------------------------------------
    def upsert_evaluation_prediction(self, data: Dict) -> Optional[Dict]:
        if data.get("source_id") is None:
            logger.warning(
                f"upsert_evaluation_prediction skipped: source_id is None "
                f"(symbol={data.get('symbol')}, mode={data.get('mode')})"
            )
            return None
        rows = self._qb("evaluation_predictions").upsert(
            data, on_conflict="source_type,source_id,mode"
        ).execute().data
        return rows[0] if rows else None

    def get_evaluation_prediction_by_source(
        self, source_type: str, source_id: int, mode: Optional[str] = None
    ) -> Optional[Dict]:
        qb = (
            self._qb("evaluation_predictions")
            .select("*")
            .eq("source_type", source_type)
            .eq("source_id", source_id)
        )
        if mode is not None:
            qb = qb.eq("mode", mode)
        rows = qb.order("created_at", desc=True).limit(1).execute().data
        return rows[0] if rows else None

    def get_evaluation_prediction(self, prediction_id: int) -> Optional[Dict]:
        rows = (
            self._qb("evaluation_predictions")
            .select("*")
            .eq("id", prediction_id)
            .limit(1)
            .execute()
            .data
        )
        return rows[0] if rows else None

    def get_evaluation_predictions(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        symbol: Optional[str] = None,
        mode: Optional[str] = None,
        source_type: Optional[str] = None,
        limit: int = 5000,
    ) -> List[Dict]:
        all_rows: List[Dict] = []
        fetched = 0
        page_size = 1000
        start_iso = start_time.isoformat() if start_time else None
        end_iso = end_time.isoformat() if end_time else None
        while fetched < limit:
            fetch_size = min(page_size, limit - fetched)
            qb = self._qb("evaluation_predictions").select("*")
            if symbol:
                qb = qb.eq("symbol", symbol)
            if mode:
                qb = qb.eq("mode", mode)
            if source_type:
                qb = qb.eq("source_type", source_type)
            if start_iso:
                qb = qb.gte("prediction_time", start_iso)
            if end_iso:
                qb = qb.lt("prediction_time", end_iso)
            chunk = qb.order("prediction_time").range(fetched, fetched + fetch_size - 1).execute().data or []
            if not chunk:
                break
            all_rows.extend(chunk)
            fetched += len(chunk)
            if len(chunk) < fetch_size:
                break
        return all_rows

    # ------------------------------------------------------------------
    # EVALUATION — OUTCOMES
    # ------------------------------------------------------------------
    def upsert_evaluation_outcome(self, data: Dict) -> Optional[Dict]:
        rows = self._qb("evaluation_outcomes").upsert(
            data, on_conflict="prediction_id,horizon_minutes"
        ).execute().data
        return rows[0] if rows else None

    def get_evaluation_outcome(
        self, prediction_id: int, horizon_minutes: int
    ) -> Optional[Dict]:
        rows = (
            self._qb("evaluation_outcomes")
            .select("*")
            .eq("prediction_id", prediction_id)
            .eq("horizon_minutes", horizon_minutes)
            .limit(1)
            .execute()
            .data
        )
        return rows[0] if rows else None

    def get_evaluation_outcomes(
        self,
        prediction_ids: Optional[List[int]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 10000,
    ) -> List[Dict]:
        start_iso = start_time.isoformat() if start_time else None
        end_iso = end_time.isoformat() if end_time else None
        all_rows: List[Dict] = []

        if prediction_ids:
            for idx in range(0, len(prediction_ids), 200):
                chunk_ids = prediction_ids[idx:idx + 200]
                qb = self._qb("evaluation_outcomes").select("*").in_("prediction_id", chunk_ids)
                if start_iso:
                    qb = qb.gte("evaluated_at", start_iso)
                if end_iso:
                    qb = qb.lt("evaluated_at", end_iso)
                rows = qb.order("evaluated_at").limit(limit).execute().data or []
                all_rows.extend(rows)
            return all_rows[:limit]

        fetched = 0
        page_size = 1000
        while fetched < limit:
            fetch_size = min(page_size, limit - fetched)
            qb = self._qb("evaluation_outcomes").select("*")
            if start_iso:
                qb = qb.gte("evaluated_at", start_iso)
            if end_iso:
                qb = qb.lt("evaluated_at", end_iso)
            chunk = qb.order("evaluated_at").range(fetched, fetched + fetch_size - 1).execute().data or []
            if not chunk:
                break
            all_rows.extend(chunk)
            fetched += len(chunk)
            if len(chunk) < fetch_size:
                break
        return all_rows

    # ------------------------------------------------------------------
    # EVALUATION — COMPONENT SCORES
    # ------------------------------------------------------------------
    def batch_upsert_evaluation_component_scores(self, data_list: List[Dict]) -> Dict:
        if not data_list:
            return SimpleNamespace(data=[])
        return self._qb("evaluation_component_scores").upsert(
            data_list,
            on_conflict="prediction_id,component_type,metric_name,scope_key",
        ).execute()

    def get_evaluation_component_scores(self, prediction_ids: List[int]) -> List[Dict]:
        all_rows: List[Dict] = []
        for idx in range(0, len(prediction_ids), 200):
            chunk_ids = prediction_ids[idx:idx + 200]
            rows = (
                self._qb("evaluation_component_scores")
                .select("*")
                .in_("prediction_id", chunk_ids)
                .order("created_at")
                .execute()
                .data or []
            )
            all_rows.extend(rows)
        return all_rows

    # ------------------------------------------------------------------
    # EVALUATION — ROLLUPS DAILY
    # ------------------------------------------------------------------
    def delete_evaluation_rollups_for_date(self, rollup_date: str) -> Dict:
        return (
            self._qb("evaluation_rollups_daily")
            .delete()
            .eq("rollup_date", rollup_date)
            .execute()
        )

    def upsert_evaluation_rollups(self, rows: List[Dict]) -> Dict:
        if not rows:
            return SimpleNamespace(data=[])
        return self._qb("evaluation_rollups_daily").upsert(
            rows,
            on_conflict="rollup_date,symbol,mode,scope,horizon_minutes,metric_name,bucket_key",
        ).execute()

    # ------------------------------------------------------------------
    # FACTOR SIGNALS / IC HISTORY / TRADE ATTRIBUTION
    # ------------------------------------------------------------------
    def insert_factor_signals(self, data: dict) -> dict:
        try:
            rows = self._qb("factor_signals").insert(data).execute().data
            return rows[0] if rows else {}
        except Exception as e:
            logger.error(f"[DB] insert_factor_signals failed: {e}")
            return {}

    def get_factor_signals_by_decision_id(self, decision_id: str) -> dict:
        try:
            if not decision_id:
                return {}
            rows = (
                self._qb("factor_signals")
                .select("*")
                .eq("decision_id", str(decision_id))
                .order("created_at", desc=True)
                .limit(1)
                .execute()
                .data
            )
            return rows[0] if rows else {}
        except Exception as e:
            logger.error(f"[DB] get_factor_signals_by_decision_id failed: {e}")
            return {}

    def get_factor_signals(
        self, symbol: str, limit: int = 100, mode: str = None
    ) -> list:
        try:
            qb = (
                self._qb("factor_signals")
                .select("*")
                .eq("symbol", symbol)
                .order("created_at", desc=True)
                .limit(limit)
            )
            if mode:
                qb = qb.eq("mode", mode)
            return qb.execute().data or []
        except Exception as e:
            logger.error(f"[DB] get_factor_signals failed: {e}")
            return []

    def upsert_factor_ic(self, data: dict) -> dict:
        try:
            rows = self._qb("factor_ic_history").insert(data).execute().data
            return rows[0] if rows else {}
        except Exception as e:
            logger.error(f"[DB] upsert_factor_ic failed: {e}")
            return {}

    def get_factor_ic_history(
        self,
        symbol: str,
        factor_name: str = None,
        regime: str = None,
        limit: int = 1,
    ) -> list:
        try:
            qb = (
                self._qb("factor_ic_history")
                .select("*")
                .eq("symbol", symbol)
                .order("computed_at", desc=True)
                .limit(limit * 8)
            )
            if factor_name:
                qb = qb.eq("factor_name", factor_name)
            if regime:
                qb = qb.eq("regime", regime)
            rows = qb.execute().data or []
            seen: set = set()
            result = []
            for row in rows:
                key = (row.get("factor_name"), row.get("regime"))
                if key not in seen:
                    seen.add(key)
                    result.append(row)
            return result
        except Exception as e:
            logger.error(f"[DB] get_factor_ic_history failed: {e}")
            return []

    def insert_trade_attribution(self, data: dict) -> dict:
        try:
            rows = self._qb("trade_attribution").insert(data).execute().data
            return rows[0] if rows else {}
        except Exception as e:
            logger.error(f"[DB] insert_trade_attribution failed: {e}")
            return {}

    def get_trade_attributions(self, symbol: str, limit: int = 90) -> list:
        try:
            return (
                self._qb("trade_attribution")
                .select("*")
                .eq("symbol", symbol)
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
                .data or []
            )
        except Exception as e:
            logger.error(f"[DB] get_trade_attributions failed: {e}")
            return []

    # ------------------------------------------------------------------
    # PAPER ORDERS
    # ------------------------------------------------------------------
    def insert_paper_order(self, data: dict) -> Optional[int]:
        try:
            rows = self._qb("paper_orders").insert(data).execute().data
            return rows[0]["id"] if rows else None
        except Exception as e:
            logger.error(f"[DB] insert_paper_order failed: {e}")
            return None

    def get_open_paper_orders(self) -> List[Dict]:
        try:
            return (
                self._qb("paper_orders")
                .select("*")
                .eq("status", "OPEN")
                .execute()
                .data or []
            )
        except Exception as e:
            logger.error(f"[DB] get_open_paper_orders failed: {e}")
            return []

    def update_paper_order_closed(self, order_id: int, data: dict) -> bool:
        try:
            self._qb("paper_orders").update(data).eq("id", order_id).execute()
            return True
        except Exception as e:
            logger.error(f"[DB] update_paper_order_closed failed: {e}")
            return False

    # ------------------------------------------------------------------
    # DATA CLEANUP (retention)
    # ------------------------------------------------------------------
    def cleanup_old_data(self) -> Dict:
        results: Dict = {}
        errors: List[Dict] = []
        now = datetime.now(timezone.utc)

        def _delete(table: str, col: str, cutoff: str, key: str) -> None:
            try:
                self._qb(table).delete().lt(col, cutoff).execute()
                results[key] = "deleted"
            except Exception as e:
                errors.append({"table": table, "error": str(e)})
                logger.error(f"[Cleanup] Failed for {table}: {e}")

        cutoff_market = (now - timedelta(days=settings.RETENTION_MARKET_DATA_DAYS)).isoformat()
        cutoff_cvd    = (now - timedelta(days=settings.RETENTION_CVD_DAYS)).isoformat()
        cutoff_long   = (now - timedelta(days=settings.RETENTION_REPORTS_DAYS)).isoformat()
        cutoff_tg     = (now - timedelta(days=settings.RETENTION_TELEGRAM_DAYS)).isoformat()
        cutoff_onchain = (now - timedelta(days=settings.RETENTION_REPORTS_DAYS)).date().isoformat()

        _delete("market_data",         "timestamp",  cutoff_market,  "market_data_deleted")
        _delete("funding_data",         "timestamp",  cutoff_market,  "funding_data_deleted")
        _delete("liquidations",         "timestamp",  cutoff_market,  "liquidations_deleted")
        _delete("cvd_data",             "timestamp",  cutoff_cvd,     "cvd_data_deleted")
        _delete("microstructure_data",  "timestamp",  cutoff_market,  "microstructure_deleted")
        _delete("macro_data",           "timestamp",  cutoff_long,    "macro_deleted")
        _delete("deribit_data",         "timestamp",  cutoff_market,  "deribit_deleted")
        _delete("fear_greed_data",      "timestamp",  cutoff_long,    "fear_greed_deleted")
        _delete("onchain_daily_snapshots", "as_of_date", cutoff_onchain, "onchain_deleted")
        _delete("telegram_messages",    "created_at", cutoff_tg,      "telegram_deleted")
        _delete("narrative_data",       "timestamp",  cutoff_long,    "narrative_deleted")
        _delete("ai_reports",           "created_at", cutoff_long,    "reports_deleted")
        _delete("dune_query_results",   "collected_at", cutoff_long,  "dune_deleted")

        if errors:
            results["cleanup_errors"] = errors
            logger.warning(f"[Cleanup] Done with {len(errors)} error(s): {[e['table'] for e in errors]}")
        else:
            logger.info(f"[Cleanup] Done cleanly: {results}")
        return results

    # ------------------------------------------------------------------
    # Compatibility shims (used by some external code)
    # ------------------------------------------------------------------
    def get_circuit_breaker_status(self) -> dict:
        """No-op shim — circuit breaker not needed for local PostgreSQL."""
        return {"quant": {"state": "CLOSED"}, "text": {"state": "CLOSED"}}


db = DatabaseClient()
