# -*- coding: utf-8 -*-
# -*- coding: cp1252 -*-
import functools
import httpx
from supabase import create_client, Client, ClientOptions
from config.settings import settings
from typing import Dict, List, Optional
from datetime import datetime, timedelta, timezone
import pandas as pd
import asyncio
from loguru import logger

# ----------------- HTTP/2 Cloudflare Patch -----------------
try:
    _orig_init = httpx.Client.__init__
    def _patched_init(self, *args, **kwargs):
        kwargs["http2"] = False
        _orig_init(self, *args, **kwargs)
    httpx.Client.__init__ = _patched_init

    _orig_async_init = httpx.AsyncClient.__init__
    def _patched_async_init(self, *args, **kwargs):
        kwargs["http2"] = False
        _orig_async_init(self, *args, **kwargs)
    httpx.AsyncClient.__init__ = _patched_async_init
except Exception:
    pass
# ------------------------------------------------------------


class _TableRouter:
    """db.client.table(...) 호출을 테이블명에 따라 올바른 Supabase 클라이언트로 자동 라우팅.

    기존 코드의 db.client.table("xxx") 패턴을 전혀 수정하지 않아도
    QUANT/TEXT 프로젝트로 자동 분기됨.
    """
    def __init__(self, db_instance: "DatabaseClient"):
        self._db = db_instance

    def table(self, table_name: str):
        return self._db._client(table_name).table(table_name)


class _CircuitBreaker:
    """Per-client circuit breaker for Supabase quota/rate-limit errors.

    States:
      CLOSED  — normal operation, all calls pass through
      OPEN    — too many consecutive quota failures; calls are blocked for
                `cooldown_seconds` to stop the retry storm
      HALF_OPEN — cooldown elapsed; next call is a probe to test recovery

    Quota errors (429 / "quota exceeded" / "over_request_rate_limit") are
    tracked separately from transient connection errors.  Connection errors
    still trigger reconnection; quota errors open the circuit so the caller
    backs off immediately rather than hammering a rate-limited endpoint.
    """

    QUOTA_KEYWORDS = (
        "429", "quota", "over_request_rate_limit", "rate limit", "too many requests",
        "rate_limit", "exceeded",
    )

    def __init__(self, failure_threshold: int = 3, cooldown_seconds: int = 60):
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._consecutive_failures = 0
        self._opened_at: Optional[datetime] = None
        self._state = "CLOSED"  # CLOSED | OPEN | HALF_OPEN

    @staticmethod
    def _is_quota_error(err_msg: str) -> bool:
        return any(k in err_msg for k in _CircuitBreaker.QUOTA_KEYWORDS)

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._state = "CLOSED"

    def record_failure(self, err_msg: str) -> None:
        if not self._is_quota_error(err_msg):
            return  # Only quota errors open the circuit
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._failure_threshold:
            if self._state != "OPEN":
                logger.warning(
                    f"[CircuitBreaker] OPEN after {self._consecutive_failures} quota errors — "
                    f"blocking DB writes for {self._cooldown_seconds}s"
                )
            self._state = "OPEN"
            self._opened_at = datetime.now(timezone.utc)

    def is_open(self) -> bool:
        if self._state == "CLOSED":
            return False
        if self._state == "OPEN":
            elapsed = (datetime.now(timezone.utc) - self._opened_at).total_seconds()
            if elapsed >= self._cooldown_seconds:
                self._state = "HALF_OPEN"
                logger.info("[CircuitBreaker] HALF_OPEN — probing Supabase recovery")
                return False  # Allow one probe call through
            return True
        return False  # HALF_OPEN: let probe through


class DatabaseClient:
    # ── QUANT Project 담당 테이블 (수치 데이터) ───────────────────────────
    QUANT_TABLES = {
        "market_data", "cvd_data", "funding_data", "liquidations",
        "microstructure_data", "liquidation_cascade_features",
        "liquidation_cascade_predictions", "macro_data", "deribit_data",
        "fear_greed_data", "onchain_daily_snapshots", "archive_manifests",
    }

    # ── TEXT Project 담당 테이블 (텍스트/AI 데이터) ───────────────────────
    TEXT_TABLES = {
        "telegram_messages", "narrative_data", "ai_reports", "feedback_logs",
        "trade_executions", "dune_query_results", "daily_playbooks",
        "monitor_logs", "market_status_events", "evaluation_predictions",
        "evaluation_outcomes", "evaluation_component_scores",
        "evaluation_rollups_daily", "paper_orders",
    }

    def __init__(self):
        options = ClientOptions(postgrest_client_timeout=60)

        # QUANT 클라이언트 — SUPABASE_URL_QUANT 없으면 기존 URL로 fallback
        quant_url = getattr(settings, "SUPABASE_URL_QUANT", "") or settings.SUPABASE_URL
        quant_key = getattr(settings, "SUPABASE_KEY_QUANT", "") or settings.SUPABASE_KEY
        self.client_quant: Client = create_client(quant_url, quant_key, options=options)

        # TEXT 클라이언트 — SUPABASE_URL_TEXT 없으면 기존 URL로 fallback
        text_url = getattr(settings, "SUPABASE_URL_TEXT", "") or settings.SUPABASE_URL
        text_key = getattr(settings, "SUPABASE_KEY_TEXT", "") or settings.SUPABASE_KEY
        self.client_text: Client = create_client(text_url, text_key, options=options)

        # db.client.table("xxx") 호출을 테이블명 기준으로 자동 라우팅
        self.client = _TableRouter(self)

        # Per-project circuit breakers — stop retry storms on quota exhaustion
        self._cb_quant = _CircuitBreaker(failure_threshold=3, cooldown_seconds=60)
        self._cb_text = _CircuitBreaker(failure_threshold=3, cooldown_seconds=60)

    def _circuit_breaker_for(self, table: str) -> _CircuitBreaker:
        return self._cb_text if table in self.TEXT_TABLES else self._cb_quant

    def get_circuit_breaker_status(self) -> dict:
        """Return current circuit breaker state for both projects — useful for health checks."""
        return {
            "quant": {
                "state": self._cb_quant._state,
                "consecutive_failures": self._cb_quant._consecutive_failures,
                "opened_at": self._cb_quant._opened_at.isoformat() if self._cb_quant._opened_at else None,
            },
            "text": {
                "state": self._cb_text._state,
                "consecutive_failures": self._cb_text._consecutive_failures,
                "opened_at": self._cb_text._opened_at.isoformat() if self._cb_text._opened_at else None,
            },
        }

    def _client(self, table: str) -> Client:
        """테이블명으로 적절한 Supabase 클라이언트 반환."""
        return self.client_text if table in self.TEXT_TABLES else self.client_quant

    def reconnect_on_error(func):
        """Transient connection error → reconnect + 1 retry.
        Quota/rate-limit error → record in circuit breaker; open circuit blocks
        further calls for 60 s to prevent retry storms.
        """
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Infer which circuit breaker to consult from the first kwarg/arg
            # that looks like a table name, or fall back to quant.
            _cb: _CircuitBreaker = self._cb_quant
            for arg in list(args) + list(kwargs.values()):
                if isinstance(arg, str) and arg in self.TEXT_TABLES:
                    _cb = self._cb_text
                    break
                if isinstance(arg, str) and arg in self.QUANT_TABLES:
                    _cb = self._cb_quant
                    break

            if _cb.is_open():
                logger.warning(f"[CircuitBreaker] OPEN — skipping {func.__name__} to avoid quota storm")
                return None

            try:
                result = func(self, *args, **kwargs)
                _cb.record_success()
                return result
            except Exception as e:
                err_msg = str(e).lower()
                err_type = type(e).__name__.lower()

                # Quota/rate-limit errors: open the circuit, do NOT reconnect
                if _CircuitBreaker._is_quota_error(err_msg):
                    _cb.record_failure(err_msg)
                    logger.warning(f"[CircuitBreaker] Quota error in {func.__name__}: {err_msg[:120]}")
                    raise

                transient_keywords = [
                    "disconnected", "closed", "connection", "eof", "protocol",
                    "pseudo-header", "timeout", "61", "104", "refused", "reset",
                    "cloudflare", "400 bad request", "json could not be generated",
                    "model_validate_json", "apierrorfromjson", "validation error",
                    "500", "502", "503", "504",
                ]

                if any(x in err_msg for x in transient_keywords) or "protocol" in err_type or "connection" in err_type:
                    logger.warning(f"Database transient error in {func.__name__} ({e.__class__.__name__}), reconnecting...")
                    try:
                        import time
                        time.sleep(1.5)
                        options = ClientOptions(postgrest_client_timeout=60)

                        quant_url = getattr(settings, "SUPABASE_URL_QUANT", "") or settings.SUPABASE_URL
                        quant_key = getattr(settings, "SUPABASE_KEY_QUANT", "") or settings.SUPABASE_KEY
                        self.client_quant = create_client(quant_url, quant_key, options=options)

                        text_url = getattr(settings, "SUPABASE_URL_TEXT", "") or settings.SUPABASE_URL
                        text_key = getattr(settings, "SUPABASE_KEY_TEXT", "") or settings.SUPABASE_KEY
                        self.client_text = create_client(text_url, text_key, options=options)
                        self.client = _TableRouter(self)

                        result = func(self, *args, **kwargs)
                        _cb.record_success()
                        return result
                    except Exception as retry_e:
                        logger.error(f"Database reconnection/retry failed: {retry_e}")
                        raise
                raise
        return wrapper

# =====================================================================
# QUANT PROJECT — 수치 데이터
# =====================================================================

# ----------------------- Market Data -----------------------

    @reconnect_on_error
    def insert_market_data(self, data: Dict) -> Dict:
        return self.client_quant.table("market_data").insert(data).execute()

    @reconnect_on_error
    def batch_insert_market_data(self, data_list: List[Dict]) -> Dict:
        """Upsert to avoid duplicate errors on (timestamp, symbol, exchange)."""
        return self.client_quant.table("market_data").upsert(
            data_list, on_conflict="timestamp,symbol,exchange"
        ).execute()

    def _fetch_paginated(self, table: str, limit: int, order_col: str, since: Optional[datetime] = None, columns: Optional[str] = None, **eq_filters) -> List[Dict]:
        """Helper to bypass Supabase 1000 row max limit using pagination."""
        client = self._client(table)
        all_rows = []
        fetched = 0
        page_size = 1000
        while fetched < limit:
            fetch_size = min(page_size, limit - fetched)
            start = fetched
            end = fetched + fetch_size - 1

            query = client.table(table).select(columns or "*")
            for k, v in eq_filters.items():
                query = query.eq(k, v)

            if since:
                query = query.gte("timestamp", since.isoformat())

            response = query.order(order_col, desc=True).range(start, end).execute()
            rows = response.data if response.data else []
            if not rows:
                break
            all_rows.extend(rows)
            fetched += len(rows)
            if len(rows) < fetch_size:
                break
        return all_rows

    # OHLCV 분석에 필요한 최소 컬럼 집합 (egress 절감)
    MARKET_DATA_OHLCV_COLUMNS = "timestamp,symbol,exchange,open,high,low,close,volume,taker_buy_volume,taker_sell_volume"

    @reconnect_on_error
    def get_latest_market_data(self, symbol: str, limit: int = 1000, exchange: str = "binance", columns: Optional[str] = None) -> pd.DataFrame:
        effective_columns = columns if columns is not None else self.MARKET_DATA_OHLCV_COLUMNS
        rows = self._fetch_paginated("market_data", limit, "timestamp", columns=effective_columns, symbol=symbol, exchange=exchange)
        if rows:
            df = pd.DataFrame(rows)
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(str), format='mixed', utc=True, errors='coerce').bfill()
            return df.sort_values('timestamp').reset_index(drop=True)
        return pd.DataFrame()

# ------------------------- CVD Data -------------------------

    @reconnect_on_error
    def batch_upsert_cvd_data(self, data_list: List[Dict]) -> Dict:
        return self.client_quant.table("cvd_data").upsert(
            data_list, on_conflict="timestamp,symbol"
        ).execute()

    @reconnect_on_error
    def get_cvd_data(self, symbol: str, limit: int = 240, since: Optional[datetime] = None, columns: Optional[str] = None) -> pd.DataFrame:
        rows = self._fetch_paginated("cvd_data", limit, "timestamp", since=since, columns=columns, symbol=symbol)
        if rows:
            df = pd.DataFrame(rows)
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(str), format='mixed', utc=True, errors='coerce').bfill()
            df = df.sort_values('timestamp').reset_index(drop=True)
            df['cvd'] = df['volume_delta'].cumsum()
            if 'whale_buy_vol' in df.columns and 'whale_sell_vol' in df.columns:
                df['whale_buy_vol'] = df['whale_buy_vol'].fillna(0)
                df['whale_sell_vol'] = df['whale_sell_vol'].fillna(0)
                df['whale_delta'] = df['whale_buy_vol'] - df['whale_sell_vol']
                df['whale_cvd'] = df['whale_delta'].cumsum()
            return df
        return pd.DataFrame()

# ----------- Whale Data (from WebSocket aggTrade) -----------

    @reconnect_on_error
    def batch_upsert_whale_data(self, data_list: List[Dict]) -> Dict:
        """Upsert whale trade data into cvd_data table (whale columns)."""
        return self.client_quant.table("cvd_data").upsert(
            data_list, on_conflict="timestamp,symbol"
        ).execute()

# --------------------- Liquidation Data ---------------------

    @reconnect_on_error
    def batch_upsert_liquidations(self, data_list: List[Dict]) -> Dict:
        return self.client_quant.table("liquidations").upsert(
            data_list, on_conflict="timestamp,symbol"
        ).execute()

    @reconnect_on_error
    def get_liquidation_data(self, symbol: str, limit: int = 240, since: Optional[datetime] = None, columns: Optional[str] = None) -> pd.DataFrame:
        rows = self._fetch_paginated("liquidations", limit, "timestamp", since=since, columns=columns, symbol=symbol)
        if rows:
            df = pd.DataFrame(rows)
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(str), format='mixed', utc=True, errors='coerce').bfill()
            return df.sort_values('timestamp').reset_index(drop=True)
        return pd.DataFrame()

# ----------------------- Funding Data -----------------------

    @reconnect_on_error
    def upsert_funding_data(self, data: Dict) -> Dict:
        return self.client_quant.table("funding_data").upsert(
            data, on_conflict="timestamp,symbol"
        ).execute()

    @reconnect_on_error
    def get_funding_history(self, symbol: str, limit: int = 100, since: Optional[datetime] = None, columns: Optional[str] = None) -> pd.DataFrame:
        rows = self._fetch_paginated("funding_data", limit, "timestamp", since=since, columns=columns, symbol=symbol)
        if rows:
            df = pd.DataFrame(rows)
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(str), format='mixed', utc=True, errors='coerce').bfill()
            return df.sort_values('timestamp').reset_index(drop=True)
        return pd.DataFrame()

# ------------------- Microstructure Data -------------------

    @reconnect_on_error
    def batch_upsert_microstructure_data(self, data_list: List[Dict]) -> Dict:
        return self.client_quant.table("microstructure_data").upsert(
            data_list, on_conflict="timestamp,symbol,exchange"
        ).execute()

    @reconnect_on_error
    def get_latest_microstructure(self, symbol: str) -> Optional[Dict]:
        response = self.client_quant.table("microstructure_data")\
                        .select("*")\
            .eq("symbol", symbol)\
            .order("timestamp", desc=True)\
            .limit(1)\
            .execute()
        return response.data[0] if response.data else None

    @reconnect_on_error
    def get_microstructure_history(self, symbol: str, limit: int = 240, since: Optional[datetime] = None, columns: Optional[str] = None) -> pd.DataFrame:
        rows = self._fetch_paginated("microstructure_data", limit, "timestamp", since=since, columns=columns, symbol=symbol, exchange="binance")
        if rows:
            df = pd.DataFrame(rows)
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(str), format='mixed', utc=True, errors='coerce').bfill()
            return df.sort_values('timestamp').reset_index(drop=True)
        return pd.DataFrame()

    @reconnect_on_error
    def batch_upsert_liquidation_cascade_features(self, data_list: List[Dict]) -> Dict:
        return self.client_quant.table("liquidation_cascade_features").upsert(
            data_list, on_conflict="timestamp,symbol,side,feature_version"
        ).execute()

    @reconnect_on_error
    def insert_liquidation_cascade_prediction(self, data: Dict) -> Dict:
        return self.client_quant.table("liquidation_cascade_predictions").insert(data).execute()

# ------------------------ Macro Data ------------------------

    @reconnect_on_error
    def upsert_macro_data(self, data: Dict) -> Dict:
        return self.client_quant.table("macro_data").upsert(
            data, on_conflict="timestamp,source"
        ).execute()

    @reconnect_on_error
    def get_latest_macro_data(self) -> Optional[Dict]:
        response = self.client_quant.table("macro_data")\
                        .select("*")\
            .order("timestamp", desc=True)\
            .limit(1)\
            .execute()
        return response.data[0] if response.data else None

# ------------------- Deribit Options Data -------------------

    @reconnect_on_error
    def upsert_deribit_data(self, data: Dict) -> Dict:
        return self.client_quant.table("deribit_data").upsert(
            data, on_conflict="symbol,timestamp"
        ).execute()

    def get_latest_deribit_data(self, symbol: str) -> Optional[Dict]:
        response = self.client_quant.table("deribit_data")\
                        .select("*")\
            .eq("symbol", symbol)\
            .order("timestamp", desc=True)\
            .limit(1)\
            .execute()
        return response.data[0] if response.data else None

# -------------------- Fear & Greed Data --------------------

    @reconnect_on_error
    def upsert_fear_greed(self, data: Dict) -> Dict:
        return self.client_quant.table("fear_greed_data").upsert(
            data, on_conflict="timestamp"
        ).execute()

    def get_latest_fear_greed(self) -> Optional[Dict]:
        response = self.client_quant.table("fear_greed_data")\
                        .select("*")\
            .order("timestamp", desc=True)\
            .limit(1)\
            .execute()
        return response.data[0] if response.data else None

# ------------------ On-Chain Daily Snapshots ----------------

    @reconnect_on_error
    def upsert_onchain_daily_snapshot(self, data: Dict) -> Dict:
        return self.client_quant.table("onchain_daily_snapshots").upsert(
            data, on_conflict="symbol,as_of_date,source"
        ).execute()

    def get_latest_onchain_snapshot(self, symbol: str, max_age_hours: Optional[int] = 48) -> Optional[Dict]:
        response = self.client_quant.table("onchain_daily_snapshots")\
                        .select("*")\
            .eq("symbol", symbol)\
            .order("as_of_date", desc=True)\
            .limit(1)\
            .execute()

        row = response.data[0] if response.data else None
        if not row or max_age_hours is None:
            return row

        try:
            as_of_date = datetime.fromisoformat(str(row["as_of_date"]))
            age_hours = (datetime.now(timezone.utc) - as_of_date.replace(tzinfo=timezone.utc)).total_seconds() / 3600.0
            row = dict(row)
            row["is_stale"] = age_hours > max_age_hours
            row["age_hours"] = round(age_hours, 2)
        except Exception:
            row = dict(row)
            row["is_stale"] = None
        return row

# ---------------------- Market Data Gap ---------------------

    @reconnect_on_error
    def get_market_data_since(
        self,
        symbol: str,
        since: datetime,
        limit: int = 120,
        exchange: str = "binance",
        columns: Optional[str] = None,
    ) -> pd.DataFrame:
        response = self.client_quant.table("market_data")\
            .select(columns or "*")\
            .eq("symbol", symbol)\
            .eq("exchange", exchange)\
            .gte("timestamp", since.isoformat())\
            .order("timestamp")\
            .limit(limit)\
            .execute()
        rows = response.data if response.data else []
        if rows:
            df = pd.DataFrame(rows)
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(str), format='mixed', utc=True, errors='coerce').bfill()
            return df.sort_values('timestamp').reset_index(drop=True)
        return pd.DataFrame()

    def get_market_data_gap(
        self,
        symbol: str,
        since: datetime,
        limit: int = 43200,
        exchange: str = "binance",
        columns: Optional[str] = None,
    ) -> pd.DataFrame:
        """GCS 로컬 캐시 이후 갭 구간만 Supabase에서 보충."""
        rows = self._fetch_paginated(
            "market_data", limit, "timestamp",
            since=since, columns=columns,
            symbol=symbol, exchange=exchange,
        )
        if rows:
            df = pd.DataFrame(rows)
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(str), format='mixed', utc=True, errors='coerce').bfill()
            return df.sort_values('timestamp').reset_index(drop=True)
        return pd.DataFrame()

# --------------------- Archive Manifests --------------------

    @reconnect_on_error
    def upsert_archive_manifest(self, data: Dict) -> Optional[Dict]:
        response = self.client_quant.table("archive_manifests").upsert(
            data,
            on_conflict="table_name,partition_key"
        ).execute()
        return response.data[0] if response.data else None

    def get_archive_manifest(self, table_name: str, partition_key: str) -> Optional[Dict]:
        response = self.client_quant.table("archive_manifests")\
                        .select("*")\
            .eq("table_name", table_name)\
            .eq("partition_key", partition_key)\
            .limit(1)\
            .execute()
        return response.data[0] if response.data else None

    @reconnect_on_error
    def update_archive_manifest(self, manifest_id: int, data: Dict) -> Optional[Dict]:
        response = self.client_quant.table("archive_manifests")\
            .update(data)\
            .eq("id", manifest_id)\
            .execute()
        return response.data[0] if response.data else None

    @reconnect_on_error
    def get_archive_manifests(
        self,
        statuses: Optional[List[str]] = None,
        cleanup_pending_only: bool = False,
        limit: int = 1000,
    ) -> List[Dict]:
        fetched = 0
        rows: List[Dict] = []
        page_size = 500
        while fetched < limit:
            fetch_size = min(page_size, limit - fetched)
            query = self.client_quant.table("archive_manifests").select("*")
            if statuses:
                query = query.in_("status", statuses)
            if cleanup_pending_only:
                query = query.is_("cleanup_completed_at", "null")
            response = query.order("archive_started_at").range(fetched, fetched + fetch_size - 1).execute()
            chunk = response.data if response.data else []
            if not chunk:
                break
            rows.extend(chunk)
            fetched += len(chunk)
            if len(chunk) < fetch_size:
                break
        return rows

# =====================================================================
# TEXT PROJECT — 텍스트/AI 데이터
# =====================================================================

# -------------------- Telegram Messages --------------------

    @reconnect_on_error
    def upsert_telegram_message(self, data: Dict) -> Dict:
        try:
            # Strip null bytes — PostgreSQL JSON rejects \x00 (causes 500 "JSON could not be generated")
            if data.get("text"):
                data = {**data, "text": data["text"].replace("\x00", "")}
            response = self.client_text.table("telegram_messages").upsert(
                data, on_conflict="channel,message_id"
            ).execute()
            if response.data:
                msg = data.get("text", "")[:30].replace("\n", " ")
                logger.info(f"DB: Saved [{data.get('channel')}] ID:{data.get('message_id')} | {msg}...")
            return response
        except Exception as e:
            logger.error(f"DB: Failed to upsert telegram message: {e}")
            raise

    def insert_telegram_message(self, data: Dict) -> Dict:
        """Backward compatible — now uses upsert."""
        return self.upsert_telegram_message(data)

    def get_recent_telegram_messages(self, hours: int = 24, limit: int = 200, columns: Optional[str] = None) -> List[Dict]:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        response = self.client_text.table("telegram_messages")\
                        .select(columns or "channel,message_id,text,created_at,timestamp")\
            .gte("created_at", cutoff)\
            .order("created_at", desc=True)\
            .limit(limit)\
            .execute()
        return response.data if response.data else []

    def get_telegram_messages_for_rag(self, days: int = 7, limit: int = 1000, columns: Optional[str] = None) -> List[Dict]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        response = self.client_text.table("telegram_messages")\
                        .select(columns or "channel,message_id,text,created_at,timestamp")\
            .gte("created_at", cutoff)\
            .order("created_at", desc=True)\
            .limit(limit)\
            .execute()
        return response.data if response.data else []

# ---------------------- Narrative Data ----------------------

    @reconnect_on_error
    def upsert_narrative_data(self, data: Dict) -> Dict:
        return self.client_text.table("narrative_data").upsert(
            data, on_conflict="timestamp,symbol,source"
        ).execute()

    def get_latest_narrative_data(self, symbol: str, source: str = "perplexity") -> Optional[Dict]:
        response = self.client_text.table("narrative_data")\
                        .select("*")\
            .eq("symbol", symbol)\
            .eq("source", source)\
            .order("timestamp", desc=True)\
            .limit(1)\
            .execute()
        return response.data[0] if response.data else None

# ------------------------ AI Reports ------------------------

    @reconnect_on_error
    def insert_ai_report(self, data: Dict) -> Optional[str]:
        try:
            response = self.client_text.table("ai_reports").insert(data).execute()
            return response.data[0]['id'] if response.data else None
        except Exception as e:
            msg = str(e).lower()
            if "onchain_context" in msg or "onchain_snapshot" in msg:
                logger.warning("ai_reports insert fallback: on-chain columns missing in target schema")
                legacy_data = {k: v for k, v in data.items() if k not in {"onchain_context", "onchain_snapshot"}}
                response = self.client_text.table("ai_reports").insert(legacy_data).execute()
                return response.data[0]['id'] if response.data else None
            raise

    def get_latest_report(self, symbol: str = None) -> Optional[Dict]:
        query = self.client_text.table("ai_reports")\
                        .select("*")\
            .order("created_at", desc=True)
        if symbol:
            query = query.eq("symbol", symbol)
        response = query.limit(1).execute()
        return response.data[0] if response.data else None

# --------------------- Dune Query Data ---------------------

    @reconnect_on_error
    def upsert_dune_query_result(self, data: Dict) -> Dict:
        return self.client_text.table("dune_query_results").upsert(
            data, on_conflict="query_id,collected_at"
        ).execute()

    def get_latest_dune_query_result(self, query_id: int) -> Optional[Dict]:
        response = self.client_text.table("dune_query_results")\
                        .select("*")\
            .eq("query_id", query_id)\
            .order("collected_at", desc=True)\
            .limit(1)\
            .execute()
        return response.data[0] if response.data else None

# ----------------------- Feedback ---------------------------

    @reconnect_on_error
    def insert_feedback(self, data: Dict) -> Dict:
        return self.client_text.table("feedback_logs").insert(data).execute()

    def get_feedback_history(self, limit: int = 10, feedback_type: Optional[str] = None) -> List[Dict]:
        query = self.client_text.table("feedback_logs").select("*")
        if feedback_type in ("positive", "negative"):
            query = query.eq("feedback_type", feedback_type)
        response = query.order("created_at", desc=True).limit(limit).execute()
        return response.data if response.data else []

# --------------------- Trade Executions ---------------------

    @reconnect_on_error
    def insert_trade_execution(self, data: Dict) -> Dict:
        return self.client_text.table("trade_executions").insert(data).execute()

    def get_trade_execution_by_order_id(self, order_id: str) -> Optional[Dict]:
        if not order_id:
            return None
        response = self.client_text.table("trade_executions")\
                        .select("*")\
            .eq("order_id", order_id)\
            .limit(1)\
            .execute()
        return response.data[0] if response.data else None

    @reconnect_on_error
    def update_trade_execution_fill_price(self, order_id: str, fill_price: float) -> Optional[Dict]:
        """[P2 - Fill Reconciliation] Patch confirmed fill price after async exchange fill."""
        if not order_id or not fill_price:
            return None
        response = self.client_text.table("trade_executions")\
            .update({"filled_price": fill_price})\
            .eq("order_id", order_id)\
            .execute()
        return response.data[0] if response.data else None

    def get_position_status(self, symbol: str) -> Optional[Dict]:
        response = self.client_text.table("trade_executions")\
                        .select("*")\
            .eq("symbol", symbol)\
            .order("created_at", desc=True)\
            .limit(1)\
            .execute()
        return response.data[0] if response.data else None

# -------------------- Market Status Events ------------------

    @reconnect_on_error
    def insert_market_status_event(self, data: Dict) -> Dict:
        return self.client_text.table("market_status_events").insert(data).execute()

    @reconnect_on_error
    def get_market_status_events(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
        hours: Optional[int] = None,
    ) -> List[Dict]:
        query = self.client_text.table("market_status_events")\
                        .select("*")\
            .order("created_at", desc=True)\
            .limit(limit)
        if symbol:
            query = query.eq("symbol", symbol)
        if hours is not None:
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
            query = query.gte("created_at", cutoff)
        response = query.execute()
        return response.data if response.data else []

    @reconnect_on_error
    def update_market_status_event_technical_snapshot(self, event_id: int, technical_snapshot: Dict) -> Dict:
        return self.client_text.table("market_status_events")\
            .update({"technical_snapshot": technical_snapshot})\
            .eq("id", event_id)\
            .execute()

# --------------------- Evaluation Storage -------------------

    @staticmethod
    def _to_iso(value) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

    @reconnect_on_error
    def upsert_evaluation_prediction(self, data: Dict) -> Optional[Dict]:
        if data.get("source_id") is None:
            logger.warning(
                f"upsert_evaluation_prediction skipped: source_id is None "
                f"(symbol={data.get('symbol')}, mode={data.get('mode')})"
            )
            return None
        response = self.client_text.table("evaluation_predictions").upsert(
            data,
            on_conflict="source_type,source_id,mode"
        ).execute()
        return response.data[0] if response.data else None

    def get_evaluation_prediction_by_source(
        self,
        source_type: str,
        source_id: int,
        mode: Optional[str] = None,
    ) -> Optional[Dict]:
        query = self.client_text.table("evaluation_predictions")\
                        .select("*")\
            .eq("source_type", source_type)\
            .eq("source_id", source_id)
        if mode is not None:
            query = query.eq("mode", mode)
        response = query.order("created_at", desc=True).limit(1).execute()
        return response.data[0] if response.data else None

    def get_evaluation_prediction(self, prediction_id: int) -> Optional[Dict]:
        response = self.client_text.table("evaluation_predictions")\
                        .select("*")\
            .eq("id", prediction_id)\
            .limit(1)\
            .execute()
        return response.data[0] if response.data else None

    def get_evaluation_predictions(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        symbol: Optional[str] = None,
        mode: Optional[str] = None,
        source_type: Optional[str] = None,
        limit: int = 5000,
    ) -> List[Dict]:
        rows: List[Dict] = []
        fetched = 0
        page_size = 1000
        start_iso = self._to_iso(start_time)
        end_iso = self._to_iso(end_time)

        while fetched < limit:
            fetch_size = min(page_size, limit - fetched)
            query = self.client_text.table("evaluation_predictions").select("*")
            if symbol:
                query = query.eq("symbol", symbol)
            if mode:
                query = query.eq("mode", mode)
            if source_type:
                query = query.eq("source_type", source_type)
            if start_iso:
                query = query.gte("prediction_time", start_iso)
            if end_iso:
                query = query.lt("prediction_time", end_iso)
            response = query.order("prediction_time").range(fetched, fetched + fetch_size - 1).execute()
            chunk = response.data if response.data else []
            if not chunk:
                break
            rows.extend(chunk)
            fetched += len(chunk)
            if len(chunk) < fetch_size:
                break
        return rows

    @reconnect_on_error
    def upsert_evaluation_outcome(self, data: Dict) -> Optional[Dict]:
        response = self.client_text.table("evaluation_outcomes").upsert(
            data,
            on_conflict="prediction_id,horizon_minutes"
        ).execute()
        return response.data[0] if response.data else None

    def get_evaluation_outcome(self, prediction_id: int, horizon_minutes: int) -> Optional[Dict]:
        response = self.client_text.table("evaluation_outcomes")\
                        .select("*")\
            .eq("prediction_id", prediction_id)\
            .eq("horizon_minutes", horizon_minutes)\
            .limit(1)\
            .execute()
        return response.data[0] if response.data else None

    def get_evaluation_outcomes(
        self,
        prediction_ids: Optional[List[int]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 10000,
    ) -> List[Dict]:
        start_iso = self._to_iso(start_time)
        end_iso = self._to_iso(end_time)
        rows: List[Dict] = []

        if prediction_ids:
            for idx in range(0, len(prediction_ids), 200):
                chunk_ids = prediction_ids[idx:idx + 200]
                query = self.client_text.table("evaluation_outcomes").select("*").in_("prediction_id", chunk_ids)
                if start_iso:
                    query = query.gte("evaluated_at", start_iso)
                if end_iso:
                    query = query.lt("evaluated_at", end_iso)
                response = query.order("evaluated_at").limit(limit).execute()
                if response.data:
                    rows.extend(response.data)
            return rows[:limit]

        fetched = 0
        page_size = 1000
        while fetched < limit:
            fetch_size = min(page_size, limit - fetched)
            query = self.client_text.table("evaluation_outcomes").select("*")
            if start_iso:
                query = query.gte("evaluated_at", start_iso)
            if end_iso:
                query = query.lt("evaluated_at", end_iso)
            response = query.order("evaluated_at").range(fetched, fetched + fetch_size - 1).execute()
            chunk = response.data if response.data else []
            if not chunk:
                break
            rows.extend(chunk)
            fetched += len(chunk)
            if len(chunk) < fetch_size:
                break
        return rows

    @reconnect_on_error
    def batch_upsert_evaluation_component_scores(self, data_list: List[Dict]) -> Dict:
        if not data_list:
            return {"data": []}
        return self.client_text.table("evaluation_component_scores").upsert(
            data_list,
            on_conflict="prediction_id,component_type,metric_name,scope_key"
        ).execute()

    def get_evaluation_component_scores(self, prediction_ids: List[int]) -> List[Dict]:
        rows: List[Dict] = []
        if not prediction_ids:
            return rows
        for idx in range(0, len(prediction_ids), 200):
            chunk_ids = prediction_ids[idx:idx + 200]
            response = self.client_text.table("evaluation_component_scores")\
                            .select("*")\
                .in_("prediction_id", chunk_ids)\
                .order("created_at")\
                .execute()
            if response.data:
                rows.extend(response.data)
        return rows

    @reconnect_on_error
    def delete_evaluation_rollups_for_date(self, rollup_date: str) -> Dict:
        return self.client_text.table("evaluation_rollups_daily")\
            .delete()\
            .eq("rollup_date", rollup_date)\
            .execute()

    @reconnect_on_error
    def upsert_evaluation_rollups(self, rows: List[Dict]) -> Dict:
        if not rows:
            return {"data": []}
        return self.client_text.table("evaluation_rollups_daily").upsert(
            rows,
            on_conflict="rollup_date,symbol,mode,scope,horizon_minutes,metric_name,bucket_key"
        ).execute()

# =====================================================================
# Data Cleanup (retention 만료 데이터 삭제)
# =====================================================================

    def cleanup_old_data(self) -> Dict:
        """[P3 - Cleanup Isolation] Delete data older than configured retention days.

        Each table is wrapped in its own try/except so a single table failure does NOT
        abort the remaining cleanups.  Partial failures are collected and returned so
        the scheduler can alert without hiding successful deletions.

        NOTE: trade_executions, feedback_logs are intentionally excluded — audit data.
        """
        results: Dict = {}
        errors: List[Dict] = []
        now = datetime.now(timezone.utc)

        # Helper: run one DELETE and record result or error
        def _delete(client, table: str, col: str, cutoff: str, key: str) -> None:
            try:
                r = client.table(table).delete().lt(col, cutoff).execute()
                results[key] = len(r.data) if r.data else 0
            except Exception as e:
                errors.append({"table": table, "error": str(e)})
                logger.error(f"[P3] Cleanup failed for {table}: {e}")

        cutoff_market = (now - timedelta(days=settings.RETENTION_MARKET_DATA_DAYS)).isoformat()
        cutoff_cvd    = (now - timedelta(days=settings.RETENTION_CVD_DAYS)).isoformat()
        cutoff_long   = (now - timedelta(days=settings.RETENTION_REPORTS_DAYS)).isoformat()
        cutoff_tg     = (now - timedelta(days=settings.RETENTION_TELEGRAM_DAYS)).isoformat()
        cutoff_onchain = (now - timedelta(days=settings.RETENTION_REPORTS_DAYS)).date().isoformat()

        # ── QUANT tables (each isolated) ──────────────────────────────
        _delete(self.client_quant, "market_data",            "timestamp",    cutoff_market,  "market_data_deleted")
        _delete(self.client_quant, "funding_data",           "timestamp",    cutoff_market,  "funding_data_deleted")
        _delete(self.client_quant, "liquidations",           "timestamp",    cutoff_market,  "liquidations_deleted")
        _delete(self.client_quant, "cvd_data",               "timestamp",    cutoff_cvd,     "cvd_data_deleted")
        _delete(self.client_quant, "microstructure_data",    "timestamp",    cutoff_market,  "microstructure_deleted")
        _delete(self.client_quant, "macro_data",             "timestamp",    cutoff_long,    "macro_deleted")
        _delete(self.client_quant, "deribit_data",           "timestamp",    cutoff_market,  "deribit_deleted")
        _delete(self.client_quant, "fear_greed_data",        "timestamp",    cutoff_long,    "fear_greed_deleted")
        _delete(self.client_quant, "onchain_daily_snapshots","as_of_date",   cutoff_onchain, "onchain_deleted")

        # ── TEXT tables (each isolated) ───────────────────────────────
        _delete(self.client_text, "telegram_messages",  "created_at",  cutoff_tg,    "telegram_deleted")
        _delete(self.client_text, "narrative_data",     "timestamp",   cutoff_long,  "narrative_deleted")
        _delete(self.client_text, "ai_reports",         "created_at",  cutoff_long,  "reports_deleted")
        _delete(self.client_text, "dune_query_results", "collected_at",cutoff_long,  "dune_deleted")

        if errors:
            results["cleanup_errors"] = errors
            logger.warning(f"[P3] Cleanup completed with {len(errors)} table failure(s): {[e['table'] for e in errors]}")
        else:
            logger.info(f"[P3] Cleanup completed cleanly: {results}")

        return results

    # ══════════════════════════════════════════════════════════════════════════
    # Quant Layer 1/6: Factor Signals, IC History, Trade Attribution
    # ══════════════════════════════════════════════════════════════════════════

    def insert_factor_signals(self, data: dict) -> dict:
        """factor_signals 테이블에 신호 스냅샷 저장."""
        try:
            r = self.client_text.table("factor_signals").insert(data).execute()
            return r.data[0] if r.data else {}
        except Exception as e:
            logger.error(f"[DB] insert_factor_signals failed: {e}")
            return {}

    def get_factor_signals_by_decision_id(self, decision_id: str) -> dict:
        """decision_id로 factor_signals 레코드 조회."""
        try:
            if not decision_id:
                return {}
            r = (
                self.client_text.table("factor_signals")
                .select("*")
                .eq("decision_id", str(decision_id))
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )
            return r.data[0] if r.data else {}
        except Exception as e:
            logger.error(f"[DB] get_factor_signals_by_decision_id failed: {e}")
            return {}

    def get_factor_signals(
        self,
        symbol: str,
        limit: int = 100,
        mode: str = None,
    ) -> list:
        """최근 factor_signals 목록 조회."""
        try:
            q = (
                self.client_text.table("factor_signals")
                .select("*")
                .eq("symbol", symbol)
                .order("created_at", desc=True)
                .limit(limit)
            )
            if mode:
                q = q.eq("mode", mode)
            r = q.execute()
            return r.data or []
        except Exception as e:
            logger.error(f"[DB] get_factor_signals failed: {e}")
            return []

    def upsert_factor_ic(self, data: dict) -> dict:
        """factor_ic_history 테이블에 IC 값 저장 (upsert)."""
        try:
            r = self.client_text.table("factor_ic_history").insert(data).execute()
            return r.data[0] if r.data else {}
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
        """최신 IC 히스토리 조회. factor_name 없으면 모든 팩터 반환."""
        try:
            q = (
                self.client_text.table("factor_ic_history")
                .select("*")
                .eq("symbol", symbol)
                .order("computed_at", desc=True)
                .limit(limit * 8)  # 팩터 수만큼 여유 있게
            )
            if factor_name:
                q = q.eq("factor_name", factor_name)
            if regime:
                q = q.eq("regime", regime)
            r = q.execute()
            if not r.data:
                return []
            # 팩터별 가장 최신 1개만
            seen = set()
            result = []
            for row in r.data:
                key = (row.get("factor_name"), row.get("regime"))
                if key not in seen:
                    seen.add(key)
                    result.append(row)
            return result
        except Exception as e:
            logger.error(f"[DB] get_factor_ic_history failed: {e}")
            return []

    def insert_trade_attribution(self, data: dict) -> dict:
        """trade_attribution 테이블에 거래 귀인 저장."""
        try:
            r = self.client_text.table("trade_attribution").insert(data).execute()
            return r.data[0] if r.data else {}
        except Exception as e:
            logger.error(f"[DB] insert_trade_attribution failed: {e}")
            return {}

    def get_trade_attributions(
        self,
        symbol: str,
        limit: int = 90,
    ) -> list:
        """최근 trade_attribution 목록 조회."""
        try:
            r = (
                self.client_text.table("trade_attribution")
                .select("*")
                .eq("symbol", symbol)
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return r.data or []
        except Exception as e:
            logger.error(f"[DB] get_trade_attributions failed: {e}")
            return []

    # ── Paper Orders ───────────────────────────────────────────────────────────

    def insert_paper_order(self, data: dict) -> Optional[int]:
        """paper_orders 테이블에 가상 포지션 기록. 생성된 id 반환."""
        try:
            r = self.client_text.table("paper_orders").insert(data).execute()
            return r.data[0]["id"] if r.data else None
        except Exception as e:
            logger.error(f"[DB] insert_paper_order failed: {e}")
            return None

    def get_open_paper_orders(self) -> List[Dict]:
        """현재 OPEN 상태인 모든 paper_orders 조회 (ws_price_feed 모니터링용)."""
        try:
            r = (
                self.client_text.table("paper_orders")
                .select("*")
                .eq("status", "OPEN")
                .execute()
            )
            return r.data or []
        except Exception as e:
            logger.error(f"[DB] get_open_paper_orders failed: {e}")
            return []

    def update_paper_order_closed(self, order_id: int, data: dict) -> bool:
        """paper_order 청산 처리 (exit_price, pnl, status 업데이트)."""
        try:
            self.client_text.table("paper_orders").update(data).eq("id", order_id).execute()
            return True
        except Exception as e:
            logger.error(f"[DB] update_paper_order_closed failed: {e}")
            return False

db = DatabaseClient()
