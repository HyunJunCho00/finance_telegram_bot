from __future__ import annotations

import json
import sqlite3
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from loguru import logger

_BASE_DIR = Path(__file__).parent.parent
_DB_PATH = str(_BASE_DIR / "data" / "local_state.db")


class AgentStateStore:
    """Persistent blackboard snapshot store.

    Two-layer architecture:
    - In-memory dict: fast reads, zero-latency hot-path access
    - SQLite backend: survives process restarts (crash recovery)

    On startup, non-expired rows are loaded from SQLite into memory,
    so TTL=6h onchain slots and TTL=30m narrative slots don't get
    re-fetched after a simple restart.
    """

    _TABLE_DDL = """
        CREATE TABLE IF NOT EXISTS agent_snapshots (
            symbol       TEXT NOT NULL,
            mode         TEXT NOT NULL,
            agent_name   TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            confidence   REAL,
            status       TEXT NOT NULL DEFAULT 'fresh',
            created_at   TEXT NOT NULL,
            expires_at   TEXT NOT NULL,
            latency_ms   REAL,
            input_hash   TEXT NOT NULL DEFAULT '',
            version      INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY (symbol, mode, agent_name)
        )
    """

    def __init__(self, db_path: str = _DB_PATH) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._snapshots: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._ensure_table()
        self._load_from_db()

    # ------------------------------------------------------------------
    # SQLite helpers
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _ensure_table(self) -> None:
        try:
            with self._conn() as conn:
                conn.execute(self._TABLE_DDL)
        except Exception as exc:
            logger.warning(f"AgentStateStore: table init failed: {exc}")

    def _load_from_db(self) -> None:
        """Restore non-expired snapshots from SQLite on startup."""
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM agent_snapshots WHERE expires_at > ?", (now,)
                ).fetchall()
            count = 0
            for row in rows:
                key = (row["symbol"], row["mode"])
                bundle = self._snapshots.setdefault(
                    key,
                    {
                        "symbol": row["symbol"],
                        "mode": row["mode"],
                        "agents": {},
                        "updated_at": row["created_at"],
                    },
                )
                bundle["agents"][row["agent_name"]] = {
                    "payload": json.loads(row["payload_json"] or "{}"),
                    "confidence": row["confidence"],
                    "status": row["status"],
                    "created_at": row["created_at"],
                    "expires_at": row["expires_at"],
                    "latency_ms": row["latency_ms"],
                    "input_hash": row["input_hash"] or "",
                    "version": row["version"] or 1,
                }
                count += 1
            if count:
                logger.info(f"AgentStateStore: restored {count} snapshot(s) from SQLite")
        except Exception as exc:
            logger.warning(f"AgentStateStore: DB load failed (cold start OK): {exc}")

    def _db_upsert(
        self,
        symbol: str,
        mode: str,
        agent_name: str,
        payload: Dict[str, Any],
        *,
        confidence: Optional[float],
        status: str,
        created_at: str,
        expires_at: str,
        latency_ms: Optional[float],
        input_hash: str,
        version: int,
    ) -> None:
        try:
            payload_json = json.dumps(payload, ensure_ascii=False, default=str)
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO agent_snapshots
                        (symbol, mode, agent_name, payload_json, confidence,
                         status, created_at, expires_at, latency_ms, input_hash, version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        symbol, mode, agent_name, payload_json,
                        confidence, status, created_at, expires_at,
                        latency_ms, input_hash, version,
                    ),
                )
        except Exception as exc:
            logger.warning(f"AgentStateStore: SQLite write failed for {agent_name}: {exc}")

    # ------------------------------------------------------------------
    # Public API (unchanged signatures)
    # ------------------------------------------------------------------

    @staticmethod
    def _key(symbol: str, mode: str) -> Tuple[str, str]:
        return (str(symbol or "").upper(), str(mode or "").lower())

    @staticmethod
    def _iso_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def upsert_agent_state(
        self,
        symbol: str,
        mode: str,
        agent_name: str,
        payload: Dict[str, Any],
        *,
        confidence: Optional[float] = None,
        status: str = "fresh",
        created_at: Optional[str] = None,
        expires_at: Optional[str] = None,
        latency_ms: Optional[float] = None,
        input_hash: str = "",
        version: int = 1,
    ) -> Dict[str, Any]:
        created = created_at or self._iso_now()
        exp = expires_at or created
        row = {
            "payload": deepcopy(payload or {}),
            "confidence": confidence,
            "status": status,
            "created_at": created,
            "expires_at": exp,
            "latency_ms": latency_ms,
            "input_hash": input_hash,
            "version": version,
        }
        key = self._key(symbol, mode)
        with self._lock:
            bundle = self._snapshots.setdefault(
                key,
                {
                    "symbol": key[0],
                    "mode": key[1],
                    "agents": {},
                    "updated_at": created,
                },
            )
            bundle["agents"][agent_name] = row
            bundle["updated_at"] = created

        # Async-safe fire-and-forget SQLite write (lock already dropped)
        self._db_upsert(
            key[0], key[1], agent_name, payload or {},
            confidence=confidence, status=status,
            created_at=created, expires_at=exp,
            latency_ms=latency_ms, input_hash=input_hash, version=version,
        )
        return deepcopy(row)

    def load_bundle(self, symbol: str, mode: str) -> Dict[str, Any]:
        key = self._key(symbol, mode)
        with self._lock:
            bundle = self._snapshots.get(
                key,
                {
                    "symbol": key[0],
                    "mode": key[1],
                    "agents": {},
                    "updated_at": self._iso_now(),
                },
            )
            return deepcopy(bundle)

    def purge_expired(self) -> int:
        """Delete expired rows from SQLite. Call from daily cleanup job."""
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._conn() as conn:
                cur = conn.execute(
                    "DELETE FROM agent_snapshots WHERE expires_at <= ?", (now,)
                )
                deleted = cur.rowcount
            if deleted:
                logger.info(f"AgentStateStore: purged {deleted} expired snapshot row(s)")
            return deleted
        except Exception as exc:
            logger.warning(f"AgentStateStore: purge failed: {exc}")
            return 0

    def delete_bundle(self, symbol: str, mode: str) -> None:
        key = self._key(symbol, mode)
        with self._lock:
            self._snapshots.pop(key, None)
        try:
            with self._conn() as conn:
                conn.execute(
                    "DELETE FROM agent_snapshots WHERE symbol=? AND mode=?",
                    (key[0], key[1]),
                )
        except Exception as exc:
            logger.warning(f"AgentStateStore: SQLite delete failed: {exc}")


agent_state_store = AgentStateStore()
