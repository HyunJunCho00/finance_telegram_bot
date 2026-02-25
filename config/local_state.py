import sqlite3
import os
import threading
from pathlib import Path
from datetime import datetime, timezone
import uuid
from loguru import logger
from typing import Dict, List, Optional

# [FIX MEDIUM-16] 절대 경로 사용 — systemd WorkingDirectory와 무관하게 동작
_BASE_DIR = Path(__file__).parent.parent
DB_PATH = str(_BASE_DIR / "data" / "local_state.db")


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS active_orders (
            intent_id         TEXT PRIMARY KEY,
            symbol            TEXT NOT NULL,
            direction         TEXT NOT NULL,
            execution_style   TEXT NOT NULL,
            total_target_amount REAL NOT NULL,
            filled_amount     REAL DEFAULT 0,
            remaining_amount  REAL NOT NULL,
            status            TEXT DEFAULT 'PENDING',
            exchange          TEXT NOT NULL,
            created_at        TEXT NOT NULL,
            expires_at        TEXT NOT NULL,
            leverage          REAL DEFAULT 1,
            tp_price          REAL DEFAULT 0,
            sl_price          REAL DEFAULT 0
        )
    ''')

    # [FIX CRITICAL-2] 기존 테이블에 새 컬럼 추가 (이미 존재하면 무시)
    for col, col_def in [
        ("leverage", "REAL DEFAULT 1"),
        ("tp_price", "REAL DEFAULT 0"),
        ("sl_price", "REAL DEFAULT 0"),
    ]:
        try:
            conn.execute(f"ALTER TABLE active_orders ADD COLUMN {col} {col_def}")
        except Exception:
            pass  # 이미 존재하는 컬럼 — 정상

    # [FIX RESOURCE-2] Index for flush_expired() and get_active_orders() performance
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_active_orders_status ON active_orders(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_active_orders_expires ON active_orders(expires_at)")
    except Exception:
        pass

    # V7: Hardened Paper Trading Engine
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS paper_wallets (
            exchange   TEXT PRIMARY KEY,
            balance    REAL NOT NULL,
            updated_at TEXT NOT NULL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS paper_positions (
            position_id TEXT PRIMARY KEY,
            exchange    TEXT NOT NULL,
            symbol      TEXT NOT NULL,
            side        TEXT NOT NULL,
            size        REAL NOT NULL,
            entry_price REAL NOT NULL,
            leverage    REAL NOT NULL,
            is_open     BOOLEAN DEFAULT 1,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL,
            tp_price    REAL DEFAULT 0,
            sl_price    REAL DEFAULT 0
        )
    ''')

    # [FIX CRITICAL-2] paper_positions에 tp/sl 컬럼 추가 (기존 DB 마이그레이션)
    for col, col_def in [
        ("tp_price", "REAL DEFAULT 0"),
        ("sl_price", "REAL DEFAULT 0"),
    ]:
        try:
            conn.execute(f"ALTER TABLE paper_positions ADD COLUMN {col} {col_def}")
        except Exception:
            pass

    # Initialize default V8 wallets if not exists
    now = datetime.now(timezone.utc).isoformat()
    cursor.execute(
        "INSERT OR IGNORE INTO paper_wallets (exchange, balance, updated_at) VALUES ('binance', 2000.0, ?)",
        (now,)
    )
    cursor.execute(
        "INSERT OR IGNORE INTO paper_wallets (exchange, balance, updated_at) VALUES ('upbit', 2000000.0, ?)",
        (now,)
    )
    conn.commit()
    conn.close()


class LocalStateManager:
    """Manages fast, temporary state for the Execution Desk."""

    def __init__(self):
        init_db()
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL;")
        # [FIX HIGH-7] 멀티스레드 동시 쓰기 방지 락
        self._lock = threading.Lock()

    def add_intent(
        self,
        symbol: str,
        direction: str,
        style: str,
        amount: float,
        exchange: str,
        ttl_hours: int = 24,
        leverage: float = 1.0,
        tp_price: float = 0.0,
        sl_price: float = 0.0,
    ) -> str:
        """Register a new execution intent.

        [FIX HIGH-8/9] leverage, tp_price, sl_price 저장 — _execute_chunk가
        올바른 레버리지로 마진 계산할 수 있도록 함.
        """
        intent_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        expires = datetime.fromtimestamp(now.timestamp() + (ttl_hours * 3600), tz=timezone.utc)

        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO active_orders
                (intent_id, symbol, direction, execution_style,
                 total_target_amount, remaining_amount, exchange,
                 created_at, expires_at, leverage, tp_price, sl_price)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                intent_id, symbol, direction, style,
                amount, amount, exchange,
                now.isoformat(), expires.isoformat(),
                leverage, tp_price, sl_price,
            ))
            self.conn.commit()

        logger.info(
            f"Intent registered: {intent_id[:8]} | {direction} {symbol} "
            f"${amount:.2f} lev={leverage}x tp={tp_price} sl={sl_price} [{exchange}]"
        )
        return intent_id

    def get_active_orders(self) -> List[Dict]:
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM active_orders WHERE status IN ('PENDING', 'ACTIVE')")
            return [dict(row) for row in cursor.fetchall()]

    def update_order_fill(self, intent_id: str, filled_chunk: float):
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                UPDATE active_orders
                SET filled_amount    = filled_amount + ?,
                    remaining_amount = remaining_amount - ?
                WHERE intent_id = ?
            ''', (filled_chunk, filled_chunk, intent_id))

            cursor.execute(
                "SELECT remaining_amount FROM active_orders WHERE intent_id = ?",
                (intent_id,)
            )
            res = cursor.fetchone()
            if res and res['remaining_amount'] <= 0.0001:
                cursor.execute(
                    "UPDATE active_orders SET status = 'COMPLETED' WHERE intent_id = ?",
                    (intent_id,)
                )
            self.conn.commit()

    def update_status(self, intent_id: str, new_status: str):
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE active_orders SET status = ? WHERE intent_id = ?",
                (new_status, intent_id)
            )
            self.conn.commit()

    def flush_expired(self):
        """Delete orders that have passed their TTL."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM active_orders WHERE expires_at < ?", (now,))
            deleted = cursor.rowcount
            self.conn.commit()
        if deleted > 0:
            logger.info(f"Local state TTL flush: Removed {deleted} stale intents.")


# Global singleton
state_manager = LocalStateManager()
