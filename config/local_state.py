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
            sl_price          REAL DEFAULT 0,
            tp2_price         REAL DEFAULT 0,
            tp1_exit_pct      REAL DEFAULT 50
        )
    ''')

    # [FIX CRITICAL-2] 기존 테이블에 새 컬럼 추가 (이미 존재하면 무시)
    for col, col_def in [
        ("leverage", "REAL DEFAULT 1"),
        ("tp_price", "REAL DEFAULT 0"),
        ("sl_price", "REAL DEFAULT 0"),
        ("tp2_price", "REAL DEFAULT 0"),
        ("tp1_exit_pct", "REAL DEFAULT 50"),
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

    # V12: Interactive Control
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_config (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    ''')

    # [NEW] Paper Trading Tables (V13.5 Support)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS paper_positions (
            position_id    TEXT PRIMARY KEY,
            exchange      TEXT NOT NULL,
            symbol        TEXT NOT NULL,
            side          TEXT NOT NULL,
            size          REAL NOT NULL,
            entry_price   REAL NOT NULL,
            leverage      REAL DEFAULT 1.0,
            is_open       INTEGER DEFAULT 1,
            created_at    TEXT NOT NULL,
            updated_at    TEXT NOT NULL,
            tp_price      REAL DEFAULT 0.0,
            sl_price      REAL DEFAULT 0.0,
            tp2_price     REAL DEFAULT 0.0,
            tp1_done      INTEGER DEFAULT 0,
            tp1_exit_pct  REAL DEFAULT 50.0
        )
    ''')

    for col, col_def in [
        ("tp2_price", "REAL DEFAULT 0.0"),
        ("tp1_done", "INTEGER DEFAULT 0"),
        ("tp1_exit_pct", "REAL DEFAULT 50.0"),
    ]:
        try:
            conn.execute(f"ALTER TABLE paper_positions ADD COLUMN {col} {col_def}")
        except Exception:
            pass

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS paper_wallets (
            exchange      TEXT PRIMARY KEY,
            balance       REAL NOT NULL,
            updated_at    TEXT NOT NULL
        )
    ''')

    # Default configs
    cursor.execute(
        "INSERT OR IGNORE INTO system_config (key, value) VALUES ('enable_ai_analysis', 'true')"
    )
    cursor.execute(
        "INSERT OR IGNORE INTO system_config (key, value) VALUES ('panic_mode', 'false')"
    )
    # [FIX] trading_mode 영구 저장 — 재시작 후에도 유지
    cursor.execute(
        "INSERT OR IGNORE INTO system_config (key, value) VALUES ('trading_mode', 'swing')"
    )

    # [NEW] Initial Paper Balance (Seed data)
    now = datetime.now(timezone.utc).isoformat()
    cursor.execute(
        "INSERT OR IGNORE INTO paper_wallets (exchange, balance, updated_at) VALUES ('binance', 100000.0, ?)",
        (now,)
    )
    cursor.execute(
        "INSERT OR IGNORE INTO paper_wallets (exchange, balance, updated_at) VALUES ('binance_spot', 100000.0, ?)",
        (now,)
    )
    cursor.execute(
        "INSERT OR IGNORE INTO paper_wallets (exchange, balance, updated_at) VALUES ('upbit', 100000.0, ?)",
        (now,)
    )

    conn.commit()
    conn.close()


class LocalStateManager:
    """Manages fast, temporary state for the Execution Desk and System Config."""

    def __init__(self):
        init_db()
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL;")
        # [FIX HIGH-7] 멀티스레드 동시 쓰기 방지 락
        self._lock = threading.Lock()

    def is_analysis_enabled(self) -> bool:
        """Check if AI analysis/reporting is currently enabled."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT value FROM system_config WHERE key = 'enable_ai_analysis'")
            row = cursor.fetchone()
            if row:
                return row['value'].lower() == 'true'
            return True

    def is_panic_mode(self) -> bool:
        """Check if market is currently in a high-volatility state."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT value FROM system_config WHERE key = 'panic_mode'")
            row = cursor.fetchone()
            if row:
                return row['value'].lower() == 'true'
            return False

    def get_system_config(self, key: str, default: str = "") -> str:
        """Generic getter for the system_config table."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT value FROM system_config WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                return row['value']
            return default

    def set_system_config(self, key: str, value: str):
        """Generic setter for the system_config table."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO system_config (key, value) VALUES (?, ?)",
                (str(key), str(value)),
            )
            self.conn.commit()

    def get_config(self, key: str, default: str = "") -> str:
        """Backward-compatible alias used by legacy callers."""
        return self.get_system_config(key, default)

    def set_config(self, key: str, value: str):
        """Backward-compatible alias used by legacy callers."""
        self.set_system_config(key, value)

    def set_panic_mode(self, enabled: bool):
        """Set or clear the market panic (high-volatility) flag."""
        val = 'true' if enabled else 'false'
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO system_config (key, value) VALUES ('panic_mode', ?)",
                (val,)
            )
            self.conn.commit()
        if enabled:
            logger.warning("System Config: Market Panic Mode ENABLED")

    def set_analysis_enabled(self, enabled: bool):
        """Enable or disable AI analysis/reporting."""
        val = 'true' if enabled else 'false'
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO system_config (key, value) VALUES ('enable_ai_analysis', ?)",
                (val,)
            )
            self.conn.commit()
        logger.info(f"System Config: AI Analysis {'ENABLED' if enabled else 'DISABLED'}")

    def get_telegram_chat_id(self, default: str = "") -> str:
        """Return the persisted Telegram target chat id, if any."""
        return self.get_system_config("telegram_chat_id", default)

    def set_telegram_chat_id(self, chat_id: str):
        """Persist the Telegram target chat id for scheduled notifications."""
        normalized = str(chat_id or "").strip()
        if not normalized:
            return
        self.set_system_config("telegram_chat_id", normalized)
        logger.info(f"System Config: Telegram chat id set to {normalized}")

    def get_trading_mode(self) -> str:
        """Get persisted trading mode (swing|position). Defaults to 'swing'."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT value FROM system_config WHERE key = 'trading_mode'")
            row = cursor.fetchone()
            if row:
                return row['value'].lower()
            return 'swing'

    def set_trading_mode(self, mode: str):
        """Persist trading mode to SQLite so it survives process restarts."""
        mode = mode.lower().strip()
        if mode not in ('swing', 'position'):
            raise ValueError(f"Invalid trading mode: {mode}")
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO system_config (key, value) VALUES ('trading_mode', ?)",
                (mode,)
            )
            self.conn.commit()
        logger.info(f"System Config: Trading Mode set to {mode.upper()}")


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
        tp2_price: float = 0.0,
        tp1_exit_pct: float = 50.0,
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
            # Deduplicate pending/active intents for same symbol+exchange to avoid order overlap.
            cursor.execute(
                """
                SELECT intent_id
                FROM active_orders
                WHERE symbol = ?
                  AND exchange = ?
                  AND status IN ('PENDING', 'ACTIVE')
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (symbol, exchange),
            )
            existing = cursor.fetchone()
            if existing:
                logger.warning(
                    f"Duplicate intent blocked for {symbol} [{exchange}] "
                    f"(existing={existing['intent_id'][:8]})"
                )
                return ""

            cursor.execute('''
                INSERT INTO active_orders
                (intent_id, symbol, direction, execution_style,
                 total_target_amount, remaining_amount, exchange,
                 created_at, expires_at, leverage, tp_price, sl_price, tp2_price, tp1_exit_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                intent_id, symbol, direction, style,
                amount, amount, exchange,
                now.isoformat(), expires.isoformat(),
                leverage, tp_price, sl_price, tp2_price, tp1_exit_pct,
            ))
            self.conn.commit()

        logger.info(
            f"Intent registered: {intent_id[:8]} | {direction} {symbol} "
            f"${amount:.2f} lev={leverage}x tp1={tp_price} tp2={tp2_price} sl={sl_price} [{exchange}]"
        )
        return intent_id

    def get_reserved_margin(self, exchange: str) -> float:
        """Estimated reserved margin from not-yet-filled intents on an exchange."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT remaining_amount, leverage
                FROM active_orders
                WHERE exchange = ?
                  AND status IN ('PENDING', 'ACTIVE')
                """,
                (exchange,),
            )
            rows = cursor.fetchall()
        reserved = 0.0
        for row in rows:
            remaining = float(row["remaining_amount"] or 0.0)
            lev = float(row["leverage"] or 1.0)
            reserved += remaining / max(lev, 1.0)
        return reserved

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
