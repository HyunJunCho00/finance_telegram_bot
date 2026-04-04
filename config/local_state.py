import sqlite3
import os
import threading
from pathlib import Path
from datetime import datetime, timezone
from loguru import logger
from typing import Dict, List, Optional
from executors.execution_repository import DuplicateActiveIntentError, execution_repository

# -------- FIX MEDIUM 16 systemd WorkingDirectory --------
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
            tp1_exit_pct      REAL DEFAULT 50,
            playbook_id       TEXT DEFAULT '',
            source_decision   TEXT DEFAULT '',
            strategy_version  TEXT DEFAULT '',
            trigger_reason    TEXT DEFAULT '',
            thesis_id         TEXT DEFAULT ''
        )
    ''')

    # ------------------ FIX CRITICAL 2 ( ) ------------------
    for col, col_def in [
        ("leverage", "REAL DEFAULT 1"),
        ("tp_price", "REAL DEFAULT 0"),
        ("sl_price", "REAL DEFAULT 0"),
        ("tp2_price", "REAL DEFAULT 0"),
        ("tp1_exit_pct", "REAL DEFAULT 50"),
        ("playbook_id", "TEXT DEFAULT ''"),
        ("source_decision", "TEXT DEFAULT ''"),
        ("strategy_version", "TEXT DEFAULT ''"),
        ("trigger_reason", "TEXT DEFAULT ''"),
        ("thesis_id", "TEXT DEFAULT ''"),
    ]:
        try:
            conn.execute(f"ALTER TABLE active_orders ADD COLUMN {col} {col_def}")
        except Exception:
            pass  # 이미 존재하는 컬럼  정상

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

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS execution_outbox (
            event_id      TEXT PRIMARY KEY,
            event_type    TEXT NOT NULL,
            idempotency_key TEXT,
            payload_json  TEXT NOT NULL,
            status        TEXT NOT NULL DEFAULT 'PENDING',
            attempts      INTEGER NOT NULL DEFAULT 0,
            last_error    TEXT,
            processing_started_at TEXT,
            created_at    TEXT NOT NULL,
            updated_at    TEXT NOT NULL,
            published_at  TEXT
        )
    ''')

    for col, col_def in [
        ("idempotency_key", "TEXT"),
        ("processing_started_at", "TEXT"),
    ]:
        try:
            conn.execute(f"ALTER TABLE execution_outbox ADD COLUMN {col} {col_def}")
        except Exception:
            pass

    try:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_execution_outbox_status_created ON execution_outbox(status, created_at)"
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_execution_outbox_idempotency ON execution_outbox(idempotency_key) WHERE idempotency_key IS NOT NULL"
        )
    except Exception:
        pass

    # Default configs
    cursor.execute(
        "INSERT OR IGNORE INTO system_config (key, value) VALUES ('enable_ai_analysis', 'true')"
    )
    cursor.execute(
        "INSERT OR IGNORE INTO system_config (key, value) VALUES ('panic_mode', 'false')"
    )
    # ------------------- FIX trading_mode -------------------
    cursor.execute(
        "INSERT OR IGNORE INTO system_config (key, value) VALUES ('trading_mode', 'swing')"
    )
    # Confluence gate threshold (auto-tuned)
    cursor.execute(
        "INSERT OR IGNORE INTO system_config (key, value) VALUES ('confluence_gate_threshold', '3.0')"
    )
    cursor.execute(
        "INSERT OR IGNORE INTO system_config (key, value) VALUES ('confluence_gate_tuner_log', '[]')"
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
        # ---------------------- FIX HIGH 7 ----------------------
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
        execution_repository.set_system_config(str(key), str(value))

    def get_config(self, key: str, default: str = "") -> str:
        """Backward-compatible alias used by legacy callers."""
        return self.get_system_config(key, default)

    def set_config(self, key: str, value: str):
        """Backward-compatible alias used by legacy callers."""
        self.set_system_config(key, value)

    def set_panic_mode(self, enabled: bool):
        """Set or clear the market panic (high-volatility) flag."""
        val = 'true' if enabled else 'false'
        execution_repository.set_system_config("panic_mode", val)
        if enabled:
            logger.warning("System Config: Market Panic Mode ENABLED")

    def set_analysis_enabled(self, enabled: bool):
        """Enable or disable AI analysis/reporting."""
        val = 'true' if enabled else 'false'
        execution_repository.set_system_config("enable_ai_analysis", val)
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
        if mode not in ('swing',):
            raise ValueError(f"Invalid trading mode: {mode}")
        execution_repository.set_system_config("trading_mode", mode)
        logger.info(f"System Config: Trading Mode set to {mode.upper()}")

    def get_confluence_gate_threshold(self) -> float:
        """Return current confluence gate threshold (default 3.0)."""
        val = self.get_system_config("confluence_gate_threshold", "3.0")
        try:
            return max(1.0, min(5.0, float(val)))
        except (ValueError, TypeError):
            return 3.0

    def set_confluence_gate_threshold(self, threshold: float, reason: str = "manual"):
        """Persist a new confluence gate threshold and log the change."""
        import json as _json
        threshold = round(max(1.0, min(5.0, float(threshold))), 1)
        execution_repository.set_system_config("confluence_gate_threshold", str(threshold))

        # Append to tuner log (keep last 20 entries)
        raw = self.get_system_config("confluence_gate_tuner_log", "[]")
        try:
            log = _json.loads(raw)
        except Exception:
            log = []
        log.append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "threshold": threshold,
            "reason": reason,
        })
        execution_repository.set_system_config(
            "confluence_gate_tuner_log",
            _json.dumps(log[-20:]),
        )
        logger.info(f"Confluence gate threshold → {threshold} ({reason})")

    def get_confluence_gate_tuner_log(self) -> list:
        import json as _json
        raw = self.get_system_config("confluence_gate_tuner_log", "[]")
        try:
            return _json.loads(raw)
        except Exception:
            return []


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
        playbook_id: str = "",
        source_decision: str = "",
        strategy_version: str = "",
        trigger_reason: str = "",
        thesis_id: str = "",
    ) -> str:
        """Register a new execution intent.

        [FIX HIGH-8/9] leverage, tp_price, sl_price 장  _execute_chunk
        올바른 레버리로 마진 계산할 수 있도록 함.
        """
        try:
            intent_id = execution_repository.create_intent(
                symbol=symbol,
                direction=direction,
                style=style,
                amount=amount,
                exchange=exchange,
                ttl_hours=ttl_hours,
                leverage=leverage,
                tp_price=tp_price,
                sl_price=sl_price,
                tp2_price=tp2_price,
                tp1_exit_pct=tp1_exit_pct,
                playbook_id=str(playbook_id or ""),
                source_decision=str(source_decision or ""),
                strategy_version=str(strategy_version or ""),
                trigger_reason=str(trigger_reason or ""),
                thesis_id=str(thesis_id or ""),
            )
        except DuplicateActiveIntentError:
            logger.warning(f"Duplicate intent blocked for {symbol} [{exchange}]")
            return ""

        logger.info(
            f"Intent registered: {intent_id[:8]} | {direction} {symbol} "
            f"${amount:.2f} lev={leverage}x tp1={tp_price} tp2={tp2_price} sl={sl_price} "
            f"playbook={playbook_id or '-'} [{exchange}]"
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
        execution_repository.update_order_fill(intent_id, filled_chunk)

    def update_status(self, intent_id: str, new_status: str):
        execution_repository.update_intent_status(intent_id, new_status)

    def flush_expired(self):
        """Delete orders that have passed their TTL."""
        now = datetime.now(timezone.utc).isoformat()
        deleted = execution_repository.flush_expired_intents(now)["deleted"]
        if deleted > 0:
            logger.info(f"Local state TTL flush: Removed {deleted} stale intents.")


# Global singleton
state_manager = LocalStateManager()
