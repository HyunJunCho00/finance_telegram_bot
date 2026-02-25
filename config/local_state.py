import sqlite3
import os
from datetime import datetime, timezone
import uuid
from loguru import logger
from typing import Dict, List

DB_PATH = "data/local_state.db"

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS active_orders (
            intent_id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            direction TEXT NOT NULL,
            execution_style TEXT NOT NULL,
            total_target_amount REAL NOT NULL,
            filled_amount REAL DEFAULT 0,
            remaining_amount REAL NOT NULL,
            status TEXT DEFAULT 'PENDING',
            exchange TEXT NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL
        )
    ''')
    
    # V7: Hardened Paper Trading Engine
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS paper_wallets (
            exchange TEXT PRIMARY KEY,
            balance REAL NOT NULL,
            updated_at TEXT NOT NULL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS paper_positions (
            position_id TEXT PRIMARY KEY,
            exchange TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            size REAL NOT NULL,
            entry_price REAL NOT NULL,
            leverage REAL NOT NULL,
            is_open BOOLEAN DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    ''')
    
    # Initialize default V8 wallets if not exists
    now = datetime.now(timezone.utc).isoformat()
    cursor.execute("INSERT OR IGNORE INTO paper_wallets (exchange, balance, updated_at) VALUES ('binance', 2000.0, ?)", (now,))
    cursor.execute("INSERT OR IGNORE INTO paper_wallets (exchange, balance, updated_at) VALUES ('upbit', 2000000.0, ?)", (now,))
    conn.commit()
    conn.close()

class LocalStateManager:
    """Manages fast, temporary state for the Execution Desk."""
    
    def __init__(self):
        init_db()
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL;")

    def add_intent(self, symbol: str, direction: str, style: str, amount: float, exchange: str, ttl_hours: int = 24) -> str:
        intent_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        expires = datetime.fromtimestamp(now.timestamp() + (ttl_hours * 3600), tz=timezone.utc)
        
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO active_orders 
            (intent_id, symbol, direction, execution_style, total_target_amount, remaining_amount, exchange, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (intent_id, symbol, direction, style, amount, amount, exchange, now.isoformat(), expires.isoformat()))
        logger.info(f"Registered generic intent {intent_id} for {direction} {symbol} on {exchange}")
        self.conn.commit()
        return intent_id

    def get_active_orders(self) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM active_orders WHERE status IN ('PENDING', 'ACTIVE')")
        return [dict(row) for row in cursor.fetchall()]

    def update_order_fill(self, intent_id: str, filled_chunk: float):
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE active_orders 
            SET filled_amount = filled_amount + ?, 
                remaining_amount = remaining_amount - ?
            WHERE intent_id = ?
        ''', (filled_chunk, filled_chunk, intent_id))
        
        # Check if completed
        cursor.execute("SELECT remaining_amount FROM active_orders WHERE intent_id = ?", (intent_id,))
        res = cursor.fetchone()
        if res and res['remaining_amount'] <= 0.0001:
            self.update_status(intent_id, 'COMPLETED')
        self.conn.commit()

    def update_status(self, intent_id: str, new_status: str):
        cursor = self.conn.cursor()
        cursor.execute("UPDATE active_orders SET status = ? WHERE intent_id = ?", (new_status, intent_id))
        self.conn.commit()

    def flush_expired(self):
        """Delete orders that have passed their TTL."""
        now = datetime.now(timezone.utc).isoformat()
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM active_orders WHERE expires_at < ?", (now,))
        deleted = cursor.rowcount
        self.conn.commit()
        if deleted > 0:
            logger.info(f"Local state TTL flush: Removed {deleted} stale intents.")

# Global
state_manager = LocalStateManager()
