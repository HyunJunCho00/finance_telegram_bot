import threading
import sqlite3
import uuid
from datetime import datetime, timezone
from loguru import logger
from config.local_state import DB_PATH


class PaperExchangeEngine:
    """
    V7 Hardened Paper Exchange.
    [FIX HIGH-7]  threading.Lock wraps every write operation.
    [FIX CRITICAL-2] simulate_execution() stores tp/sl in paper_positions.
                     check_tp_sl() closes positions on TP or SL hit every minute.
    """

    def __init__(self):
        self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._lock = threading.Lock()  # [FIX HIGH-7] shared write lock

    # ─── read helpers ───

    def get_wallet_balance(self, exchange: str) -> float:
        cursor = self._conn.cursor()
        cursor.execute("SELECT balance FROM paper_wallets WHERE exchange = ?", (exchange.lower(),))
        row = cursor.fetchone()
        return row["balance"] if row else 0.0

    # ─── write helpers ───

    def update_wallet_balance(self, exchange: str, delta: float):
        with self._lock:
            cursor = self._conn.cursor()
            now = datetime.now(timezone.utc).isoformat()
            cursor.execute(
                "UPDATE paper_wallets SET balance = balance + ?, updated_at = ? WHERE exchange = ?",
                (delta, now, exchange.lower()),
            )
            self._conn.commit()

    def simulate_execution(
        self,
        exchange: str,
        symbol: str,
        direction: str,
        amount_usd: float,
        leverage: float,
        style: str,
        raw_price: float,
        tp_price: float = 0.0,
        sl_price: float = 0.0,
    ) -> dict:
        """
        [FIX HIGH-8/9] leverage correctly passed from intent (not hard-coded 1x).
        [FIX CRITICAL-2] tp_price / sl_price stored in paper_positions.
        """
        slippage_map = {
            "MOMENTUM_SNIPER": 0.002,
            "CASINO_EXIT":     0.005,
            "PASSIVE_MAKER":  -0.0002,
            "SMART_DCA":       0.0005,
        }
        slippage_pct = slippage_map.get(style, 0.001)
        penalty = (raw_price * slippage_pct) if direction == "LONG" else -(raw_price * slippage_pct)
        filled_price = raw_price + penalty

        with self._lock:
            wallet = self.get_wallet_balance(exchange)
            required_margin = amount_usd / max(leverage, 1.0)

            if required_margin > wallet:
                return {
                    "success": False,
                    "error": f"Insufficient margin. Need ${required_margin:.2f}, have ${wallet:.2f}",
                }

            coin_size = amount_usd / filled_price
            pos_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc).isoformat()
            cursor = self._conn.cursor()
            cursor.execute(
                """INSERT INTO paper_positions
                   (position_id, exchange, symbol, side, size, entry_price,
                    leverage, is_open, created_at, updated_at, tp_price, sl_price)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)""",
                (pos_id, exchange.lower(), symbol, direction,
                 coin_size, filled_price, leverage, now, now, tp_price, sl_price),
            )
            cursor.execute(
                "UPDATE paper_wallets SET balance = balance - ?, updated_at = ? WHERE exchange = ?",
                (required_margin, now, exchange.lower()),
            )
            self._conn.commit()

        logger.info(
            f"V7 Paper | [{exchange}] {direction} {symbol} @ {filled_price:.4f} "
            f"(slip={slippage_pct*100:.2f}% lev={leverage}x margin=${required_margin:.2f} "
            f"tp={tp_price} sl={sl_price})"
        )
        return {
            "success": True,
            "order_id": pos_id,
            "filled_price": filled_price,
            "size_coin": coin_size,
            "margin_used": required_margin,
            "slippage_applied_pct": slippage_pct * 100,
        }

    def check_liquidations(self, current_prices: dict):
        """Liquidate if PnL eats 80% of initial margin."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM paper_positions WHERE is_open = 1")
        for pos in cursor.fetchall():
            symbol = pos["symbol"]
            price = current_prices.get(symbol)
            if not price:
                continue
            entry = pos["entry_price"]
            size = pos["size"]
            lev = pos["leverage"]
            side = pos["side"]
            initial_margin = (size * entry) / max(lev, 1.0)
            pnl = (price - entry) * size if side == "LONG" else (entry - price) * size
            if initial_margin > 0 and (pnl / initial_margin) <= -0.8:
                logger.warning(
                    f"LIQUIDATION: [{pos['exchange']}] {side} {symbol} entry={entry:.2f} now={price:.2f}"
                )
                with self._lock:
                    self._conn.execute(
                        "UPDATE paper_positions SET is_open = 0, updated_at = ? WHERE position_id = ?",
                        (datetime.now(timezone.utc).isoformat(), pos["position_id"]),
                    )
                    self._conn.commit()

    def check_tp_sl(self, current_prices: dict):
        """
        [FIX CRITICAL-2] Closes positions when TP or SL price is reached.
        Previously MISSING — positions never closed via TP/SL anywhere in the system.
        """
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM paper_positions WHERE is_open = 1 AND (tp_price > 0 OR sl_price > 0)"
        )
        for pos in cursor.fetchall():
            symbol = pos["symbol"]
            price = current_prices.get(symbol)
            if not price:
                continue

            side, tp, sl = pos["side"], pos["tp_price"], pos["sl_price"]
            entry, size, lev = pos["entry_price"], pos["size"], pos["leverage"]
            exchange = pos["exchange"]

            hit_reason = None
            if side == "LONG":
                if tp > 0 and price >= tp:
                    hit_reason = f"TP HIT @ {price:.2f} (target={tp:.2f})"
                elif sl > 0 and price <= sl:
                    hit_reason = f"SL HIT @ {price:.2f} (stop={sl:.2f})"
            else:
                if tp > 0 and price <= tp:
                    hit_reason = f"TP HIT @ {price:.2f} (target={tp:.2f})"
                elif sl > 0 and price >= sl:
                    hit_reason = f"SL HIT @ {price:.2f} (stop={sl:.2f})"

            if hit_reason:
                pnl = (price - entry) * size if side == "LONG" else (entry - price) * size
                initial_margin = (size * entry) / max(lev, 1.0)
                returned = max(initial_margin + pnl, 0.0)
                icon = "✅" if "TP" in hit_reason else "🛑"
                logger.info(
                    f"{icon} {hit_reason} [{exchange}] {side} {symbol} "
                    f"PnL=${pnl:+.2f} returning=${returned:.2f}"
                )
                with self._lock:
                    now = datetime.now(timezone.utc).isoformat()
                    self._conn.execute(
                        "UPDATE paper_positions SET is_open = 0, updated_at = ? WHERE position_id = ?",
                        (now, pos["position_id"]),
                    )
                    self._conn.execute(
                        "UPDATE paper_wallets SET balance = balance + ?, updated_at = ? WHERE exchange = ?",
                        (returned, now, exchange),
                    )
                    self._conn.commit()


paper_engine = PaperExchangeEngine()
