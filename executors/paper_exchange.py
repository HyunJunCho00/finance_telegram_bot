from typing import List, Dict
import threading
import sqlite3
import uuid
import sys
import asyncio
from telegram import Bot
from config.settings import settings
from datetime import datetime, timezone
from loguru import logger
from config.local_state import DB_PATH


FEE_PCT = 0.0005  # 0.05% Binance Futures Market Fee


class PaperExchangeEngine:
    """
    V7 Hardened Paper Exchange.
    [FIX HIGH-7]  threading.Lock wraps every write operation.
    [FIX CRITICAL-2] simulate_execution() stores tp/sl in paper_positions.
                     check_tp_sl() closes positions on TP or SL hit every minute.
    [V8 HARDENING] Added 0.05% per-side trading fee and Telegram notifications.
    """

    def __init__(self):
        self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._lock = threading.Lock()  # [FIX HIGH-7] shared write lock

    def _notify(self, message: str):
        """Send urgent notification via Telegram bot in a background thread."""
        async def _async_send():
            try:
                bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
                await bot.send_message(chat_id=settings.TELEGRAM_CHAT_ID, text=message, parse_mode='HTML')
            except Exception as e:
                logger.error(f"PaperEngine notification failed: {e}")

        import threading
        t = threading.Thread(target=lambda: asyncio.run(_async_send()), daemon=True)
        t.start()

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

    def get_open_positions(self) -> List[Dict]:
        """Return all currently open mock positions."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM paper_positions WHERE is_open = 1")
        return [dict(row) for row in cursor.fetchall()]

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
            entry_fee = amount_usd * FEE_PCT

            if (required_margin + entry_fee) > wallet:
                return {
                    "success": False,
                    "error": f"Insufficient margin (incl. fee). Need ${required_margin + entry_fee:.2f}, have ${wallet:.2f}",
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
            # Deduct margin + entry fee
            cursor.execute(
                "UPDATE paper_wallets SET balance = balance - ?, updated_at = ? WHERE exchange = ?",
                (required_margin + entry_fee, now, exchange.lower()),
            )
            self._conn.commit()

        logger.info(
            f"V8 Paper | [{exchange}] {direction} {symbol} @ {filled_price:.4f} "
            f"(slip={slippage_pct*100:.2f}% lev={leverage}x margin=${required_margin:.2f} fee=${entry_fee:.2f} "
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
                    now = datetime.now(timezone.utc).isoformat()
                    self._conn.execute(
                        "UPDATE paper_positions SET is_open = 0, updated_at = ? WHERE position_id = ?",
                        (now, pos["position_id"]),
                    )
                    self._conn.commit()
                
                # Notification
                self._notify(
                    f"💀 <b>LIQUIDATION</b>\n"
                    f"Exchange: {pos['exchange'].upper()}\n"
                    f"Symbol: {symbol}\n"
                    f"Side: {side}\n"
                    f"Price: {price:.2f}\n"
                    f"Entry: {entry:.2f}\n"
                    f"Margin Lost: ${initial_margin:.2f}"
                )

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
                exit_notional = size * price
                exit_fee = exit_notional * FEE_PCT
                
                initial_margin = (size * entry) / max(lev, 1.0)
                # Apply exit fee to returned amount
                returned = max(initial_margin + pnl - exit_fee, 0.0)
                
                icon = "✅" if "TP" in hit_reason else "🛑"
                logger.info(
                    f"{icon} {hit_reason} [{exchange}] {side} {symbol} "
                    f"PnL=${pnl:+.2f} fee=${exit_fee:.2f} returning=${returned:.2f}"
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

                # Notification
                pnl_pct = (pnl / initial_margin) * 100 if initial_margin > 0 else 0
                self._notify(
                    f"{icon} <b>POSITION CLOSED ({'TP' if 'TP' in hit_reason else 'SL'})</b>\n"
                    f"Exchange: {exchange.upper()}\n"
                    f"Symbol: {symbol}\n"
                    f"Side: {side}\n"
                    f"Close Price: {price:.2f}\n"
                    f"PnL: <b>${pnl:+.2f} ({pnl_pct:+.2f}%)</b>\n"
                    f"Fee: ${exit_fee:.2f}"
                )


    def handle_realtime_price(self, symbol: str, price: float):
        """
        [UPGRADE V8] Called via WebSocket for sub-millisecond precision.
        Checks TP/SL and Liquidations for a single symbol immediately.
        """
        # Maintain a local thread-safe check (using the main DB lock for simplicity)
        # In a very high-traffic scenario, we would cache active positions in RAM.
        self.check_tp_sl({symbol: price})
        self.check_liquidations({symbol: price})


paper_engine = PaperExchangeEngine()
