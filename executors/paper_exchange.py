from loguru import logger
import sqlite3
import uuid
from datetime import datetime, timezone
from config.local_state import DB_PATH

class PaperExchangeEngine:
    """
    V7 Hardened Paper Exchange.
    Replaces fake "zero-slippage" naive paper trading with realistic margin rules,
    wallet deductions, and execution penalties.
    """
    def __init__(self):
        self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")

    def get_wallet_balance(self, exchange: str) -> float:
        cursor = self._conn.cursor()
        cursor.execute("SELECT balance FROM paper_wallets WHERE exchange = ?", (exchange.lower(),))
        row = cursor.fetchone()
        return row['balance'] if row else 0.0

    def update_wallet_balance(self, exchange: str, delta: float):
        cursor = self._conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        cursor.execute('''
            UPDATE paper_wallets 
            SET balance = balance + ?, updated_at = ? 
            WHERE exchange = ?
        ''', (delta, now, exchange.lower()))
        self._conn.commit()

    def simulate_execution(self, exchange: str, symbol: str, direction: str, amount_usd: float, leverage: float, style: str, raw_price: float) -> dict:
        """
        Calculates execution price with realistic slippage based on style.
        """
        # 1. Slippage Penalty
        slippage_pct = 0.0
        if style == "MOMENTUM_SNIPER":
            slippage_pct = 0.002 # 0.2% slippage for market sweeps
        elif style == "CASINO_EXIT":
            slippage_pct = 0.005 # 0.5% slippage during panic
        elif style == "PASSIVE_MAKER":
            slippage_pct = -0.0002 # 0.02% rebate (negative slippage)
        elif style == "SMART_DCA":
            slippage_pct = 0.0005 # 0.05% mild impact
            
        penalty = (raw_price * slippage_pct) if direction == "LONG" else -(raw_price * slippage_pct)
        filled_price = raw_price + penalty
        
        # 2. Check Margin
        wallet = self.get_wallet_balance(exchange)
        required_margin = amount_usd / leverage
        
        if required_margin > wallet:
            return {"success": False, "error": f"Insufficient margin. Need ${required_margin:.2f}, have ${wallet:.2f}"}
            
        # 3. Size calculation based on filled price
        coin_size = amount_usd / filled_price
        
        # 4. Open Position & Deduct active margin
        pos_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        
        cursor = self._conn.cursor()
        cursor.execute('''
            INSERT INTO paper_positions 
            (position_id, exchange, symbol, side, size, entry_price, leverage, is_open, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
        ''', (pos_id, exchange.lower(), symbol, direction, coin_size, filled_price, leverage, now, now))
        self._conn.commit()
        
        # Deduct margin from available balance (We treat paper_wallets as 'Available Balance')
        self.update_wallet_balance(exchange, -required_margin)
        
        logger.info(f"V7 Paper Execution: [{exchange}] {direction} {symbol} @ {filled_price:.4f} (Raw: {raw_price}, Slippage: {slippage_pct*100}%)")
        
        return {
            "success": True,
            "order_id": pos_id,
            "filled_price": filled_price,
            "size_coin": coin_size,
            "margin_used": required_margin,
            "slippage_applied_pct": slippage_pct * 100
        }
        
    def check_liquidations(self, current_prices: dict):
        """
        Scan all open positions against current market prices to enforce margin calls.
        """
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM paper_positions WHERE is_open = 1")
        open_positions = cursor.fetchall()
        
        for pos in open_positions:
            symbol = pos['symbol']
            price = current_prices.get(symbol)
            if not price: continue
            
            # Simple margin call logic: if PnL eats 80% of margin, liquidate.
            entry = pos['entry_price']
            size = pos['size']
            lev = pos['leverage']
            side = pos['side']
            
            initial_margin = (size * entry) / lev
            
            pnl = 0
            if side == 'LONG':
                pnl = (price - entry) * size
            else:
                pnl = (entry - price) * size
                
            pnl_pct_of_margin = pnl / initial_margin
            
            if pnl_pct_of_margin <= -0.8: # 80% margin depleted
                logger.warning(f"🚨 LIQUIDATION TRIGGERED: [{pos['exchange']}] {side} {symbol} at {price}")
                # Close position
                cursor.execute("UPDATE paper_positions SET is_open = 0, updated_at = ? WHERE position_id = ?", 
                               (datetime.now(timezone.utc).isoformat(), pos['position_id']))
                # The remaining 20% is lost to exchange fees in liquidation, wallet gets 0 back.
                self._conn.commit()

paper_engine = PaperExchangeEngine()
