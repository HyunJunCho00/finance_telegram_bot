"""Microstructure collector (1m cadence).

Collects lightweight execution-quality proxies from Binance futures orderbook:
- Bid/Ask spread (bps)
- Top-N orderbook imbalance
- Simulated slippage for notional order size

Designed to be cheap enough for frequent collection while avoiding full depth storage.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List

import ccxt
from loguru import logger

from config.database import db
from config.settings import settings


class MicrostructureCollector:
    def __init__(self):
        self.binance_futures = ccxt.binance({
            "apiKey": settings.BINANCE_API_KEY,
            "secret": settings.BINANCE_API_SECRET,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        self.symbols = ["BTC/USDT", "ETH/USDT"]
        self.depth_levels = 5
        self.slippage_notional_usd = 100_000.0

    def _calc_slippage_bps(self, asks: List[List[float]], bids: List[List[float]], notional_usd: float) -> float:
        if not asks or not bids:
            return 0.0

        mid = (asks[0][0] + bids[0][0]) / 2.0
        if mid <= 0:
            return 0.0

        remaining = notional_usd
        filled_notional = 0.0
        filled_qty = 0.0

        for price, qty in asks:
            if remaining <= 0:
                break
            level_notional = price * qty
            take_notional = min(level_notional, remaining)
            take_qty = take_notional / price
            filled_notional += take_notional
            filled_qty += take_qty
            remaining -= take_notional

        if filled_qty <= 0:
            return 0.0

        avg_fill = filled_notional / filled_qty
        return round(((avg_fill - mid) / mid) * 10_000, 4)

    def _fetch_symbol(self, symbol: str) -> Dict:
        book = self.binance_futures.fetch_order_book(symbol, limit=self.depth_levels)
        bids = book.get("bids", [])[: self.depth_levels]
        asks = book.get("asks", [])[: self.depth_levels]

        if not bids or not asks:
            return {}

        bid1 = float(bids[0][0])
        ask1 = float(asks[0][0])
        mid = (bid1 + ask1) / 2.0
        spread_bps = ((ask1 - bid1) / mid) * 10_000 if mid > 0 else 0.0

        bid_qty_sum = sum(float(level[1]) for level in bids)
        ask_qty_sum = sum(float(level[1]) for level in asks)
        denom = bid_qty_sum + ask_qty_sum
        imbalance = (bid_qty_sum - ask_qty_sum) / denom if denom > 0 else 0.0

        slippage_buy_bps = self._calc_slippage_bps(asks, bids, self.slippage_notional_usd)

        return {
            "timestamp": datetime.now(timezone.utc).replace(second=0, microsecond=0).isoformat(),
            "symbol": symbol.replace("/", ""),
            "exchange": "binance",
            "bid_price": round(bid1, 4),
            "ask_price": round(ask1, 4),
            "spread_bps": round(spread_bps, 4),
            "bid_qty_top5": round(bid_qty_sum, 6),
            "ask_qty_top5": round(ask_qty_sum, 6),
            "orderbook_imbalance": round(imbalance, 6),
            "slippage_buy_100k_bps": slippage_buy_bps,
            "depth_levels": self.depth_levels,
        }

    def run(self) -> None:
        rows = []
        for symbol in self.symbols:
            try:
                row = self._fetch_symbol(symbol)
                if row:
                    rows.append(row)
            except Exception as e:
                logger.error(f"Microstructure fetch error for {symbol}: {e}")

        if not rows:
            return

        try:
            db.batch_upsert_microstructure_data(rows)
            logger.info(f"Saved {len(rows)} microstructure rows")
        except Exception as e:
            logger.error(f"Microstructure DB save error: {e}")


microstructure_collector = MicrostructureCollector()
