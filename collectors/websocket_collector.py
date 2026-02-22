"""Real-time WebSocket collector for Liquidation + Whale CVD.

Two streams from Binance Futures WebSocket:
1. !forceOrder@arr  → Liquidation events (force-closed positions)
2. aggTrade         → Individual trades, filtered for $100K+ (whale detection)

Architecture:
  WebSocket → in-memory buffer (1 min) → batch insert to PostgreSQL
  - Crash-safe: max 1 minute of data loss (acceptable for supplementary data)
  - Low resource: single asyncio event loop, no heavy dependencies

Usage:
  Called from scheduler.py as a background thread.
  Runs indefinitely, flushing to DB every 60 seconds.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

import websockets
from loguru import logger

from config.database import db
from config.settings import settings


# ─────────────── Constants ───────────────

BINANCE_WS_BASE = "wss://fstream.binance.com/ws"

# Whale threshold: trades >= $100,000 USD
WHALE_THRESHOLD_USD = 100_000

# Flush interval in seconds
FLUSH_INTERVAL = 60

# Symbols to track — derived from central TRADING_SYMBOLS config
SYMBOLS = [s.lower() for s in settings.trading_symbols]


# ─────────────── Buffer Classes ───────────────

class LiquidationBuffer:
    """Accumulates liquidation events per minute per symbol."""

    def __init__(self):
        self._lock = threading.Lock()
        self._buffer: Dict[str, Dict] = {}  # key: "symbol"

    def add(self, symbol: str, side: str, qty: float, price: float):
        """Add a liquidation event.
        side: "BUY" means short was liquidated, "SELL" means long was liquidated.
        """
        usd_value = qty * price
        key = symbol.upper()

        with self._lock:
            if key not in self._buffer:
                self._buffer[key] = {
                    "long_liq_usd": 0.0,
                    "short_liq_usd": 0.0,
                    "long_liq_count": 0,
                    "short_liq_count": 0,
                    "largest_single_usd": 0.0,
                    "largest_single_side": "",
                    "largest_single_price": 0.0,
                }

            entry = self._buffer[key]

            # SELL order in forceOrder = LONG position was liquidated
            # BUY order in forceOrder = SHORT position was liquidated
            if side == "SELL":
                entry["long_liq_usd"] += usd_value
                entry["long_liq_count"] += 1
                liq_side = "LONG"
            else:
                entry["short_liq_usd"] += usd_value
                entry["short_liq_count"] += 1
                liq_side = "SHORT"

            if usd_value > entry["largest_single_usd"]:
                entry["largest_single_usd"] = usd_value
                entry["largest_single_side"] = liq_side
                entry["largest_single_price"] = price

    def flush(self) -> List[Dict]:
        """Return accumulated data and reset buffer."""
        with self._lock:
            if not self._buffer:
                return []

            now = datetime.now(timezone.utc).replace(second=0, microsecond=0).isoformat()
            records = []
            for symbol, data in self._buffer.items():
                total = data["long_liq_usd"] + data["short_liq_usd"]
                if total > 0:
                    records.append({
                        "timestamp": now,
                        "symbol": symbol,
                        "long_liq_usd": round(data["long_liq_usd"], 2),
                        "short_liq_usd": round(data["short_liq_usd"], 2),
                        "long_liq_count": data["long_liq_count"],
                        "short_liq_count": data["short_liq_count"],
                        "largest_single_usd": round(data["largest_single_usd"], 2),
                        "largest_single_side": data["largest_single_side"],
                        "largest_single_price": round(data["largest_single_price"], 2),
                    })
            self._buffer.clear()
            return records


class WhaleBuffer:
    """Accumulates whale trades (>= $100K) per minute per symbol."""

    def __init__(self):
        self._lock = threading.Lock()
        self._buffer: Dict[str, Dict] = {}

    def add(self, symbol: str, is_buyer_maker: bool, qty: float, price: float):
        """Add a whale trade.
        is_buyer_maker=True  → taker is SELLER (aggressive sell)
        is_buyer_maker=False → taker is BUYER  (aggressive buy)
        """
        usd_value = qty * price
        if usd_value < WHALE_THRESHOLD_USD:
            return  # Not a whale trade

        key = symbol.upper()
        with self._lock:
            if key not in self._buffer:
                self._buffer[key] = {
                    "whale_buy_vol": 0.0,
                    "whale_sell_vol": 0.0,
                    "whale_buy_count": 0,
                    "whale_sell_count": 0,
                }

            entry = self._buffer[key]
            if not is_buyer_maker:
                # Taker is buyer = aggressive buying
                entry["whale_buy_vol"] += usd_value
                entry["whale_buy_count"] += 1
            else:
                # Taker is seller = aggressive selling
                entry["whale_sell_vol"] += usd_value
                entry["whale_sell_count"] += 1

    def flush(self) -> Dict[str, Dict]:
        """Return accumulated whale data per symbol and reset."""
        with self._lock:
            result = {}
            for symbol, data in self._buffer.items():
                if data["whale_buy_vol"] > 0 or data["whale_sell_vol"] > 0:
                    result[symbol] = {
                        "whale_buy_vol": round(data["whale_buy_vol"], 2),
                        "whale_sell_vol": round(data["whale_sell_vol"], 2),
                        "whale_buy_count": data["whale_buy_count"],
                        "whale_sell_count": data["whale_sell_count"],
                    }
            self._buffer.clear()
            return result


# ─────────────── WebSocket Collector ───────────────

class WebSocketCollector:
    """Manages Binance Futures WebSocket connections for liquidation and whale data."""

    def __init__(self):
        self.liq_buffer = LiquidationBuffer()
        self.whale_buffer = WhaleBuffer()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _build_ws_url(self) -> str:
        """Combined stream URL for all symbols + liquidation."""
        streams = ["!forceOrder@arr"]
        for sym in SYMBOLS:
            streams.append(f"{sym}@aggTrade")
        stream_path = "/".join(streams)
        return f"wss://fstream.binance.com/stream?streams={stream_path}"

    async def _handle_message(self, message: str):
        """Parse and route incoming WebSocket messages."""
        try:
            data = json.loads(message)

            # Combined stream format: {"stream": "...", "data": {...}}
            stream = data.get("stream", "")
            payload = data.get("data", {})

            if "forceOrder" in stream:
                # Liquidation event
                order = payload.get("o", {})
                symbol = order.get("s", "")  # e.g. "BTCUSDT"
                side = order.get("S", "")  # "BUY" or "SELL"
                qty = float(order.get("q", 0))
                price = float(order.get("p", 0))

                if symbol and side and qty > 0 and price > 0:
                    self.liq_buffer.add(symbol, side, qty, price)

            elif "aggTrade" in stream:
                # Aggregated trade
                symbol = payload.get("s", "")
                price = float(payload.get("p", 0))
                qty = float(payload.get("q", 0))
                is_buyer_maker = payload.get("m", False)

                if symbol and qty > 0 and price > 0:
                    self.whale_buffer.add(symbol, is_buyer_maker, qty, price)

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.debug(f"WS message parse error: {e}")

    async def _flush_to_db(self):
        """Periodically flush buffers to database."""
        while self._running:
            await asyncio.sleep(FLUSH_INTERVAL)
            try:
                # Flush liquidations
                liq_records = self.liq_buffer.flush()
                if liq_records:
                    db.batch_upsert_liquidations(liq_records)
                    total_liq = sum(r["long_liq_usd"] + r["short_liq_usd"] for r in liq_records)
                    logger.info(f"WS flush: {len(liq_records)} liquidation records (${total_liq:,.0f})")

                # Flush whale CVD (merge into existing cvd_data minute row)
                whale_data = self.whale_buffer.flush()
                if whale_data:
                    now = datetime.now(timezone.utc).replace(second=0, microsecond=0).isoformat()
                    whale_records = []
                    for symbol, wdata in whale_data.items():
                        whale_records.append({
                            "timestamp": now,
                            "symbol": symbol,
                            "whale_buy_vol": wdata["whale_buy_vol"],
                            "whale_sell_vol": wdata["whale_sell_vol"],
                            "whale_buy_count": wdata["whale_buy_count"],
                            "whale_sell_count": wdata["whale_sell_count"],
                        })
                    db.batch_upsert_whale_data(whale_records)
                    logger.info(f"WS flush: {len(whale_records)} whale CVD records")

            except Exception as e:
                logger.error(f"WS flush error: {e}")

    async def _connect_and_listen(self):
        """Main WebSocket connection loop with reconnect."""
        url = self._build_ws_url()

        while self._running:
            try:
                logger.info(f"WebSocket connecting to {len(SYMBOLS)} streams + liquidation...")
                async with websockets.connect(
                    url,
                    ping_interval=30,
                    ping_timeout=10,
                    close_timeout=5,
                    max_size=2**20,  # 1MB max message
                ) as ws:
                    logger.info("WebSocket connected successfully")
                    async for message in ws:
                        if not self._running:
                            break
                        await self._handle_message(message)

            except websockets.ConnectionClosedError as e:
                logger.warning(f"WebSocket connection closed: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"WebSocket error: {e}. Reconnecting in 10s...")
                await asyncio.sleep(10)

    async def _run_async(self):
        """Run both the listener and the flusher concurrently."""
        self._running = True
        await asyncio.gather(
            self._connect_and_listen(),
            self._flush_to_db(),
        )

    def start_background(self):
        """Start WebSocket collector in a background thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("WebSocket collector already running")
            return

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._run_async())
            except Exception as e:
                logger.error(f"WebSocket background thread error: {e}")
            finally:
                loop.close()

        self._running = True
        self._thread = threading.Thread(target=_run, daemon=True, name="ws-collector")
        self._thread.start()
        logger.info("WebSocket collector started in background thread")

    def stop(self):
        """Stop the WebSocket collector."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
            logger.info("WebSocket collector stopped")


# Singleton
websocket_collector = WebSocketCollector()
