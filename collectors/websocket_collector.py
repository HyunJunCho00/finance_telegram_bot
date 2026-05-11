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
import socket
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

import websockets
from loguru import logger

from config.database import db
from config.settings import settings
from executors.paper_exchange import paper_engine


# ----------------------- Constants -----------------------

BINANCE_WS_BASE = "wss://fstream.binance.com/ws"

# Whale threshold: trades >= $100,000 USD
WHALE_THRESHOLD_USD = 100_000

# Flush interval in seconds
FLUSH_INTERVAL = 60

# ---- Symbols to track derived from central TRADING_SYMBOLS config ----
SYMBOLS = [s.lower() for s in settings.trading_symbols]


# -------------------- Buffer Classes --------------------

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

    def get_data(self) -> List[Dict]:
        """Return accumulated data without clearing the buffer."""
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
            return records

    def clear(self):
        """Reset the buffer."""
        with self._lock:
            self._buffer.clear()

    def flush(self) -> List[Dict]:
        """[DEPRECATED] Use get_data() + clear() for safe DB flush."""
        data = self.get_data()
        self.clear()
        return data


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

    def get_data(self) -> Dict[str, Dict]:
        """Return accumulated whale data per symbol without clearing."""
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
            return result

    def clear(self):
        """Reset the buffer."""
        with self._lock:
            self._buffer.clear()

    def flush(self) -> Dict[str, Dict]:
        """[DEPRECATED] Use get_data() + clear() for safe DB flush."""
        data = self.get_data()
        self.clear()
        return data


# ------------------ WebSocket Collector ------------------

class WebSocketCollector:
    """Manages Binance Futures WebSocket connections for liquidation and whale data."""

    def __init__(self):
        self.liq_buffer = LiquidationBuffer()
        self.whale_buffer = WhaleBuffer()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._last_message_time: float = 0.0
        self._started_at: float = 0.0

    def _build_ws_url(self) -> str:
        """Combined stream URL for all symbols + liquidation."""
        streams = ["!forceOrder@arr"]
        for sym in SYMBOLS:
            streams.append(f"{sym}@aggTrade")
        stream_path = "/".join(streams)
        return f"wss://fstream.binance.com/stream?streams={stream_path}"

    async def _handle_message(self, message: str):
        """Parse and route incoming WebSocket messages."""
        self._last_message_time = time.time()
        try:
            data = json.loads(message)

            # Combined stream format: {"stream": "...", "data": {...}}
            stream = data.get("stream", "")
            payload = data.get("data", {})

            if "forceOrder" in stream:
                # Liquidation event
                order = payload.get("o", {})
                symbol = order.get("s", "")  # e.g. "BTCUSDT"
                
                # [FIX] Filter only tracked symbols to prevent Supabase insert errors & DB bloat
                if not symbol or symbol.lower() not in SYMBOLS:
                    return
                    
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
                    
                    # [UPGRADE V8] Real-time Paper Trading Trigger
                    if settings.PAPER_TRADING_MODE:
                        # Map Binance symbol (BTCUSDT) to our config if necessary, 
                        # but PaperEngine's check_tp_sl handles raw symbols.
                        paper_engine.handle_realtime_price(symbol, price)

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.debug(f"WS message parse error: {e}")

    async def _flush_to_db(self):
        """Periodically flush buffers to database."""
        while self._running:
            await asyncio.sleep(FLUSH_INTERVAL)
            try:
                # 1. Flush liquidations: Only clear if DB write succeeds
                liq_records = self.liq_buffer.get_data()
                if liq_records:
                    # ── 1순위: 로컬 파케이 캐시 ────────────────────────────
                    try:
                        import pandas as pd
                        from processors.gcs_parquet import gcs_parquet_store
                        df_liq = pd.DataFrame(liq_records)
                        if "timestamp" in df_liq.columns:
                            df_liq["timestamp"] = pd.to_datetime(df_liq["timestamp"], utc=True, errors="coerce")
                        for sym in df_liq["symbol"].unique() if "symbol" in df_liq.columns else []:
                            gcs_parquet_store.write_timeseries_to_local(
                                "liquidations", sym,
                                df_liq[df_liq["symbol"] == sym].copy(),
                                ["timestamp", "symbol"],
                            )
                    except Exception as e:
                        logger.debug(f"[LocalCache] liquidations local write skipped: {e}")

                    # ── 2순위: Supabase ──────────────────────────────────────
                    try:
                        db.batch_upsert_liquidations(liq_records)
                        self.liq_buffer.clear()  # Success
                        total_liq = sum(r["long_liq_usd"] + r["short_liq_usd"] for r in liq_records)
                        logger.info(f"WS flush: {len(liq_records)} liquidation records (${total_liq:,.0f})")
                    except Exception as e:
                        self.liq_buffer.clear()  # 로컬에 저장됐으니 버퍼 비움
                        logger.warning(f"WS liquidation flush → Supabase skipped (local cache OK): {e}")

                # 2. Flush whale CVD: Only clear if DB write succeeds
                whale_data = self.whale_buffer.get_data()
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
                    try:
                        db.batch_upsert_whale_data(whale_records)
                        self.whale_buffer.clear() # Success
                        logger.info(f"WS flush: {len(whale_records)} whale CVD records")
                    except Exception as e:
                        logger.warning(f"WS whale flush failed (data kept in buffer): {e}")

            except Exception as e:
                logger.error(f"WS flush error: {e}")

    @staticmethod
    def _apply_tcp_keepalive(ws) -> None:
        """OS 레벨 TCP Keepalive 설정. app-level ping과 독립적으로 동작하는 2차 방어선."""
        try:
            sock = ws.transport.get_extra_info("socket")
            if sock is None:
                return
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 10)   # 10초 idle 후 첫 probe
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 5)   # probe 간격 5초
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)     # 3회 무응답 → 연결 사망
        except Exception as e:
            logger.debug(f"[WS] TCP keepalive 설정 실패 (무시): {e}")

    async def _backfill_missed_liquidations(self, gap_start: float) -> None:
        """재연결 후 REST로 누락 구간 청산 데이터 복구.
        !forceOrder@arr 스트림은 시장 전체 청산이라 공개 REST 엔드포인트가 없음.
        Binance aggTrades 기반 대안: cvd_data 버퍼는 1분 유실 허용 설계이므로 스킵.
        청산 버퍼는 gap 구간 데이터를 0으로 기록하는 대신 경고 로그만 남김.
        """
        gap_secs = time.time() - gap_start
        if gap_secs < 3:
            return
        logger.warning(
            f"[WS Backfill] 청산 스트림 {gap_secs:.0f}s 공백 발생. "
            f"시장 전체 청산은 공개 REST 히스토리 없음 — 해당 구간 버퍼 공백으로 처리."
        )

    async def _connect_and_listen(self):
        """Main WebSocket connection loop with reconnect."""
        url = self._build_ws_url()
        backoff = 5.0
        _disconnect_at: float = 0.0

        while self._running:
            try:
                logger.info(f"WebSocket connecting to {len(SYMBOLS)} streams + liquidation...")
                async with websockets.connect(
                    url,
                    ping_interval=20,   # 20초마다 ping → soft failure (NAT 타임아웃 등) 감지
                    ping_timeout=10,    # 10초 내 pong 없으면 ConnectionClosedError
                    close_timeout=5,
                    max_size=2**20,
                ) as ws:
                    self._apply_tcp_keepalive(ws)  # OS 레벨 keepalive 추가

                    if _disconnect_at > 0:
                        await self._backfill_missed_liquidations(_disconnect_at)
                        _disconnect_at = 0.0

                    logger.info("WebSocket connected successfully")
                    self._last_message_time = time.time()
                    _connected_at = time.time()
                    backoff = 5.0
                    msg_count = 0

                    async for message in ws:
                        if not self._running:
                            break
                        # Binance 24h TTL 강제 종료 전에 23h에서 proactive reconnect
                        if time.time() - _connected_at > 23 * 3600:
                            logger.info("[WS] 23h 경과 — Binance 24h TTL 전 proactive reconnect")
                            break
                        msg_count += 1
                        if msg_count == 1:
                            logger.info("WebSocket receiving data (first message OK)")
                        self._last_message_time = time.time()
                        await self._handle_message(message)

            except asyncio.CancelledError:
                logger.warning("WebSocket task cancelled, retrying in 5s...")
                _disconnect_at = time.time()
                try:
                    await asyncio.sleep(5)
                except asyncio.CancelledError:
                    break  # 진짜 종료 신호
            except websockets.ConnectionClosedError as e:
                logger.warning(f"WebSocket connection closed: {e}. Reconnecting in {backoff:.0f}s...")
                _disconnect_at = time.time()
                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:
                    break
                backoff = min(backoff * 2, 60.0)
            except Exception as e:
                logger.error(f"WebSocket error: {e}. Reconnecting in {backoff:.0f}s...")
                _disconnect_at = time.time()
                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:
                    break
                backoff = min(backoff * 2, 60.0)

    async def _message_watchdog(self):
        """Soft failure 감지: 메시지가 60초 이상 없으면 연결 강제 종료 → 재연결 트리거."""
        await asyncio.sleep(60)  # 최초 연결 안정화 유예
        while self._running:
            await asyncio.sleep(15)
            if self._last_message_time == 0.0:
                continue
            silent = time.time() - self._last_message_time
            if silent > 60:
                logger.warning(f"[WS Watchdog] No message for {silent:.0f}s — forcing reconnect")
                for task in list(getattr(self, '_tasks', [])):
                    if not task.done():
                        task.cancel()
                return

    async def _run_async(self):
        """Run listener, flusher, and watchdog concurrently."""
        self._running = True
        tasks = [
            asyncio.ensure_future(self._connect_and_listen()),
            asyncio.ensure_future(self._flush_to_db()),
            asyncio.ensure_future(self._message_watchdog()),
        ]
        self._tasks = tasks
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            self._tasks = []

    def start_background(self):
        """Start WebSocket collector in a background thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("WebSocket collector already running")
            return

        def _run():
            backoff = 5.0
            while self._running:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._loop = loop
                try:
                    loop.run_until_complete(self._run_async())
                    backoff = 5.0  # 정상 종료 시 백오프 리셋
                except Exception as e:
                    logger.error(f"WebSocket event loop crashed: {e}. Restarting in {backoff:.0f}s...")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60.0)
                finally:
                    self._loop = None
                    try:
                        loop.close()
                    except Exception:
                        pass
            logger.info("WebSocket collector thread exiting (running=False)")

        self._running = True
        self._started_at = time.time()
        self._last_message_time = 0.0
        self._tasks = []
        self._thread = threading.Thread(target=_run, daemon=True, name="ws-collector")
        self._thread.start()
        logger.info("WebSocket collector started in background thread")

    def stop(self):
        """Stop the WebSocket collector by cancelling asyncio tasks."""
        self._running = False
        loop = self._loop
        if loop and loop.is_running():
            for task in list(getattr(self, '_tasks', [])):
                loop.call_soon_threadsafe(task.cancel)
        if self._thread:
            self._thread.join(timeout=5)
        self._thread = None
        self._loop = None
        logger.info("WebSocket collector stopped")


# Singleton
websocket_collector = WebSocketCollector()
