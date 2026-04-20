"""Real-time Binance price feed via WebSocket.

Streams aggTrade (last trade price) and markPriceUpdate (futures mark price)
in a background daemon thread. Provides zero-latency in-memory price reads
so order execution does not need a blocking REST call at fire time.

Usage:
    from collectors.ws_price_feed import ws_price_feed
    ws_price_feed.start()          # called once at bot startup
    price = ws_price_feed.get("BTCUSDT")   # O(1) dict lookup, thread-safe
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Dict, List, Optional

import websockets
from loguru import logger


_FUTURES_WS = "wss://fstream.binance.com/stream"
_STALE_SECONDS = 5.0   # price older than this → fall back to REST

# 재연결 백오프: 3s → 6s → 12s → 최대 30s
_RECONNECT_BASE = 3.0
_RECONNECT_MAX  = 30.0


class WsPriceFeed:
    def __init__(self) -> None:
        self._lock = threading.Lock()                  # Fix 3: 공유 상태 보호

        self._prices: Dict[str, float] = {}
        self._mark_prices: Dict[str, float] = {}
        self._updated_at: Dict[str, float] = {}
        self._symbols: List[str] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Paper order monitoring
        self._last_fetch: Dict[str, float] = {}
        self._open_orders: Dict[str, List[dict]] = {}

    # ── Public API ──────────────────────────────────────────────────────────────

    def start(self, symbols: Optional[List[str]] = None) -> None:
        """Start background streaming. Safe to call multiple times."""
        if self._running:
            return
        from config.settings import settings
        self._symbols = symbols or [
            s.strip().upper().replace("/", "")
            for s in settings.trading_symbols
        ]
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            name="ws-price-feed",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"[WsPriceFeed] Started for {self._symbols}")

    def stop(self) -> None:
        self._running = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

    def get(self, symbol: str) -> Optional[float]:
        """Latest trade price from memory. None if stale (>5s) or not yet received."""
        key = symbol.upper().replace("/", "")
        with self._lock:                               # Fix 3
            ts = self._updated_at.get(key, 0.0)
            price = self._prices.get(key)
        if time.monotonic() - ts < _STALE_SECONDS:
            return price
        return None

    def get_mark(self, symbol: str) -> Optional[float]:
        """Latest futures mark price from memory."""
        key = symbol.upper().replace("/", "")
        with self._lock:                               # Fix 3
            return self._mark_prices.get(key)

    def is_live(self, symbol: str) -> bool:
        key = symbol.upper().replace("/", "")
        with self._lock:                               # Fix 3
            ts = self._updated_at.get(key, 0.0)
        return (time.monotonic() - ts) < _STALE_SECONDS

    # ── Fix 1: 주문 즉시 등록 ───────────────────────────────────────────────────

    def register_order(self, symbol: str, order: dict) -> None:
        """포지션 오픈 즉시 메모리에 등록 — 60초 DB 갱신 대기 없이 바로 모니터링.

        order 필드: id, symbol, side, entry_price, sl_price, tp1_price
        """
        sym = symbol.upper().replace("/", "")
        with self._lock:
            orders = self._open_orders.setdefault(sym, [])
            # 중복 방지
            existing_ids = {o.get("id") for o in orders}
            if order.get("id") not in existing_ids:
                orders.append(order)
                logger.debug(f"[WsPriceFeed] Registered order {order.get('id')} for {sym} monitoring")

    def deregister_order(self, symbol: str, order_id) -> None:
        """포지션 청산 시 메모리에서 제거."""
        sym = symbol.upper().replace("/", "")
        with self._lock:
            orders = self._open_orders.get(sym, [])
            self._open_orders[sym] = [o for o in orders if o.get("id") != order_id]

    # ── Internal ────────────────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        backoff = _RECONNECT_BASE                      # Fix 2: 지수 백오프
        while self._running:
            try:
                self._loop.run_until_complete(self._stream())
                backoff = _RECONNECT_BASE              # 정상 종료 시 백오프 리셋
            except Exception as exc:
                logger.warning(
                    f"[WsPriceFeed] Disconnected ({exc}), "
                    f"reconnecting in {backoff:.0f}s…"
                )
                # Fix 2: 재연결 공백 동안 REST fallback으로 SL 긴급 체크
                self._fallback_sl_check_on_reconnect()
                time.sleep(backoff)
                backoff = min(backoff * 2, _RECONNECT_MAX)

    async def _stream(self) -> None:
        streams: List[str] = []
        for sym in self._symbols:
            s = sym.lower()
            streams.append(f"{s}@aggTrade")
            streams.append(f"{s}@markPrice")

        url = f"{_FUTURES_WS}?streams={'/'.join(streams)}"
        async with websockets.connect(
            url,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5,
        ) as ws:
            logger.info(f"[WsPriceFeed] Connected ({len(streams)} streams)")
            async for raw in ws:
                if not self._running:
                    break
                self._handle(raw)

    def _handle(self, raw: str) -> None:
        try:
            data = json.loads(raw).get("data", {})
            event = data.get("e")
            sym = str(data.get("s", "")).upper()
            if not sym:
                return
            if event == "aggTrade":
                price = float(data["p"])
                with self._lock:                       # Fix 3
                    self._prices[sym] = price
                    self._updated_at[sym] = time.monotonic()
                self._check_paper_orders(sym, price)   # 락 밖에서 호출
            elif event == "markPriceUpdate":
                with self._lock:                       # Fix 3
                    self._mark_prices[sym] = float(data["p"])
        except Exception:
            pass  # single bad frame — keep going

    # ── Fix 2: 재연결 공백 동안 REST fallback ────────────────────────────────────

    def _fallback_sl_check_on_reconnect(self) -> None:
        """WS 연결이 끊긴 동안 REST로 현재가 조회해서 SL/TP 긴급 체크."""
        with self._lock:
            all_orders = {
                sym: list(orders)
                for sym, orders in self._open_orders.items()
                if orders
            }
        if not all_orders:
            return

        logger.warning("[WsPriceFeed] WS down — running REST fallback SL check")
        try:
            import ccxt
            from config.settings import settings
            exchange = ccxt.binance({
                "apiKey": settings.BINANCE_API_KEY,
                "secret": settings.BINANCE_API_SECRET,
                "options": {"defaultType": "future"},
            })
            for sym, orders in all_orders.items():
                try:
                    ccxt_sym = sym.replace("USDT", "/USDT") if "/" not in sym else sym
                    ticker = exchange.fetch_ticker(ccxt_sym)
                    price = float(ticker.get("last") or 0)
                    if price > 0:
                        self._check_paper_orders(sym, price)
                        logger.info(f"[WsPriceFeed] REST fallback check {sym} @ {price:.2f}")
                except Exception as e:
                    logger.warning(f"[WsPriceFeed] REST fallback failed for {sym}: {e}")
        except Exception as e:
            logger.error(f"[WsPriceFeed] REST fallback init failed: {e}")

    # ── Paper Order Monitoring ──────────────────────────────────────────────────

    _FETCH_INTERVAL = 60.0  # DB 전체 재조회 주기 (register_order로 즉시 등록 보완)

    def _check_paper_orders(self, symbol: str, current_price: float) -> None:
        """실시간 가격이 들어올 때마다 paper_orders SL/TP 조건 확인."""
        now = time.monotonic()

        # 주기적으로 DB에서 전체 OPEN 목록 동기화 (register_order 누락분 보완)
        if now - self._last_fetch.get(symbol, 0) > self._FETCH_INTERVAL:
            self._last_fetch[symbol] = now
            try:
                from config.database import db
                all_open = db.get_open_paper_orders()
                db_orders = [o for o in all_open if o.get("symbol") == symbol]
                with self._lock:
                    existing = self._open_orders.get(symbol, [])
                    existing_ids = {o.get("id") for o in existing}
                    # DB에만 있고 메모리에 없는 것 추가 (재시작 후 복원)
                    for o in db_orders:
                        if o.get("id") not in existing_ids:
                            existing.append(o)
                    self._open_orders[symbol] = existing
            except Exception as e:
                logger.warning(f"[PaperMonitor] DB fetch failed for {symbol}: {e}")

        with self._lock:
            orders = list(self._open_orders.get(symbol, []))

        if not orders:
            return

        still_open = []
        for order in orders:
            closed = self._try_close_order(order, current_price)
            if not closed:
                still_open.append(order)

        with self._lock:
            self._open_orders[symbol] = still_open

    def _try_close_order(self, order: dict, current_price: float) -> bool:
        """SL 또는 TP 조건 충족 시 paper_order 청산 처리. 청산됐으면 True 반환."""
        try:
            from config.database import db
            from datetime import datetime, timezone

            side = str(order.get("side", "")).upper()
            sl   = float(order.get("sl_price") or 0)
            tp1  = float(order.get("tp1_price") or 0)
            entry = float(order.get("entry_price") or current_price)
            order_id = order.get("id")

            hit_sl = hit_tp = False
            if side == "LONG":
                hit_sl = sl > 0 and current_price <= sl
                hit_tp = tp1 > 0 and current_price >= tp1
            elif side == "SHORT":
                hit_sl = sl > 0 and current_price >= sl
                hit_tp = tp1 > 0 and current_price <= tp1

            if not (hit_sl or hit_tp):
                return False

            status = "CLOSED_SL" if hit_sl else "CLOSED_TP1"
            pnl_pct = (
                (current_price - entry) / entry * 100
                if side == "LONG"
                else (entry - current_price) / entry * 100
            )

            created_at_str = order.get("created_at", "")
            try:
                created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                holding_hours = (datetime.now(timezone.utc) - created_at).total_seconds() / 3600
            except Exception:
                holding_hours = None

            db.update_paper_order_closed(order_id, {
                "status":           status,
                "exit_price":       current_price,
                "exit_at":          datetime.now(timezone.utc).isoformat(),
                "realized_pnl_pct": round(pnl_pct, 4),
                "holding_hours":    round(holding_hours, 2) if holding_hours else None,
            })
            logger.info(
                f"[PaperMonitor] {order.get('symbol')} {side} id={order_id} "
                f"→ {status} @ {current_price:.2f} | PnL: {pnl_pct:+.2f}%"
            )
            return True

        except Exception as e:
            logger.error(f"[PaperMonitor] close failed for order {order.get('id')}: {e}")
            return False


ws_price_feed = WsPriceFeed()
