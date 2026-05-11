"""Binance Futures User Data Stream — real-time fill confirmation.

Listens for ORDER_TRADE_UPDATE events and updates DB fill price immediately,
replacing the delayed outbox REST polling path.

Architecture:
  - Background daemon thread with its own asyncio event loop
  - listenKey obtained via REST, renewed every 29 minutes
  - On FILLED event: db.update_trade_execution_fill_price() called inline
  - outbox _enqueue_fill_check remains as safety net for missed events

Binance docs:
  POST /fapi/v1/listenKey  → get key
  PUT  /fapi/v1/listenKey  → keep-alive (every <30min)
  wss://fstream.binance.com/ws/{listenKey}
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import socket
import threading
import time
import urllib.parse
from typing import Optional

import requests
import websockets
from loguru import logger

from config.database import db
from config.settings import settings


_FUTURES_REST  = "https://fapi.binance.com/fapi/v1/listenKey"
_FUTURES_WS    = "wss://fstream.binance.com/ws"
_RENEW_EVERY_S = 29 * 60   # 29분 (Binance TTL = 60분, 여유있게)
_RECONNECT_BASE = 5.0
_RECONNECT_MAX  = 60.0


class WsUserStream:
    """Binance Futures User Data Stream listener."""

    def __init__(self) -> None:
        self._running  = False
        self._thread: Optional[threading.Thread] = None
        self._loop:   Optional[asyncio.AbstractEventLoop] = None
        self._listen_key: Optional[str] = None
        self._disconnect_at: float = 0.0

    # ── Public API ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start background stream. Safe to call multiple times."""
        if self._running:
            return
        if not settings.BINANCE_API_KEY or settings.PAPER_TRADING_MODE:
            logger.info("[UserStream] Skipped (paper mode or no API key)")
            return

        self._running = True
        self._thread  = threading.Thread(
            target=self._run_loop,
            name="ws-user-stream",
            daemon=True,
        )
        self._thread.start()
        logger.info("[UserStream] Started")

    def stop(self) -> None:
        self._running = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

    # ── Internal ────────────────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        backoff = _RECONNECT_BASE
        while self._running:
            try:
                self._loop.run_until_complete(self._stream())
                backoff = _RECONNECT_BASE
            except Exception as exc:
                self._disconnect_at = time.time()
                logger.warning(f"[UserStream] Disconnected ({exc}), reconnecting in {backoff:.0f}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2, _RECONNECT_MAX)

    async def _stream(self) -> None:
        listen_key = self._get_listen_key()
        if not listen_key:
            logger.error("[UserStream] Could not get listenKey — retrying in 60s")
            await asyncio.sleep(60)
            return

        self._listen_key = listen_key
        url = f"{_FUTURES_WS}/{listen_key}"
        last_renewed = time.monotonic()

        async with websockets.connect(
            url,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5,
        ) as ws:
            # OS 레벨 TCP keepalive
            try:
                sock = ws.transport.get_extra_info("socket")
                if sock:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 10)
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 5)
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)
            except Exception as e:
                logger.debug(f"[UserStream] TCP keepalive 설정 실패 (무시): {e}")

            # 재연결 직후 공백 구간 체결 이벤트 복구
            if self._disconnect_at > 0:
                await self._reconcile_missed_fills(self._disconnect_at)
                self._disconnect_at = 0.0

            logger.info("[UserStream] Connected to User Data Stream")
            async for raw in ws:
                if not self._running:
                    break

                # listenKey 갱신
                if time.monotonic() - last_renewed > _RENEW_EVERY_S:
                    self._renew_listen_key(listen_key)
                    last_renewed = time.monotonic()

                self._handle(raw)

    def _handle(self, raw: str) -> None:
        try:
            msg  = json.loads(raw)
            if msg.get("e") != "ORDER_TRADE_UPDATE":
                return

            order  = msg["o"]
            status = order.get("X", "")   # order status
            if status not in ("FILLED", "PARTIALLY_FILLED"):
                return

            order_id  = str(order.get("i", ""))
            avg_price = float(order.get("ap", 0) or 0)
            symbol    = order.get("s", "")

            if not order_id or avg_price <= 0:
                return

            logger.info(
                f"[UserStream] {status} | {symbol} order={order_id} "
                f"avg_price={avg_price}"
            )
            self._update_fill(order_id, avg_price)

        except Exception as exc:
            logger.warning(f"[UserStream] Handle error: {exc}")

    @staticmethod
    def _update_fill(order_id: str, avg_price: float) -> None:
        """DB fill price 즉시 업데이트 — outbox polling 대체."""
        try:
            db.update_trade_execution_fill_price(order_id, avg_price)
            logger.debug(f"[UserStream] Fill recorded: {order_id} @ {avg_price}")
        except Exception as exc:
            logger.warning(f"[UserStream] DB update failed: {exc}")

    # ── 재연결 후 OMS 동기화 ────────────────────────────────────────────────────

    async def _reconcile_missed_fills(self, gap_start: float) -> None:
        """재연결 후 공백 구간 체결 이벤트를 Binance REST로 복구.

        ws_user_stream이 끊긴 동안 FILLED된 주문을 outbox 폴링 주기 전에 즉시 반영.
        공백이 3초 미만이거나 1시간 초과이면 스킵 (의미 없거나 너무 오래됨).
        """
        gap_secs = time.time() - gap_start
        if gap_secs < 3 or gap_secs > 3600:
            return

        try:
            start_ms = int(gap_start * 1000)
            end_ms   = int(time.time() * 1000)
            ts       = str(end_ms)
            params   = f"startTime={start_ms}&endTime={end_ms}&limit=50&timestamp={ts}"
            sig = hmac.new(
                settings.BINANCE_API_SECRET.encode(),
                params.encode(),
                hashlib.sha256,
            ).hexdigest()

            import aiohttp
            url = f"https://fapi.binance.com/fapi/v1/userTrades?{params}&signature={sig}"
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers={"X-MBX-APIKEY": settings.BINANCE_API_KEY},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"[UserStream] Reconcile REST {resp.status}")
                        return
                    trades = await resp.json()

            recovered = 0
            for trade in trades:
                order_id  = str(trade.get("orderId", ""))
                avg_price = float(trade.get("price", 0) or 0)
                if order_id and avg_price > 0:
                    self._update_fill(order_id, avg_price)
                    recovered += 1

            if recovered:
                logger.info(f"[UserStream] Reconcile: {recovered}개 체결 복구 ({gap_secs:.0f}s 공백)")
        except Exception as e:
            logger.warning(f"[UserStream] Reconcile 실패 (outbox fallback 활성): {e}")

    # ── listenKey 관리 ──────────────────────────────────────────────────────────

    def _get_listen_key(self) -> Optional[str]:
        try:
            resp = requests.post(
                _FUTURES_REST,
                headers={"X-MBX-APIKEY": settings.BINANCE_API_KEY},
                timeout=10,
            )
            resp.raise_for_status()
            key = resp.json().get("listenKey")
            logger.info(f"[UserStream] listenKey obtained: {key[:8]}...")
            return key
        except Exception as exc:
            logger.error(f"[UserStream] Failed to get listenKey: {exc}")
            return None

    def _renew_listen_key(self, listen_key: str) -> None:
        try:
            requests.put(
                _FUTURES_REST,
                headers={"X-MBX-APIKEY": settings.BINANCE_API_KEY},
                params={"listenKey": listen_key},
                timeout=10,
            )
            logger.debug(f"[UserStream] listenKey renewed: {listen_key[:8]}...")
        except Exception as exc:
            logger.warning(f"[UserStream] listenKey renewal failed: {exc}")


ws_user_stream = WsUserStream()
