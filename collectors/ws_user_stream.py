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
import json
import threading
import time
from typing import Optional

import requests
import websockets
from loguru import logger

from config.database import db
from config.settings import settings


_FUTURES_REST  = "https://fapi.binance.com/fapi/v1/listenKey"
_FUTURES_WS    = "wss://fstream.binance.com/ws"
_RENEW_EVERY_S = 29 * 60   # 29분 (Binance TTL = 60분, 여유있게)


class WsUserStream:
    """Binance Futures User Data Stream listener."""

    def __init__(self) -> None:
        self._running  = False
        self._thread: Optional[threading.Thread] = None
        self._loop:   Optional[asyncio.AbstractEventLoop] = None
        self._listen_key: Optional[str] = None

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
        while self._running:
            try:
                self._loop.run_until_complete(self._stream())
            except Exception as exc:
                logger.warning(f"[UserStream] Disconnected ({exc}), reconnecting in 5s...")
                time.sleep(5)

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
