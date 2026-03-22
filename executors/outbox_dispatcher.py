from __future__ import annotations

import asyncio
import json
import threading
from typing import Dict, List

import ccxt
from loguru import logger
from telegram import Bot

from config.database import db
from config.settings import settings
from executors.execution_repository import execution_repository


class OutboxDispatcher:
    @staticmethod
    def _run_async(async_fn, *args, **kwargs):
        """Run async Telegram SDK calls from both sync and async caller contexts."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(async_fn(*args, **kwargs))

        result: Dict[str, object] = {}
        error: Dict[str, BaseException] = {}

        def _runner():
            try:
                result["value"] = asyncio.run(async_fn(*args, **kwargs))
            except BaseException as exc:
                error["exc"] = exc

        worker = threading.Thread(target=_runner, name="outbox-async-runner", daemon=True)
        worker.start()
        worker.join()

        if "exc" in error:
            raise error["exc"]
        return result.get("value")

    def publish_pending(self, limit: int = 50, stale_after_seconds: int = 300) -> Dict:
        rows = execution_repository.claim_pending_outbox_events(limit=limit, stale_after_seconds=stale_after_seconds)
        published = 0
        failed = 0
        errors: List[str] = []

        for row in rows:
            event_id = str(row["event_id"])
            try:
                payload = json.loads(row["payload_json"])
                self._publish_event(str(row["event_type"]), payload)
                execution_repository.mark_outbox_event_published(event_id)
                published += 1
            except Exception as e:
                execution_repository.mark_outbox_event_failed(event_id, str(e))
                failed += 1
                errors.append(f"{event_id}: {e}")
                logger.error(f"Outbox publish failed for {event_id}: {e}")

        return {
            "published": published,
            "failed": failed,
            "errors": errors,
            "processed": len(rows),
        }

    def _publish_event(self, event_type: str, payload: Dict) -> None:
        if event_type == "trade_execution_record":
            order_id = str(payload.get("order_id") or "")
            if order_id and db.get_trade_execution_by_order_id(order_id):
                return
            db.insert_trade_execution(payload)
            return

        if event_type == "telegram_message":
            self._send_telegram_message(payload)
            return

        if event_type == "telegram_payload":
            self._send_telegram_payload(payload)
            return

        # [P1 - ACID] A preflight that reaches dispatch in PENDING/PROCESSING state means the
        # process crashed after the exchange API call but before mark_outbox_event_published.
        # The trade MAY have executed on the exchange without a DB record — alert immediately.
        if event_type == "live_trade_preflight":
            logger.critical(
                f"[P1 ORPHAN ALERT] live_trade_preflight reached dispatcher in unresolved state! "
                f"A live trade may have executed without a DB record. "
                f"Manual reconciliation required. "
                f"Details: symbol={payload.get('symbol')} exchange={payload.get('exchange')} "
                f"side={payload.get('side')} amount={payload.get('amount')} "
                f"intent_id={payload.get('intent_id')} preflight_at={payload.get('preflight_at')}"
            )
            # Mark as SENT so it stops being retried — human must reconcile manually.
            return

        # [P2 - Fill Reconciliation] Query exchange for actual fill price of async market orders.
        if event_type == "live_fill_check":
            self._check_and_update_fill(payload)
            return

        raise ValueError(f"Unsupported outbox event type: {event_type}")

    def _check_and_update_fill(self, payload: Dict) -> None:
        """[P2 - Fill Reconciliation] Fetch confirmed fill price from exchange and update DB.

        Market orders on Binance Futures may return price=None immediately; the actual
        average fill price becomes available via fetch_order() after matching completes.
        Raises on failure so the outbox dispatcher retries on next cycle.
        """
        exchange_name = str(payload.get("exchange", "binance"))
        order_id = str(payload.get("order_id", ""))
        symbol = str(payload.get("symbol", ""))

        if not order_id or not symbol:
            raise ValueError(f"live_fill_check missing required fields: {payload}")

        # Normalize symbol for ccxt (BTCUSDT → BTC/USDT)
        ccxt_symbol = symbol
        if "USDT" in symbol and "/" not in symbol:
            ccxt_symbol = symbol.replace("USDT", "/USDT")
        elif "KRW" in symbol and "/" not in symbol:
            ccxt_symbol = symbol.replace("KRW", "/KRW")

        ex = self._build_ccxt_exchange(exchange_name)
        order_detail = ex.fetch_order(order_id, ccxt_symbol)
        fill_price = order_detail.get("average") or order_detail.get("price")

        if not fill_price:
            raise RuntimeError(
                f"[P2] Order {order_id} on {exchange_name} still shows no fill price. "
                f"Status: {order_detail.get('status')}. Will retry."
            )

        db.update_trade_execution_fill_price(order_id, float(fill_price))
        logger.info(f"[P2] Fill reconciled: order_id={order_id} fill_price={fill_price} exchange={exchange_name}")

    @staticmethod
    def _build_ccxt_exchange(exchange_name: str):
        """Build a one-off ccxt instance for fill_check. Avoids circular import with trade_executor."""
        if exchange_name in ("binance", "binance_spot"):
            return ccxt.binance({
                "apiKey": settings.BINANCE_API_KEY,
                "secret": settings.BINANCE_API_SECRET,
                "enableRateLimit": True,
                "options": {"defaultType": "future" if exchange_name == "binance" else "spot"},
            })
        if exchange_name == "upbit":
            return ccxt.upbit({
                "apiKey": settings.UPBIT_ACCESS_KEY,
                "secret": settings.UPBIT_SECRET_KEY,
                "enableRateLimit": True,
            })
        if exchange_name == "coinbase":
            return ccxt.coinbase({
                "apiKey": settings.COINBASE_API_KEY,
                "secret": settings.COINBASE_API_SECRET,
                "enableRateLimit": True,
            })
        raise ValueError(f"[P2] Unknown exchange for fill_check: {exchange_name}")

    def _send_telegram_message(self, payload: Dict) -> None:
        from config.local_state import state_manager

        text = str(payload.get("text") or "")
        if not text:
            raise ValueError("telegram_message missing text")

        chat_id = (
            payload.get("chat_id")
            or state_manager.get_telegram_chat_id(settings.TELEGRAM_CHAT_ID)
            or settings.TELEGRAM_CHAT_ID
        )
        parse_mode = payload.get("parse_mode")
        bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
        try:
            self._run_async(bot.send_message, chat_id=chat_id, text=text, parse_mode=parse_mode)
        except Exception as e:
            # Telegram HTML parser rejects chars like <= in AI-generated text.
            # Retry with a fresh Bot instance (asyncio.run closes the event loop after
            # each call, so reusing the same Bot object causes 'Event loop is closed').
            if parse_mode and ("parse" in str(e).lower() or "entities" in str(e).lower() or "tag" in str(e).lower()):
                logger.warning(f"Telegram HTML parse error, retrying as plain text: {e}")
                retry_bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
                self._run_async(retry_bot.send_message, chat_id=chat_id, text=text, parse_mode=None)
            else:
                raise

    def _send_telegram_payload(self, payload: Dict) -> None:
        from executors.report_generator import report_generator

        message_payload = payload.get("payload")
        chat_id = payload.get("chat_id")
        if not isinstance(message_payload, dict) or not chat_id:
            raise ValueError("telegram_payload missing payload or chat_id")

        bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
        self._run_async(report_generator._send_payload, bot, str(chat_id), message_payload)


outbox_dispatcher = OutboxDispatcher()
