from __future__ import annotations

import asyncio
import json
from typing import Dict, List

from loguru import logger
from telegram import Bot

from config.database import db
from config.settings import settings
from executors.execution_repository import execution_repository


class OutboxDispatcher:
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

        raise ValueError(f"Unsupported outbox event type: {event_type}")

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
        asyncio.run(bot.send_message(chat_id=chat_id, text=text, parse_mode=parse_mode))

    def _send_telegram_payload(self, payload: Dict) -> None:
        from executors.report_generator import report_generator

        message_payload = payload.get("payload")
        chat_id = payload.get("chat_id")
        if not isinstance(message_payload, dict) or not chat_id:
            raise ValueError("telegram_payload missing payload or chat_id")

        bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
        asyncio.run(report_generator._send_payload(bot, str(chat_id), message_payload))


outbox_dispatcher = OutboxDispatcher()
