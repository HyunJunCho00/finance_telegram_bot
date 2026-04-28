from __future__ import annotations

import json
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor
from config.settings import settings
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

FEE_PCT = 0.0005  # 0.05% Binance Futures Market Fee

_pool = None

def _get_pool():
    global _pool
    if _pool is None:
        _pool = ThreadedConnectionPool(1, 20, dsn=settings.EXECUTION_DB_URL)
    return _pool


class DuplicateActiveIntentError(RuntimeError):
    def __init__(self, symbol: str, exchange: str, existing_intent_id: str):
        self.symbol = symbol
        self.exchange = exchange
        self.existing_intent_id = existing_intent_id
        super().__init__(
            f"Duplicate active intent for {symbol} [{exchange}] (existing={existing_intent_id[:8]})"
        )


class ExecutionRepository:
    @contextmanager
    def transaction(self):
        pool = _get_pool()
        conn = pool.getconn()
        try:
            with conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    yield cur
        finally:
            pool.putconn(conn)

    def create_intent(self, **intent_spec) -> str:
        return self.create_intents([intent_spec])[0]

    def create_intents(self, intent_specs: Iterable[Dict]) -> List[str]:
        specs = [dict(spec) for spec in intent_specs]
        if not specs:
            return []

        created_ids: List[str] = []
        with self.transaction() as cur:
            for spec in specs:
                created_ids.append(self._insert_intent_locked(cur, spec))
        return created_ids

    def open_paper_position(
        self,
        *,
        exchange: str,
        symbol: str,
        direction: str,
        amount_usd: float,
        leverage: float,
        style: str,
        raw_price: float,
        tp_price: float = 0.0,
        sl_price: float = 0.0,
        tp2_price: float = 0.0,
        tp1_exit_pct: float = 50.0,
        execution_record_base: Optional[Dict] = None,
        outbox_events: Optional[Iterable[Dict]] = None,
    ) -> Dict:
        with self.transaction() as cur:
            result = self._open_paper_position_locked(
                conn,
                exchange=exchange,
                symbol=symbol,
                direction=direction,
                amount_usd=amount_usd,
                leverage=leverage,
                style=style,
                raw_price=raw_price,
                tp_price=tp_price,
                sl_price=sl_price,
                tp2_price=tp2_price,
                tp1_exit_pct=tp1_exit_pct,
            )
            if result.get("success"):
                if execution_record_base:
                    payload = self._build_trade_execution_payload(
                        base=execution_record_base,
                        result=result,
                        exchange=exchange,
                        symbol=symbol,
                        side=direction,
                        leverage=leverage,
                        notional=amount_usd,
                    )
                    event_id, _ = self._enqueue_outbox_event_locked(
                        conn,
                        "trade_execution_record",
                        payload,
                        idempotency_key=f"trade_execution:{result['order_id']}",
                    )
                    result["execution_outbox_enqueued"] = True
                    result["execution_outbox_event_id"] = event_id
                self._enqueue_outbox_events_locked(cur, outbox_events)
            return result

    def execute_paper_fill(
        self,
        *,
        intent_id: str,
        exchange: str,
        symbol: str,
        direction: str,
        amount_usd: float,
        leverage: float,
        style: str,
        raw_price: float,
        tp_price: float = 0.0,
        sl_price: float = 0.0,
        tp2_price: float = 0.0,
        tp1_exit_pct: float = 50.0,
        execution_record_base: Optional[Dict] = None,
        outbox_events: Optional[Iterable[Dict]] = None,
    ) -> Dict:
        with self.transaction() as cur:
            order = cur.execute(
                """
                SELECT remaining_amount, status
                FROM active_orders
                WHERE intent_id = %s
                """,
                (intent_id,),
            ).fetchone()
            if not order or order["status"] not in ("PENDING", "ACTIVE"):
                return {"success": False, "error": f"Intent not executable: {intent_id}"}

            remaining_amount = float(order["remaining_amount"] or 0.0)
            amount_to_fill = min(float(amount_usd or 0.0), remaining_amount)
            if amount_to_fill <= 0:
                return {"success": False, "error": f"Intent already fully filled: {intent_id}"}

            result = self._open_paper_position_locked(
                conn,
                exchange=exchange,
                symbol=symbol,
                direction=direction,
                amount_usd=amount_to_fill,
                leverage=leverage,
                style=style,
                raw_price=raw_price,
                tp_price=tp_price,
                sl_price=sl_price,
                tp2_price=tp2_price,
                tp1_exit_pct=tp1_exit_pct,
            )
            if not result.get("success"):
                return result

            intent_state = self._update_intent_fill_locked(cur, intent_id, amount_to_fill)
            result["intent_id"] = intent_id
            result["filled_notional_usd"] = amount_to_fill
            result["intent_status"] = intent_state["status"]
            result["intent_remaining_amount"] = intent_state["remaining_amount"]
            if execution_record_base:
                payload = self._build_trade_execution_payload(
                    base=execution_record_base,
                    result=result,
                    exchange=exchange,
                    symbol=symbol,
                    side=direction,
                    leverage=leverage,
                    notional=amount_to_fill,
                )
                event_id, _ = self._enqueue_outbox_event_locked(
                    conn,
                    "trade_execution_record",
                    payload,
                    idempotency_key=f"trade_execution:{result['order_id']}",
                )
                result["execution_outbox_enqueued"] = True
                result["execution_outbox_event_id"] = event_id
            self._enqueue_outbox_events_locked(cur, outbox_events)
            return result

    def apply_tp1(
        self,
        position_id: str,
        current_price: float,
        *,
        outbox_events: Optional[Iterable[Dict]] = None,
    ) -> Dict:
        with self.transaction() as cur:
            pos = self._get_open_position_locked(cur, position_id)
            if not pos:
                return {"success": False, "error": "Position not found"}

            tp = float(pos["tp_price"] or 0.0)
            tp2 = float(pos["tp2_price"] or 0.0)
            tp1_done = int(pos["tp1_done"] or 0)
            tp1_exit_pct = float(pos["tp1_exit_pct"] or 50.0)
            if tp1_done or tp <= 0:
                return {"success": False, "error": "TP1 is not available"}

            fraction = max(0.0, min(tp1_exit_pct / 100.0, 0.99))
            close_result = self._close_position_fraction_locked(cur, pos, current_price, fraction)
            self._apply_tp1_followup_locked(
                conn,
                position_id=position_id,
                entry_price=float(pos["entry_price"]),
                next_tp=(tp2 if tp2 > 0 else 0.0),
            )

            close_result["success"] = True
            close_result["tp1_exit_pct"] = tp1_exit_pct
            close_result["next_tp_price"] = (tp2 if tp2 > 0 else 0.0)
            close_result["new_sl_price"] = float(pos["entry_price"])
            self._enqueue_outbox_events_locked(cur, outbox_events)
            return close_result

    def close_position_fraction(
        self,
        position_id: str,
        current_price: float,
        fraction: float,
        *,
        outbox_events: Optional[Iterable[Dict]] = None,
    ) -> Dict:
        with self.transaction() as cur:
            pos = self._get_open_position_locked(cur, position_id)
            if not pos:
                return {"success": False, "error": "Position not found"}

            result = self._close_position_fraction_locked(cur, pos, current_price, fraction)
            result["success"] = True
            self._enqueue_outbox_events_locked(cur, outbox_events)
            return result

    def liquidate_position(
        self,
        position_id: str,
        *,
        outbox_events: Optional[Iterable[Dict]] = None,
    ) -> Dict:
        with self.transaction() as cur:
            pos = self._get_open_position_locked(cur, position_id)
            if not pos:
                return {"success": False, "error": "Position not found"}
            now = datetime.now(timezone.utc).isoformat()
            cur.execute(
                "UPDATE paper_positions SET is_open = 0, updated_at = %s WHERE position_id = %s",
                (now, position_id),
            )
            self._enqueue_outbox_events_locked(cur, outbox_events)
            return {"success": True, "position_id": position_id}

    def update_position_sl(
        self,
        position_id: str,
        sl_price: float,
        *,
        outbox_events: Optional[Iterable[Dict]] = None,
    ) -> Dict:
        with self.transaction() as cur:
            pos = self._get_open_position_locked(cur, position_id)
            if not pos:
                return {"success": False, "error": "Position not found"}
            now = datetime.now(timezone.utc).isoformat()
            cur.execute(
                "UPDATE paper_positions SET sl_price = %s, updated_at = %s WHERE position_id = %s",
                (sl_price, now, position_id),
            )
            self._enqueue_outbox_events_locked(cur, outbox_events)
            return {"success": True, "new_sl": sl_price}

    def adjust_wallet_balance(self, exchange: str, delta: float) -> Dict:
        with self.transaction() as cur:
            now = datetime.now(timezone.utc).isoformat()
            if delta >= 0:
                self._credit_wallet_locked(cur, exchange, float(delta), now)
            else:
                self._debit_wallet_locked(cur, exchange, abs(float(delta)), now)
            return {"success": True, "exchange": exchange.lower(), "delta": float(delta)}

    def apply_funding_fees(self, symbol_funding_rates: dict, current_prices: dict) -> List[Dict]:
        applied: List[Dict] = []
        with self.transaction() as cur:
            rows = cur.execute("SELECT * FROM paper_positions WHERE is_open = 1").fetchall()
            now = datetime.now(timezone.utc).isoformat()
            for pos in rows:
                symbol = str(pos["symbol"])
                rate = float(symbol_funding_rates.get(symbol, 0.0) or 0.0)
                if rate == 0.0:
                    continue

                price = current_prices.get(symbol)
                if not price:
                    continue

                size = float(pos["size"])
                side = str(pos["side"])
                exchange = str(pos["exchange"])
                notional = size * float(price)
                fee = notional * rate if side == "LONG" else notional * -rate
                self._debit_wallet_locked(cur, exchange, fee, now)
                applied.append(
                    {
                        "exchange": exchange,
                        "symbol": symbol,
                        "side": side,
                        "rate": rate,
                        "fee": fee,
                    }
                )
        return applied

    def set_system_config(self, key: str, value: str) -> Dict:
        with self.transaction() as cur:
            cur.execute(
                "INSERT INTO system_config (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                (str(key), str(value)),
            )
            return {"success": True, "key": str(key), "value": str(value)}

    def enqueue_outbox_event(self, event_type: str, payload: Dict, *, idempotency_key: Optional[str] = None) -> Dict:
        with self.transaction() as cur:
            event_id, deduped = self._enqueue_outbox_event_locked(
                conn,
                event_type,
                payload,
                idempotency_key=idempotency_key,
            )
            return {"success": True, "event_id": event_id, "deduped": deduped}

    def register_live_trade_preflight(self, payload: Dict, idempotency_key: str) -> str:
        """[P1 - ACID] Pre-register a live trade attempt BEFORE calling the exchange API.

        Inserts directly with status='PROCESSING' + processing_started_at=now so the
        normal publish_pending() loop does NOT pick it up as a regular PENDING event.

        Lifecycle:
          - Trade succeeds  → caller calls mark_outbox_event_published(event_id)  → SENT
          - Trade fails     → caller calls mark_outbox_event_failed(event_id, err)  → PENDING (alerts on next dispatch)
          - Process crashes → record stays in PROCESSING; after stale_after_seconds
                              claim_pending_outbox_events re-claims it and handler logs orphan alert
        """
        with self.transaction() as cur:
            existing = cur.execute(
                "SELECT event_id FROM execution_outbox WHERE idempotency_key = %s LIMIT 1",
                (idempotency_key,),
            ).fetchone()
            if existing:
                return str(existing["event_id"])

            event_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc).isoformat()
            cur.execute(
                """
                INSERT INTO execution_outbox
                (event_id, event_type, idempotency_key, payload_json,
                 status, attempts, last_error, processing_started_at,
                 created_at, updated_at, published_at)
                VALUES (%s, 'live_trade_preflight', %s, %s, 'PROCESSING', 1,
                        NULL, %s, %s, %s, NULL)
                """,
                (
                    event_id,
                    idempotency_key,
                    json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str),
                    now,  # processing_started_at — prevents premature dispatch
                    now,  # created_at
                    now,  # updated_at
                ),
            )
            return event_id

    def claim_pending_outbox_events(self, limit: int = 50, stale_after_seconds: int = 300) -> List[Dict]:
        with self.transaction() as cur:
            now = datetime.now(timezone.utc)
            stale_before = datetime.fromtimestamp(
                now.timestamp() - max(int(stale_after_seconds), 0),
                tz=timezone.utc,
            ).isoformat()
            rows = cur.execute(
                """
                SELECT event_id, event_type, idempotency_key, payload_json, attempts, last_error, created_at
                FROM execution_outbox
                WHERE status = 'PENDING'
                   OR (status = 'PROCESSING' AND processing_started_at IS NOT NULL AND processing_started_at <= %s)
                ORDER BY created_at ASC
                LIMIT %s
                """,
                (stale_before, int(limit)),
            ).fetchall()
            claimed = [dict(row) for row in rows]
            if not claimed:
                return []

            claim_time = now.isoformat()
            for row in claimed:
                cur.execute(
                    """
                    UPDATE execution_outbox
                    SET status = 'PROCESSING',
                        attempts = attempts + 1,
                        processing_started_at = %s,
                        updated_at = %s
                    WHERE event_id = %s
                    """,
                    (claim_time, claim_time, row["event_id"]),
                )
            return claimed

    def mark_outbox_event_published(self, event_id: str) -> Dict:
        with self.transaction() as cur:
            now = datetime.now(timezone.utc).isoformat()
            cur.execute(
                """
                UPDATE execution_outbox
                SET status = 'SENT',
                    published_at = %s,
                    updated_at = %s,
                    last_error = NULL,
                    processing_started_at = NULL
                WHERE event_id = %s
                """,
                (now, now, event_id),
            )
            return {"success": True, "event_id": event_id}

    def mark_outbox_event_failed(self, event_id: str, error: str) -> Dict:
        with self.transaction() as cur:
            now = datetime.now(timezone.utc).isoformat()
            cur.execute(
                """
                UPDATE execution_outbox
                SET status = 'PENDING',
                    last_error = %s,
                    updated_at = %s,
                    processing_started_at = NULL
                WHERE event_id = %s
                """,
                (str(error), now, event_id),
            )
            return {"success": True, "event_id": event_id}

    def get_oldest_pending_outbox_age_seconds(self) -> float:
        """Return age in seconds of the oldest PENDING outbox event, 0.0 if none."""
        with _WRITE_LOCK:
            try:
                conn = _get_connection()
                row = cur.execute(
                    "SELECT created_at FROM execution_outbox WHERE status = 'PENDING' "
                    "ORDER BY created_at ASC LIMIT 1"
                ).fetchone()
                if not row:
                    return 0.0
                created = datetime.fromisoformat(str(row["created_at"]).replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                return max(0.0, (now - created).total_seconds())
            except Exception:
                return 0.0

    def update_intent_status(self, intent_id: str, new_status: str) -> Dict:
        with self.transaction() as cur:
            cur.execute(
                "UPDATE active_orders SET status = %s WHERE intent_id = %s",
                (new_status, intent_id),
            )
            return {"success": True, "intent_id": intent_id, "status": new_status}

    def update_order_fill(self, intent_id: str, filled_chunk: float) -> Dict:
        with self.transaction() as cur:
            state = self._update_intent_fill_locked(cur, intent_id, filled_chunk)
            state["success"] = True
            state["intent_id"] = intent_id
            return state

    def flush_expired_intents(self, now_iso: Optional[str] = None) -> Dict:
        cutoff = now_iso or datetime.now(timezone.utc).isoformat()
        with self.transaction() as cur:
            cursor = cur.execute("DELETE FROM active_orders WHERE expires_at < %s", (cutoff,))
            return {"success": True, "deleted": int(cursor.rowcount or 0), "cutoff": cutoff}

    def _insert_intent_locked(self, cur, spec: Dict) -> str:
        symbol = str(spec["symbol"])
        exchange = str(spec["exchange"])
        existing = cur.execute(
            """
            SELECT intent_id
            FROM active_orders
            WHERE symbol = %s
              AND exchange = %s
              AND status IN ('PENDING', 'ACTIVE')
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (symbol, exchange),
        ).fetchone()
        if existing:
            raise DuplicateActiveIntentError(symbol, exchange, existing["intent_id"])

        intent_id = str(spec.get("intent_id") or uuid.uuid4())
        now = datetime.now(timezone.utc)
        ttl_hours = int(spec.get("ttl_hours", 24) or 24)
        expires = datetime.fromtimestamp(now.timestamp() + (ttl_hours * 3600), tz=timezone.utc)
        cur.execute(
            """
            INSERT INTO active_orders
            (intent_id, symbol, direction, execution_style,
             total_target_amount, remaining_amount, exchange,
             created_at, expires_at, leverage, tp_price, sl_price, tp2_price, tp1_exit_pct,
             playbook_id, source_decision, strategy_version, trigger_reason, thesis_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                intent_id,
                symbol,
                str(spec["direction"]),
                str(spec["style"]),
                float(spec["amount"]),
                float(spec["amount"]),
                exchange,
                now.isoformat(),
                expires.isoformat(),
                float(spec.get("leverage", 1.0) or 1.0),
                float(spec.get("tp_price", 0.0) or 0.0),
                float(spec.get("sl_price", 0.0) or 0.0),
                float(spec.get("tp2_price", 0.0) or 0.0),
                float(spec.get("tp1_exit_pct", 50.0) or 50.0),
                str(spec.get("playbook_id", "") or ""),
                str(spec.get("source_decision", "") or ""),
                str(spec.get("strategy_version", "") or ""),
                str(spec.get("trigger_reason", "") or ""),
                str(spec.get("thesis_id", "") or ""),
            ),
        )
        return intent_id

    def _open_paper_position_locked(
        self,
        cur,
        *,
        exchange: str,
        symbol: str,
        direction: str,
        amount_usd: float,
        leverage: float,
        style: str,
        raw_price: float,
        tp_price: float,
        sl_price: float,
        tp2_price: float,
        tp1_exit_pct: float,
    ) -> Dict:
        slippage_map = {
            "MOMENTUM_SNIPER": 0.002,
            "CASINO_EXIT": 0.005,
            "PASSIVE_MAKER": -0.0002,
            "SMART_DCA": 0.0005,
        }
        slippage_pct = slippage_map.get(style, 0.001)
        penalty = (raw_price * slippage_pct) if direction == "LONG" else -(raw_price * slippage_pct)
        filled_price = raw_price + penalty

        wallet = self._get_wallet_balance_locked(cur, exchange)
        required_margin = float(amount_usd) / max(float(leverage or 1.0), 1.0)
        entry_fee = float(amount_usd) * FEE_PCT
        if (required_margin + entry_fee) > wallet:
            return {
                "success": False,
                "error": (
                    f"Insufficient margin (incl. fee). Need ${required_margin + entry_fee:.2f}, "
                    f"have ${wallet:.2f}"
                ),
            }

        coin_size = float(amount_usd) / filled_price
        pos_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        self._insert_position_locked(
            conn,
            position_id=pos_id,
            exchange=exchange,
            symbol=symbol,
            direction=direction,
            coin_size=coin_size,
            filled_price=filled_price,
            leverage=leverage,
            now=now,
            tp_price=tp_price,
            sl_price=sl_price,
            tp2_price=tp2_price,
            tp1_exit_pct=tp1_exit_pct,
        )
        self._debit_wallet_locked(cur, exchange, required_margin + entry_fee, now)
        return {
            "success": True,
            "order_id": pos_id,
            "filled_price": filled_price,
            "size_coin": coin_size,
            "margin_used": required_margin,
            "slippage_applied_pct": slippage_pct * 100,
            "entry_fee": entry_fee,
        }

    def _insert_position_locked(
        self,
        cur,
        *,
        position_id: str,
        exchange: str,
        symbol: str,
        direction: str,
        coin_size: float,
        filled_price: float,
        leverage: float,
        now: str,
        tp_price: float,
        sl_price: float,
        tp2_price: float,
        tp1_exit_pct: float,
    ) -> None:
        cur.execute(
            """
            INSERT INTO paper_positions
            (position_id, exchange, symbol, side, size, entry_price,
             leverage, is_open, created_at, updated_at, tp_price, sl_price, tp2_price, tp1_done, tp1_exit_pct)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 1, %s, %s, %s, %s, %s, 0, %s)
            """,
            (
                position_id,
                exchange.lower(),
                symbol,
                direction,
                coin_size,
                filled_price,
                leverage,
                now,
                now,
                tp_price,
                sl_price,
                tp2_price,
                tp1_exit_pct,
            ),
        )

    def _debit_wallet_locked(self, cur, exchange: str, delta: float, now: str) -> None:
        cur.execute(
            "UPDATE paper_wallets SET balance = balance - %s, updated_at = %s WHERE exchange = %s",
            (delta, now, exchange.lower()),
        )

    def _credit_wallet_locked(self, cur, exchange: str, delta: float, now: str) -> None:
        cur.execute(
            "UPDATE paper_wallets SET balance = balance + %s, updated_at = %s WHERE exchange = %s",
            (delta, now, exchange.lower()),
        )

    def _get_wallet_balance_locked(self, cur, exchange: str) -> float:
        row = cur.execute(
            "SELECT balance FROM paper_wallets WHERE exchange = %s",
            (exchange.lower(),),
        ).fetchone()
        return float(row["balance"]) if row else 0.0

    def _update_intent_fill_locked(self, cur, intent_id: str, filled_chunk: float) -> Dict:
        cur.execute(
            """
            UPDATE active_orders
            SET filled_amount = filled_amount + %s,
                remaining_amount = MAX(remaining_amount - %s, 0),
                status = CASE
                    WHEN remaining_amount - %s <= 0.0001 THEN 'COMPLETED'
                    ELSE 'ACTIVE'
                END
            WHERE intent_id = %s
              AND status IN ('PENDING', 'ACTIVE')
            """,
            (filled_chunk, filled_chunk, filled_chunk, intent_id),
        )
        row = cur.execute(
            "SELECT remaining_amount, status FROM active_orders WHERE intent_id = %s",
            (intent_id,),
        ).fetchone()
        if not row:
            raise RuntimeError(f"Intent disappeared during fill: {intent_id}")
        return {
            "remaining_amount": float(row["remaining_amount"] or 0.0),
            "status": str(row["status"]),
        }

    def _get_open_position_locked(self, cur, position_id: str):
        return cur.execute(
            "SELECT * FROM paper_positions WHERE is_open = 1 AND position_id = %s",
            (position_id,),
        ).fetchone()

    def _close_position_fraction_locked(
        self,
        cur,
        pos: sqlite3.Row,
        current_price: float,
        fraction: float,
    ) -> Dict:
        close_fraction = max(0.0, min(float(fraction), 1.0))
        if close_fraction <= 0:
            raise ValueError("Invalid fraction")

        size = float(pos["size"])
        close_size = size * close_fraction
        if close_size <= 0:
            raise ValueError("Invalid fraction")

        side = str(pos["side"])
        entry = float(pos["entry_price"])
        leverage = float(pos["leverage"] or 1.0)
        exchange = str(pos["exchange"])
        pnl = (current_price - entry) * close_size if side == "LONG" else (entry - current_price) * close_size
        exit_notional = close_size * current_price
        exit_fee = exit_notional * FEE_PCT
        initial_margin = (close_size * entry) / max(leverage, 1.0)
        returned = max(initial_margin + pnl - exit_fee, 0.0)
        now = datetime.now(timezone.utc).isoformat()

        if close_fraction >= 0.99:
            cur.execute(
                "UPDATE paper_positions SET is_open = 0, updated_at = %s WHERE position_id = %s",
                (now, pos["position_id"]),
            )
        else:
            cur.execute(
                "UPDATE paper_positions SET size = size - %s, updated_at = %s WHERE position_id = %s",
                (close_size, now, pos["position_id"]),
            )

        self._credit_wallet_locked(cur, exchange, returned, now)
        return {
            "position_id": str(pos["position_id"]),
            "exchange": exchange,
            "symbol": str(pos["symbol"]),
            "side": side,
            "pnl": pnl,
            "returned": returned,
            "exit_fee": exit_fee,
            "initial_margin": initial_margin,
            "close_fraction": close_fraction,
        }

    def _apply_tp1_followup_locked(
        self,
        cur,
        *,
        position_id: str,
        entry_price: float,
        next_tp: float,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        cur.execute(
            """
            UPDATE paper_positions
            SET tp1_done = 1,
                tp_price = %s,
                sl_price = %s,
                updated_at = %s
            WHERE position_id = %s
            """,
            (next_tp, entry_price, now, position_id),
        )

    def _enqueue_outbox_events_locked(
        self,
        cur,
        outbox_events: Optional[Iterable[Dict]],
    ) -> List[str]:
        event_ids: List[str] = []
        if not outbox_events:
            return event_ids
        for spec in outbox_events:
            event_type = str(spec["event_type"])
            payload = dict(spec.get("payload") or {})
            idempotency_key = spec.get("idempotency_key")
            event_id, _ = self._enqueue_outbox_event_locked(cur, event_type, payload, idempotency_key=idempotency_key)
            event_ids.append(event_id)
        return event_ids

    def _enqueue_outbox_event_locked(
        self,
        cur,
        event_type: str,
        payload: Dict,
        *,
        idempotency_key: Optional[str] = None,
    ) -> tuple[str, bool]:
        if idempotency_key:
            existing = cur.execute(
                """
                SELECT event_id
                FROM execution_outbox
                WHERE idempotency_key = %s
                LIMIT 1
                """,
                (str(idempotency_key),),
            ).fetchone()
            if existing:
                return str(existing["event_id"]), True

        event_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        cur.execute(
            """
            INSERT INTO execution_outbox
            (event_id, event_type, idempotency_key, payload_json, status, attempts, last_error, processing_started_at, created_at, updated_at, published_at)
            VALUES (%s, %s, %s, %s, 'PENDING', 0, NULL, NULL, %s, %s, NULL)
            """,
            (
                event_id,
                str(event_type),
                str(idempotency_key) if idempotency_key else None,
                json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str),
                now,
                now,
            ),
        )
        return event_id, False

    def _build_trade_execution_payload(
        self,
        *,
        base: Dict,
        result: Dict,
        exchange: str,
        symbol: str,
        side: str,
        leverage: float,
        notional: float,
    ) -> Dict:
        payload = dict(base or {})
        payload.update(
            {
                "success": True,
                "paper": True,
                "exchange": exchange,
                "symbol": symbol,
                "side": side,
                "amount": result["size_coin"],
                "leverage": leverage,
                "order_id": result["order_id"],
                "filled_price": result["filled_price"],
                "notional": notional,
                "timestamp": payload.get("timestamp") or datetime.now(timezone.utc).isoformat(),
            }
        )
        return payload


execution_repository = ExecutionRepository()
