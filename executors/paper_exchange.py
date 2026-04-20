from typing import Dict, List
import sqlite3
import threading
from datetime import datetime, timezone

from loguru import logger

from config.local_state import DB_PATH
from executors.execution_repository import FEE_PCT, execution_repository, _WRITE_LOCK, _get_connection
from executors.outbox_dispatcher import outbox_dispatcher


class PaperExchangeEngine:
    """
    Paper trading state manager.
    DB writes go through execution_repository so position changes and outbox events
    can commit atomically.
    Reads use the shared thread-local connection from execution_repository directly.
    """

    def _fetch_one(self, query: str, params: tuple = ()):
        with _WRITE_LOCK:
            conn = _get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchone()

    def _fetch_all(self, query: str, params: tuple = ()) -> List[Dict]:
        with _WRITE_LOCK:
            conn = _get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def _queue_notification(self, message: str, idempotency_key: str) -> Dict:
        return {
            "event_type": "telegram_message",
            "idempotency_key": idempotency_key,
            "payload": {
                "text": message,
                "parse_mode": "HTML",
            },
        }

    def _flush_outbox(self):
        outbox_dispatcher.publish_pending(limit=20)

    def get_wallet_balance(self, exchange: str) -> float:
        row = self._fetch_one(
            "SELECT balance FROM paper_wallets WHERE exchange = ?",
            (exchange.lower(),),
        )
        return row["balance"] if row else 0.0

    def update_wallet_balance(self, exchange: str, delta: float):
        result = execution_repository.adjust_wallet_balance(exchange, delta)
        self._flush_outbox()
        return result

    def get_open_positions(self) -> List[Dict]:
        return self._fetch_all("SELECT * FROM paper_positions WHERE is_open = 1")

    def get_open_position(self, position_id: str) -> Dict | None:
        row = self._fetch_one(
            "SELECT * FROM paper_positions WHERE is_open = 1 AND position_id = ?",
            (position_id,),
        )
        return dict(row) if row else None

    def simulate_execution(
        self,
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
    ) -> dict:
        result = execution_repository.open_paper_position(
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
            # Fix 1: 포지션 오픈 즉시 WS 모니터에 등록 (60초 DB 갱신 대기 제거)
            try:
                from collectors.ws_price_feed import ws_price_feed
                ws_price_feed.register_order(symbol, {
                    "id":          result["order_id"],
                    "symbol":      symbol,
                    "side":        direction,
                    "entry_price": result["filled_price"],
                    "sl_price":    sl_price,
                    "tp1_price":   tp_price,
                    "created_at":  __import__("datetime").datetime.now(
                                       __import__("datetime").timezone.utc
                                   ).isoformat(),
                })
            except Exception:
                pass  # non-blocking — WS monitor 등록 실패해도 주문은 유효
            logger.info(
                f"V8 Paper | [{exchange}] {direction} {symbol} @ {result['filled_price']:.4f} "
                f"(slip={result['slippage_applied_pct']:.2f}% lev={leverage}x margin=${result['margin_used']:.2f} "
                f"fee=${result['entry_fee']:.2f} tp1={tp_price} tp2={tp2_price} sl={sl_price})"
            )
        return result

    def check_liquidations(self, current_prices: dict):
        for pos in self.get_open_positions():
            symbol = pos["symbol"]
            price = current_prices.get(symbol)
            if not price:
                continue

            entry = float(pos["entry_price"])
            size = float(pos["size"])
            lev = float(pos["leverage"] or 1.0)
            side = str(pos["side"])
            initial_margin = (size * entry) / max(lev, 1.0)
            pnl = (price - entry) * size if side == "LONG" else (entry - price) * size
            if initial_margin <= 0 or (pnl / initial_margin) > -0.8:
                continue

            logger.warning(
                f"LIQUIDATION: [{pos['exchange']}] {side} {symbol} entry={entry:.2f} now={price:.2f}"
            )
            message = (
                "<b>LIQUIDATION</b>\n"
                f"Exchange: {str(pos['exchange']).upper()}\n"
                f"Symbol: {symbol}\n"
                f"Side: {side}\n"
                f"Price: {price:.2f}\n"
                f"Entry: {entry:.2f}\n"
                f"Margin Lost: ${initial_margin:.2f}"
            )
            res = execution_repository.liquidate_position(
                str(pos["position_id"]),
                outbox_events=[self._queue_notification(message, f"telegram:position:{pos['position_id']}:liquidation")],
            )
            if res.get("success"):
                self._flush_outbox()
            else:
                logger.error(f"Liquidation persistence failed for {pos['position_id']}: {res.get('error')}")

    def check_tp_sl(self, current_prices: dict):
        positions = self._fetch_all(
            "SELECT * FROM paper_positions WHERE is_open = 1 AND (tp_price > 0 OR sl_price > 0)"
        )
        for pos in positions:
            symbol = pos["symbol"]
            price = current_prices.get(symbol)
            if not price:
                continue

            side = str(pos["side"])
            tp = float(pos["tp_price"] or 0.0)
            sl = float(pos["sl_price"] or 0.0)
            tp1_done = int(pos["tp1_done"] or 0)
            tp1_exit_pct = float(pos["tp1_exit_pct"] or 50.0)
            entry = float(pos["entry_price"])
            size = float(pos["size"])
            lev = float(pos["leverage"] or 1.0)
            exchange = str(pos["exchange"])

            if not tp1_done and tp > 0:
                tp_hit = (side == "LONG" and price >= tp) or (side == "SHORT" and price <= tp)
                if tp_hit:
                    message = (
                        "<b>TP1 HIT</b>\n"
                        f"Exchange: {exchange.upper()}\n"
                        f"Symbol: {symbol}\n"
                        f"Side: {side}\n"
                        f"TP1 Price: {price:.2f}\n"
                        f"Partial Exit: {tp1_exit_pct:.0f}%\n"
                        f"SL moved to BE: {entry:.2f}"
                    )
                    tp1_res = execution_repository.apply_tp1(
                        str(pos["position_id"]),
                        float(price),
                        outbox_events=[self._queue_notification(message, f"telegram:position:{pos['position_id']}:tp1")],
                    )
                    if tp1_res.get("success"):
                        self._flush_outbox()
                    else:
                        logger.error(f"TP1 apply failed for {pos['position_id']}: {tp1_res.get('error')}")
                    continue

            hit_reason = None
            if side == "LONG":
                if tp > 0 and price >= tp:
                    hit_reason = f"TP HIT @ {price:.2f} (target={tp:.2f})"
                elif sl > 0 and price <= sl:
                    hit_reason = f"SL HIT @ {price:.2f} (stop={sl:.2f})"
            else:
                if tp > 0 and price <= tp:
                    hit_reason = f"TP HIT @ {price:.2f} (target={tp:.2f})"
                elif sl > 0 and price >= sl:
                    hit_reason = f"SL HIT @ {price:.2f} (stop={sl:.2f})"

            if not hit_reason:
                continue

            pnl = (price - entry) * size if side == "LONG" else (entry - price) * size
            exit_notional = size * price
            exit_fee = exit_notional * FEE_PCT
            initial_margin = (size * entry) / max(lev, 1.0)
            returned = max(initial_margin + pnl - exit_fee, 0.0)
            close_type = "TP" if "TP" in hit_reason else "SL"
            logger.info(
                f"[{close_type}] {hit_reason} [{exchange}] {side} {symbol} "
                f"PnL=${pnl:+.2f} fee=${exit_fee:.2f} returning=${returned:.2f}"
            )
            pnl_pct = (pnl / initial_margin) * 100 if initial_margin > 0 else 0
            message = (
                f"<b>POSITION CLOSED ({close_type})</b>\n"
                f"Exchange: {exchange.upper()}\n"
                f"Symbol: {symbol}\n"
                f"Side: {side}\n"
                f"Close Price: {price:.2f}\n"
                f"PnL: <b>${pnl:+.2f} ({pnl_pct:+.2f}%)</b>\n"
                f"Fee: ${exit_fee:.2f}"
            )
            close_res = execution_repository.close_position_fraction(
                str(pos["position_id"]),
                float(price),
                1.0,
                outbox_events=[self._queue_notification(message, f"telegram:position:{pos['position_id']}:close:{close_type.lower()}")],
            )
            if close_res.get("success"):
                self._flush_outbox()
            else:
                logger.error(f"Position close failed for {pos['position_id']}: {close_res.get('error')}")

    def apply_funding_fees(self, symbol_funding_rates: dict, current_prices: dict):
        applied = execution_repository.apply_funding_fees(symbol_funding_rates, current_prices)
        for item in applied:
            logger.info(
                f"Funding Fee Applied: [{item['exchange']}] {item['side']} {item['symbol']} "
                f"Rate: {item['rate']*100:.4f}% Fee: ${item['fee']:+.4f}"
            )

    def close_position_partial(self, pos_id: str, current_price: float, fraction: float):
        close_label = "CLOSED" if fraction >= 0.99 else f"PARTIAL CLOSE ({fraction*100:.0f}%)"
        message_template = (
            f"<b>POSITION {close_label} (MANUAL)</b>\n"
            "Symbol: {symbol}\n"
            "Side: {side}\n"
            f"Close Price: {current_price:.2f}\n"
            "PnL: <b>${pnl:+.2f} ({pnl_pct:+.2f}%)</b>\n"
            "Fee: ${fee:.2f}"
        )
        res = execution_repository.close_position_fraction(
            pos_id,
            current_price,
            fraction,
            outbox_events=[],
        )
        if not res.get("success"):
            return res

        pnl_pct = (res["pnl"] / res["initial_margin"]) * 100 if res["initial_margin"] > 0 else 0
        message = message_template.format(
            symbol=res["symbol"],
            side=res["side"],
            pnl=res["pnl"],
            pnl_pct=pnl_pct,
            fee=res["exit_fee"],
        )
        execution_repository.enqueue_outbox_event(
            "telegram_message",
            {"text": message, "parse_mode": "HTML"},
            idempotency_key=f"telegram:position:{pos_id}:manual:{int(fraction * 1000)}:{int(round(current_price * 100))}",
        )
        self._flush_outbox()

        logger.info(
            f"[MANUAL] {close_label} [{res['exchange']}] {res['side']} {res['symbol']} "
            f"PnL=${res['pnl']:+.2f} fee=${res['exit_fee']:.2f} returning=${res['returned']:.2f}"
        )
        return {"success": True, "pnl": res["pnl"], "returned": res["returned"]}

    def update_sl_to_breakeven(self, pos_id: str):
        pos = self.get_open_position(pos_id)
        if not pos:
            return {"success": False, "error": "Position not found"}

        entry = float(pos["entry_price"])
        message = f"<b>BREAK-EVEN SL SET</b>\nSymbol: {pos['symbol']}\nNew SL: {entry:.2f}"
        res = execution_repository.update_position_sl(
            pos_id,
            entry,
            outbox_events=[self._queue_notification(message, f"telegram:position:{pos_id}:breakeven")],
        )
        if res.get("success"):
            self._flush_outbox()
        return res

    def handle_realtime_price(self, symbol: str, price: float):
        self.check_tp_sl({symbol: price})
        self.check_liquidations({symbol: price})


paper_engine = PaperExchangeEngine()
