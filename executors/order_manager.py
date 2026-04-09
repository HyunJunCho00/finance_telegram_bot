from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Dict, Optional

from loguru import logger

from config.local_state import state_manager
from config.settings import settings
from executors.trade_executor import trade_executor


class ExecutionDesk:
    """Stateful Order Manager.

    매 1분마다 PENDING/ACTIVE intent를 처리한다.

    SMART_DCA 실행 방식:
      - 첫 실행: 1번 진입(40%) 시장가 즉시 + 2·3번 진입 지정가 동시 등록
      - 이후:    만료된 지정가 자동 취소 (TTL = SMART_DCA_LIMIT_TTL_MINUTES)
    """

    def __init__(self) -> None:
        # {exchange_order_id: {symbol, exchange, side, expires_at_ts, intent_id}}
        self._pending_limits: Dict[str, dict] = {}

    # ── 메인 루프 (1분마다 호출) ─────────────────────────────────────────────────

    def process_intents(self) -> None:
        # 만료된 지정가부터 정리
        self._cancel_expired_limits()

        active_orders = state_manager.get_active_orders()

        for order in active_orders:
            intent_id    = order["intent_id"]
            symbol       = order["symbol"]
            direction    = order["direction"]
            style        = order["execution_style"]
            remaining    = order["remaining_amount"]
            total        = order["total_target_amount"]
            exchange     = order["exchange"]
            status       = order["status"]
            leverage     = order.get("leverage", 1)
            tp_price     = order.get("tp_price", 0.0)
            sl_price     = order.get("sl_price", 0.0)
            tp2_price    = order.get("tp2_price", 0.0)
            tp1_exit_pct = order.get("tp1_exit_pct", 50.0)
            lineage = {
                "playbook_id":      order.get("playbook_id", ""),
                "source_decision":  order.get("source_decision", ""),
                "strategy_version": order.get("strategy_version", ""),
                "trigger_reason":   order.get("trigger_reason", ""),
                "thesis_id":        order.get("thesis_id", ""),
            }

            if status == "PENDING":
                state_manager.update_status(intent_id, "ACTIVE")
                status = "ACTIVE"

            if status != "ACTIVE":
                continue

            logger.info(
                f"ExecutionDesk: {style} {symbol} on {exchange} "
                f"remaining=${remaining:.2f} lev={leverage}x"
            )

            try:
                if style == "MOMENTUM_SNIPER":
                    self._execute_chunk(
                        intent_id, symbol, direction, remaining,
                        exchange, style, leverage, tp_price, sl_price,
                        tp2_price, tp1_exit_pct, lineage,
                    )

                elif style == "SMART_DCA":
                    self._process_smart_dca(
                        intent_id, symbol, direction, remaining, total,
                        exchange, leverage, tp_price, sl_price,
                        tp2_price, tp1_exit_pct, lineage,
                    )

                elif style in ("PASSIVE_MAKER", "CASINO_EXIT"):
                    self._execute_chunk(
                        intent_id, symbol, direction, remaining,
                        exchange, style, leverage, tp_price, sl_price,
                        tp2_price, tp1_exit_pct, lineage,
                    )

            except Exception as e:
                logger.error(f"Error processing intent {intent_id}: {e}")

        state_manager.flush_expired()

    # ── SMART_DCA ────────────────────────────────────────────────────────────────

    def _process_smart_dca(
        self,
        intent_id: str,
        symbol: str,
        direction: str,
        remaining: float,
        total: float,
        exchange: str,
        leverage: float,
        tp_price: float,
        sl_price: float,
        tp2_price: float,
        tp1_exit_pct: float,
        lineage: dict,
    ) -> None:
        """SMART_DCA: 1번 시장가 즉시 + 2·3번 지정가 등록.

        첫 실행 감지: remaining >= total * 0.95
        이후 호출:    pending limit 취소 확인만 (이미 위에서 처리됨)
        """
        is_first = remaining >= total * 0.95
        if not is_first:
            # 이미 1번 진입 완료, 지정가 대기 중 — 취소 루프만 실행
            return

        chunk1 = total * 0.40  # 시장가 즉시
        chunk2 = total * 0.35  # 지정가 #2
        chunk3 = total * 0.25  # 지정가 #3

        # 시장가 1번 진입
        self._execute_chunk(
            intent_id, symbol, direction, chunk1,
            exchange, "SMART_DCA", leverage,
            tp_price, sl_price, tp2_price, tp1_exit_pct, lineage,
        )

        # Paper 모드: 나머지도 즉시 시장가로 시뮬레이션 (기존 동작 유지)
        if settings.PAPER_TRADING_MODE:
            for chunk in (chunk2, chunk3):
                self._execute_chunk(
                    intent_id, symbol, direction, chunk,
                    exchange, "SMART_DCA", leverage,
                    tp_price, sl_price, tp2_price, tp1_exit_pct, lineage,
                )
            return

        # Live 모드: 지정가 2·3번 등록
        current_price = trade_executor._get_reference_price(symbol, exchange)
        if current_price <= 0:
            logger.warning(f"[SMART_DCA] Cannot get price for limit orders ({symbol})")
            return

        atr = current_price * 0.02  # 2% fallback (ATR anchor 없을 때)
        ttl_min = int(settings.SMART_DCA_LIMIT_TTL_MINUTES)

        limit_configs = [
            (chunk2, 0.5),   # 현재가 - 0.5 ATR
            (chunk3, 1.0),   # 현재가 - 1.0 ATR
        ]

        for chunk, atr_mult in limit_configs:
            if direction.upper() == "LONG":
                limit_price = round(current_price - atr_mult * atr, 1)
            else:
                limit_price = round(current_price + atr_mult * atr, 1)

            res = trade_executor.execute_limit(
                symbol=symbol,
                side=direction,
                amount=chunk,
                leverage=int(leverage),
                limit_price=limit_price,
                exchange=exchange,
            )

            if res.get("success") and res.get("order_id"):
                self._track_limit(
                    order_id=res["order_id"],
                    symbol=symbol,
                    exchange=exchange,
                    side=direction,
                    intent_id=intent_id,
                    ttl_minutes=ttl_min,
                )

    # ── 지정가 추적 / 취소 ───────────────────────────────────────────────────────

    def _track_limit(
        self,
        order_id: str,
        symbol: str,
        exchange: str,
        side: str,
        intent_id: str,
        ttl_minutes: int,
    ) -> None:
        expires_ts = time.monotonic() + ttl_minutes * 60
        self._pending_limits[order_id] = {
            "symbol":    symbol,
            "exchange":  exchange,
            "side":      side,
            "intent_id": intent_id,
            "expires_ts": expires_ts,
            "placed_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.info(
            f"[LimitTracker] Tracking {order_id} ({symbol} {side}) "
            f"TTL={ttl_minutes}min"
        )

    def _cancel_expired_limits(self) -> None:
        now = time.monotonic()
        expired = [oid for oid, meta in self._pending_limits.items() if meta["expires_ts"] <= now]

        for order_id in expired:
            meta = self._pending_limits.pop(order_id)
            logger.warning(
                f"[LimitTracker] TTL expired for {order_id} "
                f"({meta['symbol']} {meta['side']}) — cancelling"
            )
            trade_executor.cancel_order(
                exchange=meta["exchange"],
                order_id=order_id,
                symbol=meta["symbol"],
            )

    # ── 청크 실행 (시장가) ────────────────────────────────────────────────────────

    def _execute_chunk(
        self,
        intent_id: str,
        symbol: str,
        direction: str,
        amount: float,
        exchange: str,
        style: str,
        leverage: float = 1.0,
        tp_price: float = 0.0,
        sl_price: float = 0.0,
        tp2_price: float = 0.0,
        tp1_exit_pct: float = 50.0,
        lineage: Optional[dict] = None,
    ) -> None:
        if amount <= 0:
            return

        res = trade_executor.execute(
            symbol=symbol,
            side=direction,
            amount=amount,
            leverage=int(leverage),
            exchange=exchange,
            style=style,
            tp_price=tp_price,
            sl_price=sl_price,
            tp2_price=tp2_price,
            tp1_exit_pct=tp1_exit_pct,
            lineage=lineage or {},
            intent_id=intent_id,
        )

        if res.get("success"):
            logger.info(f"ExecutionDesk filled ${amount:.2f} of {intent_id[:8]} via {exchange}")
        else:
            logger.error(f"ExecutionDesk fill failed: {res.get('error') or res.get('note')}")


# Global instance
execution_desk = ExecutionDesk()
