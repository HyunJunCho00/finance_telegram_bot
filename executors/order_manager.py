from loguru import logger
from config.local_state import state_manager
from executors.trade_executor import trade_executor
from config.settings import settings


class ExecutionDesk:
    """The Stateful Order Manager.
    Processes PENDING and ACTIVE intents from local_state.db.
    """

    def __init__(self):
        self.dca_steps = 3

    def process_intents(self):
        """Called every minute by job_1min_execution."""
        active_orders = state_manager.get_active_orders()

        for order in active_orders:
            intent_id   = order["intent_id"]
            symbol      = order["symbol"]
            direction   = order["direction"]
            style       = order["execution_style"]
            remaining   = order["remaining_amount"]
            exchange    = order["exchange"]
            status      = order["status"]
            # [FIX HIGH-8/9] read leverage/tp/sl stored by add_intent()
            leverage    = order.get("leverage", 1)
            tp_price    = order.get("tp_price", 0.0)
            sl_price    = order.get("sl_price", 0.0)
            tp2_price   = order.get("tp2_price", 0.0)
            tp1_exit_pct = order.get("tp1_exit_pct", 50.0)
            lineage = {
                "playbook_id": order.get("playbook_id", ""),
                "source_decision": order.get("source_decision", ""),
                "strategy_version": order.get("strategy_version", ""),
                "trigger_reason": order.get("trigger_reason", ""),
                "thesis_id": order.get("thesis_id", ""),
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
                        exchange, style, leverage, tp_price, sl_price, tp2_price, tp1_exit_pct, lineage
                    )

                elif style == "SMART_DCA":
                    chunk_size   = order["total_target_amount"] / self.dca_steps
                    actual_chunk = min(chunk_size, remaining)
                    self._execute_chunk(
                        intent_id, symbol, direction, actual_chunk,
                        exchange, style, leverage, tp_price, sl_price, tp2_price, tp1_exit_pct, lineage
                    )

                elif style == "PASSIVE_MAKER":
                    self._execute_chunk(
                        intent_id, symbol, direction, remaining,
                        exchange, style, leverage, tp_price, sl_price, tp2_price, tp1_exit_pct, lineage
                    )

                elif style == "CASINO_EXIT":
                    self._execute_chunk(
                        intent_id, symbol, direction, remaining,
                        exchange, style, leverage, tp_price, sl_price, tp2_price, tp1_exit_pct, lineage
                    )

            except Exception as e:
                logger.error(f"Error processing intent {intent_id}: {e}")

        state_manager.flush_expired()

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
        lineage: dict | None = None,
    ):
        if amount <= 0:
            return

        # [FIX HIGH-8/9] leverage no longer hard-coded to 1 — passes the CRO-approved value
        res = trade_executor.execute(
            symbol=symbol,
            side=direction,
            amount=amount,
            leverage=leverage,
            exchange=exchange,
            style=style,
            tp_price=tp_price,
            sl_price=sl_price,
            tp2_price=tp2_price,
            tp1_exit_pct=tp1_exit_pct,
            lineage=lineage or {},
        )

        if res.get("success"):
            state_manager.update_order_fill(intent_id, amount)
            logger.info(f"ExecutionDesk filled ${amount:.2f} of {intent_id[:8]} via {exchange}")
        else:
            logger.error(f"ExecutionDesk fill failed: {res.get('error')}")


# Global instance
execution_desk = ExecutionDesk()
