import ccxt
from typing import Dict, Optional
from datetime import datetime, timezone
from config.settings import settings
from config.database import db
from config.local_state import state_manager
from loguru import logger
from executors.execution_repository import DuplicateActiveIntentError, execution_repository
from executors.outbox_dispatcher import outbox_dispatcher


class TradeExecutor:
    def __init__(self):
        self.binance = ccxt.binance({
            'apiKey': settings.BINANCE_API_KEY,
            'secret': settings.BINANCE_API_SECRET,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        self.binance_spot = ccxt.binance({
            'apiKey': settings.BINANCE_API_KEY,
            'secret': settings.BINANCE_API_SECRET,
            'enableRateLimit': True,
        })

        if settings.BINANCE_USE_TESTNET:
            self.binance.set_sandbox_mode(True)

        self.upbit = ccxt.upbit({
            'apiKey': settings.UPBIT_ACCESS_KEY,
            'secret': settings.UPBIT_SECRET_KEY,
            'enableRateLimit': True
        })

        self.coinbase = ccxt.coinbase({
            'apiKey': settings.COINBASE_API_KEY,
            'secret': settings.COINBASE_API_SECRET,
            'enableRateLimit': True
        })

    @staticmethod
    def _build_lineage(final_decision: dict, playbook_context: Optional[dict] = None) -> dict:
        context = dict(playbook_context or final_decision.get("playbook_context", {}) or {})
        return {
            "playbook_id": str(context.get("playbook_id") or ""),
            "source_decision": str(context.get("source_decision") or final_decision.get("decision", "")),
            "strategy_version": str(context.get("strategy_version") or ""),
            "trigger_reason": str(context.get("trigger_reason") or ""),
            "thesis_id": str(context.get("thesis_id") or ""),
        }

    def execute_from_decision(
        self,
        final_decision: dict,
        mode: str,
        symbol: str,
        *,
        playbook_context: Optional[dict] = None,
    ) -> dict:
        """Calculate order sizes and register execution intents safely."""
        try:
            direction = final_decision.get("decision", "HOLD")
            mode_upper = (mode or "SWING").upper()
            lineage = self._build_lineage(final_decision, playbook_context)

            if direction == "CANCEL_AND_CLOSE":
                active = state_manager.get_active_orders()
                cancelled_count = 0
                for order in active:
                    if order["symbol"] == symbol:
                        state_manager.update_status(order["intent_id"], "CANCELLED")
                        cancelled_count += 1
                return {
                    "success": True,
                    "receipts": [{"note": f"CANCEL_AND_CLOSE executed. {cancelled_count} pending intents cancelled."}],
                    "strategy_applied": "CASINO_EXIT",
                    "total_notional": 0,
                    **lineage,
                }

            if direction not in ["LONG", "SHORT"]:
                return {"success": False, "note": "No valid trade direction"}

            allocation_pct = float(final_decision.get("allocation_pct", 0) or 0)
            leverage = float(final_decision.get("leverage", 1) or 1)
            target_exchange = str(final_decision.get("target_exchange", "BINANCE")).lower()
            exec_style = str(final_decision.get("recommended_execution_style", "MOMENTUM_SNIPER"))
            tp_price = float(final_decision.get("tp1_price", final_decision.get("take_profit", 0)) or 0)
            sl_price = float(final_decision.get("stop_loss", 0) or 0)
            tp2_price = float(final_decision.get("take_profit", 0) or 0)
            tp1_exit_pct = float(final_decision.get("tp1_exit_pct", 50.0) or 50.0)

            if allocation_pct <= 0:
                return {"success": False, "note": "Allocation is 0% (Vetoed or No Confidence)"}

            ref_exchange = target_exchange if target_exchange != "split" else "binance"
            price = self._get_reference_price(symbol, exchange=ref_exchange)
            if price <= 0:
                return {"success": False, "error": "Could not fetch price"}

            if mode_upper == "POSITION" and direction == "SHORT":
                return {"success": False, "note": "POSITION mode does not allow SHORT."}

            if settings.PAPER_TRADING_MODE:
                from executors.paper_exchange import paper_engine
                open_pos = paper_engine.get_open_positions()
                open_keys = {(p["symbol"], p["exchange"]) for p in open_pos}
                if target_exchange == "split":
                    exchanges_to_check = ["binance_spot", "upbit"] if mode_upper == "POSITION" else ["binance", "upbit"]
                else:
                    exchanges_to_check = [target_exchange]
                for ex in exchanges_to_check:
                    if (symbol, ex) in open_keys:
                        logger.info(f"Skipping {direction} {symbol} on {ex}: position already open")
                        return {"success": False, "note": f"Position already open for {symbol} on {ex.upper()}"}

            if target_exchange == "split":
                alloc_ratio = allocation_pct / 100.0
                split_lev = 1 if mode_upper == "POSITION" else leverage
                binance_notional = (settings.BINANCE_PAPER_BALANCE_USD * alloc_ratio * split_lev) * 0.5
                upbit_notional = (settings.UPBIT_PAPER_BALANCE_USD * alloc_ratio) * 0.5
                binance_exchange = "binance_spot" if mode_upper == "POSITION" else "binance"

                ok_b, msg_b = self._check_paper_budget(binance_exchange, binance_notional, split_lev)
                if not ok_b:
                    return {"success": False, "note": msg_b}
                ok_u, msg_u = self._check_paper_budget("upbit", upbit_notional, 1)
                if not ok_u:
                    return {"success": False, "note": msg_u}

                try:
                    intent_b, intent_u = execution_repository.create_intents(
                        [
                            {
                                "symbol": symbol,
                                "direction": direction,
                                "style": exec_style,
                                "amount": binance_notional,
                                "exchange": binance_exchange,
                                "leverage": split_lev,
                                "tp_price": tp_price,
                                "sl_price": sl_price,
                                "tp2_price": tp2_price,
                                "tp1_exit_pct": tp1_exit_pct,
                                **lineage,
                            },
                            {
                                "symbol": symbol,
                                "direction": direction,
                                "style": exec_style,
                                "amount": upbit_notional,
                                "exchange": "upbit",
                                "leverage": 1,
                                "tp_price": tp_price,
                                "sl_price": sl_price,
                                "tp2_price": tp2_price,
                                "tp1_exit_pct": tp1_exit_pct,
                                **lineage,
                            },
                        ]
                    )
                except DuplicateActiveIntentError:
                    return {"success": False, "note": f"Duplicate intent blocked for {symbol} (split route)"}

                receipts = [
                    {
                        "order_id": intent_b,
                        "exchange": binance_exchange.upper(),
                        "side": direction,
                        "notional": binance_notional,
                        "paper": settings.PAPER_TRADING_MODE,
                        "note": f"SPLIT Intent: {exec_style}",
                        "tp1_price": tp_price,
                        "tp2_price": tp2_price,
                        **lineage,
                    },
                    {
                        "order_id": intent_u,
                        "exchange": "UPBIT",
                        "side": direction,
                        "notional": upbit_notional,
                        "paper": settings.PAPER_TRADING_MODE,
                        "note": f"SPLIT Intent: {exec_style} (lev=1x)",
                        "tp1_price": tp_price,
                        "tp2_price": tp2_price,
                        **lineage,
                    },
                ]
                total_not_usd = binance_notional
            else:
                if mode_upper == "POSITION" and target_exchange == "binance":
                    target_exchange = "binance_spot"

                lev_for_exchange = 1 if target_exchange in ("upbit", "binance_spot") or mode_upper == "POSITION" else leverage
                if target_exchange in ("binance", "binance_spot"):
                    wallet_balance = settings.BINANCE_PAPER_BALANCE_USD
                else:
                    wallet_balance = settings.UPBIT_PAPER_BALANCE_USD
                target_notional = wallet_balance * (allocation_pct / 100.0) * lev_for_exchange

                ok_budget, msg_budget = self._check_paper_budget(target_exchange, target_notional, lev_for_exchange)
                if not ok_budget:
                    return {"success": False, "note": msg_budget}

                try:
                    intent_id = execution_repository.create_intent(
                        symbol=symbol,
                        direction=direction,
                        style=exec_style,
                        amount=target_notional,
                        exchange=target_exchange,
                        leverage=lev_for_exchange,
                        tp_price=tp_price,
                        sl_price=sl_price,
                        tp2_price=tp2_price,
                        tp1_exit_pct=tp1_exit_pct,
                        **lineage,
                    )
                except DuplicateActiveIntentError:
                    return {"success": False, "note": f"Duplicate intent blocked for {symbol} [{target_exchange}]"}

                receipts = [
                    {
                        "order_id": intent_id,
                        "exchange": target_exchange.upper(),
                        "side": direction,
                        "notional": target_notional,
                        "paper": settings.PAPER_TRADING_MODE,
                        "note": f"Registered Intent: {exec_style}",
                        "tp1_price": tp_price,
                        "tp2_price": tp2_price,
                        **lineage,
                    }
                ]
                total_not_usd = target_notional

            return {
                "success": True,
                "receipts": receipts,
                "strategy_applied": exec_style,
                "total_notional": total_not_usd,
                **lineage,
            }
        except Exception as e:
            logger.error(f"Execution formatting error: {e}")
            return {"success": False, "error": str(e)}

    def _check_paper_budget(self, exchange: str, target_notional: float, leverage: float) -> tuple[bool, str]:
        """Pre-check free paper wallet after reserving pending/active intent margin."""
        if not settings.PAPER_TRADING_MODE:
            return True, ""
        try:
            from executors.paper_exchange import paper_engine
            wallet = float(paper_engine.get_wallet_balance(exchange))
            reserved = float(state_manager.get_reserved_margin(exchange))
            required = float(target_notional) / max(float(leverage or 1.0), 1.0)
            available = wallet - reserved
            if required > max(available, 0.0):
                msg = (
                    f"Insufficient paper budget on {exchange}: "
                    f"required=${required:.2f}, available=${available:.2f} "
                    f"(wallet=${wallet:.2f}, reserved=${reserved:.2f})"
                )
                logger.warning(msg)
                return False, msg
            return True, ""
        except Exception as e:
            logger.warning(f"Paper budget pre-check failed ({exchange}): {e}")
            return True, ""

    def execute(
        self,
        symbol: str,
        side: str,
        amount: float,
        leverage: int = 1,
        exchange: str = 'binance',
        style: str = "SMART_DCA",
        tp_price: float = 0.0,
        sl_price: float = 0.0,
        tp2_price: float = 0.0,
        tp1_exit_pct: float = 50.0,
        lineage: Optional[dict] = None,
        intent_id: Optional[str] = None,
    ) -> Dict:
        try:
            side = side.upper()
            exchange = exchange.lower()
            execution_record_base = {
                "note": f"PAPER/LIVE EXECUTION | Style: {style}",
            }
            if lineage:
                execution_record_base.update({
                    "playbook_id": str(lineage.get("playbook_id") or ""),
                    "source_decision": str(lineage.get("source_decision") or ""),
                    "strategy_version": str(lineage.get("strategy_version") or ""),
                    "trigger_reason": str(lineage.get("trigger_reason") or ""),
                    "thesis_id": str(lineage.get("thesis_id") or ""),
                })

            # Default-safe path: no real order API call
            if settings.PAPER_TRADING_MODE:
                if intent_id:
                    price = self._get_reference_price(symbol, exchange)
                    if not price:
                        return {"success": False, "error": "Could not get reference price"}
                    result = execution_repository.execute_paper_fill(
                        intent_id=intent_id,
                        exchange=exchange,
                        symbol=symbol,
                        direction=side,
                        amount_usd=amount,
                        leverage=leverage,
                        style=style,
                        raw_price=price,
                        tp_price=tp_price,
                        sl_price=sl_price,
                        tp2_price=tp2_price,
                        tp1_exit_pct=tp1_exit_pct,
                        execution_record_base=execution_record_base,
                    )
                else:
                    result = self._simulate_order(symbol, side, amount, leverage, exchange, style, tp_price, sl_price, tp2_price, tp1_exit_pct)
            else:
                if exchange == 'binance':
                    result = self._execute_binance(symbol, side, amount, leverage)
                elif exchange == 'binance_spot':
                    result = self._execute_binance_spot(symbol, side, amount)
                elif exchange == 'upbit':
                    if settings.UPBIT_PAPER_ONLY:
                        result = self._simulate_order(symbol, side, amount, leverage, exchange, style, tp_price, sl_price, tp2_price, tp1_exit_pct)
                    else:
                        result = self._execute_upbit(symbol, side, amount)
                elif exchange == 'coinbase':
                    result = self._execute_coinbase(symbol, side, amount)
                else:
                    return {"error": "Invalid exchange"}

            if lineage:
                result.update({
                    "playbook_id": str(lineage.get("playbook_id") or ""),
                    "source_decision": str(lineage.get("source_decision") or ""),
                    "strategy_version": str(lineage.get("strategy_version") or ""),
                    "trigger_reason": str(lineage.get("trigger_reason") or ""),
                    "thesis_id": str(lineage.get("thesis_id") or ""),
                })

            self._save_execution_record(result)
            outbox_dispatcher.publish_pending(limit=20)
            return result

        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return {"success": False, "error": str(e)}

    def _get_reference_price(self, symbol: str, exchange: str = 'binance') -> float:
        try:
            if settings.PAPER_TRADING_PRICE_SOURCE == "last_report":
                report = db.get_latest_report(symbol=symbol)
                if report and report.get('final_decision'):
                    fd = report['final_decision']
                    if isinstance(fd, str):
                        import json
                        fd = json.loads(fd)
                    p = fd.get('entry_price')
                    if p:
                        return float(p)

            # [FIX SILENT-3] CCXT requires slash format: BTCUSDT ??BTC/USDT
            ccxt_symbol = symbol
            if 'USDT' in symbol and '/' not in symbol:
                ccxt_symbol = symbol.replace('USDT', '/USDT')
            elif 'KRW' in symbol and '/' not in symbol:
                ccxt_symbol = symbol.replace('KRW', '/KRW')

            # default: live ticker reference
            ex = self.binance if exchange == 'binance' else self.binance_spot if exchange == 'binance_spot' else self.upbit if exchange == 'upbit' else self.coinbase
            ticker = ex.fetch_ticker(ccxt_symbol)
            return float(ticker.get('last') or ticker.get('close') or 0)
        except Exception as e:
            logger.warning(f"Reference price fetch failed ({exchange}, {symbol}): {e}")
            return 0.0

    def _simulate_order(self, symbol: str, side: str, amount: float, leverage: int, exchange: str, style: str, tp_price: float = 0.0, sl_price: float = 0.0, tp2_price: float = 0.0, tp1_exit_pct: float = 50.0) -> Dict:
        price = self._get_reference_price(symbol, exchange)
        if not price:
            return {"success": False, "error": "Could not get reference price"}

        sim_res = execution_repository.open_paper_position(
            exchange=exchange,
            symbol=symbol,
            direction=side,
            amount_usd=amount,
            leverage=leverage,
            style=style,
            raw_price=price,
            tp_price=tp_price,
            sl_price=sl_price,
            tp2_price=tp2_price,
            tp1_exit_pct=tp1_exit_pct,
        )
        
        if not sim_res.get('success'):
            return sim_res

        return {
            "success": True,
            "paper": True,
            "exchange": exchange,
            "symbol": symbol,
            "side": side,
            "amount": sim_res['size_coin'],
            "leverage": leverage,
            "order_id": sim_res['order_id'],
            "filled_price": sim_res['filled_price'],
            "notional": amount,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "note": f"V8 PAPER ENGINE | Style: {style} | Slippage: {sim_res.get('slippage_applied_pct', 0):.3f}% | Price: {sim_res.get('filled_price', 0):.2f}",
            "tp1_price": tp_price,
            "tp2_price": tp2_price,
        }

    def _execute_binance(self, symbol: str, side: str, amount: float, leverage: int) -> Dict:
        try:
            self.binance.set_leverage(leverage, symbol)

            order = self.binance.create_order(
                symbol=symbol,
                type='market',
                side=side.lower(),
                amount=amount
            )

            return {
                "success": True,
                "paper": False,
                "exchange": "binance",
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "leverage": leverage,
                "order_id": order.get('id'),
                "filled_price": order.get('price'),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Binance execution error: {e}")
            return {"success": False, "paper": False, "error": str(e)}

    def _execute_upbit(self, symbol: str, side: str, amount: float) -> Dict:
        try:
            order = self.upbit.create_order(
                symbol=symbol,
                type='market',
                side=side.lower(),
                amount=amount
            )

            return {
                "success": True,
                "paper": False,
                "exchange": "upbit",
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "order_id": order.get('id'),
                "filled_price": order.get('price'),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Upbit execution error: {e}")
            return {"success": False, "paper": False, "error": str(e)}

    def _execute_binance_spot(self, symbol: str, side: str, amount: float) -> Dict:
        try:
            if side.upper() == "SHORT":
                return {"success": False, "paper": False, "error": "SHORT is not allowed on Binance spot"}
            order = self.binance_spot.create_order(
                symbol=symbol,
                type='market',
                side=side.lower(),
                amount=amount
            )

            return {
                "success": True,
                "paper": False,
                "exchange": "binance_spot",
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "order_id": order.get('id'),
                "filled_price": order.get('price'),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Binance spot execution error: {e}")
            return {"success": False, "paper": False, "error": str(e)}

    def _execute_coinbase(self, symbol: str, side: str, amount: float) -> Dict:
        try:
            order = self.coinbase.create_order(
                symbol=symbol,
                type='market',
                side=side.lower(),
                amount=amount
            )
            return {
                "success": True,
                "paper": False,
                "exchange": "coinbase",
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "order_id": order.get('id'),
                "filled_price": order.get('price'),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"Coinbase execution error: {e}")
            return {"success": False, "paper": False, "error": str(e)}

    def _save_execution_record(self, execution_result: Dict) -> None:
        try:
            if execution_result.get("execution_outbox_enqueued"):
                return
            order_id = str(execution_result.get("order_id") or "")
            execution_repository.enqueue_outbox_event(
                "trade_execution_record",
                execution_result,
                idempotency_key=(f"trade_execution:{order_id}" if order_id else None),
            )
            logger.info("Trade execution outbox event enqueued")
        except Exception as e:
            logger.error(f"Failed to enqueue trade execution outbox event: {e}")


trade_executor = TradeExecutor()
