import ccxt
from typing import Dict
from datetime import datetime, timezone
from config.settings import settings
from config.database import db
from config.local_state import state_manager
from loguru import logger


class TradeExecutor:
    def __init__(self):
        self.binance = ccxt.binance({
            'apiKey': settings.BINANCE_API_KEY,
            'secret': settings.BINANCE_API_SECRET,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
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
    def execute_from_decision(self, final_decision: dict, mode: str, symbol: str) -> dict:
        """Calculate order sizes and execute based on PM/CRO decision and trading mode."""
        try:
            direction = final_decision.get("decision", "HOLD")
            
            # V5: Handle Emergency Cancel Hook
            if direction == "CANCEL_AND_CLOSE":
                active = state_manager.get_active_orders()
                cancelled_count = 0
                for o in active:
                    if o['symbol'] == symbol:
                        state_manager.update_status(o['intent_id'], 'CANCELLED')
                        cancelled_count += 1
                return {
                    "success": True, 
                    "receipts": [{"note": f"CANCEL_AND_CLOSE executed. {cancelled_count} pending intents cancelled."}],
                    "strategy_applied": "CASINO_EXIT",
                    "total_notional": 0
                }
                
            if direction not in ["LONG", "SHORT"]:
                return {"success": False, "note": "No valid trade direction"}
                
            allocation_pct = final_decision.get("allocation_pct", 0)
            leverage = final_decision.get("leverage", 1)
            target_exchange = final_decision.get("target_exchange", "BINANCE").lower()
            exec_style = final_decision.get("recommended_execution_style", "MOMENTUM_SNIPER")
            
            if allocation_pct <= 0:
                return {"success": False, "note": "Allocation is 0% (Vetoed or No Confidence)"}
                
            price = self._get_reference_price(symbol, exchange=target_exchange)
            if price <= 0: return {"success": False, "error": "Could not fetch price"}
            
            # V8 Multi-Exchange Wallet Balances
            tp_price = float(final_decision.get("take_profit", 0))
            sl_price = float(final_decision.get("stop_loss", 0))

            if target_exchange == 'split':
                alloc_pct = allocation_pct / 100.0
                binance_notional = (settings.BINANCE_PAPER_BALANCE_USD * alloc_pct * leverage) * 0.5
                # [FIX HIGH-8] Upbit is spot-only — no leverage
                upbit_notional = (settings.UPBIT_PAPER_BALANCE_KRW * alloc_pct * 1) * 0.5
                
                intent_b = state_manager.add_intent(
                    symbol=symbol, direction=direction, style=exec_style,
                    amount=binance_notional, exchange="binance",
                    leverage=leverage, tp_price=tp_price, sl_price=sl_price,
                )
                intent_u = state_manager.add_intent(
                    symbol=symbol, direction=direction, style=exec_style,
                    amount=upbit_notional, exchange="upbit",
                    leverage=1, tp_price=tp_price, sl_price=sl_price,
                )
                
                receipts = [
                    {"order_id": intent_b, "exchange": "BINANCE", "side": direction, "notional": binance_notional, "paper": settings.PAPER_TRADING_MODE, "note": f"SPLIT Intent: {exec_style}"},
                    {"order_id": intent_u, "exchange": "UPBIT", "side": direction, "notional": upbit_notional, "paper": settings.PAPER_TRADING_MODE, "note": f"SPLIT Intent: {exec_style} (lev=1x)"}
                ]
                total_not_usd = binance_notional # Reporting only USD side for summary
            else:
                # Upbit spot: force leverage=1
                lev_for_exchange = 1 if target_exchange == 'upbit' else leverage
                wallet_balance = settings.BINANCE_PAPER_BALANCE_USD if target_exchange == "binance" else settings.UPBIT_PAPER_BALANCE_KRW
                target_notional = wallet_balance * (allocation_pct / 100.0) * lev_for_exchange
                
                # V5: Register Intent with Local State Manager instead of immediate naive execution
                intent_id = state_manager.add_intent(
                    symbol=symbol,
                    direction=direction,
                    style=exec_style,
                    amount=target_notional,
                    exchange=target_exchange,
                    leverage=lev_for_exchange,
                    tp_price=tp_price,
                    sl_price=sl_price,
                )
                
                # For logging in Telegram
                receipts = [{
                    "order_id": intent_id,
                    "exchange": target_exchange.upper(),
                    "side": direction,
                    "notional": target_notional,
                    "paper": settings.PAPER_TRADING_MODE,
                    "note": f"Registered Intent: {exec_style}"
                }]
                total_not_usd = target_notional
            
            return {
                "success": True,
                "receipts": receipts,
                "strategy_applied": exec_style,
                "total_notional": total_not_usd
            }
                
        except Exception as e:
            logger.error(f"Execution formatting error: {e}")
            return {"success": False, "error": str(e)}

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
    ) -> Dict:
        try:
            side = side.upper()
            exchange = exchange.lower()

            # Default-safe path: no real order API call
            if settings.PAPER_TRADING_MODE:
                result = self._simulate_order(symbol, side, amount, leverage, exchange, style, tp_price, sl_price)
            else:
                if exchange == 'binance':
                    result = self._execute_binance(symbol, side, amount, leverage)
                elif exchange == 'upbit':
                    if settings.UPBIT_PAPER_ONLY:
                        result = self._simulate_order(symbol, side, amount, leverage, exchange, style, tp_price, sl_price)
                    else:
                        result = self._execute_upbit(symbol, side, amount)
                elif exchange == 'coinbase':
                    result = self._execute_coinbase(symbol, side, amount)
                else:
                    return {"error": "Invalid exchange"}

            self._save_execution_record(result)
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

            # [FIX SILENT-3] CCXT requires slash format: BTCUSDT → BTC/USDT
            ccxt_symbol = symbol
            if 'USDT' in symbol and '/' not in symbol:
                ccxt_symbol = symbol.replace('USDT', '/USDT')
            elif 'KRW' in symbol and '/' not in symbol:
                ccxt_symbol = symbol.replace('KRW', '/KRW')

            # default: live ticker reference
            ex = self.binance if exchange == 'binance' else self.upbit if exchange == 'upbit' else self.coinbase
            ticker = ex.fetch_ticker(ccxt_symbol)
            return float(ticker.get('last') or ticker.get('close') or 0)
        except Exception as e:
            logger.warning(f"Reference price fetch failed ({exchange}, {symbol}): {e}")
            return 0.0

    def _simulate_order(self, symbol: str, side: str, amount: float, leverage: int, exchange: str, style: str, tp_price: float = 0.0, sl_price: float = 0.0) -> Dict:
        price = self._get_reference_price(symbol, exchange)
        if not price:
            return {"success": False, "error": "Could not get reference price"}

        # V7: Paper Exchange Engine provides realistic slippage and wallet deduction
        from executors.paper_exchange import paper_engine
        
        sim_res = paper_engine.simulate_execution(
            exchange=exchange,
            symbol=symbol,
            direction=side,
            amount_usd=amount,
            leverage=leverage,
            style=style,
            raw_price=price,
            tp_price=tp_price,
            sl_price=sl_price,
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
            "note": f"V7 PAPER ENGINE - Slippage: {sim_res['slippage_applied_pct']:.3f}% applied",
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
            db.insert_trade_execution(execution_result)
            logger.info("Trade execution record saved")
        except Exception as e:
            logger.error(f"Failed to save execution record: {e}")


trade_executor = TradeExecutor()
