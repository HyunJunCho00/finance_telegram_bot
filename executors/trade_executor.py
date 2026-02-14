import ccxt
from typing import Dict
from datetime import datetime, timezone
from config.settings import settings
from config.database import db
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

    def execute(
        self,
        symbol: str,
        side: str,
        amount: float,
        leverage: int = 1,
        exchange: str = 'binance'
    ) -> Dict:
        try:
            side = side.upper()
            exchange = exchange.lower()

            # Default-safe path: no real order API call
            if settings.PAPER_TRADING_MODE:
                result = self._simulate_order(symbol, side, amount, leverage, exchange)
            else:
                if exchange == 'binance':
                    result = self._execute_binance(symbol, side, amount, leverage)
                elif exchange == 'upbit':
                    if settings.UPBIT_PAPER_ONLY:
                        result = self._simulate_order(symbol, side, amount, leverage, exchange)
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

            # default: live ticker reference
            ex = self.binance if exchange == 'binance' else self.upbit if exchange == 'upbit' else self.coinbase
            ticker = ex.fetch_ticker(symbol)
            return float(ticker.get('last') or ticker.get('close') or 0)
        except Exception as e:
            logger.warning(f"Reference price fetch failed ({exchange}, {symbol}): {e}")
            return 0.0

    def _simulate_order(self, symbol: str, side: str, amount: float, leverage: int, exchange: str) -> Dict:
        price = self._get_reference_price(symbol, exchange)
        notional = round(price * amount, 4) if price > 0 else 0
        return {
            "success": True,
            "paper": True,
            "exchange": exchange,
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "leverage": leverage,
            "order_id": f"paper-{exchange}-{int(datetime.now(timezone.utc).timestamp())}",
            "filled_price": price,
            "notional": notional,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "note": "PAPER_TRADING_MODE enabled - no live order sent",
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
