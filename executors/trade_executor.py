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

        self.upbit = ccxt.upbit({
            'apiKey': settings.UPBIT_ACCESS_KEY,
            'secret': settings.UPBIT_SECRET_KEY,
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
            if exchange == 'binance':
                result = self._execute_binance(symbol, side, amount, leverage)
            elif exchange == 'upbit':
                result = self._execute_upbit(symbol, side, amount)
            else:
                return {"error": "Invalid exchange"}

            self._save_execution_record(result)

            return result

        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return {"error": str(e)}

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
            return {"success": False, "error": str(e)}

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
            return {"success": False, "error": str(e)}

    def _save_execution_record(self, execution_result: Dict) -> None:
        try:
            db.insert_trade_execution(execution_result)
            logger.info("Trade execution record saved")
        except Exception as e:
            logger.error(f"Failed to save execution record: {e}")


trade_executor = TradeExecutor()
