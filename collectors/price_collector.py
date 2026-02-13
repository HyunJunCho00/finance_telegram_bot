import ccxt
from datetime import datetime, timezone
from typing import Dict, List
import pandas as pd
import pytz
from config.settings import settings
from config.database import db
from loguru import logger


class PriceCollector:
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

        # BTC and ETH only
        self.symbols = {
            'binance': ['BTC/USDT', 'ETH/USDT'],
            'upbit': ['BTC/KRW', 'ETH/KRW']
        }

    def fetch_binance_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 1) -> pd.DataFrame:
        try:
            ohlcv = self.binance.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            # All timestamps stored as UTC
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df['exchange'] = 'binance'
            df['symbol'] = symbol.replace('/', '')
            return df
        except Exception as e:
            logger.error(f"Binance fetch error for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_upbit_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 1) -> pd.DataFrame:
        try:
            ohlcv = self.upbit.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            # All timestamps stored as UTC
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df['exchange'] = 'upbit'
            df['symbol'] = symbol.replace('/', '')
            return df
        except Exception as e:
            logger.error(f"Upbit fetch error for {symbol}: {e}")
            return pd.DataFrame()

    def collect_all_prices(self) -> List[Dict]:
        all_data = []

        for symbol in self.symbols['binance']:
            df = self.fetch_binance_ohlcv(symbol)
            if not df.empty:
                records = df.to_dict('records')
                for r in records:
                    r['timestamp'] = r['timestamp'].isoformat()
                all_data.extend(records)

        for symbol in self.symbols['upbit']:
            df = self.fetch_upbit_ohlcv(symbol)
            if not df.empty:
                records = df.to_dict('records')
                for r in records:
                    r['timestamp'] = r['timestamp'].isoformat()
                all_data.extend(records)

        return all_data

    def save_to_database(self, data: List[Dict]) -> None:
        if data:
            try:
                db.batch_insert_market_data(data)
                logger.info(f"Saved {len(data)} market data records")
            except Exception as e:
                logger.error(f"Database save error: {e}")

    def run(self) -> None:
        data = self.collect_all_prices()
        self.save_to_database(data)


collector = PriceCollector()
