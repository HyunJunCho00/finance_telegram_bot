import ccxt
from datetime import datetime, timezone
from typing import Dict, List
from config.settings import settings
from config.database import db
from loguru import logger


class FundingCollector:
    def __init__(self):
        self.binance = ccxt.binance({
            'apiKey': settings.BINANCE_API_KEY,
            'secret': settings.BINANCE_API_SECRET,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })

        # BTC and ETH only
        self.symbols = ['BTC/USDT', 'ETH/USDT']

    def fetch_funding_rate(self, symbol: str) -> Dict:
        try:
            funding = self.binance.fetch_funding_rate(symbol)
            return {
                'funding_rate': float(funding.get('fundingRate', 0) or 0),
                'next_funding_time': funding.get('fundingTimestamp'),
                'mark_price': float(funding.get('markPrice', 0) or 0),
                'index_price': float(funding.get('indexPrice', 0) or 0),
            }
        except Exception as e:
            logger.error(f"Funding rate fetch error for {symbol}: {e}")
            return {}

    def fetch_open_interest(self, symbol: str) -> Dict:
        try:
            binance_symbol = symbol.replace('/', '')
            response = self.binance.fapiPublicGetOpenInterest({'symbol': binance_symbol})
            oi_amount = float(response.get('openInterest', 0))

            ticker = self.binance.fetch_ticker(symbol)
            last_price = float(ticker.get('last', 0) or 0)
            oi_value = oi_amount * last_price

            return {
                'open_interest': oi_amount,
                'open_interest_value': oi_value,
            }
        except Exception as e:
            logger.error(f"Open interest fetch error for {symbol}: {e}")
            return {}

    def fetch_long_short_ratio(self, symbol: str) -> Dict:
        try:
            binance_symbol = symbol.replace('/', '')
            response = self.binance.fapiPublicGetGlobalLongShortAccountRatio({
                'symbol': binance_symbol,
                'period': '5m',
                'limit': 1
            })
            if response and len(response) > 0:
                return {
                    'long_short_ratio': float(response[0].get('longShortRatio', 1.0)),
                    'long_account': float(response[0].get('longAccount', 0.5)),
                    'short_account': float(response[0].get('shortAccount', 0.5)),
                }
        except Exception as e:
            logger.error(f"Long/short ratio fetch error for {symbol}: {e}")
        return {}

    def collect_all_funding_data(self) -> List[Dict]:
        results = []
        # Truncate to minute for dedup (UNIQUE constraint on timestamp, symbol)
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0).isoformat()

        for symbol in self.symbols:
            funding = self.fetch_funding_rate(symbol)
            oi = self.fetch_open_interest(symbol)
            ls_ratio = self.fetch_long_short_ratio(symbol)

            if funding or oi:
                combined = {
                    'symbol': symbol.replace('/', ''),
                    'timestamp': now,
                    'funding_rate': funding.get('funding_rate', 0),
                    'next_funding_time': funding.get('next_funding_time'),
                    'open_interest': oi.get('open_interest', 0),
                    'open_interest_value': oi.get('open_interest_value', 0),
                    **ls_ratio,
                }
                results.append(combined)

        return results

    def save_to_database(self, data: List[Dict]) -> None:
        if data:
            try:
                for record in data:
                    db.upsert_funding_data(record)
                logger.info(f"Saved {len(data)} funding data records")
            except Exception as e:
                logger.error(f"Database save error: {e}")

    def run(self) -> None:
        data = self.collect_all_funding_data()
        self.save_to_database(data)


funding_collector = FundingCollector()
