"""Funding & Global OI collector.

Collects from 3 major exchanges (Binance, Bybit, OKX) via ccxt:
- Funding rates (Binance primary)
- Global Open Interest = sum of Binance + Bybit + OKX OI in USD
- Long/Short ratio (Binance)

Global OI eliminates single-exchange noise and measures total market energy.
OI-price divergence is a key professional signal.
"""

import ccxt
from datetime import datetime, timezone
from typing import Dict, List, Optional
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
        self.binance_spot = ccxt.binance({
            'enableRateLimit': True
        })

        # Bybit and OKX - public API only (no keys needed for OI)
        self.bybit = ccxt.bybit({
            'enableRateLimit': True,
            'options': {'defaultType': 'swap'}
        })

        self.okx = ccxt.okx({
            'enableRateLimit': True,
            'options': {'defaultType': 'swap'}
        })

        self.symbols = settings.trading_symbols_slash

        # Symbol mappings per exchange — auto-generated from settings
        # Binance futures: BTC/USDT stays as-is
        # Bybit/OKX perpetual swaps: BTC/USDT → BTC/USDT:USDT
        self._symbol_map = {
            'binance': {s: s for s in self.symbols},
            'bybit':   {s: f"{s.split('/')[0]}/USDT:USDT" for s in self.symbols},
            'okx':     {s: f"{s.split('/')[0]}/USDT:USDT" for s in self.symbols},
        }

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

    def _fetch_oi_single(self, exchange, exchange_name: str, symbol: str) -> Optional[float]:
        """Fetch OI from a single exchange in USD value."""
        try:
            mapped = self._symbol_map.get(exchange_name, {}).get(symbol)
            if not mapped:
                return None

            if exchange_name == 'binance':
                binance_sym = symbol.replace('/', '')
                response = self.binance.fapiPublicGetOpenInterest({'symbol': binance_sym})
                oi_amount = float(response.get('openInterest', 0))
                ticker = self.binance.fetch_ticker(symbol)
                last_price = float(ticker.get('last', 0) or 0)
                return oi_amount * last_price

            elif exchange_name == 'bybit':
                # Bybit uses fetch_open_interest
                response = exchange.fetch_open_interest(mapped)
                if response and hasattr(response, 'openInterestValue'):
                    return float(response.openInterestValue or 0)
                elif isinstance(response, dict):
                    oi_val = response.get('openInterestValue') or response.get('info', {}).get('openInterestValue', 0)
                    return float(oi_val)
                return None

            elif exchange_name == 'okx':
                response = exchange.fetch_open_interest(mapped)
                if response and hasattr(response, 'openInterestValue'):
                    return float(response.openInterestValue or 0)
                elif isinstance(response, dict):
                    oi_val = response.get('openInterestValue') or response.get('info', {}).get('openInterestValue', 0)
                    return float(oi_val)
                return None

        except Exception as e:
            logger.warning(f"OI fetch error ({exchange_name}, {symbol}): {e}")
            return None

    def fetch_global_open_interest(self, symbol: str) -> Dict:
        """Fetch OI from Binance + Bybit + OKX and sum in USD.

        Global OI = Binance_OI + Bybit_OI + OKX_OI (all in USD value)
        Eliminates single-exchange noise.
        """
        oi_breakdown = {}
        total_oi = 0.0

        exchanges = [
            (self.binance, 'binance'),
            (self.bybit, 'bybit'),
            (self.okx, 'okx'),
        ]

        for exchange, name in exchanges:
            oi_value = self._fetch_oi_single(exchange, name, symbol)
            if oi_value is not None and oi_value > 0:
                oi_breakdown[f'oi_{name}'] = round(oi_value, 2)
                total_oi += oi_value

        return {
            'open_interest_global': round(total_oi, 2),
            'open_interest_binance': oi_breakdown.get('oi_binance', 0),
            'open_interest_bybit': oi_breakdown.get('oi_bybit', 0),
            'open_interest_okx': oi_breakdown.get('oi_okx', 0),
        }

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

    def fetch_basis_pct(self, symbol: str) -> float:
        """Perp-vs-spot basis in percent."""
        try:
            perp_last = float(self.binance.fetch_ticker(symbol).get('last', 0) or 0)
            spot_last = float(self.binance_spot.fetch_ticker(symbol).get('last', 0) or 0)
            if perp_last <= 0 or spot_last <= 0:
                return 0.0
            return round(((perp_last - spot_last) / spot_last) * 100, 6)
        except Exception as e:
            logger.warning(f"Basis fetch error for {symbol}: {e}")
            return 0.0

    def collect_all_funding_data(self) -> List[Dict]:
        results = []
        # Truncate to minute for dedup (UNIQUE constraint on timestamp, symbol)
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0).isoformat()

        for symbol in self.symbols:
            funding = self.fetch_funding_rate(symbol)
            global_oi = self.fetch_global_open_interest(symbol)
            ls_ratio = self.fetch_long_short_ratio(symbol)
            basis_pct = self.fetch_basis_pct(symbol)

            if funding or global_oi:
                combined = {
                    'symbol': symbol.replace('/', ''),
                    'timestamp': now,
                    'funding_rate': funding.get('funding_rate', 0),
                    'next_funding_time': funding.get('next_funding_time'),
                    'open_interest': global_oi.get('open_interest_binance', 0),
                    'open_interest_value': global_oi.get('open_interest_global', 0),
                    'oi_binance': global_oi.get('open_interest_binance', 0),
                    'oi_bybit': global_oi.get('open_interest_bybit', 0),
                    'oi_okx': global_oi.get('open_interest_okx', 0),
                    'basis_pct': basis_pct,
                    **ls_ratio,
                }
                results.append(combined)

        return results

    def save_to_database(self, data: List[Dict]) -> None:
        if data:
            try:
                for record in data:
                    db.upsert_funding_data(record)
                logger.info(f"Saved {len(data)} funding data records (global OI)")
            except Exception as e:
                logger.error(f"Database save error: {e}")

    def run(self) -> None:
        data = self.collect_all_funding_data()
        self.save_to_database(data)


funding_collector = FundingCollector()
