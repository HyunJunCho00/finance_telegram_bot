"""Price collector with CVD (Cumulative Volume Delta) calculation.

Collects from:
- Binance Futures: BTC/USDT, ETH/USDT (1m candles)
- Upbit Spot: BTC/KRW, ETH/KRW (1m candles)

CVD Calculation:
  Uses Binance candle API's hidden field: Taker Buy Base Asset Volume
  - Taker Sell Vol = Total Vol - Taker Buy Vol
  - Volume Delta = Taker Buy Vol - Taker Sell Vol
  - CVD = cumulative sum of Volume Delta

  CVD reveals whale accumulation/distribution that price alone cannot show.
  Rising CVD + flat price = hidden accumulation (bullish)
  Falling CVD + rising price = distribution (bearish divergence)
"""

import ccxt
from datetime import datetime, timezone
from typing import Dict, List, Optional
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

        self.symbols = {
            'binance': settings.trading_symbols_slash,
            'upbit': settings.trading_symbols_krw,
        }

    def fetch_binance_ohlcv_with_cvd(
        self,
        symbol: str,
        timeframe: str = '1m',
        limit: int = 1,
        since_ms: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV from Binance Futures with Taker Buy Volume for CVD calculation.

        Binance klines API returns 12 fields per candle:
        [0] Open time, [1] Open, [2] High, [3] Low, [4] Close, [5] Volume,
        [6] Close time, [7] Quote asset volume, [8] Number of trades,
        [9] Taker buy base asset volume, [10] Taker buy quote asset volume,
        [11] Ignore
        """
        try:
            binance_symbol = symbol.replace('/', '')
            # Use raw API to get taker buy volume (field [9])
            raw_klines = self.binance.fapiPublicGetKlines({
                'symbol': binance_symbol,
                'interval': timeframe,
                'limit': limit,
                **({'startTime': since_ms} if since_ms is not None else {}),
            })

            if not raw_klines:
                return pd.DataFrame()

            rows = []
            for k in raw_klines:
                total_vol = float(k[5])
                taker_buy_vol = float(k[9])
                taker_sell_vol = total_vol - taker_buy_vol
                volume_delta = taker_buy_vol - taker_sell_vol

                rows.append({
                    'timestamp': pd.Timestamp(int(k[0]), unit='ms', tz='UTC'),
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': total_vol,
                    'taker_buy_volume': taker_buy_vol,
                    'taker_sell_volume': taker_sell_vol,
                    'volume_delta': volume_delta,
                    'exchange': 'binance',
                    'symbol': binance_symbol,
                })

            df = pd.DataFrame(rows)
            return df

        except Exception as e:
            logger.error(f"Binance fetch error for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_binance_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 1) -> pd.DataFrame:
        """Standard OHLCV fetch (backward compatible)."""
        return self.fetch_binance_ohlcv_with_cvd(symbol, timeframe, limit)

    def fetch_upbit_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1m',
        limit: int = 1,
        since_ms: Optional[int] = None,
    ) -> pd.DataFrame:
        try:
            ohlcv = self.upbit.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            # All timestamps stored as UTC
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df['exchange'] = 'upbit'
            df['symbol'] = symbol.replace('/', '')
            return df
        except Exception as e:
            logger.error(f"Upbit fetch error for {symbol}: {e}")
            return pd.DataFrame()

    def _latest_db_timestamp_ms(self, db_symbol: str, exchange: str) -> Optional[int]:
        """Read last stored candle timestamp (UTC ms) for symbol/exchange."""
        df = db.get_latest_market_data(db_symbol, limit=1, exchange=exchange)
        if df.empty:
            return None
        return int(df.iloc[-1]['timestamp'].timestamp() * 1000)

    def _collect_binance_symbol_gap(self, symbol: str) -> List[Dict]:
        """Backfill gap from latest DB candle to now, using 1m pagination."""
        db_symbol = symbol.replace('/', '')
        last_ms = self._latest_db_timestamp_ms(db_symbol, 'binance')

        if last_ms is None:
            # [FIX] If DB is empty, check GCS for the last available archived candle
            try:
                from processors.gcs_parquet import gcs_parquet_store
                if gcs_parquet_store.enabled:
                    # Look back 1 month for GCS archive to find a 'seed' for backfill
                    gcs_df = gcs_parquet_store.load_ohlcv("1m", db_symbol, months_back=1)
                    if not gcs_df.empty:
                        last_ms = int(gcs_df.iloc[-1]['timestamp'].timestamp() * 1000)
                        logger.info(f"DB empty for {symbol}, seeding from GCS @ {gcs_df.iloc[-1]['timestamp']}")
            except Exception as e:
                logger.warning(f"GCS seed lookup failed for {symbol}: {e}")

        if last_ms is None:
            # Final fallback: just get the last 2 candles
            df = self.fetch_binance_ohlcv_with_cvd(symbol, timeframe='1m', limit=2)
            return df.to_dict('records') if not df.empty else []

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        current_since = last_ms + 60000  # next minute
        if current_since > now_ms:
            return []

        rows: List[Dict] = []
        while current_since <= now_ms:
            batch = self.fetch_binance_ohlcv_with_cvd(
                symbol, timeframe='1m', limit=1000, since_ms=current_since
            )
            if batch.empty:
                break

            recs = batch.to_dict('records')
            rows.extend(recs)
            last_batch_ms = int(batch.iloc[-1]['timestamp'].timestamp() * 1000)
            next_since = last_batch_ms + 60000
            if next_since <= current_since:
                break
            current_since = next_since

            if len(batch) < 1000:
                break

        return rows

    def _collect_upbit_symbol_gap(self, symbol: str) -> List[Dict]:
        """Backfill gap from latest DB candle to now, using 1m pagination."""
        db_symbol = symbol.replace('/', '')
        last_ms = self._latest_db_timestamp_ms(db_symbol, 'upbit')

        if last_ms is None:
            # [FIX] If DB is empty, check GCS for seed
            try:
                from processors.gcs_parquet import gcs_parquet_store
                if gcs_parquet_store.enabled:
                    gcs_df = gcs_parquet_store.load_ohlcv("1m", db_symbol, months_back=1)
                    if not gcs_df.empty:
                        last_ms = int(gcs_df.iloc[-1]['timestamp'].timestamp() * 1000)
                        logger.info(f"DB empty for {symbol} (Upbit), seeding from GCS @ {gcs_df.iloc[-1]['timestamp']}")
            except Exception as e:
                logger.debug(f"GCS seed lookup failed for {symbol}: {e}")

        if last_ms is None:
            df = self.fetch_upbit_ohlcv(symbol, timeframe='1m', limit=2)
            return df.to_dict('records') if not df.empty else []

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        current_since = last_ms + 60000
        if current_since > now_ms:
            return []

        rows: List[Dict] = []
        while current_since <= now_ms:
            batch = self.fetch_upbit_ohlcv(
                symbol, timeframe='1m', limit=200, since_ms=current_since
            )
            if batch.empty:
                break

            recs = batch.to_dict('records')
            rows.extend(recs)
            last_batch_ms = int(batch.iloc[-1]['timestamp'].timestamp() * 1000)
            next_since = last_batch_ms + 60000
            if next_since <= current_since:
                break
            current_since = next_since

            if len(batch) < 200:
                break

        return rows

    def collect_all_prices(self) -> List[Dict]:
        all_data = []

        for symbol in self.symbols['binance']:
            records = self._collect_binance_symbol_gap(symbol)
            for r in records:
                r['timestamp'] = r['timestamp'].isoformat()
            all_data.extend(records)

        for symbol in self.symbols['upbit']:
            records = self._collect_upbit_symbol_gap(symbol)
            for r in records:
                r['timestamp'] = r['timestamp'].isoformat()
            all_data.extend(records)

        return all_data

    def save_to_database(self, data: List[Dict]) -> None:
        if data:
            try:
                # Separate market_data fields from CVD fields
                market_records = []
                cvd_records = []

                for r in data:
                    # Standard market data (backward compatible)
                    market_record = {
                        'timestamp': r['timestamp'],
                        'symbol': r['symbol'],
                        'exchange': r['exchange'],
                        'open': r['open'],
                        'high': r['high'],
                        'low': r['low'],
                        'close': r['close'],
                        'volume': r['volume'],
                    }
                    market_records.append(market_record)

                    # CVD data (Binance only)
                    if r.get('volume_delta') is not None:
                        cvd_records.append({
                            'timestamp': r['timestamp'],
                            'symbol': r['symbol'],
                            'taker_buy_volume': r.get('taker_buy_volume', 0),
                            'taker_sell_volume': r.get('taker_sell_volume', 0),
                            'volume_delta': r.get('volume_delta', 0),
                        })

                db.batch_insert_market_data(market_records)

                # Save CVD data
                if cvd_records:
                    db.batch_upsert_cvd_data(cvd_records)

                logger.info(f"Saved {len(market_records)} market + {len(cvd_records)} CVD records")
            except Exception as e:
                logger.error(f"Database save error: {e}")

    def run(self) -> None:
        data = self.collect_all_prices()
        self.save_to_database(data)


collector = PriceCollector()
