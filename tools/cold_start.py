"""Cold Start: One-time bulk data loader for initial system setup.

Downloads historical data and stores in both PostgreSQL (hot cache) and GCS Parquet (permanent).

Loadable data sources (REST API, no WebSocket needed):
  [O] OHLCV 1m     — Binance klines API (max 1000 per request, paginated)
  [O] CVD           — Extracted from same Binance klines (taker_buy_volume)
  [O] Funding Rate  — Binance fundingRate API (max 1000 per request)
  [O] OI History    — Binance openInterestHist API (5m granularity)
  [O] Telegram      — Telethon historical messages (unlimited)
  [X] Liquidations  — WebSocket only (no historical REST API)
  [X] Whale CVD     — WebSocket only (aggTrade not available historically in bulk)
  [X] Perplexity    — Real-time search only

Usage:
  python -m tools.cold_start --mode all --days 210 --symbol BTCUSDT
  python -m tools.cold_start --mode ohlcv --days 30 --symbol BTCUSDT
  python -m tools.cold_start --mode funding --days 365
  python -m tools.cold_start --mode telegram --days 30
  python -m tools.cold_start --mode resample --symbol BTCUSDT  # Generate 1h/4h/1d/1w from 1m

Note on data volume:
  1 day of 1m candles = 1,440 rows
  210 days (7 months) = ~302,400 rows per symbol ≈ 30MB Parquet
  Binance rate limit: 1200 req/min → 1000 candles/req → ~303 requests → ~1 minute
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import ccxt
import pandas as pd
import requests
import yfinance as yf
from loguru import logger

# Add project root to path
sys.path.insert(0, ".")

from config.settings import settings
from config.database import db
from processors.gcs_parquet import gcs_parquet_store


# ─────────────── Resume Helper ───────────────

def _get_latest_timestamp(table: str, symbol: str = None, exchange: str = None) -> Optional[datetime]:
    """Query Supabase for the latest timestamp in a table, for resume support."""
    try:
        query = db.client.table(table).select("timestamp").order("timestamp", desc=True).limit(1)
        if symbol:
            query = query.eq("symbol", symbol)
        if exchange:
            query = query.eq("exchange", exchange)
        response = query.execute()
        if response.data:
            return pd.Timestamp(response.data[0]['timestamp']).to_pydatetime().replace(tzinfo=timezone.utc)
    except Exception as e:
        logger.warning(f"Resume check failed for {table}: {e}")
    return None


class OHLCVBulkLoader:
    """Load historical 1m OHLCV + CVD from Binance Futures."""

    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': settings.BINANCE_API_KEY,
            'secret': settings.BINANCE_API_SECRET,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        self.batch_size = 1000  # Binance max per request

    def fetch_range(self, symbol: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        """Fetch 1m candles for a date range, paginated."""
        binance_symbol = symbol.replace("/", "")
        all_rows = []
        current_start = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        total_expected = int((end_dt - start_dt).total_seconds() / 60)
        logger.info(f"Fetching {symbol} 1m: {start_dt.date()} → {end_dt.date()} "
                     f"(~{total_expected:,} candles expected)")

        request_count = 0
        while current_start < end_ms:
            try:
                raw = self.exchange.fapiPublicGetKlines({
                    'symbol': binance_symbol,
                    'interval': '1m',
                    'startTime': current_start,
                    'limit': self.batch_size,
                })

                if not raw:
                    break

                for k in raw:
                    total_vol = float(k[5])
                    taker_buy = float(k[9])
                    taker_sell = total_vol - taker_buy

                    all_rows.append({
                        'timestamp': pd.Timestamp(int(k[0]), unit='ms', tz='UTC'),
                        'open': float(k[1]),
                        'high': float(k[2]),
                        'low': float(k[3]),
                        'close': float(k[4]),
                        'volume': total_vol,
                        'exchange': 'binance',
                        'symbol': binance_symbol,
                        'taker_buy_volume': taker_buy,
                        'taker_sell_volume': taker_sell,
                        'volume_delta': taker_buy - taker_sell,
                    })

                # Move to next batch
                last_ts = int(raw[-1][0])
                current_start = last_ts + 60000  # +1 minute

                request_count += 1
                if request_count % 50 == 0:
                    logger.info(f"  ... {len(all_rows):,} candles fetched ({request_count} requests)")

                if len(raw) < self.batch_size:
                    break  # No more data

            except Exception as e:
                logger.error(f"Fetch error at {current_start}: {e}")
                time.sleep(2)
                continue

        df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
        if not df.empty:
            logger.info(f"Fetched {len(df):,} candles for {symbol}")
        return df

    def save_to_db(self, df: pd.DataFrame, db_days: int = 3) -> int:
        """Save recent N days to PostgreSQL (hot cache only)."""
        if df.empty:
            return 0

        cutoff = datetime.now(timezone.utc) - timedelta(days=db_days)
        recent = df[df['timestamp'] >= cutoff].copy()
        if recent.empty:
            return 0

        # Market data
        market_records = []
        cvd_records = []

        for _, r in recent.iterrows():
            market_records.append({
                'timestamp': r['timestamp'].isoformat(),
                'symbol': r['symbol'],
                'exchange': r['exchange'],
                'open': r['open'],
                'high': r['high'],
                'low': r['low'],
                'close': r['close'],
                'volume': r['volume'],
            })
            if pd.notna(r.get('volume_delta')):
                cvd_records.append({
                    'timestamp': r['timestamp'].isoformat(),
                    'symbol': r['symbol'],
                    'taker_buy_volume': r['taker_buy_volume'],
                    'taker_sell_volume': r['taker_sell_volume'],
                    'volume_delta': r['volume_delta'],
                })

        # Batch upsert in chunks of 500
        chunk_size = 500
        for i in range(0, len(market_records), chunk_size):
            db.batch_insert_market_data(market_records[i:i + chunk_size])

        for i in range(0, len(cvd_records), chunk_size):
            db.batch_upsert_cvd_data(cvd_records[i:i + chunk_size])

        logger.info(f"Saved {len(market_records)} market + {len(cvd_records)} CVD records to PostgreSQL")
        return len(market_records)

    def save_to_gcs(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Save full history to GCS Parquet, partitioned by month."""
        if df.empty or not gcs_parquet_store.enabled:
            return {"saved": False}

        df = df.copy()
        df['month'] = df['timestamp'].dt.strftime('%Y-%m')

        total_bytes = 0
        total_rows = 0

        for month, group in df.groupby('month'):
            # OHLCV (without CVD columns)
            ohlcv_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'exchange', 'symbol']
            ohlcv_df = group[ohlcv_cols].copy()
            path = f"ohlcv/1m/{symbol}/{month}.parquet"
            size = gcs_parquet_store._merge_upload_parquet(path, ohlcv_df,
                                                           ["timestamp", "symbol", "exchange"])
            total_bytes += size
            total_rows += len(group)

            # CVD data
            cvd_cols = ['timestamp', 'symbol', 'taker_buy_volume', 'taker_sell_volume', 'volume_delta']
            cvd_df = group[cvd_cols].copy()
            cvd_path = f"cvd/{symbol}/{month}.parquet"
            gcs_parquet_store._merge_upload_parquet(cvd_path, cvd_df, ["timestamp", "symbol"])

        logger.info(f"GCS: {total_rows:,} rows, {total_bytes:,} bytes for {symbol}")
        return {"rows": total_rows, "bytes": total_bytes}


# ─────────────── Resample & Save Higher Timeframes ───────────────

class ResampleUploader:
    """Generate 1h/4h/1d/1w from 1m data and save to GCS."""

    TIMEFRAME_MAP = {
        '1h': '1h',
        '4h': '4h',
        '1d': '1D',
        '1w': '1W-MON',  # Week starts Monday (aligns with Upbit KST 09:00 = UTC 00:00)
    }

    def resample_and_upload(self, symbol: str) -> Dict:
        """Load 1m from GCS, resample to all higher timeframes, save back."""
        if not gcs_parquet_store.enabled:
            return {"enabled": False}

        # Load all 1m data from GCS
        df_1m = gcs_parquet_store.load_ohlcv("1m", symbol, months_back=60)  # Up to 5 years
        if df_1m.empty:
            logger.warning(f"No 1m data in GCS for {symbol}")
            return {"error": "no_1m_data"}

        logger.info(f"Loaded {len(df_1m):,} 1m candles for {symbol} from GCS")

        # Ensure proper types
        df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'], utc=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df_1m.columns:
                df_1m[col] = df_1m[col].astype(float)

        df_1m = df_1m.set_index('timestamp')
        results = {}

        for tf_name, rule in self.TIMEFRAME_MAP.items():
            try:
                resampled = df_1m.resample(rule).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum',
                }).dropna().reset_index()

                resampled.rename(columns={'timestamp': 'timestamp'}, inplace=True)
                if 'index' in resampled.columns:
                    resampled.rename(columns={'index': 'timestamp'}, inplace=True)

                # Upload partitioned
                if tf_name in ('1d', '1w'):
                    # Yearly partitions
                    resampled['year'] = resampled['timestamp'].dt.strftime('%Y')
                    for year, group in resampled.groupby('year'):
                        path = f"ohlcv/{tf_name}/{symbol}/{year}.parquet"
                        gcs_parquet_store._upload_parquet(path, group.drop(columns=['year']))
                else:
                    # Monthly partitions
                    resampled['month'] = resampled['timestamp'].dt.strftime('%Y-%m')
                    for month, group in resampled.groupby('month'):
                        path = f"ohlcv/{tf_name}/{symbol}/{month}.parquet"
                        gcs_parquet_store._upload_parquet(path, group.drop(columns=['month']))

                results[tf_name] = {"rows": len(resampled)}
                logger.info(f"Resampled {symbol} {tf_name}: {len(resampled)} candles")

            except Exception as e:
                logger.error(f"Resample {tf_name} error: {e}")
                results[tf_name] = {"error": str(e)}

        return results


# ─────────────── Funding Rate Bulk Loader ───────────────

class FundingBulkLoader:
    """Load historical funding rates from Binance."""

    def __init__(self):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })

    def fetch_range(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """Fetch historical funding rates. Binance provides 8h intervals."""
        binance_symbol = symbol.replace("/", "").replace("USDT", "") + "USDT"
        all_rows = []
        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_time = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
        current_start = start_time

        logger.info(f"Fetching {symbol} funding rates: {days} days")

        while current_start < end_time:
            try:
                response = self.exchange.fapiPublicGetFundingRate({
                    'symbol': binance_symbol,
                    'startTime': current_start,
                    'limit': 1000,
                })

                if not response:
                    break

                for r in response:
                    all_rows.append({
                        'timestamp': pd.Timestamp(int(r['fundingTime']), unit='ms', tz='UTC'),
                        'symbol': binance_symbol,
                        'funding_rate': float(r['fundingRate']),
                    })

                last_ts = int(response[-1]['fundingTime'])
                current_start = last_ts + 1

                if len(response) < 1000:
                    break

            except Exception as e:
                logger.error(f"Funding fetch error: {e}")
                time.sleep(1)
                continue

        df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
        if not df.empty:
            logger.info(f"Fetched {len(df)} funding rate records for {symbol}")
        return df

    def save_to_db(self, df: pd.DataFrame, db_days: int = 7) -> int:
        """Save recent N days to PostgreSQL."""
        if df.empty:
            return 0

        cutoff = datetime.now(timezone.utc) - timedelta(days=db_days)
        recent = df[df['timestamp'] >= cutoff].copy()

        for _, r in recent.iterrows():
            db.upsert_funding_data({
                'timestamp': r['timestamp'].isoformat(),
                'symbol': r['symbol'],
                'funding_rate': r['funding_rate'],
            })

        logger.info(f"Saved {len(recent)} funding records to PostgreSQL")
        return len(recent)

    def save_to_gcs(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Save full funding history to GCS Parquet."""
        if df.empty or not gcs_parquet_store.enabled:
            return {"saved": False}

        df = df.copy()
        df['month'] = df['timestamp'].dt.strftime('%Y-%m')

        for month, group in df.groupby('month'):
            path = f"funding/{symbol}/{month}.parquet"
            gcs_parquet_store._merge_upload_parquet(path, group.drop(columns=['month']),
                                                     ["timestamp", "symbol"])

        logger.info(f"GCS: {len(df)} funding records for {symbol}")
        return {"rows": len(df)}


# ─────────────── Telegram Bulk Loader ───────────────

class TelegramBulkLoader:
    """Load historical Telegram messages."""

    def load(self, days: int = 30) -> int:
        """Use existing TelegramCollector with extended lookback."""
        from collectors.telegram_collector import telegram_collector
        import asyncio

        logger.info(f"Fetching Telegram messages: last {days} days")
        hours = days * 24

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            messages = loop.run_until_complete(
                telegram_collector.fetch_recent_messages(hours=hours)
            )
            loop.close()

            if messages:
                telegram_collector.save_to_database(messages)
                logger.info(f"Saved {len(messages)} Telegram messages")
                return len(messages)
            return 0
        except Exception as e:
            logger.error(f"Telegram bulk load error: {e}")
            return 0


# ─────────────── Upbit Spot Bulk Loader ───────────────

class UpbitBulkLoader:
    """Load historical Upbit spot data (BTC/KRW, ETH/KRW)."""

    def __init__(self):
        self.exchange = ccxt.upbit({'enableRateLimit': True})

    def fetch_range(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Upbit's fetch_ohlcv supports pagination via 'since' parameter.

        Timezone note:
          - Upbit candles are exchange-local (KST) intervals.
          - ccxt returns epoch milliseconds; we normalize/store as UTC tz-aware timestamps.
          - This keeps Binance + Upbit merge/resample on one unified UTC timeline.
        """
        all_rows = []
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)
        current_since = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)

        upbit_symbol = symbol  # e.g. "BTC/KRW"
        db_symbol = symbol.replace("/", "")
        logger.info(f"Fetching {symbol} from Upbit: {days} days")

        consecutive_empty = 0

        while current_since < end_ms:
            try:
                ohlcv = self.exchange.fetch_ohlcv(upbit_symbol, '1m',
                                                   since=current_since, limit=200)
                if not ohlcv:
                    consecutive_empty += 1
                    if consecutive_empty >= 5:
                        logger.warning(f"Upbit {symbol}: 5 consecutive empty responses, stopping")
                        break
                    current_since += 60000 * 200  # skip ahead ~200 minutes
                    time.sleep(1)
                    continue

                consecutive_empty = 0

                for candle in ohlcv:
                    ts_utc = pd.Timestamp(candle[0], unit='ms', tz='UTC').floor('min')
                    all_rows.append({
                        'timestamp': ts_utc,
                        'open': candle[1],
                        'high': candle[2],
                        'low': candle[3],
                        'close': candle[4],
                        'volume': candle[5],
                        'exchange': 'upbit',
                        'symbol': db_symbol,
                    })

                last_ts = ohlcv[-1][0]
                current_since = last_ts + 60000

                if len(all_rows) % 50000 == 0 and len(all_rows) > 0:
                    logger.info(f"  ... {len(all_rows):,} Upbit candles fetched")

                # Only stop on short response if we've actually reached end
                if len(ohlcv) < 200 and current_since >= end_ms:
                    break

                time.sleep(0.2)  # Upbit rate limit is stricter

            except Exception as e:
                logger.error(f"Upbit fetch error: {e}")
                time.sleep(2)
                continue

        df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
        if not df.empty:
            logger.info(f"Fetched {len(df):,} Upbit candles for {symbol}")
        return df

    def save_to_db(self, df: pd.DataFrame, db_days: int = 3) -> int:
        """Save recent data to PostgreSQL."""
        if df.empty:
            return 0

        cutoff = datetime.now(timezone.utc) - timedelta(days=db_days)
        recent = df[df['timestamp'] >= cutoff].copy()
        if recent.empty:
            return 0

        records = []
        for _, r in recent.iterrows():
            records.append({
                'timestamp': r['timestamp'].isoformat(),
                'symbol': r['symbol'],
                'exchange': r['exchange'],
                'open': r['open'],
                'high': r['high'],
                'low': r['low'],
                'close': r['close'],
                'volume': r['volume'],
            })

        chunk_size = 500
        for i in range(0, len(records), chunk_size):
            db.batch_insert_market_data(records[i:i + chunk_size])

        logger.info(f"Saved {len(records)} Upbit records to PostgreSQL")
        return len(records)

    def save_to_gcs(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Save to GCS Parquet."""
        if df.empty or not gcs_parquet_store.enabled:
            return {"saved": False}

        db_symbol = symbol.replace("/", "")
        df = df.copy()
        df['month'] = df['timestamp'].dt.strftime('%Y-%m')

        for month, group in df.groupby('month'):
            path = f"ohlcv/1m/{db_symbol}/{month}.parquet"
            ohlcv_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'exchange', 'symbol']
            gcs_parquet_store._merge_upload_parquet(path, group[ohlcv_cols],
                                                     ["timestamp", "symbol", "exchange"])

        logger.info(f"GCS: {len(df)} Upbit records for {symbol}")
        return {"rows": len(df)}


# ─────────────── Fear & Greed Bulk Loader ───────────────

class FearGreedBulkLoader:
    """Load historical Fear & Greed Index from alternative.me (no API key, free)."""

    FNG_URL = "https://api.alternative.me/fng/"

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({'Accept': 'application/json'})

    def fetch_range(self, days: int = 2190) -> pd.DataFrame:
        """Fetch historical F&G. alternative.me supports up to ~2000+ days via ?limit=."""
        try:
            resp = self._session.get(
                self.FNG_URL,
                params={'limit': days, 'format': 'json'},
                timeout=30,
            )
            resp.raise_for_status()
            entries = resp.json().get('data', [])
            if not entries:
                logger.warning("Fear & Greed: empty response")
                return pd.DataFrame()

            rows = []
            for i, e in enumerate(entries):
                value = int(e['value'])
                prev = entries[i + 1] if i + 1 < len(entries) else None
                ts = pd.Timestamp(int(e['timestamp']), unit='s', tz='UTC').normalize()
                rows.append({
                    'timestamp': ts,
                    'value': value,
                    'classification': e['value_classification'],
                    'value_prev': int(prev['value']) if prev else None,
                    'classification_prev': prev['value_classification'] if prev else None,
                    'change': (value - int(prev['value'])) if prev else None,
                })

            df = pd.DataFrame(rows)
            logger.info(f"Fetched {len(df)} Fear & Greed records (oldest: {df['timestamp'].min().date()})")
            return df

        except Exception as e:
            logger.error(f"Fear & Greed bulk fetch error: {e}")
            return pd.DataFrame()

    def save_to_db(self, df: pd.DataFrame) -> int:
        """Save all records to PostgreSQL (small dataset, save everything)."""
        if df.empty:
            return 0

        count = 0
        for _, r in df.iterrows():
            try:
                db.upsert_fear_greed({
                    'timestamp': r['timestamp'].isoformat(),
                    'value': int(r['value']),
                    'classification': r['classification'],
                    'value_prev': int(r['value_prev']) if pd.notna(r.get('value_prev')) else None,
                    'classification_prev': r.get('classification_prev'),
                    'change': int(r['change']) if pd.notna(r.get('change')) else None,
                })
                count += 1
            except Exception as e:
                logger.debug(f"Fear & Greed upsert error: {e}")

        logger.info(f"Saved {count} Fear & Greed records to PostgreSQL")
        return count


# ─────────────── Deribit DVOL Bulk Loader ───────────────

class DeribitDVOLBulkLoader:
    """Load historical DVOL index from Deribit public API (no auth required).

    Only DVOL is available historically.
    PCR / IV Term Structure / Skew require live option chain → real-time only.
    """

    DERIBIT_URL = "https://www.deribit.com/api/v2/public"
    RESOLUTION = 3600  # 1h in seconds (Deribit resolution values: 1, 60, 3600, 43200, 86400)

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({'Accept': 'application/json'})

    def fetch_range(self, currency: str, days: int = 2190) -> pd.DataFrame:
        """Fetch historical DVOL hourly candles with pagination."""
        all_rows = []
        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
        resolution_ms = self.RESOLUTION * 1000
        current_start = start_ms
        request_count = 0

        logger.info(f"Fetching Deribit DVOL {currency}: {days} days (1h resolution)")

        while current_start < end_ms:
            try:
                resp = self._session.get(
                    f"{self.DERIBIT_URL}/get_volatility_index_data",
                    params={
                        'currency': currency,
                        'start_timestamp': current_start,
                        'end_timestamp': end_ms,
                        'resolution': str(self.RESOLUTION),
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                result = resp.json().get('result', {})
                data = result.get('data', [])

                if not data:
                    break

                for point in data:
                    # [ts_ms, open, high, low, close]
                    all_rows.append({
                        'timestamp': pd.Timestamp(int(point[0]), unit='ms', tz='UTC'),
                        'symbol': currency,
                        'dvol': round(float(point[4]), 4),  # close value
                    })

                last_ts = int(data[-1][0])
                current_start = last_ts + resolution_ms
                request_count += 1

                if request_count % 10 == 0:
                    logger.info(f"  ... {len(all_rows):,} DVOL records ({request_count} requests)")

                # Deribit returns fewer points when approaching end_ms
                if len(data) < 10:
                    break

                time.sleep(0.2)

            except Exception as e:
                logger.error(f"Deribit DVOL fetch error at ts={current_start}: {e}")
                time.sleep(2)
                current_start += resolution_ms * 100  # skip ahead on error
                continue

        df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
        if not df.empty:
            logger.info(f"Fetched {len(df):,} DVOL records for {currency} "
                        f"(oldest: {df['timestamp'].min().date()})")
        return df

    def save_to_db(self, df: pd.DataFrame) -> int:
        """Save DVOL records to deribit_data table (only dvol field, others NULL)."""
        if df.empty:
            return 0

        count = 0
        for _, r in df.iterrows():
            try:
                db.upsert_deribit_data({
                    'timestamp': r['timestamp'].isoformat(),
                    'symbol': r['symbol'],
                    'dvol': float(r['dvol']),
                    # PCR / IV / Skew are NULL for historical rows (live option chain required)
                })
                count += 1
            except Exception as e:
                logger.debug(f"Deribit DVOL upsert error: {e}")

        logger.info(f"Saved {count} Deribit DVOL records to PostgreSQL")
        return count


# ─────────────── Macro Bulk Loader ───────────────

class MacroBulkLoader:
    """Load historical macro data from FRED API and yfinance.

    FRED series (daily/monthly): DGS2, DGS10, CPIAUCSL, FEDFUNDS
    yfinance (daily close): DXY, NASDAQ, Gold, Oil
    Merged into daily snapshots with forward-fill for weekends/holidays.
    """

    FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
    FRED_SERIES = {
        "dgs2": "DGS2",
        "dgs10": "DGS10",
        "cpiaucsl": "CPIAUCSL",
        "fedfunds": "FEDFUNDS",
    }
    TICKERS = {
        "dxy": "DX-Y.NYB",
        "nasdaq": "^IXIC",
        "gold": "GC=F",
        "oil": "CL=F",
    }

    def __init__(self):
        self.fred_api_key = settings.FRED_API_KEY
        self._session = requests.Session()

    def _fetch_fred(self, series_id: str, days: int) -> pd.Series:
        if not self.fred_api_key:
            logger.warning(f"FRED_API_KEY not set — skipping {series_id}")
            return pd.Series(dtype=float, name=series_id.lower())

        start_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime('%Y-%m-%d')
        try:
            resp = self._session.get(self.FRED_BASE, params={
                "series_id": series_id,
                "api_key": self.fred_api_key,
                "file_type": "json",
                "sort_order": "asc",
                "observation_start": start_date,
            }, timeout=30)
            resp.raise_for_status()
            obs = resp.json().get("observations", [])
            data = {o["date"]: float(o["value"]) for o in obs
                    if o.get("value") not in (None, ".")}
            s = pd.Series(data, name=series_id.lower())
            s.index = pd.to_datetime(s.index)
            logger.info(f"FRED {series_id}: {len(s)} records")
            return s
        except Exception as e:
            logger.warning(f"FRED {series_id} error: {e}")
            return pd.Series(dtype=float, name=series_id.lower())

    def _fetch_yf(self, ticker: str, name: str) -> pd.Series:
        try:
            hist = yf.Ticker(ticker).history(period="max")["Close"].dropna()
            hist.index = hist.index.tz_localize(None).normalize()
            hist.name = name
            logger.info(f"yfinance {ticker} ({name}): {len(hist)} records")
            return hist
        except Exception as e:
            logger.warning(f"yfinance {ticker} error: {e}")
            return pd.Series(dtype=float, name=name)

    def build_snapshots(self, days: int = 2190) -> pd.DataFrame:
        """Merge FRED + yfinance into daily macro snapshots with forward-fill."""
        all_series = []

        for col, sid in self.FRED_SERIES.items():
            s = self._fetch_fred(sid, days)
            s.name = col
            all_series.append(s)

        for col, ticker in self.TICKERS.items():
            s = self._fetch_yf(ticker, col)
            all_series.append(s)

        df = pd.concat(all_series, axis=1)

        # Forward-fill weekends/holidays (FRED updates only on business days)
        df = df.ffill()

        # Filter to requested period
        start = pd.Timestamp(datetime.now(timezone.utc) - timedelta(days=days)).tz_localize(None)
        df = df[df.index >= start]

        # Yield curve spread
        if 'dgs2' in df.columns and 'dgs10' in df.columns:
            df['ust_2s10s_spread'] = (df['dgs10'] - df['dgs2']).round(4)

        df = df.dropna(how='all').reset_index().rename(columns={'index': 'date'})
        logger.info(f"Built {len(df)} macro daily snapshots "
                    f"(oldest: {df['date'].min().date() if not df.empty else 'N/A'})")
        return df

    def save_to_db(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0

        def _safe(r, col):
            v = r.get(col)
            return float(v) if pd.notna(v) else None

        count = 0
        for _, r in df.iterrows():
            try:
                ts = pd.Timestamp(r['date'])
                if ts.tzinfo is None:
                    ts = ts.tz_localize('UTC')
                db.upsert_macro_data({
                    'timestamp': ts.isoformat(),
                    'source': 'macro_bulk_loader',
                    'dgs2': _safe(r, 'dgs2'),
                    'dgs10': _safe(r, 'dgs10'),
                    'ust_2s10s_spread': _safe(r, 'ust_2s10s_spread'),
                    'cpiaucsl': _safe(r, 'cpiaucsl'),
                    'fedfunds': _safe(r, 'fedfunds'),
                    'dxy': _safe(r, 'dxy'),
                    'nasdaq': _safe(r, 'nasdaq'),
                    'gold': _safe(r, 'gold'),
                    'oil': _safe(r, 'oil'),
                })
                count += 1
            except Exception as e:
                logger.debug(f"Macro upsert error: {e}")

        logger.info(f"Saved {count} macro snapshots to PostgreSQL")
        return count


# ─────────────── Main Entry Point ───────────────

def run_cold_start(mode: str = "all", days: int = 210,
                   symbols: Optional[List[str]] = None,
                   db_days: int = 7,
                   ohlcv_days: Optional[int] = None,
                   funding_days: Optional[int] = None,
                   telegram_days: Optional[int] = None,
                   upbit_days: Optional[int] = None,
                   fear_greed_days: Optional[int] = None,
                   deribit_days: Optional[int] = None,
                   macro_days: Optional[int] = None):
    """Run cold start data loading.

    Args:
        mode: "all", "ohlcv", "funding", "telegram", "resample", "upbit",
              "fear_greed", "deribit_dvol", "macro"
        days: Default days of history for all loaders
        symbols: List of Binance Futures symbols (default: ["BTCUSDT", "ETHUSDT"])
        db_days: Recent days to keep in PostgreSQL hot cache (rest → GCS)
        ohlcv_days: Override for OHLCV loader
        funding_days: Override for Funding loader
        telegram_days: Override for Telegram loader (no default cap — fetch all)
        upbit_days: Override for Upbit loader
        fear_greed_days: Override for Fear & Greed loader
        deribit_days: Override for Deribit DVOL loader
        macro_days: Override for Macro loader
    """
    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT"]

    logger.info(f"=== COLD START: mode={mode}, days={days}, db_days={db_days}, symbols={symbols} ===")
    results = {}

    effective_ohlcv_days = ohlcv_days if ohlcv_days is not None else days
    effective_funding_days = funding_days if funding_days is not None else days
    effective_telegram_days = telegram_days if telegram_days is not None else days  # no cap
    effective_upbit_days = upbit_days if upbit_days is not None else days
    effective_fear_greed_days = fear_greed_days if fear_greed_days is not None else days
    effective_deribit_days = deribit_days if deribit_days is not None else days
    effective_macro_days = macro_days if macro_days is not None else days

    # ── OHLCV (Binance Futures) ──
    if mode in ("all", "ohlcv"):
        loader = OHLCVBulkLoader()
        for symbol in symbols:
            end = datetime.now(timezone.utc)
            default_start = end - timedelta(days=effective_ohlcv_days)
            # Resume: check DB for latest existing data
            latest = _get_latest_timestamp("market_data", symbol=symbol, exchange="binance")
            if latest and latest > default_start:
                start = latest + timedelta(minutes=1)
                logger.info(f"[RESUME] {symbol} OHLCV: DB has data up to {latest}, fetching from {start}")
            else:
                start = default_start
            if start >= end:
                logger.info(f"[SKIP] {symbol} OHLCV: already up to date")
                results[f"ohlcv_{symbol}"] = {"status": "already_up_to_date"}
                continue
            df = loader.fetch_range(symbol, start, end)
            db_count = loader.save_to_db(df, db_days=db_days)
            gcs_result = loader.save_to_gcs(df, symbol)
            results[f"ohlcv_{symbol}"] = {
                "total_candles": len(df),
                "db_rows": db_count,
                "gcs": gcs_result,
            }

    # ── Upbit Spot ──
    if mode in ("all", "upbit"):
        loader = UpbitBulkLoader()
        for symbol in ["BTC/KRW", "ETH/KRW"]:
            db_symbol = symbol.replace("/", "")
            latest = _get_latest_timestamp("market_data", symbol=db_symbol, exchange="upbit")
            if latest:
                elapsed = (datetime.now(timezone.utc) - latest).total_seconds() / 86400
                effective_days = max(1, int(elapsed) + 1)  # fetch from last data + 1 day buffer
                logger.info(f"[RESUME] {symbol} Upbit: DB has data up to {latest}, fetching last {effective_days} days")
            else:
                effective_days = effective_upbit_days
            df = loader.fetch_range(symbol, effective_days)
            db_count = loader.save_to_db(df, db_days=db_days)
            gcs_result = loader.save_to_gcs(df, symbol)
            results[f"upbit_{symbol}"] = {
                "total_candles": len(df),
                "db_rows": db_count,
                "gcs": gcs_result,
            }

    # ── Funding Rates ──
    if mode in ("all", "funding"):
        loader = FundingBulkLoader()
        for symbol in symbols:
            latest = _get_latest_timestamp("funding_data", symbol=symbol)
            if latest:
                elapsed = (datetime.now(timezone.utc) - latest).total_seconds() / 86400
                effective_days_f = max(1, int(elapsed) + 1)
                logger.info(f"[RESUME] {symbol} Funding: DB has data up to {latest}, fetching last {effective_days_f} days")
            else:
                effective_days_f = effective_funding_days
            df = loader.fetch_range(symbol, days=effective_days_f)
            db_count = loader.save_to_db(df, db_days=db_days)
            gcs_result = loader.save_to_gcs(df, symbol)
            results[f"funding_{symbol}"] = {
                "total_records": len(df),
                "db_rows": db_count,
                "gcs": gcs_result,
            }

    # ── Telegram Messages ──
    if mode in ("all", "telegram"):
        loader = TelegramBulkLoader()
        latest = _get_latest_timestamp("telegram_messages")
        if latest:
            elapsed = (datetime.now(timezone.utc) - latest).total_seconds() / 86400
            effective_days_t = max(1, int(elapsed) + 1)
            logger.info(f"[RESUME] Telegram: DB has data up to {latest}, fetching last {effective_days_t} days")
        else:
            effective_days_t = effective_telegram_days
        count = loader.load(days=effective_days_t)
        results["telegram"] = {"messages": count}

    # ── Fear & Greed Index ──
    if mode in ("all", "fear_greed"):
        loader = FearGreedBulkLoader()
        latest = _get_latest_timestamp("fear_greed_data")
        if latest:
            elapsed = (datetime.now(timezone.utc) - latest).total_seconds() / 86400
            effective_days_fg = max(1, int(elapsed) + 1)
            logger.info(f"[RESUME] Fear & Greed: DB has data up to {latest}, fetching last {effective_days_fg} days")
        else:
            effective_days_fg = effective_fear_greed_days
        df = loader.fetch_range(days=effective_days_fg)
        count = loader.save_to_db(df)
        results["fear_greed"] = {"total_records": len(df), "db_rows": count}

    # ── Deribit DVOL History ──
    if mode in ("all", "deribit_dvol"):
        loader = DeribitDVOLBulkLoader()
        deribit_currencies = [s[:-4] for s in symbols if s.endswith("USDT")
                              and s[:-4] in ("BTC", "ETH")]
        for currency in deribit_currencies:
            latest = _get_latest_timestamp("deribit_data", symbol=currency)
            if latest:
                elapsed = (datetime.now(timezone.utc) - latest).total_seconds() / 86400
                effective_days_d = max(1, int(elapsed) + 1)
                logger.info(f"[RESUME] Deribit DVOL {currency}: DB has data up to {latest}, fetching last {effective_days_d} days")
            else:
                effective_days_d = effective_deribit_days
            df = loader.fetch_range(currency, days=effective_days_d)
            count = loader.save_to_db(df)
            results[f"deribit_dvol_{currency}"] = {"total_records": len(df), "db_rows": count}

    # ── Macro (FRED + yfinance) ──
    if mode in ("all", "macro"):
        loader = MacroBulkLoader()
        latest = _get_latest_timestamp("macro_data")
        if latest:
            elapsed = (datetime.now(timezone.utc) - latest).total_seconds() / 86400
            effective_days_m = max(1, int(elapsed) + 1)
            logger.info(f"[RESUME] Macro: DB has data up to {latest}, fetching last {effective_days_m} days")
        else:
            effective_days_m = effective_macro_days
        df = loader.build_snapshots(days=effective_days_m)
        count = loader.save_to_db(df)
        results["macro"] = {"total_records": len(df), "db_rows": count}

    # ── Resample (generate 1h/4h/1d/1w from 1m in GCS) ──
    if mode in ("all", "resample"):
        uploader = ResampleUploader()
        for symbol in symbols:
            res = uploader.resample_and_upload(symbol)
            results[f"resample_{symbol}"] = res

    logger.info(f"=== COLD START COMPLETE ===")
    for key, val in results.items():
        logger.info(f"  {key}: {val}")

    return results


# ─────────────── CLI ───────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cold Start Data Loader")
    parser.add_argument("--mode", default="all",
                        choices=["all", "ohlcv", "funding", "telegram", "resample",
                                 "upbit", "fear_greed", "deribit_dvol", "macro"],
                        help="What to load (default: all)")
    _default_days = (datetime.now(timezone.utc) - datetime(2020, 1, 1, tzinfo=timezone.utc)).days
    parser.add_argument("--days", type=int, default=_default_days,
                        help=f"Days of history for all loaders (default: since 2020-01-01 = {_default_days} days)")
    parser.add_argument("--symbol", type=str, default=None,
                        help="Single symbol e.g. BTCUSDT (default: BTCUSDT + ETHUSDT)")
    parser.add_argument("--db-days", type=int, default=7,
                        help="Days to keep in PostgreSQL hot cache (rest → GCS)")
    parser.add_argument("--ohlcv-days", type=int, default=None,
                        help="Override --days for OHLCV loader")
    parser.add_argument("--funding-days", type=int, default=None,
                        help="Override --days for Funding loader")
    parser.add_argument("--telegram-days", type=int, default=None,
                        help="Override --days for Telegram loader")
    parser.add_argument("--upbit-days", type=int, default=None,
                        help="Override --days for Upbit loader")
    parser.add_argument("--fear-greed-days", type=int, default=None,
                        help="Override --days for Fear & Greed loader")
    parser.add_argument("--deribit-days", type=int, default=None,
                        help="Override --days for Deribit DVOL loader")
    parser.add_argument("--macro-days", type=int, default=None,
                        help="Override --days for Macro (FRED + yfinance) loader")

    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else None
    run_cold_start(
        mode=args.mode,
        days=args.days,
        symbols=symbols,
        db_days=args.db_days,
        ohlcv_days=args.ohlcv_days,
        funding_days=args.funding_days,
        telegram_days=args.telegram_days,
        upbit_days=args.upbit_days,
        fear_greed_days=args.fear_greed_days,
        deribit_days=args.deribit_days,
        macro_days=args.macro_days,
    )
