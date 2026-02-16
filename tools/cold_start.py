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
from loguru import logger

# Add project root to path
sys.path.insert(0, ".")

from config.settings import settings
from config.database import db
from processors.gcs_parquet import gcs_parquet_store


# ─────────────── OHLCV Bulk Loader ───────────────

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

        while current_since < end_ms:
            try:
                ohlcv = self.exchange.fetch_ohlcv(upbit_symbol, '1m',
                                                   since=current_since, limit=200)
                if not ohlcv:
                    break

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

                current_since = ohlcv[-1][0] + 60000
                if len(ohlcv) < 200:
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


# ─────────────── Main Entry Point ───────────────

def run_cold_start(mode: str = "all", days: int = 210,
                   symbols: Optional[List[str]] = None,
                   db_days: int = 3,
                   ohlcv_days: Optional[int] = None,
                   funding_days: Optional[int] = None,
                   telegram_days: Optional[int] = None,
                   upbit_days: Optional[int] = None):
    """Run cold start data loading.

    Args:
        mode: "all", "ohlcv", "funding", "telegram", "resample", "upbit"
        days: How many days of history to fetch
        symbols: List of symbols (default: ["BTCUSDT", "ETHUSDT"])
        db_days: How many recent days to keep in PostgreSQL
        ohlcv_days: Days override for OHLCV loader (default: days)
        funding_days: Days override for funding loader (default: days)
        telegram_days: Days override for Telegram loader (default: min(days, 90))
        upbit_days: Days override for Upbit loader (default: days)
    """
    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT"]

    logger.info(f"=== COLD START: mode={mode}, days={days}, symbols={symbols} ===")
    results = {}

    effective_ohlcv_days = ohlcv_days if ohlcv_days is not None else days
    effective_funding_days = funding_days if funding_days is not None else days
    effective_telegram_days = telegram_days if telegram_days is not None else min(days, 90)
    effective_upbit_days = upbit_days if upbit_days is not None else days

    # ── OHLCV (Binance Futures) ──
    if mode in ("all", "ohlcv"):
        loader = OHLCVBulkLoader()
        for symbol in symbols:
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=effective_ohlcv_days)
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
            df = loader.fetch_range(symbol, effective_upbit_days)
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
            df = loader.fetch_range(symbol, days=effective_funding_days)
            db_count = loader.save_to_db(df, db_days=7)
            gcs_result = loader.save_to_gcs(df, symbol)
            results[f"funding_{symbol}"] = {
                "total_records": len(df),
                "db_rows": db_count,
                "gcs": gcs_result,
            }

    # ── Telegram Messages ──
    if mode in ("all", "telegram"):
        loader = TelegramBulkLoader()
        count = loader.load(days=effective_telegram_days)
        results["telegram"] = {"messages": count}

    # ── Resample (generate 1h/4h/1d/1w from 1m) ──
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
                        choices=["all", "ohlcv", "funding", "telegram", "resample", "upbit"],
                        help="What to load")
    parser.add_argument("--days", type=int, default=210,
                        help="Days of history (default: 210 for 1d EMA200)")
    parser.add_argument("--symbol", type=str, default=None,
                        help="Single symbol (default: BTCUSDT + ETHUSDT)")
    parser.add_argument("--db-days", type=int, default=3,
                        help="Days to keep in PostgreSQL hot cache")
    parser.add_argument("--ohlcv-days", type=int, default=None,
                        help="Override days only for OHLCV loader")
    parser.add_argument("--funding-days", type=int, default=None,
                        help="Override days only for funding loader")
    parser.add_argument("--telegram-days", type=int, default=None,
                        help="Override days only for Telegram loader")
    parser.add_argument("--upbit-days", type=int, default=None,
                        help="Override days only for Upbit loader")

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
    )
