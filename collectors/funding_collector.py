"""Funding & Global OI collector.

Collects from 3 major exchanges (Binance, Bybit, OKX) via ccxt:
- Funding rates (Binance primary)
- Global Open Interest = sum of Binance + Bybit + OKX OI in USD
- Long/Short ratio (Binance)

Global OI eliminates single-exchange noise and measures total market energy.
OI-price divergence is a key professional signal.

Historical data availability (Binance):
- Funding rates : Full history via /fapi/v1/fundingRate (8h intervals, back to 2019)
- Open Interest : Only last ~30 days via /futures/data/openInterestHist (API limit)
  → OI beyond 30 days is not available from free Binance API.
  → Use backfill_funding_history() to populate GCS Parquet for long-term charts.
"""

import ccxt
import requests
import time
import pandas as pd
from datetime import datetime, timezone, timedelta
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
            'open_interest_okx': 0, # zeroed out to maintain DB compatibility
        }

    def fetch_long_short_ratio(self, symbol: str) -> Dict:
        try:
            import requests
            binance_symbol = symbol.replace('/', '')
            url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
            params = {'symbol': binance_symbol, 'period': '5m', 'limit': 1}
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data and len(data) > 0:
                return {
                    'long_short_ratio': float(data[0].get('longShortRatio', 1.0)),
                    'long_account': float(data[0].get('longAccount', 0.5)),
                    'short_account': float(data[0].get('shortAccount', 0.5)),
                }
        except Exception as e:
            logger.warning(f"Long/short ratio fetch error for {symbol}: {e}")
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

    # ─────────────── Historical Backfill (one-time / on-demand) ───────────────

    def backfill_funding_history(self,
                                  symbol: str,
                                  start_date: str = "2021-01-01",
                                  dry_run: bool = False) -> Dict:
        """Backfill full Binance funding rate history into GCS Parquet.

        Binance provides 8h-interval funding rates back to contract inception:
          - BTCUSDT: ~2019-09-10
          - ETHUSDT: ~2021-05-12
        This data is NOT limited by DB retention — written directly to GCS.

        OI history: Binance only provides ~30 days via openInterestHist API.
        This method fetches OI for whatever Binance returns and stores alongside
        the funding rates so the chart panels show real data for recent months.

        Args:
            symbol:     e.g. "BTC/USDT" (slash format)
            start_date: ISO date string "YYYY-MM-DD"
            dry_run:    If True, fetch and log only, do not write to GCS.

        Returns:
            summary dict with row counts and GCS paths written.
        """
        from processors.gcs_parquet import gcs_parquet_store

        if not gcs_parquet_store.enabled:
            logger.warning("GCS archive disabled — set ENABLE_GCS_ARCHIVE=true")
            return {"error": "gcs_disabled"}

        binance_sym = symbol.replace("/", "")  # "BTCUSDT"
        start_ms = int(datetime.fromisoformat(start_date)
                       .replace(tzinfo=timezone.utc).timestamp() * 1000)
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        # ── 1. Fetch paginated funding rates ──────────────────────────────────
        logger.info(f"[Backfill] Fetching funding rate history for {binance_sym} from {start_date}")
        url = "https://fapi.binance.com/fapi/v1/fundingRate"
        all_funding: List[Dict] = []
        page_start = start_ms

        while page_start < now_ms:
            try:
                resp = requests.get(url, params={
                    "symbol": binance_sym,
                    "startTime": page_start,
                    "limit": 1000,
                }, timeout=15)
                resp.raise_for_status()
                rows = resp.json()
            except Exception as e:
                logger.error(f"[Backfill] Funding fetch error at {page_start}: {e}")
                break

            if not rows:
                break

            all_funding.extend(rows)
            last_ts = rows[-1]["fundingTime"]
            if last_ts <= page_start or len(rows) < 1000:
                break
            page_start = last_ts + 1
            time.sleep(0.3)  # respect rate limit

        logger.info(f"[Backfill] Fetched {len(all_funding)} funding rate rows for {binance_sym}")

        # ── 2. Fetch historical OI (Binance: ~30 days available, 1h interval) ─
        logger.info(f"[Backfill] Fetching OI history for {binance_sym} (last 30 days max)")
        oi_url = "https://fapi.binance.com/futures/data/openInterestHist"
        all_oi: List[Dict] = []
        oi_start = now_ms - (30 * 24 * 3600 * 1000)  # 30 days back

        while oi_start < now_ms:
            try:
                resp = requests.get(oi_url, params={
                    "symbol": binance_sym,
                    "period": "1h",
                    "startTime": oi_start,
                    "limit": 500,
                }, timeout=15)
                resp.raise_for_status()
                rows = resp.json()
            except Exception as e:
                logger.error(f"[Backfill] OI fetch error at {oi_start}: {e}")
                break

            if not rows or not isinstance(rows, list):
                break

            all_oi.extend(rows)
            last_ts = rows[-1]["timestamp"]
            if last_ts <= oi_start or len(rows) < 500:
                break
            oi_start = last_ts + 1
            time.sleep(0.3)

        logger.info(f"[Backfill] Fetched {len(all_oi)} OI rows for {binance_sym}")

        # ── 3. Build merged DataFrame (funding + OI joined by hour) ──────────
        if not all_funding:
            return {"error": "no_funding_data"}

        df_fund = pd.DataFrame(all_funding)
        df_fund["timestamp"] = pd.to_datetime(df_fund["fundingTime"], unit="ms", utc=True)
        df_fund["funding_rate"] = df_fund["fundingRate"].astype(float)
        df_fund["symbol"] = binance_sym
        df_fund = df_fund[["symbol", "timestamp", "funding_rate"]].copy()

        # OI: build lookup dict {hour_floor_ms: open_interest}
        oi_lookup: Dict[pd.Timestamp, float] = {}
        if all_oi:
            df_oi = pd.DataFrame(all_oi)
            df_oi["ts"] = pd.to_datetime(df_oi["timestamp"], unit="ms", utc=True).dt.floor("h")
            df_oi["oi"] = df_oi["sumOpenInterestValue"].astype(float)
            oi_lookup = dict(zip(df_oi["ts"], df_oi["oi"]))

        # Attach OI to funding rows (nearest hour, 0 if not available)
        df_fund["open_interest"] = df_fund["timestamp"].dt.floor("h").map(oi_lookup).fillna(0)
        df_fund["open_interest_value"] = df_fund["open_interest"]

        # ── 4. Partition by month and write to GCS ───────────────────────────
        df_fund["month"] = df_fund["timestamp"].dt.strftime("%Y-%m")
        paths_written = []
        total_rows = 0

        for (sym, month), group in df_fund.groupby(["symbol", "month"]):
            path = f"funding/{sym}/{month}.parquet"
            rows_in_group = len(group)

            if dry_run:
                logger.info(f"[DRY RUN] Would write {rows_in_group} rows → {path}")
            else:
                g = group.drop(columns=["month"]).copy()
                size = gcs_parquet_store._merge_upload_parquet(
                    path, g, dedup_cols=["timestamp"]
                )
                logger.info(f"[Backfill] Written {rows_in_group} rows → gs://{gcs_parquet_store.bucket_name}/{path} ({size:,} bytes)")
                paths_written.append(path)

            total_rows += rows_in_group

        return {
            "symbol": binance_sym,
            "funding_rows": len(all_funding),
            "oi_rows": len(all_oi),
            "months_written": len(paths_written),
            "total_rows": total_rows,
            "paths": paths_written,
            "dry_run": dry_run,
        }

    def backfill_all_symbols(self, start_date: str = "2021-01-01",
                              dry_run: bool = False) -> Dict:
        """Run backfill for all configured trading symbols."""
        results = {}
        for sym in self.symbols:
            logger.info(f"[Backfill] Starting {sym}")
            results[sym] = self.backfill_funding_history(sym, start_date=start_date,
                                                          dry_run=dry_run)
        return results


funding_collector = FundingCollector()
