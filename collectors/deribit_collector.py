"""Deribit Options Data Collector.

Collects from public Deribit REST API (no API key required):
- DVOL: BTC/ETH 30-day implied volatility index (crypto VIX equivalent)
- PCR:  Put/Call Ratio by OI and by volume
- IV Term Structure: average ATM IV across expiry buckets (1w/2w/1m/3m/6m)
- 25-delta Skew: put_avg_IV - call_avg_IV per bucket
  (positive = fear/downside hedging, negative = greed/upside demand)

Rate limit: Deribit public API is generous, no auth needed.
"""

import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from config.settings import settings
from config.database import db
from loguru import logger

DERIBIT_URL = "https://www.deribit.com/api/v2/public"


class DeribitCollector:
    def __init__(self):
        self.enabled = getattr(settings, 'DERIBIT_ENABLED', True)
        # Deribit options exist only for BTC and ETH (exchange constraint)
        self.currencies = settings.deribit_currencies
        self._session = requests.Session()
        self._session.headers.update({'Accept': 'application/json'})

    # ─────────────── Raw API Calls ───────────────

    def _get(self, method: str, params: Dict = None) -> Optional[object]:
        """Generic GET wrapper for Deribit public API."""
        try:
            resp = self._session.get(
                f"{DERIBIT_URL}/{method}",
                params=params or {},
                timeout=15,
            )
            resp.raise_for_status()
            body = resp.json()
            if body.get('error'):
                logger.warning(f"Deribit API error for {method}: {body['error']}")
                return None
            return body.get('result')
        except Exception as e:
            logger.warning(f"Deribit request failed ({method}): {e}")
            return None

    def fetch_dvol(self, currency: str) -> Optional[float]:
        """Fetch latest DVOL (30-day IV index) close value."""
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ms = now_ms - 2 * 3600 * 1000  # last 2 hours to ensure data
        result = self._get('get_volatility_index_data', {
            'currency': currency,
            'start_timestamp': start_ms,
            'end_timestamp': now_ms,
            'resolution': '60',  # hourly candles
        })
        # result['data'] = [[ts_ms, open, high, low, close], ...]
        if result and isinstance(result.get('data'), list) and result['data']:
            last = result['data'][-1]
            return round(float(last[4]), 4)  # close
        return None

    def fetch_index_price(self, currency: str) -> Optional[float]:
        """Fetch current Deribit index price (spot proxy)."""
        result = self._get('get_index_price', {'index_name': f'{currency.lower()}_usd'})
        if result:
            return float(result.get('index_price', 0) or 0)
        return None

    def fetch_options_summary(self, currency: str) -> List[Dict]:
        """Fetch all option book summaries for a currency (single call)."""
        result = self._get('get_book_summary_by_currency', {
            'currency': currency,
            'kind': 'option',
        })
        return result if isinstance(result, list) else []

    # ─────────────── Expiry Bucketing ───────────────

    def _expiry_bucket(self, exp_ts_ms: int) -> Optional[str]:
        """Classify an expiry timestamp into a term bucket."""
        now = datetime.now(timezone.utc)
        exp = datetime.fromtimestamp(exp_ts_ms / 1000, tz=timezone.utc)
        days = (exp - now).days
        if days < 1:
            return None
        if days <= 10:
            return '1w'
        if days <= 18:
            return '2w'
        if days <= 45:
            return '1m'
        if days <= 100:
            return '3m'
        if days <= 200:
            return '6m'
        return None

    # ─────────────── PCR ───────────────

    def calculate_pcr(self, summaries: List[Dict]) -> Dict:
        """Calculate Put/Call Ratio by OI and by volume."""
        put_oi = call_oi = put_vol = call_vol = 0.0

        for s in summaries:
            name = s.get('instrument_name', '')
            oi  = float(s.get('open_interest', 0) or 0)
            vol = float(s.get('volume', 0) or 0)
            if name.endswith('-P'):
                put_oi  += oi
                put_vol += vol
            elif name.endswith('-C'):
                call_oi  += oi
                call_vol += vol

        return {
            'pcr_oi':   round(put_oi  / call_oi,  4) if call_oi  > 0 else None,
            'pcr_vol':  round(put_vol / call_vol, 4) if call_vol > 0 else None,
            'put_oi':   round(put_oi,  4),
            'call_oi':  round(call_oi, 4),
            'put_vol':  round(put_vol,  6),
            'call_vol': round(call_vol, 6),
        }

    # ─────────────── IV Term Structure ───────────────

    def calculate_term_structure(self, summaries: List[Dict]) -> Dict:
        """Build IV term structure: average mark_iv per expiry bucket.

        Deribit's mark_iv is the model IV for each option — averaging across
        all strikes gives a crude ATM proxy that is sufficient for directional bias.
        """
        buckets: Dict[str, List[float]] = {}

        for s in summaries:
            exp_ts = s.get('expiration_timestamp')
            iv = s.get('mark_iv')
            if not exp_ts or iv is None:
                continue
            b = self._expiry_bucket(int(exp_ts))
            if not b:
                continue
            buckets.setdefault(b, []).append(float(iv))

        term: Dict = {}
        for b, ivs in buckets.items():
            if ivs:
                term[b] = round(sum(ivs) / len(ivs), 4)

        # Inversion = front-end IV > back-end IV → near-term panic (potential bottom)
        order = ['1w', '2w', '1m', '3m', '6m']
        vals = [term[k] for k in order if term.get(k) is not None]
        term['inverted'] = bool(len(vals) >= 2 and vals[0] > vals[-1])

        return term

    # ─────────────── 25-delta Skew ───────────────

    def calculate_skew(self, summaries: List[Dict], spot: float) -> Dict:
        """Calculate approx 25-delta skew per expiry bucket.

        25d skew = put_IV - call_IV
        Positive → puts more expensive → market fears downside (bearish)
        Negative → calls more expensive → market bets on upside (bullish)

        Approximation: use options with strike within ±10-25% of spot
        as a proxy for the 25-delta wing.
        """
        if not spot or spot <= 0:
            return {}

        lower_bound = spot * 0.75
        upper_bound = spot * 1.25
        bucket_data: Dict[str, Dict[str, List[float]]] = {}

        for s in summaries:
            name = s.get('instrument_name', '')
            exp_ts = s.get('expiration_timestamp')
            iv = s.get('mark_iv')
            if not exp_ts or iv is None:
                continue

            # Parse strike from name: CURRENCY-DDMONYY-STRIKE-C/P
            parts = name.split('-')
            if len(parts) < 4:
                continue
            try:
                strike = float(parts[2])
            except ValueError:
                continue

            if not (lower_bound <= strike <= upper_bound):
                continue

            b = self._expiry_bucket(int(exp_ts))
            if not b:
                continue

            opt_type = 'put' if name.endswith('-P') else 'call'
            bucket_data.setdefault(b, {'put': [], 'call': []})
            bucket_data[b][opt_type].append(float(iv))

        skew: Dict = {}
        for b, data in bucket_data.items():
            puts  = data.get('put', [])
            calls = data.get('call', [])
            if puts and calls:
                skew[f'skew_{b}'] = round(sum(puts)/len(puts) - sum(calls)/len(calls), 4)

        return skew

    # ─────────────── Main Collection ───────────────

    def collect_for_currency(self, currency: str) -> Optional[Dict]:
        """Collect all Deribit options signals for one currency."""
        try:
            dvol      = self.fetch_dvol(currency)
            spot      = self.fetch_index_price(currency)
            summaries = self.fetch_options_summary(currency)

            if not summaries:
                logger.warning(f"Deribit: no options data for {currency}")
                return None

            pcr  = self.calculate_pcr(summaries)
            term = self.calculate_term_structure(summaries)
            skew = self.calculate_skew(summaries, spot or 0.0)

            now = datetime.now(timezone.utc).replace(second=0, microsecond=0).isoformat()

            record = {
                'symbol':       currency,
                'timestamp':    now,
                'dvol':         dvol,
                'spot_price':   round(spot, 4) if spot else None,
                # PCR
                **pcr,
                # Term Structure
                'iv_1w':        term.get('1w'),
                'iv_2w':        term.get('2w'),
                'iv_1m':        term.get('1m'),
                'iv_3m':        term.get('3m'),
                'iv_6m':        term.get('6m'),
                'term_inverted': term.get('inverted', False),
                # Skew
                'skew_1w':      skew.get('skew_1w'),
                'skew_2w':      skew.get('skew_2w'),
                'skew_1m':      skew.get('skew_1m'),
                'skew_3m':      skew.get('skew_3m'),
            }
            return record
        except Exception as e:
            logger.error(f"Deribit collection error ({currency}): {e}")
            return None

    def run(self) -> None:
        if not self.enabled:
            return
        for currency in self.currencies:
            data = self.collect_for_currency(currency)
            if data:
                try:
                    db.upsert_deribit_data(data)
                    logger.info(
                        f"Deribit {currency}: DVOL={data.get('dvol')} "
                        f"PCR_OI={data.get('pcr_oi')} "
                        f"IV_1m={data.get('iv_1m')} "
                        f"Skew_1m={data.get('skew_1m')}"
                    )
                except Exception as e:
                    logger.error(f"Deribit DB save error ({currency}): {e}")


deribit_collector = DeribitCollector()
