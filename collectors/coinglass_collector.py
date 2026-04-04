"""
CoinGlass Institutional Positioning Collector.

Fetches per-exchange aggregated data not available from free sources:
  - Long/Short account ratio (% of accounts long vs short)
  - Futures Open Interest by exchange

These are institutional positioning signals: extreme LSR values (>0.6 long
or <0.4 long) correlate with crowded trades subject to reversal.

Requires COINGLASS_API_KEY in .env.
  Sign up free at: https://www.coinglass.com/pricing (free tier: 100 req/day)

API v1 (free tier):
  Base: https://open-api.coinglass.com/public/v1/
  LSR: GET /lsr?symbol=BTC&interval=4h&limit=1
  OI:  GET /openInterest?symbol=BTC&interval=4h&limit=1

Stores to narrative_data (source="coinglass_positioning") per symbol.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

import requests
from loguru import logger

from config.database import db
from config.settings import settings

_BASE_URL = "https://open-api.coinglass.com/public/v1"
_TIMEOUT = 20
_HEADERS = {
    "User-Agent": "finance-bot/1.0",
    "Accept": "application/json",
}

_SYMBOL_TO_ASSET = {
    "BTCUSDT": "BTC",
    "ETHUSDT": "ETH",
}


class CoinglassCollector:
    """Fetches LSR and OI from CoinGlass API (requires free API key)."""

    def __init__(self):
        self._api_key: str = getattr(settings, "COINGLASS_API_KEY", "")
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)

    def _is_enabled(self) -> bool:
        if not self._api_key:
            logger.debug("CoinglassCollector: COINGLASS_API_KEY not set, skipping")
            return False
        return True

    def _get(self, endpoint: str, params: Dict) -> Optional[Dict]:
        params["apiKey"] = self._api_key
        url = f"{_BASE_URL}/{endpoint}"
        try:
            resp = self._session.get(url, params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
            body = resp.json()
            if not body.get("success", True):
                logger.warning(f"CoinglassCollector: API error {endpoint}: {body.get('msg')}")
                return None
            return body.get("data")
        except Exception as e:
            logger.warning(f"CoinglassCollector: request failed {endpoint}: {e}")
            return None

    def _fetch_lsr(self, asset: str) -> Optional[Dict]:
        """Long/Short account ratio: latest 4h candle."""
        data = self._get("lsr", {"symbol": asset, "interval": "4h", "limit": 1})
        if not data or not isinstance(data, list) or not data:
            return None
        latest = data[-1]
        try:
            long_pct = float(latest.get("longAccount", 0)) * 100
            short_pct = float(latest.get("shortAccount", 0)) * 100
            ratio = float(latest.get("longShortRatio", 0))
            return {
                "long_pct": round(long_pct, 2),
                "short_pct": round(short_pct, 2),
                "long_short_ratio": round(ratio, 4),
                "timestamp_ms": latest.get("createTime"),
            }
        except (TypeError, ValueError) as e:
            logger.warning(f"CoinglassCollector: LSR parse error for {asset}: {e}")
            return None

    def _fetch_oi(self, asset: str) -> Optional[Dict]:
        """Open interest (USD) across all exchanges."""
        data = self._get("openInterest", {"symbol": asset, "interval": "4h", "limit": 1})
        if not data or not isinstance(data, list) or not data:
            return None
        latest = data[-1]
        try:
            oi_usd = float(latest.get("openInterest", 0))
            return {
                "open_interest_usd": round(oi_usd, 2),
                "timestamp_ms": latest.get("createTime"),
            }
        except (TypeError, ValueError) as e:
            logger.warning(f"CoinglassCollector: OI parse error for {asset}: {e}")
            return None

    def _format_summary(self, asset: str, lsr: Optional[Dict], oi: Optional[Dict]) -> str:
        parts = [f"[COINGLASS {asset}]"]
        if lsr:
            bias = "LONG-HEAVY" if lsr["long_pct"] > 60 else ("SHORT-HEAVY" if lsr["long_pct"] < 40 else "NEUTRAL")
            parts.append(f"LSR={lsr['long_short_ratio']:.2f} ({lsr['long_pct']:.1f}%L/{lsr['short_pct']:.1f}%S → {bias})")
        if oi:
            oi_bn = oi["open_interest_usd"] / 1e9
            parts.append(f"OI=${oi_bn:.2f}B")
        return " | ".join(parts)

    def collect_symbol(self, binance_symbol: str) -> Optional[Dict]:
        if not self._is_enabled():
            return None
        asset = _SYMBOL_TO_ASSET.get(binance_symbol)
        if not asset:
            return None

        lsr = self._fetch_lsr(asset)
        oi = self._fetch_oi(asset)

        if lsr is None and oi is None:
            return None

        summary = self._format_summary(asset, lsr, oi)
        result = {
            "asset": asset,
            "symbol": binance_symbol,
            "lsr": lsr,
            "open_interest": oi,
            "collected_at": datetime.now(timezone.utc).isoformat(),
        }

        now = datetime.now(timezone.utc).replace(second=0, microsecond=0).isoformat()
        db.upsert_narrative_data({
            "timestamp": now,
            "symbol": binance_symbol,
            "source": "coinglass_positioning",
            "status": "ok",
            "sentiment": "neutral",
            "summary": summary,
            "reasoning": "",
            "macro_context": "",
            "key_events": [],
            "bullish_factors": [],
            "bearish_factors": [],
            "sources": [],
            "raw_payload": result,
        })
        logger.info(f"CoinglassCollector: {summary}")
        return result

    def run(self) -> Dict[str, Optional[Dict]]:
        results: Dict[str, Optional[Dict]] = {}
        for symbol in settings.trading_symbols:
            try:
                results[symbol] = self.collect_symbol(symbol)
            except Exception as e:
                logger.error(f"CoinglassCollector: failed for {symbol}: {e}")
                results[symbol] = None
        return results


coinglass_collector = CoinglassCollector()
