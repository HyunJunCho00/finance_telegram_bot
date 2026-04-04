"""
Binance Futures Positioning Collector (API key-free).

Uses Binance public futures endpoints — no authentication required:
  - Global Long/Short Account Ratio (all retail accounts on Binance Futures)
  - Top Trader Long/Short Ratio (large accounts / whales)
  - Open Interest (USD notional)

Signal interpretation:
  LSR > 1.0  → more longs than shorts → crowded long → reversal risk
  LSR < 1.0  → more shorts than longs → crowded short → squeeze risk
  Extreme readings (>1.5 or <0.7) are historically mean-reverting.

Stores to narrative_data (source="coinglass_positioning") per symbol.
Note: source key kept as "coinglass_positioning" for orchestrator compatibility.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

import requests
from loguru import logger

from config.database import db
from config.settings import settings

_BASE = "https://fapi.binance.com"
_TIMEOUT = 20
_HEADERS = {"User-Agent": "finance-bot/1.0", "Accept": "application/json"}

_PERIOD = "4h"   # 4h granularity matches analysis cycle


class BinancePositioningCollector:
    """Fetches LSR + OI from Binance public futures API (no API key needed)."""

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)

    # ------------------------------------------------------------------
    # Fetch helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: Dict) -> Optional[object]:
        try:
            resp = self._session.get(f"{_BASE}{path}", params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"BinancePositioning: request failed {path}: {e}")
            return None

    def _fetch_global_lsr(self, symbol: str) -> Optional[Dict]:
        """Global long/short account ratio — all retail accounts."""
        data = self._get(
            "/futures/data/globalLongShortAccountRatio",
            {"symbol": symbol, "period": _PERIOD, "limit": 1},
        )
        if not data or not isinstance(data, list):
            return None
        row = data[0]
        try:
            ratio = float(row["longShortRatio"])
            long_pct = float(row["longAccount"]) * 100
            short_pct = float(row["shortAccount"]) * 100
            return {
                "long_short_ratio": round(ratio, 4),
                "long_pct": round(long_pct, 2),
                "short_pct": round(short_pct, 2),
                "timestamp_ms": int(row.get("timestamp", 0)),
            }
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"BinancePositioning: LSR parse error {symbol}: {e}")
            return None

    def _fetch_top_trader_lsr(self, symbol: str) -> Optional[Dict]:
        """Top trader long/short ratio — large accounts (whale proxy)."""
        data = self._get(
            "/futures/data/topLongShortAccountRatio",
            {"symbol": symbol, "period": _PERIOD, "limit": 1},
        )
        if not data or not isinstance(data, list):
            return None
        row = data[0]
        try:
            ratio = float(row["longShortRatio"])
            long_pct = float(row["longAccount"]) * 100
            short_pct = float(row["shortAccount"]) * 100
            return {
                "long_short_ratio": round(ratio, 4),
                "long_pct": round(long_pct, 2),
                "short_pct": round(short_pct, 2),
            }
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"BinancePositioning: top-trader LSR parse error {symbol}: {e}")
            return None

    def _fetch_oi(self, symbol: str) -> Optional[Dict]:
        """Current open interest in USD."""
        data = self._get("/fapi/v1/openInterest", {"symbol": symbol})
        if not data or not isinstance(data, dict):
            return None
        try:
            oi = float(data["openInterest"])
            # OI from this endpoint is in contracts (BTC or ETH qty), not USD
            # Get mark price to convert
            ticker = self._get("/fapi/v1/premiumIndex", {"symbol": symbol})
            mark_price = float(ticker.get("markPrice", 0)) if ticker else 0.0
            oi_usd = oi * mark_price if mark_price > 0 else None
            return {
                "open_interest_contracts": round(oi, 4),
                "open_interest_usd": round(oi_usd, 2) if oi_usd else None,
                "mark_price": round(mark_price, 2),
            }
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"BinancePositioning: OI parse error {symbol}: {e}")
            return None

    # ------------------------------------------------------------------
    # Format
    # ------------------------------------------------------------------

    def _lsr_bias(self, ratio: float) -> str:
        if ratio > 1.5:
            return "EXTREME-LONG (squeeze risk)"
        if ratio > 1.0:
            return "LONG-HEAVY"
        if ratio < 0.7:
            return "EXTREME-SHORT (squeeze risk)"
        if ratio < 1.0:
            return "SHORT-HEAVY"
        return "NEUTRAL"

    def _format_summary(self, symbol: str, global_lsr: Optional[Dict],
                        top_lsr: Optional[Dict], oi: Optional[Dict]) -> str:
        asset = symbol.replace("USDT", "")
        parts = [f"[BINANCE_POSITIONING {asset}]"]

        if global_lsr:
            bias = self._lsr_bias(global_lsr["long_short_ratio"])
            parts.append(
                f"Global LSR={global_lsr['long_short_ratio']:.2f} "
                f"({global_lsr['long_pct']:.1f}%L/{global_lsr['short_pct']:.1f}%S → {bias})"
            )
        if top_lsr:
            top_bias = self._lsr_bias(top_lsr["long_short_ratio"])
            parts.append(
                f"Whale LSR={top_lsr['long_short_ratio']:.2f} "
                f"({top_lsr['long_pct']:.1f}%L → {top_bias})"
            )
        if oi and oi.get("open_interest_usd"):
            oi_bn = oi["open_interest_usd"] / 1e9
            parts.append(f"OI=${oi_bn:.2f}B")

        return " | ".join(parts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect_symbol(self, symbol: str) -> Optional[Dict]:
        global_lsr = self._fetch_global_lsr(symbol)
        top_lsr = self._fetch_top_trader_lsr(symbol)
        oi = self._fetch_oi(symbol)

        if global_lsr is None and top_lsr is None and oi is None:
            logger.warning(f"BinancePositioning: no data for {symbol}")
            return None

        summary = self._format_summary(symbol, global_lsr, top_lsr, oi)
        result = {
            "symbol": symbol,
            "global_lsr": global_lsr,
            "top_trader_lsr": top_lsr,
            "open_interest": oi,
            "collected_at": datetime.now(timezone.utc).isoformat(),
        }

        now = datetime.now(timezone.utc).replace(second=0, microsecond=0).isoformat()
        db.upsert_narrative_data({
            "timestamp": now,
            "symbol": symbol,
            "source": "coinglass_positioning",  # kept for orchestrator compat
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
        logger.info(f"BinancePositioning: {summary}")
        return result

    def run(self) -> Dict[str, Optional[Dict]]:
        results: Dict[str, Optional[Dict]] = {}
        for symbol in settings.trading_symbols:
            try:
                results[symbol] = self.collect_symbol(symbol)
            except Exception as e:
                logger.error(f"BinancePositioning: failed for {symbol}: {e}")
                results[symbol] = None
        return results


coinglass_collector = BinancePositioningCollector()
