"""
Stablecoin Supply Collector — DefiLlama free API.

Tracks USDT + USDC circulating supply and 7-day changes.
Rising supply = new fiat capital entering crypto (bullish liquidity).
Declining supply = capital exiting or moving to risk-off (bearish).

Endpoint: https://stablecoins.llama.fi/stablecoins?includePrices=true
Free, no API key required.

Stores to narrative_data (source="stablecoin_defillama") for each trading symbol
so the orchestrator can inject it into the agent context.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests
from loguru import logger

from config.database import db
from config.settings import settings

_API_URL = "https://stablecoins.llama.fi/stablecoins?includePrices=true"
_TIMEOUT = 20
_TARGETS = {"usdt", "usdc"}  # symbols to track (lowercase)

_HEADERS = {
    "User-Agent": "Mozilla/5.0 finance-bot/1.0",
    "Accept": "application/json",
}


class StablecoinCollector:
    """Fetches USDT + USDC circulating supply snapshot from DefiLlama."""

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------

    def _fetch_raw(self) -> Optional[List[Dict]]:
        try:
            resp = self._session.get(_API_URL, timeout=_TIMEOUT)
            resp.raise_for_status()
            return resp.json().get("peggedAssets", [])
        except Exception as e:
            logger.warning(f"StablecoinCollector: fetch failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Parse
    # ------------------------------------------------------------------

    def _extract(self, assets: List[Dict]) -> Dict:
        """Pull USDT/USDC circulating USD and pct changes from response."""
        result: Dict[str, Dict] = {}
        for asset in assets:
            symbol = str(asset.get("symbol", "")).lower()
            if symbol not in _TARGETS:
                continue

            circ = asset.get("circulating", {})
            # DefiLlama stores circulating supply as {"peggedUSD": ...}
            supply_usd = circ.get("peggedUSD") if isinstance(circ, dict) else None

            # 7d and 1d change fields (pct or absolute depending on API version)
            change_1d = asset.get("change_1d")   # may be None
            change_7d = asset.get("change_7d")   # may be None
            change_1m = asset.get("change_1m")   # may be None

            result[symbol] = {
                "supply_usd": supply_usd,
                "supply_bn": round(supply_usd / 1e9, 2) if supply_usd else None,
                "change_1d_pct": change_1d,
                "change_7d_pct": change_7d,
                "change_1m_pct": change_1m,
                "name": asset.get("name", symbol.upper()),
            }
        return result

    def _total_supply(self, data: Dict) -> Optional[float]:
        """Sum USDT + USDC circulating supply in USD."""
        total = 0.0
        for v in data.values():
            s = v.get("supply_usd")
            if s is not None:
                total += s
        return total if total > 0 else None

    def _format_summary(self, data: Dict) -> str:
        lines = []
        for sym, v in sorted(data.items()):
            bn = v.get("supply_bn")
            c7 = v.get("change_7d_pct")
            trend = ""
            if c7 is not None:
                trend = f" 7d:{'+' if c7 >= 0 else ''}{c7:.2f}%"
            lines.append(f"{sym.upper()}=${bn:.1f}B{trend}" if bn else f"{sym.upper()}=N/A")

        total = self._total_supply(data)
        total_str = f" | Combined=${total/1e9:.1f}B" if total else ""
        return f"[STABLECOIN_SUPPLY] {' | '.join(lines)}{total_str}"

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> Optional[Dict]:
        assets = self._fetch_raw()
        if not assets:
            return None

        try:
            data = self._extract(assets)
            if not data:
                logger.warning("StablecoinCollector: no USDT/USDC data in response")
                return None

            summary = self._format_summary(data)
            total_usd = self._total_supply(data)

            payload_base = {
                "source": "stablecoin_defillama",
                "status": "ok",
                "sentiment": "neutral",
                "summary": summary,
                "reasoning": "",
                "macro_context": "",
                "key_events": [],
                "bullish_factors": [],
                "bearish_factors": [],
                "sources": [],
                "raw_payload": {
                    "stablecoins": data,
                    "total_supply_usd": total_usd,
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                },
            }

            now = datetime.now(timezone.utc).replace(second=0, microsecond=0).isoformat()

            # Store once per trading symbol (same global signal)
            for symbol in settings.trading_symbols:
                db.upsert_narrative_data({
                    **payload_base,
                    "timestamp": now,
                    "symbol": symbol,
                })

            logger.info(f"StablecoinCollector: {summary}")
            return data

        except Exception as e:
            logger.error(f"StablecoinCollector: processing failed: {e}")
            return None


stablecoin_collector = StablecoinCollector()
