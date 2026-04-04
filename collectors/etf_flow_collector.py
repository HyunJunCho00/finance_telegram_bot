"""
ETF Flow Collector — Farside Investors daily BTC/ETH ETF net flow data.

Sources:
  BTC: https://farside.co.uk/bitcoin-etf-flow-all-data/
  ETH: https://farside.co.uk/ethereum-etf-flow-all-data/

Free, daily-updated. No API key required.
Stores to narrative_data (source="etf_flow_farside") per symbol.

Signal interpretation:
  Positive total → net inflow → institutional demand (bullish)
  Negative total → net outflow → distribution (bearish)
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests
from loguru import logger

from config.database import db

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}
_TIMEOUT = 20
_BTC_URL = "https://farside.co.uk/bitcoin-etf-flow-all-data/"
_ETH_URL = "https://farside.co.uk/ethereum-etf-flow-all-data/"

_SYMBOL_MAP = {
    "BTCUSDT": ("BTC", _BTC_URL),
    "ETHUSDT": ("ETH", _ETH_URL),
}


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------

def _extract_table_rows(html: str) -> List[List[str]]:
    """Extract rows from the first <table> in the HTML using stdlib regex."""
    # Strip scripts/styles to avoid false positives
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)

    table_match = re.search(r"<table[^>]*>(.*?)</table>", html, re.DOTALL | re.IGNORECASE)
    if not table_match:
        return []

    table_html = table_match.group(1)
    rows: List[List[str]] = []
    for row_match in re.finditer(r"<tr[^>]*>(.*?)</tr>", table_html, re.DOTALL | re.IGNORECASE):
        row_html = row_match.group(1)
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row_html, re.DOTALL | re.IGNORECASE)
        # Strip inner HTML tags and whitespace
        cleaned = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        if cleaned:
            rows.append(cleaned)
    return rows


def _parse_flow_value(raw: str) -> Optional[float]:
    """Parse a cell value like '150.5', '-42.1', '(30)', '' → float or None."""
    raw = raw.replace(",", "").replace("$", "").strip()
    if not raw or raw in ("-", "N/A", "n/a", "—"):
        return None
    # Farside sometimes uses parentheses for negative: (30.5)
    neg = raw.startswith("(") and raw.endswith(")")
    raw = raw.strip("()")
    try:
        val = float(raw)
        return -val if neg else val
    except ValueError:
        return None


def _parse_table(rows: List[List[str]]) -> Optional[Dict]:
    """Find the most recent data row and extract per-ETF flows + total."""
    if len(rows) < 2:
        return None

    # First row = headers
    headers = [h.upper() for h in rows[0]]

    # The "TOTAL" column is usually the last column
    total_idx = next((i for i, h in enumerate(headers) if "TOTAL" in h), len(headers) - 1)

    # Walk rows in reverse to find the most recent non-empty row
    for row in reversed(rows[1:]):
        if not row or len(row) < 2:
            continue
        date_raw = row[0].strip()
        if not date_raw or date_raw.upper() in ("DATE", "TOTAL", ""):
            continue

        total_val = _parse_flow_value(row[total_idx]) if total_idx < len(row) else None
        if total_val is None:
            # Skip rows where total is blank (usually future dates)
            continue

        per_etf: Dict[str, Optional[float]] = {}
        for i, h in enumerate(headers):
            if i == 0 or i == total_idx:
                continue
            if i < len(row):
                per_etf[h] = _parse_flow_value(row[i])

        return {
            "date": date_raw,
            "total_net_flow_usd_m": total_val,
            "per_etf": per_etf,
        }

    return None


# ---------------------------------------------------------------------------
# Collector class
# ---------------------------------------------------------------------------

class ETFFlowCollector:
    """Scrapes Farside Investors for daily BTC/ETH ETF net flow data."""

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)

    def _fetch(self, url: str) -> Optional[str]:
        try:
            resp = self._session.get(url, timeout=_TIMEOUT)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            logger.warning(f"ETFFlowCollector: fetch failed {url}: {e}")
            return None

    def collect_symbol(self, binance_symbol: str) -> Optional[Dict]:
        """Scrape and return parsed flow data for one symbol."""
        if binance_symbol not in _SYMBOL_MAP:
            return None

        asset, url = _SYMBOL_MAP[binance_symbol]
        html = self._fetch(url)
        if not html:
            return None

        rows = _extract_table_rows(html)
        data = _parse_table(rows)
        if not data:
            logger.warning(f"ETFFlowCollector: could not parse table for {asset}")
            return None

        data["asset"] = asset
        data["symbol"] = binance_symbol
        data["collected_at"] = datetime.now(timezone.utc).isoformat()
        return data

    def _format_summary(self, data: Dict) -> str:
        asset = data["asset"]
        net = data["total_net_flow_usd_m"]
        direction = "INFLOW" if net >= 0 else "OUTFLOW"
        sign = "+" if net >= 0 else ""
        top_etfs = sorted(
            [(k, v) for k, v in data.get("per_etf", {}).items() if v is not None],
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:3]
        etf_str = ", ".join(f"{k}:{'+' if v>=0 else ''}{v:.1f}M" for k, v in top_etfs)
        return (
            f"[ETF_FLOW {asset}] {data['date']}: Net {direction} {sign}{net:.1f}M USD | "
            f"Top movers: {etf_str}"
        )

    def run(self) -> Dict[str, Optional[Dict]]:
        results: Dict[str, Optional[Dict]] = {}
        for symbol in _SYMBOL_MAP:
            try:
                data = self.collect_symbol(symbol)
                if not data:
                    results[symbol] = None
                    continue

                summary = self._format_summary(data)
                now = datetime.now(timezone.utc).replace(second=0, microsecond=0).isoformat()
                payload = {
                    "timestamp": now,
                    "symbol": symbol,
                    "source": "etf_flow_farside",
                    "status": "ok",
                    "sentiment": "bullish" if data["total_net_flow_usd_m"] > 0 else "bearish",
                    "summary": summary,
                    "reasoning": "",
                    "macro_context": "",
                    "key_events": [],
                    "bullish_factors": [],
                    "bearish_factors": [],
                    "sources": [],
                    "raw_payload": data,
                }
                db.upsert_narrative_data(payload)
                logger.info(f"ETFFlowCollector: {summary}")
                results[symbol] = data
            except Exception as e:
                logger.error(f"ETFFlowCollector failed for {symbol}: {e}")
                results[symbol] = None
        return results


etf_flow_collector = ETFFlowCollector()
