"""Macro collector (4h cadence) using free data providers.

- FRED series via direct API request
- Cross-asset prices via yfinance
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

import requests
import yfinance as yf
from loguru import logger

from config.database import db
from config.settings import settings


class MacroCollector:
    def __init__(self):
        self.fred_api_key = getattr(settings, "FRED_API_KEY", "")
        self.fred_base = "https://api.stlouisfed.org/fred/series/observations"
        # DGS2/DGS10 (US Treasury yields), CPIAUCSL (CPI), FEDFUNDS
        self.fred_series = ["DGS2", "DGS10", "CPIAUCSL", "FEDFUNDS"]
        # yfinance proxies: DXY, Nasdaq, Gold, Oil
        self.tickers = {
            "dxy": "DX-Y.NYB",
            "nasdaq": "^IXIC",
            "gold": "GC=F",
            "oil": "CL=F",
        }

    def _fetch_fred_last(self, series_id: str) -> Optional[float]:
        if not self.fred_api_key:
            return None
        params = {
            "series_id": series_id,
            "api_key": self.fred_api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 1,
        }
        try:
            res = requests.get(self.fred_base, params=params, timeout=20)
            res.raise_for_status()
            data = res.json().get("observations", [])
            if not data:
                return None
            value = data[0].get("value")
            if value in (None, "."):
                return None
            return float(value)
        except Exception as e:
            logger.warning(f"FRED fetch failed {series_id}: {e}")
            return None

    def _fetch_yfinance_last(self, ticker: str) -> Optional[float]:
        try:
            hist = yf.Ticker(ticker).history(period="5d", interval="1h")
            if hist.empty:
                return None
            return float(hist["Close"].dropna().iloc[-1])
        except Exception as e:
            logger.warning(f"yfinance fetch failed {ticker}: {e}")
            return None

    def run(self) -> None:
        # 1. Fetch cross-asset data
        payload: Dict = {
            "timestamp": datetime.now(timezone.utc).replace(second=0, microsecond=0).isoformat(),
            "source": "macro_collector",
            "dgs2": self._fetch_fred_last("DGS2"),
            "dgs10": self._fetch_fred_last("DGS10"),
            "cpiaucsl": self._fetch_fred_last("CPIAUCSL"),
            "fedfunds": self._fetch_fred_last("FEDFUNDS"),
            "dxy": self._fetch_yfinance_last(self.tickers["dxy"]),
            "nasdaq": self._fetch_yfinance_last(self.tickers["nasdaq"]),
            "gold": self._fetch_yfinance_last(self.tickers["gold"]),
            "oil": self._fetch_yfinance_last(self.tickers["oil"]),
        }

        # 2. CME Basis Calculation (Institutional Sentiment)
        # Pairs: BTC=F (CME) vs BTC-USD (Spot), ETH=F (CME) vs ETH-USD (Spot)
        assets = {"btc": ("BTC=F", "BTC-USD"), "eth": ("ETH=F", "ETH-USD")}
        for asset, (f_ticker, s_ticker) in assets.items():
            try:
                cme_f = self._fetch_yfinance_last(f_ticker)
                spot = self._fetch_yfinance_last(s_ticker)
                
                key_basis = f"{asset}_cme_basis"
                key_pct = f"{asset}_cme_basis_pct"
                
                if cme_f and spot:
                    basis = cme_f - spot
                    basis_pct = (basis / spot) * 100
                    payload[key_basis] = round(basis, 2)
                    payload[key_pct] = round(basis_pct, 4)
                    logger.info(f"CME {asset.upper()} Basis: ${payload[key_basis]} ({payload[key_pct]}%)")
                else:
                    payload[key_basis] = None
                    payload[key_pct] = None
            except Exception as e:
                logger.warning(f"CME {asset.upper()} Basis calc failed: {e}")
                payload[f"{asset}_cme_basis"] = None
                payload[f"{asset}_cme_basis_pct"] = None

        # 3. Spread Calculation
        if payload["dgs2"] is not None and payload["dgs10"] is not None:
            payload["ust_2s10s_spread"] = round(payload["dgs10"] - payload["dgs2"], 4)
        else:
            payload["ust_2s10s_spread"] = None

        try:
            db.upsert_macro_data(payload)
            logger.info("Saved macro data snapshot")
        except Exception as e:
            logger.error(f"Macro DB save error: {e}")


macro_collector = MacroCollector()
