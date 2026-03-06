from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List
from urllib.parse import urlencode
from urllib.request import urlopen
from loguru import logger

from config.database import db
from config.settings import settings
from processors.onchain_signal_engine import onchain_signal_engine


class CoinMetricsCollector:
    """Daily Coin Metrics collector used for on-chain regime snapshots."""

    SYMBOL_TO_ASSET = {
        "BTCUSDT": "btc",
        "ETHUSDT": "eth",
    }

    METRIC_CANDIDATES = {
        "mvrv": ["CapMVRVCur"],
        "realized_cap_usd": ["CapRealUSD"],
        "current_supply": ["SplyCur"],
        "exchange_supply": ["SplyExNtv", "SplyExPct"],
        "tx_count": ["TxCnt"],
        "new_addresses": ["AdrNew", "AdrActCnt"],
        "tx_eip1559_count": ["TxEIP1559Cnt"],
    }

    def __init__(self):
        self.base_url = settings.COINMETRICS_BASE_URL.rstrip("/")
        self.timeout = settings.COINMETRICS_TIMEOUT_SECONDS
        self.lookback_days = settings.COINMETRICS_LOOKBACK_DAYS

    def _fetch_timeseries(self, asset: str, metrics: List[str]) -> List[Dict]:
        start_time = (datetime.now(timezone.utc) - timedelta(days=self.lookback_days)).date().isoformat()
        url = f"{self.base_url}/timeseries/asset-metrics"
        params = {
            "assets": asset,
            "metrics": ",".join(metrics),
            "frequency": "1d",
            "start_time": start_time,
            "page_size": 200,
        }
        full_url = f"{url}?{urlencode(params)}"
        with urlopen(full_url, timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        data = payload.get("data", []) if isinstance(payload, dict) else []
        return data if isinstance(data, list) else []

    def _get_spot_price(self, symbol: str) -> float | None:
        try:
            df = db.get_latest_market_data(symbol, limit=1)
            if df is not None and not df.empty and "close" in df.columns:
                return float(df["close"].iloc[-1])
        except Exception as e:
            logger.warning(f"Coin Metrics spot price lookup failed for {symbol}: {e}")
        return None

    def _resolve_metric_aliases(self, rows: List[Dict]) -> Dict[str, str]:
        aliases: Dict[str, str] = {}
        for concept, candidates in self.METRIC_CANDIDATES.items():
            for metric_id in candidates:
                if any(row.get(metric_id) not in (None, "", "null") for row in rows):
                    aliases[concept] = metric_id
                    break
            if concept not in aliases and candidates:
                aliases[concept] = candidates[0]
        return aliases

    def collect_symbol(self, symbol: str) -> Dict | None:
        if not settings.COINMETRICS_ENABLED:
            logger.info("Coin Metrics collection skipped: disabled")
            return None

        asset = self.SYMBOL_TO_ASSET.get(symbol)
        if not asset:
            logger.debug(f"Coin Metrics not configured for symbol={symbol}")
            return None

        metric_set = sorted({metric for items in self.METRIC_CANDIDATES.values() for metric in items})
        rows = self._fetch_timeseries(asset, metric_set)
        if not rows:
            logger.warning(f"Coin Metrics returned no rows for {symbol} ({asset})")
            return None

        metric_aliases = self._resolve_metric_aliases(rows)
        spot_price = self._get_spot_price(symbol)
        snapshot = onchain_signal_engine.build_snapshot(
            symbol=symbol,
            series_rows=rows,
            spot_price=spot_price,
            metric_aliases=metric_aliases,
            source="coinmetrics",
        )
        db.upsert_onchain_daily_snapshot(snapshot)
        logger.info(
            f"Coin Metrics snapshot saved for {symbol}: bias={snapshot.get('risk_bias')} "
            f"score={snapshot.get('bias_score')}"
        )
        return snapshot

    def run(self) -> Dict[str, Dict]:
        results: Dict[str, Dict] = {}
        if not settings.COINMETRICS_ENABLED:
            logger.info("Coin Metrics collection skipped: disabled")
            return results

        for symbol in self.SYMBOL_TO_ASSET:
            try:
                snapshot = self.collect_symbol(symbol)
                if snapshot:
                    results[symbol] = snapshot
            except Exception as e:
                logger.error(f"Coin Metrics collection failed for {symbol}: {e}")
        return results


coinmetrics_collector = CoinMetricsCollector()
