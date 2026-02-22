"""Crypto Fear & Greed Index Collector.

Uses the free alternative.me API — no API key required, daily updates.
Endpoint: https://api.alternative.me/fng/

Index range: 0 (Extreme Fear) → 100 (Extreme Greed)
Classifications: Extreme Fear / Fear / Neutral / Greed / Extreme Greed

Usage notes:
- Data updates once per day (around 00:00 UTC)
- Scheduled daily, but safe to run more often (API is idempotent)
- Stored with daily midnight UTC timestamp for clean dedup
"""

import requests
from datetime import datetime, timezone
from typing import Optional, Dict
from config.database import db
from loguru import logger

FNG_URL = "https://api.alternative.me/fng/"


class FearGreedCollector:
    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({'Accept': 'application/json'})

    def fetch(self) -> Optional[Dict]:
        """Fetch today's and yesterday's Fear & Greed values."""
        try:
            resp = self._session.get(
                FNG_URL,
                params={'limit': 2, 'format': 'json'},
                timeout=10,
            )
            resp.raise_for_status()
            entries = resp.json().get('data', [])
            if not entries:
                logger.warning("Fear & Greed: empty response")
                return None

            current = entries[0]
            prev    = entries[1] if len(entries) > 1 else None

            value          = int(current.get('value', 0))
            classification = current.get('value_classification', 'Unknown')
            value_prev     = int(prev['value']) if prev else None
            class_prev     = prev.get('value_classification') if prev else None
            change         = (value - value_prev) if value_prev is not None else None

            # Daily: key by midnight UTC so upsert is idempotent on re-runs
            day_ts = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            ).isoformat()

            return {
                'timestamp':          day_ts,
                'value':              value,
                'classification':     classification,
                'value_prev':         value_prev,
                'classification_prev': class_prev,
                'change':             change,
            }
        except Exception as e:
            logger.error(f"Fear & Greed fetch error: {e}")
            return None

    def run(self) -> None:
        data = self.fetch()
        if data:
            try:
                db.upsert_fear_greed(data)
                logger.info(
                    f"Fear & Greed: {data['value']} ({data['classification']})"
                    f"  prev={data.get('value_prev')} Δ={data.get('change')}"
                )
            except Exception as e:
                logger.error(f"Fear & Greed DB save error: {e}")


fear_greed_collector = FearGreedCollector()
