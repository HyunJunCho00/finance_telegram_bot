"""Dune collector with cadence-aware execution and DB persistence."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from collectors.dune_sanitizer import sanitize_dune_rows
from config.database import db
from config.settings import settings
from loguru import logger

BASE_URL = "https://api.dune.com/api/v1"
STALE_HOURS = 24


@dataclass(frozen=True)
class DuneQueryConfig:
    query_id: int
    query_name: str
    category: str
    cadence_minutes: int


# Cadence policy by data characteristics
# - Real-time DEX flows: 15m
# - Hourly exchange netflow: 60m
# - Daily staking/on-chain snapshots: 360m (~6h)
QUERY_CONFIGS = [
    DuneQueryConfig(6638261, "ETH Exchange Netflow", "netflow_hourly", 60),
    DuneQueryConfig(5924114, "BTC On-chain Netflow", "netflow_daily", 360),
    DuneQueryConfig(21689, "DEX Aggregator Volume", "dex_realtime", 15),
    DuneQueryConfig(4319, "DEX Volume", "dex_realtime", 15),
    DuneQueryConfig(3383110, "Lido Staking", "staking_daily", 360),
]
QUERY_CONFIGS_BY_ID = {cfg.query_id: cfg for cfg in QUERY_CONFIGS}


class DuneCollector:
    def __init__(self, api_key: str, timeout: int = 30, save_dir: str | None = None):
        self.api_key = api_key
        self.timeout = timeout
        self.save_dir = Path(save_dir).resolve() if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def _request(self, method: str, endpoint: str) -> dict[str, Any]:
        url = f"{BASE_URL}{endpoint}"
        req = Request(url, headers={"X-Dune-API-Key": self.api_key}, method=method)
        with urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def fetch_results(self, query_id: int, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        params = urlencode({"limit": limit, "offset": offset})
        return self._request("GET", f"/query/{query_id}/results?{params}")

    def execute_query(self, query_id: int) -> str:
        data = self._request("POST", f"/query/{query_id}/execute")
        return data["execution_id"]

    def get_execution_status(self, execution_id: str) -> dict[str, Any]:
        return self._request("GET", f"/execution/{execution_id}/status")

    def get_execution_results(self, execution_id: str, limit: int = 100) -> dict[str, Any]:
        return self._request("GET", f"/execution/{execution_id}/results?limit={limit}")

    def _is_stale_payload(self, payload: dict[str, Any]) -> bool:
        ended = payload.get("execution_ended_at")
        if not ended:
            return True
        try:
            ended_at = datetime.fromisoformat(ended.replace("Z", "+00:00"))
            return (datetime.now(timezone.utc) - ended_at) > timedelta(hours=STALE_HOURS)
        except ValueError:
            return True

    def _run_execution_cycle(self, query_id: int, limit: int = 100) -> dict[str, Any]:
        execution_id = self.execute_query(query_id)
        while True:
            status = self.get_execution_status(execution_id)
            state = status.get("state")
            if state == "QUERY_STATE_COMPLETED":
                break
            if state in {"QUERY_STATE_FAILED", "QUERY_STATE_CANCELLED"}:
                raise RuntimeError(f"Execution failed: {state}")
            time.sleep(3)
        return self.get_execution_results(execution_id, limit=limit)

    def resolve_query_configs(self, query_ids: list[int] | None = None) -> list[DuneQueryConfig]:
        if not query_ids:
            return QUERY_CONFIGS

        resolved: list[DuneQueryConfig] = []
        for query_id in query_ids:
            cfg = QUERY_CONFIGS_BY_ID.get(query_id)
            if cfg:
                resolved.append(cfg)
            else:
                resolved.append(DuneQueryConfig(query_id, f"query_{query_id}", "custom", 60))
        return resolved

    def _should_run_now(self, query_cfg: DuneQueryConfig) -> bool:
        latest = db.get_latest_dune_query_result(query_cfg.query_id)
        if not latest:
            return True

        collected_at = latest.get("collected_at")
        if not collected_at:
            return True

        try:
            ts = datetime.fromisoformat(collected_at.replace("Z", "+00:00"))
        except ValueError:
            return True

        due_delta = timedelta(minutes=query_cfg.cadence_minutes)
        return datetime.now(timezone.utc) - ts >= due_delta

    def _persist(self, query_cfg: DuneQueryConfig, payload: dict[str, Any]) -> None:
        raw_rows = payload.get("result", {}).get("rows", [])
        rows, dropped_keys = sanitize_dune_rows(raw_rows)
        execution_ended_at = payload.get("execution_ended_at")
        collected_at = datetime.now(timezone.utc).replace(second=0, microsecond=0).isoformat()

        db.upsert_dune_query_result(
            {
                "query_id": query_cfg.query_id,
                "query_name": query_cfg.query_name,
                "category": query_cfg.category,
                "cadence_minutes": query_cfg.cadence_minutes,
                "execution_ended_at": execution_ended_at,
                "row_count": len(rows),
                "rows": rows,
                "collected_at": collected_at,
            }
        )

        if self.save_dir:
            out = self.save_dir / f"dune_data_{query_cfg.query_id}.json"
            out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

        if dropped_keys:
            logger.info(f"Dune sanitized query_id={query_cfg.query_id} dropped={dropped_keys}")
        logger.info(
            f"Dune saved query_id={query_cfg.query_id} rows={len(rows)} cadence={query_cfg.cadence_minutes}m"
        )

    def collect_query(self, query_cfg: DuneQueryConfig, limit: int = 100, offset: int = 0) -> bool:
        payload: dict[str, Any]
        try:
            payload = self.fetch_results(query_cfg.query_id, limit=limit, offset=offset)
            if self._is_stale_payload(payload):
                payload = self._run_execution_cycle(query_cfg.query_id, limit=limit)
        except HTTPError as exc:
            if exc.code == 404:
                payload = self._run_execution_cycle(query_cfg.query_id, limit=limit)
            else:
                raise

        self._persist(query_cfg, payload)
        return True

    def run_due_queries(
        self,
        limit: int = 100,
        offset: int = 0,
        query_ids: list[int] | None = None,
        force: bool = False,
    ) -> dict[str, int]:
        stats = {"selected": 0, "ran": 0, "skipped": 0, "failed": 0}

        for query_cfg in self.resolve_query_configs(query_ids):
            stats["selected"] += 1

            if not force and not self._should_run_now(query_cfg):
                stats["skipped"] += 1
                continue

            try:
                self.collect_query(query_cfg, limit=limit, offset=offset)
                stats["ran"] += 1
            except Exception as exc:
                stats["failed"] += 1
                logger.error(f"Dune collection error query_id={query_cfg.query_id}: {exc}")

        return stats


# Optional singleton for scheduler
if settings.DUNE_ENABLED and settings.DUNE_API_KEY:
    dune_collector = DuneCollector(settings.DUNE_API_KEY)
else:
    dune_collector = None
