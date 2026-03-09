from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


class PerformanceTelemetry:
    """Append-only JSONL telemetry for before/after latency comparisons."""

    def __init__(self, log_dir: str = "data/performance_telemetry") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.refresh_file = self.log_dir / "snapshot_refresh.jsonl"
        self.hot_path_file = self.log_dir / "hot_path_runs.jsonl"
        self.full_path_file = self.log_dir / "full_path_runs.jsonl"

    @staticmethod
    def _base_event(event_type: str) -> Dict[str, Any]:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
        }

    def _append_jsonl(self, path: Path, payload: Dict[str, Any]) -> None:
        try:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            logger.error(f"Failed to append performance telemetry {path}: {exc}")

    def log_snapshot_refresh(
        self,
        *,
        symbol: str,
        mode: str,
        stage: str,
        allow_perplexity: bool,
        latency_ms: float,
        success: bool = True,
        details: Dict[str, Any] | None = None,
    ) -> None:
        event = self._base_event("snapshot_refresh")
        event.update(
            {
                "symbol": symbol,
                "mode": mode,
                "stage": stage,
                "allow_perplexity": bool(allow_perplexity),
                "latency_ms": round(float(latency_ms), 4),
                "success": bool(success),
                "details": details or {},
            }
        )
        self._append_jsonl(self.refresh_file, event)

    def log_hot_path_run(
        self,
        *,
        symbol: str,
        mode: str,
        notification_context: str,
        execute_trades: bool,
        snapshot_hit: bool,
        missing_agents: list[str],
        latency_ms: float,
        stage_latencies_ms: Dict[str, float],
        final_decision: Dict[str, Any] | None = None,
        report_generated: bool = False,
    ) -> None:
        event = self._base_event("hot_path_run")
        event.update(
            {
                "symbol": symbol,
                "mode": mode,
                "notification_context": notification_context,
                "execute_trades": bool(execute_trades),
                "snapshot_hit": bool(snapshot_hit),
                "missing_agents": list(missing_agents or []),
                "missing_agent_count": len(list(missing_agents or [])),
                "latency_ms": round(float(latency_ms), 4),
                "stage_latencies_ms": {
                    key: round(float(value), 4) for key, value in (stage_latencies_ms or {}).items()
                },
                "decision": str((final_decision or {}).get("decision", "")),
                "allocation_pct": (final_decision or {}).get("allocation_pct"),
                "report_generated": bool(report_generated),
            }
        )
        self._append_jsonl(self.hot_path_file, event)

    def log_full_path_run(
        self,
        *,
        symbol: str,
        mode: str,
        notification_context: str,
        execution_path: str,
        allow_perplexity: bool,
        execute_trades: bool,
        latency_ms: float,
        final_decision: Dict[str, Any] | None = None,
    ) -> None:
        event = self._base_event("full_path_run")
        event.update(
            {
                "symbol": symbol,
                "mode": mode,
                "notification_context": notification_context,
                "execution_path": execution_path,
                "allow_perplexity": bool(allow_perplexity),
                "execute_trades": bool(execute_trades),
                "latency_ms": round(float(latency_ms), 4),
                "decision": str((final_decision or {}).get("decision", "")),
                "allocation_pct": (final_decision or {}).get("allocation_pct"),
            }
        )
        self._append_jsonl(self.full_path_file, event)


performance_telemetry = PerformanceTelemetry()
