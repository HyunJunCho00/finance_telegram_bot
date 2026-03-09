from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, Optional, Tuple


class AgentStateStore:
    """In-process blackboard snapshot store.

    This is the first step toward a persistent async-DAG state layer.
    The API is intentionally small so we can swap the backend to DB/Redis later.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._snapshots: Dict[Tuple[str, str], Dict[str, Any]] = {}

    @staticmethod
    def _key(symbol: str, mode: str) -> Tuple[str, str]:
        return (str(symbol or "").upper(), str(mode or "").lower())

    @staticmethod
    def _iso_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def upsert_agent_state(
        self,
        symbol: str,
        mode: str,
        agent_name: str,
        payload: Dict[str, Any],
        *,
        confidence: Optional[float] = None,
        status: str = "fresh",
        created_at: Optional[str] = None,
        expires_at: Optional[str] = None,
        latency_ms: Optional[float] = None,
        input_hash: str = "",
        version: int = 1,
    ) -> Dict[str, Any]:
        created = created_at or self._iso_now()
        row = {
            "payload": deepcopy(payload or {}),
            "confidence": confidence,
            "status": status,
            "created_at": created,
            "expires_at": expires_at or created,
            "latency_ms": latency_ms,
            "input_hash": input_hash,
            "version": version,
        }
        key = self._key(symbol, mode)
        with self._lock:
            bundle = self._snapshots.setdefault(
                key,
                {
                    "symbol": key[0],
                    "mode": key[1],
                    "agents": {},
                    "updated_at": created,
                },
            )
            bundle["agents"][agent_name] = row
            bundle["updated_at"] = created
            return deepcopy(row)

    def load_bundle(self, symbol: str, mode: str) -> Dict[str, Any]:
        key = self._key(symbol, mode)
        with self._lock:
            bundle = self._snapshots.get(
                key,
                {
                    "symbol": key[0],
                    "mode": key[1],
                    "agents": {},
                    "updated_at": self._iso_now(),
                },
            )
            return deepcopy(bundle)

    def delete_bundle(self, symbol: str, mode: str) -> None:
        key = self._key(symbol, mode)
        with self._lock:
            self._snapshots.pop(key, None)


agent_state_store = AgentStateStore()
