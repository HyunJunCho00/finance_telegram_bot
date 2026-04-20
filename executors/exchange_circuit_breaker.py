"""
Exchange Circuit Breaker — CLOSED / OPEN / HALF_OPEN state machine.

Trip condition : ccxt NetworkError 계열 (거래소 헬스 이상) N회 연속 실패
Business errors: InsufficientFunds, InvalidOrder 등은 트리핑 대상 아님
                 (거래소는 살아 있고 내 요청이 잘못된 것)
"""

from __future__ import annotations

import threading
import time
from enum import Enum
from typing import Dict, Optional

from loguru import logger


class State(str, Enum):
    CLOSED    = "CLOSED"
    OPEN      = "OPEN"
    HALF_OPEN = "HALF_OPEN"


# ccxt NetworkError 계열 — 거래소 헬스 이상으로 판단하는 예외 클래스명
# MRO 전체를 검사하므로 서브클래스도 자동 커버됨
_TRIP_NAMES = frozenset({
    "NetworkError",
    "RequestTimeout",
    "ExchangeNotAvailable",
    "DDoSProtection",
    "RateLimitExceeded",
    "OnMaintenance",
})


def is_trip_worthy(error: Exception) -> bool:
    """거래소 헬스 이상으로 판단해 Circuit Breaker를 트리핑해야 하는 예외인지 확인."""
    return any(cls.__name__ in _TRIP_NAMES for cls in type(error).__mro__)


class ExchangeCircuitBreaker:
    """
    상태 전이:
      CLOSED ──(N연속 헬스 실패)──► OPEN ──(reset_timeout 경과)──► HALF_OPEN
        ▲                                                               │
        └─────────────(probe 성공)──────────────────────────────────────┘
                                         (probe 실패)──► OPEN (타이머 리셋)

    Thread-safe: 모든 상태 읽기/쓰기는 self._lock 아래에서 수행.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        reset_timeout: float = 60.0,
    ) -> None:
        self._name = name
        self._failure_threshold = failure_threshold
        self._reset_timeout = reset_timeout

        self._lock = threading.Lock()
        self._state: State = State.CLOSED
        self._failures: int = 0
        self._opened_at: Optional[float] = None

    # ── 외부 API ────────────────────────────────────────────────────────────────

    @property
    def state(self) -> State:
        with self._lock:
            self._maybe_promote()
            return self._state

    def allow_request(self) -> bool:
        """
        요청을 허용할지 반환.
        CLOSED  → True (정상 통과)
        OPEN    → False (즉시 Fail-Fast)
        HALF_OPEN → True, 단 probe 슬롯을 원자적으로 클레임하고 상태를 OPEN으로 전환.
                   동시 호출 중 오직 하나만 probe를 얻음.
        """
        with self._lock:
            self._maybe_promote()
            if self._state == State.CLOSED:
                return True
            if self._state == State.HALF_OPEN:
                # Probe 클레임: 즉시 OPEN으로 재잠금 → 동시 호출자는 모두 차단
                self._state = State.OPEN
                self._opened_at = time.monotonic()  # probe 실패 시 타이머 리셋 효과
                logger.info(f"[CB:{self._name}] Probe 발사 (HALF_OPEN → OPEN)")
                return True
            return False  # OPEN

    def record_success(self) -> None:
        """거래소 호출 성공 시 반드시 호출. CLOSED로 복귀."""
        with self._lock:
            prev = self._failures
            self._state = State.CLOSED
            self._failures = 0
            self._opened_at = None
        if prev > 0:
            logger.info(f"[CB:{self._name}] 성공 → CLOSED (cleared {prev} failures)")

    def record_failure(self, error: Exception) -> None:
        """
        거래소 호출 실패 시 호출.
        헬스 이상 예외만 카운트; 비즈니스 예외는 무시.
        """
        if not is_trip_worthy(error):
            logger.debug(
                f"[CB:{self._name}] 비즈니스 에러 (트리핑 안 함): {type(error).__name__}"
            )
            return

        with self._lock:
            self._failures += 1
            logger.warning(
                f"[CB:{self._name}] 헬스 실패 #{self._failures}/{self._failure_threshold}: "
                f"{type(error).__name__}"
            )
            if self._failures >= self._failure_threshold and self._state != State.OPEN:
                self._state = State.OPEN
                self._opened_at = time.monotonic()
                logger.error(
                    f"[CB:{self._name}] → OPEN (연속 {self._failures}회 실패, "
                    f"{self._reset_timeout:.0f}s 차단)"
                )

    def status(self) -> dict:
        with self._lock:
            self._maybe_promote()
            return {
                "exchange": self._name,
                "state": self._state.value,
                "consecutive_failures": self._failures,
                "seconds_until_probe": (
                    max(0.0, self._reset_timeout - (time.monotonic() - self._opened_at))
                    if self._opened_at and self._state == State.OPEN
                    else 0.0
                ),
            }

    # ── 내부 ──────────────────────────────────────────────────────────────────

    def _maybe_promote(self) -> None:
        """self._lock 아래에서만 호출. OPEN → HALF_OPEN 자동 전이."""
        if (
            self._state == State.OPEN
            and self._opened_at is not None
            and time.monotonic() - self._opened_at >= self._reset_timeout
        ):
            self._state = State.HALF_OPEN
            logger.info(f"[CB:{self._name}] OPEN → HALF_OPEN (timeout 경과)")


# ── 전역 레지스트리 (거래소별 싱글턴) ──────────────────────────────────────────

_registry: Dict[str, ExchangeCircuitBreaker] = {}
_registry_lock = threading.Lock()


def get_breaker(
    exchange: str,
    failure_threshold: int = 3,
    reset_timeout: float = 60.0,
) -> ExchangeCircuitBreaker:
    """거래소 이름으로 Circuit Breaker 싱글턴을 반환."""
    with _registry_lock:
        if exchange not in _registry:
            _registry[exchange] = ExchangeCircuitBreaker(
                name=exchange,
                failure_threshold=failure_threshold,
                reset_timeout=reset_timeout,
            )
        return _registry[exchange]
