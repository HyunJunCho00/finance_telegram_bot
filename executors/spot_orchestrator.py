"""
Spot Trading Orchestrator
=========================
쉽알남(leverage SWING bot)과 독립적으로 동작하는 현물 거래 모듈.

SpotMode.SWING    : 쉽알남 LONG APPROVED 신호를 현물 거래소에 미러링
SpotMode.POSITION : 독립 분석 사이클 — 1w/1d 구조적 thesis, 기간 설정 없음,
                    thesis invalidation 시 청산

- SHORT 불가
- 레버리지 없음 (1x 강제)
- 거래소: upbit / binance_spot / split
"""

from __future__ import annotations

from loguru import logger

from config.settings import SpotMode, TradingMode

# ── 고정 설정값 ────────────────────────────────────────────────────────────────
SPOT_MODE_ENABLED: bool = False               # True로 바꾸면 활성화
SPOT_MODE: SpotMode = SpotMode.SWING          # SpotMode.SWING | SpotMode.POSITION
SPOT_EXCHANGE: str = "upbit"                  # "upbit" | "binance_spot" | "split"
SPOT_MIRROR_ALLOCATION_PCT: float = 10.0      # spot_swing: 현물 지갑 대비 미러 비중
SPOT_POSITION_ANALYSIS_HOUR_UTC: int = 2      # spot_position: 매일 분석 시각 (UTC)


class SpotOrchestrator:

    # ── SpotMode.SWING ─────────────────────────────────────────────────────

    def maybe_mirror_swing_long(self, symbol: str, swing_decision: dict) -> None:
        """쉽알남이 LONG APPROVED를 냈을 때 현물 미러 진입을 시도한다.

        run_analysis_with_mode() 반환 직후에 호출됨.
        SHORT / HOLD / VETO 결정은 조용히 무시.
        """
        if not SPOT_MODE_ENABLED:
            return
        if SPOT_MODE != SpotMode.SWING:
            return
        if not isinstance(swing_decision, dict):
            return

        direction = str(swing_decision.get("decision", "HOLD")).upper()
        if direction != "LONG":
            return

        policy = swing_decision.get("policy_checks", {}) or {}
        if str(policy.get("status", "")).upper() != "APPROVED":
            logger.debug(f"SpotOrchestrator: skipping mirror — policy status={policy.get('status')}")
            return

        stop_loss = swing_decision.get("stop_loss")
        take_profit = swing_decision.get("take_profit")

        spot_decision = {
            "decision": "LONG",
            "allocation_pct": SPOT_MIRROR_ALLOCATION_PCT,
            "leverage": 1,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "reasoning": {
                "final_logic": (
                    f"[SPOT MIRROR] 쉽알남 SWING LONG 미러링 | "
                    f"SL={stop_loss} TP={take_profit} | "
                    f"원본: {str(swing_decision.get('reasoning', {}).get('final_logic', ''))[:200]}"
                )
            },
        }

        try:
            from executors.trade_executor import trade_executor
            result = trade_executor.execute_from_decision(
                spot_decision,
                mode="POSITION",   # trade_executor POSITION 분기: lev=1, spot 거래소 라우팅
                symbol=symbol,
            )
            if result.get("success"):
                logger.info(
                    f"SpotOrchestrator: LONG mirror executed for {symbol} "
                    f"on {SPOT_EXCHANGE} | alloc={SPOT_MIRROR_ALLOCATION_PCT}%"
                )
            else:
                logger.warning(
                    f"SpotOrchestrator: mirror skipped for {symbol} — {result.get('note') or result.get('error')}"
                )
        except Exception as e:
            logger.error(f"SpotOrchestrator: mirror execution error for {symbol}: {e}")

    # ── SpotMode.POSITION ──────────────────────────────────────────────────

    def run_position_analysis(self, symbol: str) -> dict:
        """SpotMode.POSITION용 독립 분석 사이클.

        execute_trades=False로 분석만 실행 후 spot_orchestrator가 직접 spot 거래소에 라우팅.
        """
        if not SPOT_MODE_ENABLED:
            return {}
        if SPOT_MODE != SpotMode.POSITION:
            return {}

        logger.info(f"SpotOrchestrator: starting spot_position analysis for {symbol}")
        try:
            from executors.orchestrator import orchestrator
            result = orchestrator.run_analysis_with_mode(
                symbol,
                TradingMode.SWING,
                execute_trades=False,
                allow_perplexity=True,
                notification_context="spot_position",
            )

            decision = str(result.get("decision", "HOLD")).upper()
            if decision == "LONG":
                self._execute_spot_position_entry(symbol, result)
            elif decision == "CANCEL_AND_CLOSE":
                self._execute_spot_exit(symbol, reason="Judge CANCEL_AND_CLOSE")

            return result
        except Exception as e:
            logger.error(f"SpotOrchestrator: spot_position analysis error for {symbol}: {e}")
            return {}

    def check_position_invalidation(self, symbol: str) -> None:
        """저장된 현물 포지션의 thesis invalidation 조건을 실시간 가격과 비교.

        hourly monitor 사이클에서 호출됨. SL 도달 시 자동 청산.
        """
        if not SPOT_MODE_ENABLED:
            return
        if SPOT_MODE != SpotMode.POSITION:
            return

        try:
            from executors.paper_exchange import paper_engine
            positions = paper_engine.get_open_positions()
            spot_positions = [
                p for p in positions
                if p.get("exchange") in ("upbit", "binance_spot")
                and p.get("symbol") == symbol
                and p.get("side", "").upper() == "LONG"
            ]
            if not spot_positions:
                return

            from executors.trade_executor import trade_executor
            price = trade_executor._get_reference_price(symbol, exchange=SPOT_EXCHANGE)
            if price <= 0:
                return

            for pos in spot_positions:
                sl = pos.get("stop_loss") or pos.get("sl_price")
                if sl and float(sl) > 0 and price <= float(sl):
                    logger.warning(
                        f"SpotOrchestrator: invalidation triggered for {symbol} — "
                        f"price={price} <= SL={sl}"
                    )
                    self._execute_spot_exit(symbol, reason=f"SL hit: price={price:.2f} <= sl={sl:.2f}")
                    break
        except Exception as e:
            logger.error(f"SpotOrchestrator: invalidation check error for {symbol}: {e}")

    def _execute_spot_position_entry(self, symbol: str, decision: dict) -> None:
        try:
            from executors.trade_executor import trade_executor
            spot_decision = {
                "decision": "LONG",
                "allocation_pct": float(decision.get("allocation_pct") or SPOT_MIRROR_ALLOCATION_PCT),
                "leverage": 1,
                "stop_loss": decision.get("stop_loss"),
                "take_profit": decision.get("take_profit"),
                "reasoning": decision.get("reasoning", {}),
            }
            result = trade_executor.execute_from_decision(
                spot_decision, mode="POSITION", symbol=symbol
            )
            logger.info(f"SpotOrchestrator: spot_position entry for {symbol}: {result.get('note') or result.get('success')}")
        except Exception as e:
            logger.error(f"SpotOrchestrator: spot entry error for {symbol}: {e}")

    def _execute_spot_exit(self, symbol: str, reason: str = "") -> None:
        try:
            from executors.trade_executor import trade_executor
            result = trade_executor.close_position(symbol, exchange=SPOT_EXCHANGE)
            logger.info(f"SpotOrchestrator: spot exit {symbol} — {reason} | result={result}")
        except Exception as e:
            logger.error(f"SpotOrchestrator: spot exit error for {symbol}: {e}")


spot_orchestrator = SpotOrchestrator()
