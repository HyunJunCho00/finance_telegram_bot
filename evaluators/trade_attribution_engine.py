# -*- coding: utf-8 -*-
"""Layer 6: Post-Trade Attribution & IC Feedback Loop.

Institutional quant 방식:
  거래 종료 후 각 팩터가 수익에 얼마나 기여했는지 수치로 분해.
  → 이 결과가 Layer 1(Factor IC Tracker)의 IC를 업데이트하는 피드백 루프를 완성.

Attribution 공식:
  factor_contribution_i = signal_value_i × realized_return × (ic_i / Σic)

IC 업데이트:
  새 (signal, return) 쌍을 추가 → 롤링 Pearson 상관계수 재계산 → DB 저장
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple
from loguru import logger


# 팩터 이름 목록
FACTOR_NAMES = [
    "funding_rate_signal",
    "oi_change_signal",
    "liquidation_trap_signal",
    "onchain_flow_signal",
    "narrative_sentiment_signal",
    "rsi_divergence_signal",
    "macro_regime_signal",
    "microstructure_signal",
]

# IC 업데이트 지수 감쇠 (하루 경과 시 5% 가중치 감소)
IC_DECAY_PER_HOUR = 0.998  # 0.998^24 ≈ 0.953 (하루 5% 감소)


def _weighted_pearson(
    pairs: List[Tuple[float, float]],
    weights: List[float],
) -> Optional[float]:
    """가중 Pearson 상관계수 계산. 샘플 부족 시 None 반환."""
    if len(pairs) < 3:
        return None
    x_vals = [p[0] for p in pairs]
    y_vals = [p[1] for p in pairs]
    w = weights

    w_sum = sum(w)
    if w_sum < 1e-12:
        return None

    x_mean = sum(wi * xi for wi, xi in zip(w, x_vals)) / w_sum
    y_mean = sum(wi * yi for wi, yi in zip(w, y_vals)) / w_sum

    cov = sum(wi * (xi - x_mean) * (yi - y_mean)
              for wi, xi, yi in zip(w, x_vals, y_vals)) / w_sum
    var_x = sum(wi * (xi - x_mean) ** 2
                for wi, xi in zip(w, x_vals)) / w_sum
    var_y = sum(wi * (yi - y_mean) ** 2
                for wi, yi in zip(w, y_vals)) / w_sum

    denom = math.sqrt(max(var_x, 1e-12) * max(var_y, 1e-12))
    return cov / denom if denom > 1e-12 else None


class TradeAttributionEngine:
    """거래 종료 후 팩터별 기여도 분해 + IC 피드백 루프."""

    # ── 메인 엔트리포인트 ─────────────────────────────────────────────────────

    def attribute(
        self,
        symbol: str,
        mode: str,
        realized_pnl_pct: float,       # 수수료 차감 실현 수익률 (%)
        decision_id: Optional[str],    # ai_reports.id
        entry_price: float = 0.0,
        exit_price: float = 0.0,
        holding_hours: float = 0.0,
        regime_at_entry: str = "UNKNOWN",
    ) -> Optional[Dict]:
        """
        1. decision_id로 factor_signals 조회
        2. 각 팩터 기여도 계산
        3. IC 히스토리 업데이트
        4. trade_attribution 저장
        """
        try:
            from config.database import db

            # 1. 진입 시점의 팩터 신호 조회
            signals_record = db.get_factor_signals_by_decision_id(decision_id)
            if not signals_record:
                logger.warning(
                    f"[Attribution] No factor_signals found for decision_id={decision_id}"
                )
                return None

            signals_snapshot = {
                fname: signals_record.get(fname, 0.0) or 0.0
                for fname in FACTOR_NAMES
            }
            ic_weights_at_entry = signals_record.get("ic_weights") or {}
            factor_signals_id = signals_record.get("id")

            # 2. 팩터별 기여도 계산
            contributions = self._compute_contributions(
                signals=signals_snapshot,
                ic_weights=ic_weights_at_entry,
                realized_pnl_pct=realized_pnl_pct,
            )

            # 3. dominant factor
            dominant = max(
                contributions,
                key=lambda k: abs(contributions[k]),
                default=None,
            )
            dominant_correct = None
            if dominant and realized_pnl_pct != 0:
                sig_sign = math.copysign(1, signals_snapshot.get(dominant, 0))
                ret_sign = math.copysign(1, realized_pnl_pct)
                dominant_correct = (sig_sign == ret_sign)

            # 4. 설명 불가 잔여분 (residual)
            total_explained = sum(contributions.values())
            residual = realized_pnl_pct - total_explained

            # 5. trade_attribution 저장
            attribution_record = {
                "symbol": symbol,
                "mode": mode,
                "decision_id": decision_id,
                "factor_signals_id": factor_signals_id,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "realized_pnl_pct": realized_pnl_pct,
                "holding_hours": holding_hours,
                "regime_at_entry": regime_at_entry,
                "factor_contributions": contributions,
                "factor_signals_snapshot": signals_snapshot,
                "dominant_factor": dominant,
                "dominant_factor_correct": dominant_correct,
                "execution_cost_pct": self._estimate_execution_cost(holding_hours),
                "residual_pct": residual,
            }
            db.insert_trade_attribution(attribution_record)

            # 6. IC 히스토리 업데이트 (피드백 루프 핵심)
            self._update_ic_history(
                symbol=symbol,
                regime=regime_at_entry,
                signals_snapshot=signals_snapshot,
                realized_pnl_pct=realized_pnl_pct,
            )

            logger.info(
                f"[Attribution] {symbol} pnl={realized_pnl_pct:+.2f}% "
                f"dominant={dominant} correct={dominant_correct} "
                f"residual={residual:+.2f}%"
            )
            return attribution_record

        except Exception as e:
            logger.error(f"[Attribution] attribute() failed: {e}")
            return None

    # ── 기여도 계산 ────────────────────────────────────────────────────────────

    def _compute_contributions(
        self,
        signals: Dict[str, float],
        ic_weights: Dict[str, float],
        realized_pnl_pct: float,
    ) -> Dict[str, float]:
        """각 팩터의 수익 기여도(%) 계산.

        contribution_i = signal_i × realized_return × (IC_i / Σ|IC_j|)
        """
        total_ic = sum(abs(ic_weights.get(f, 0.01)) for f in FACTOR_NAMES)
        if total_ic < 1e-9:
            total_ic = 1.0

        contributions: Dict[str, float] = {}
        for fname in FACTOR_NAMES:
            sig = float(signals.get(fname, 0.0) or 0.0)
            ic = abs(float(ic_weights.get(fname, 0.01) or 0.01))
            ic_share = ic / total_ic
            # 기여도: 신호 방향이 맞으면 양수, 틀리면 음수
            contribution = sig * realized_pnl_pct * ic_share
            contributions[fname] = round(float(contribution), 4)

        return contributions

    # ── 실행 비용 추정 ────────────────────────────────────────────────────────

    @staticmethod
    def _estimate_execution_cost(holding_hours: float) -> float:
        """슬리피지 + 수수료 추정 (간략 모델).
        Binance 선물: Maker 0.02%, Taker 0.05%
        SMART_DCA 가정: 0.02% × 2 (진입 + 청산) = 0.04%
        """
        return 0.04  # 고정 추정값

    # ── IC 히스토리 업데이트 ──────────────────────────────────────────────────

    def _update_ic_history(
        self,
        symbol: str,
        regime: str,
        signals_snapshot: Dict[str, float],
        realized_pnl_pct: float,
    ) -> None:
        """DB에서 과거 (signal, return) 쌍 로드 → 새 쌍 추가 → IC 재계산 → 저장."""
        try:
            from config.database import db

            # 최근 90개 trade_attribution 로드 (최대 90 trades)
            attributions = db.get_trade_attributions(symbol=symbol, limit=90)

            for fname in FACTOR_NAMES:
                # 과거 기록 수집: (signal_value, realized_return, hours_ago)
                pairs: List[Tuple[float, float]] = []
                hours_ago_list: List[float] = []

                for attr in attributions:
                    snap = attr.get("factor_signals_snapshot") or {}
                    pnl = attr.get("realized_pnl_pct")
                    created_at = attr.get("created_at")
                    if pnl is None:
                        continue
                    sig_val = snap.get(fname, 0.0) or 0.0
                    pairs.append((float(sig_val), float(pnl)))

                    # 경과 시간 계산 (감쇠 가중치용)
                    try:
                        from datetime import datetime, timezone
                        if isinstance(created_at, str):
                            ts = datetime.fromisoformat(
                                created_at.replace("Z", "+00:00")
                            )
                        else:
                            ts = created_at
                        elapsed = (
                            datetime.now(timezone.utc) - ts
                        ).total_seconds() / 3600.0
                        hours_ago_list.append(max(0.0, float(elapsed)))
                    except Exception:
                        hours_ago_list.append(24.0)

                # 현재 거래의 신호/수익 추가
                current_sig = float(signals_snapshot.get(fname, 0.0) or 0.0)
                pairs.append((current_sig, float(realized_pnl_pct)))
                hours_ago_list.append(0.0)

                # 감쇠 가중치 계산
                weights = [
                    IC_DECAY_PER_HOUR ** h for h in hours_ago_list
                ]

                n = len(pairs)
                # IC 계산 (윈도우별)
                def _ic_window(window: int) -> Optional[float]:
                    if n < 3:
                        return None
                    sliced = pairs[-window:]
                    w_sliced = weights[-window:]
                    return _weighted_pearson(sliced, w_sliced)

                ic_7t = _ic_window(7)
                ic_30t = _ic_window(30)
                ic_all = _weighted_pearson(pairs, weights)
                ic_decay = _weighted_pearson(pairs, weights)  # decay already applied

                # 모두 None이면 저장 불필요
                if all(v is None for v in [ic_7t, ic_30t, ic_all]):
                    continue

                ic_record = {
                    "factor_name": fname,
                    "symbol": symbol,
                    "regime": regime,
                    "ic_7t": round(ic_7t, 5) if ic_7t is not None else None,
                    "ic_30t": round(ic_30t, 5) if ic_30t is not None else None,
                    "ic_all": round(ic_all, 5) if ic_all is not None else None,
                    "ic_decay_weighted": round(ic_decay, 5) if ic_decay is not None else None,
                    "sample_count": n,
                }
                db.upsert_factor_ic(ic_record)
                logger.debug(
                    f"[Attribution/IC] {fname} n={n} "
                    f"ic_30t={ic_30t:.4f if ic_30t else 'N/A'} "
                    f"ic_decay={ic_decay:.4f if ic_decay else 'N/A'}"
                )

        except Exception as e:
            logger.error(f"[Attribution] _update_ic_history failed: {e}")

    # ── 피드백 루프 트리거 (feedback_generator에서 호출) ──────────────────────

    def run_attribution_for_evaluation(
        self,
        evaluation: Dict,
    ) -> Optional[Dict]:
        """performance_evaluator 결과를 받아 attribution 실행.

        evaluation dict 구조:
          - symbol, mode, realized_return_pct, decision_id
          - entry_price, exit_price, horizon_minutes
        """
        try:
            symbol = evaluation.get("symbol", "BTCUSDT")
            mode = evaluation.get("mode", "swing")
            realized_pnl = float(evaluation.get("realized_return_pct", 0.0) or 0.0)
            decision_id = str(evaluation.get("decision_id") or "")
            entry_price = float(evaluation.get("entry_price", 0.0) or 0.0)
            exit_price = float(evaluation.get("exit_price", 0.0) or 0.0)
            horizon_minutes = float(evaluation.get("horizon_minutes", 240) or 240)
            holding_hours = horizon_minutes / 60.0
            regime = str(evaluation.get("regime_at_entry", "UNKNOWN") or "UNKNOWN")

            if not decision_id:
                return None

            return self.attribute(
                symbol=symbol,
                mode=mode,
                realized_pnl_pct=realized_pnl,
                decision_id=decision_id,
                entry_price=entry_price,
                exit_price=exit_price,
                holding_hours=holding_hours,
                regime_at_entry=regime,
            )
        except Exception as e:
            logger.error(f"[Attribution] run_attribution_for_evaluation failed: {e}")
            return None

    # ── 요약 리포트 ──────────────────────────────────────────────────────────

    def format_attribution_report(self, attribution: Dict) -> str:
        """Attribution 결과를 읽기 쉬운 문자열로 포맷."""
        if not attribution:
            return "[Attribution] No data available"

        pnl = attribution.get("realized_pnl_pct", 0.0)
        dominant = attribution.get("dominant_factor", "N/A")
        correct = attribution.get("dominant_factor_correct")
        contributions = attribution.get("factor_contributions") or {}

        lines = [
            f"[POST-TRADE ATTRIBUTION] PnL: {pnl:+.2f}%",
            f"Dominant Factor: {dominant} ({'CORRECT ✓' if correct else 'WRONG ✗' if correct is False else 'N/A'})",
            "",
            "Factor Contributions:",
        ]
        sorted_contribs = sorted(
            contributions.items(), key=lambda x: abs(x[1]), reverse=True
        )
        for fname, contrib in sorted_contribs:
            bar = "+" * int(abs(contrib) / 0.1) if abs(contrib) > 0.1 else "·"
            lines.append(f"  {fname:<32} {contrib:+.3f}%  {bar}")

        lines.append(
            f"\nExecution Cost: -{attribution.get('execution_cost_pct', 0.04):.2f}%  "
            f"Residual: {attribution.get('residual_pct', 0.0):+.3f}%"
        )
        return "\n".join(lines)


# ── Singleton ──────────────────────────────────────────────────────────────────
trade_attribution_engine = TradeAttributionEngine()
