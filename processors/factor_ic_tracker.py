# -*- coding: utf-8 -*-
"""Layer 1: Factor Signal Engine + IC Tracker.

Institutional quant 방식: 각 신호(팩터)의 예측력(IC, Information Coefficient)을 추적하고,
IC 가중 합산으로 방향성 알파 스코어를 계산한다.

IC = Pearson correlation(signal_at_entry, subsequent_realized_return)

높은 IC → 이 신호가 실제 수익에 기여 → 가중치 올림
낮은 IC → 이 신호가 노이즈 → 가중치 내림

도메인 사전 확률(prior):
  거래 이력이 충분하지 않을 때 사용하는 기본 IC 값.
  크립토 관련 학술 연구와 실전 경험에 근거.
"""

from __future__ import annotations
import math
from typing import Dict, Optional, List, Tuple
from loguru import logger


# ── 도메인 사전 IC (거래 이력 부족 시 기본값) ─────────────────────────────────
# 양수 = 신호 방향이 수익 방향과 일치하는 경향
# (예: funding_rate 신호는 역추세 → 실제로 약하게 작동)
_IC_PRIORS: Dict[str, float] = {
    "funding_rate_signal":       0.05,  # 역추세: 극단 펀딩 → 반전 신호
    "oi_change_signal":          0.03,  # OI 급증 = 포지션 과열 → 약한 역추세
    "liquidation_trap_signal":   0.07,  # 청산 트랩 후 반전 → 강한 신호
    "onchain_flow_signal":       0.06,  # 거래소 유출 = 축적 → 강한 신호
    "narrative_sentiment_signal":0.03,  # 내러티브는 후행 → 약한 신호
    "rsi_divergence_signal":     0.05,  # RSI 다이버전스 → 중간 신호
    "macro_regime_signal":       0.04,  # 레짐 방향성 → 중간 신호
    "microstructure_signal":     0.04,  # 오더북 불균형 → 중간 신호
}

# 레짐별 IC 조정 배수 (특정 레짐에서 특정 신호가 더/덜 유효)
_REGIME_IC_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    "BULL_MOMENTUM": {
        "funding_rate_signal": 0.8,       # 추세장에서 역추세 덜 작동
        "macro_regime_signal": 1.3,       # 레짐 신호 더 작동
        "onchain_flow_signal": 1.2,
    },
    "BEAR_MOMENTUM": {
        "funding_rate_signal": 0.8,
        "macro_regime_signal": 1.3,
        "liquidation_trap_signal": 1.2,   # 청산 연쇄 후 반전 잦음
    },
    "RANGE_BOUND": {
        "rsi_divergence_signal": 1.4,     # 레인지에서 RSI 다이버전스 강함
        "microstructure_signal": 1.3,
        "funding_rate_signal": 1.2,       # 레인지에서 역추세 잘 작동
    },
    "VOLATILITY_PANIC": {
        "macro_regime_signal": 1.5,       # 패닉장 = 매크로 판단이 핵심
        "narrative_sentiment_signal": 0.5, # 패닉 내러티브는 노이즈
        "onchain_flow_signal": 0.7,       # 온체인 데이터 지연됨
    },
    "SIDEWAYS_ACCUMULATION": {
        "onchain_flow_signal": 1.5,       # 축적 감지에 온체인이 핵심
        "microstructure_signal": 1.2,
        "narrative_sentiment_signal": 0.8,
    },
}

# 최소 샘플 수 (이 미만이면 사전 확률 비중 높임)
_MIN_SAMPLES_FOR_FULL_IC = 20
_MIN_SAMPLES_FOR_ANY_IC = 5


class FactorICTracker:
    """Factor IC 추적 및 IC-가중 알파 스코어 계산."""

    def __init__(self) -> None:
        # 인메모리 캐시: {(factor_name, symbol, regime): [(signal, return), ...]}
        self._ic_cache: Dict[Tuple, List[Tuple[float, float]]] = {}
        # 마지막으로 DB에서 로드한 IC 값 캐시
        self._loaded_ic: Dict[str, float] = {}

    # ── 신호 추출 ──────────────────────────────────────────────────────────────

    def extract_signals(
        self,
        state: Dict,
        regime: str = "UNKNOWN",
    ) -> Dict[str, float]:
        """AnalysisState에서 각 팩터의 정규화 신호값(-1 ~ +1)을 추출."""
        signals: Dict[str, float] = {}

        # 1. funding_rate_signal
        try:
            raw_funding = state.get("raw_funding") or {}
            funding_rate = float(raw_funding.get("funding_rate", 0) or 0)
            # 극단 펀딩 = 포지션 과열 → 역추세 신호
            # +0.03% 이상(롱 과열) → 숏 방향 = 음수
            # -0.03% 이하(숏 과열) → 롱 방향 = 양수
            normalized = -math.tanh(funding_rate / 0.015)  # 0.015% = 1 std
            signals["funding_rate_signal"] = round(float(normalized), 4)
        except Exception:
            signals["funding_rate_signal"] = 0.0

        # 2. oi_change_signal (OI 4h 변화율)
        try:
            raw_funding = state.get("raw_funding") or {}
            oi_binance = float(raw_funding.get("oi_binance", 0) or 0)
            # OI 컨텍스트에서 변화율 파싱 (funding_context 문자열에서)
            fc = state.get("funding_context", "") or ""
            oi_chg = 0.0
            for part in fc.split():
                if "oi_chg" in part.lower() or "oichg" in part.lower():
                    try:
                        oi_chg = float(part.split("=")[-1].replace("%",""))
                    except Exception:
                        pass
            # OI 급증 = 포지션 과열 = 약한 역추세 신호
            normalized = -math.tanh(oi_chg / 3.0)  # 3% = 1 std
            signals["oi_change_signal"] = round(float(normalized), 4)
        except Exception:
            signals["oi_change_signal"] = 0.0

        # 3. liquidation_trap_signal (Liquidity Agent 결과)
        try:
            blackboard = state.get("blackboard") or {}
            liq_result = blackboard.get("liquidity") or {}
            bias = str(liq_result.get("directional_bias", "NEUTRAL")).upper()
            conf = float(liq_result.get("confidence", 0.3) or 0.3)
            bias_map = {
                "BULLISH": +1.0, "CAUTIOUS_LONG": +0.5,
                "BEARISH": -1.0, "CAUTIOUS_SHORT": -0.5,
                "NEUTRAL": 0.0,
            }
            signals["liquidation_trap_signal"] = round(
                bias_map.get(bias, 0.0) * min(conf * 2, 1.0), 4
            )
        except Exception:
            signals["liquidation_trap_signal"] = 0.0

        # 4. onchain_flow_signal (온체인 게이트 + 스냅샷)
        try:
            onchain_gate = state.get("onchain_gate") or {}
            onchain_snap = state.get("onchain_snapshot") or {}
            raw = onchain_snap.get("raw_metrics") or {}
            # 거래소 유입/유출 (exchange_net_flow: 양수 = 유입 = 매도 압력 = 베어리시)
            net_flow = float(raw.get("exchange_net_flow", 0) or 0)
            # 정규화: 유출(음수) → 양수 신호, 유입(양수) → 음수 신호
            if net_flow != 0:
                normalized = -math.tanh(net_flow / (abs(net_flow) * 5 + 1e-9))
            else:
                normalized = 0.0
            # onchain_gate의 allow_long/allow_short도 반영
            if not onchain_gate.get("allow_long", True):
                normalized = min(normalized, -0.3)
            if not onchain_gate.get("allow_short", True):
                normalized = max(normalized, +0.3)
            signals["onchain_flow_signal"] = round(float(normalized), 4)
        except Exception:
            signals["onchain_flow_signal"] = 0.0

        # 5. narrative_sentiment_signal (Perplexity + RAG)
        try:
            # unified_narrative에서 bullish/bearish 키워드 카운트
            narrative = (state.get("unified_narrative") or
                         state.get("narrative_text") or "")
            narrative_lower = narrative.lower()
            bull_kw = ["bullish", "accumulation", "breakout", "rally", "etf inflow",
                       "institutional buy", "whale buy", "support held"]
            bear_kw = ["bearish", "distribution", "breakdown", "dump", "etf outflow",
                       "institutional sell", "whale sell", "resistance"]
            bull_count = sum(narrative_lower.count(kw) for kw in bull_kw)
            bear_count = sum(narrative_lower.count(kw) for kw in bear_kw)
            total = bull_count + bear_count
            if total > 0:
                normalized = (bull_count - bear_count) / total
            else:
                normalized = 0.0
            signals["narrative_sentiment_signal"] = round(float(normalized), 4)
        except Exception:
            signals["narrative_sentiment_signal"] = 0.0

        # 6. rsi_divergence_signal (market_data_compact에서)
        try:
            mdc = state.get("market_data_compact") or ""
            mdc_lower = mdc.lower()
            if "bullish divergence" in mdc_lower or "positive divergence" in mdc_lower:
                signals["rsi_divergence_signal"] = +0.7
            elif "bearish divergence" in mdc_lower or "negative divergence" in mdc_lower:
                signals["rsi_divergence_signal"] = -0.7
            else:
                signals["rsi_divergence_signal"] = 0.0
        except Exception:
            signals["rsi_divergence_signal"] = 0.0

        # 7. macro_regime_signal (Meta Agent 결과)
        try:
            regime_map = {
                "BULL_MOMENTUM": +1.0,
                "SIDEWAYS_ACCUMULATION": +0.3,
                "RANGE_BOUND": 0.0,
                "BEAR_MOMENTUM": -1.0,
                "VOLATILITY_PANIC": -0.5,
            }
            signals["macro_regime_signal"] = float(
                regime_map.get(regime.upper(), 0.0)
            )
        except Exception:
            signals["macro_regime_signal"] = 0.0

        # 8. microstructure_signal (Microstructure Agent)
        try:
            blackboard = state.get("blackboard") or {}
            micro = blackboard.get("microstructure") or {}
            imbalance = float(micro.get("imbalance", 0.5) or 0.5)
            # imbalance: 0.5 = neutral, > 0.5 = bid-heavy (bullish), < 0.5 = ask-heavy
            normalized = (imbalance - 0.5) * 2.0  # → -1 ~ +1
            normalized = max(-1.0, min(1.0, normalized))
            signals["microstructure_signal"] = round(float(normalized), 4)
        except Exception:
            signals["microstructure_signal"] = 0.0

        return signals

    # ── IC 로드 ────────────────────────────────────────────────────────────────

    def load_ic_from_db(
        self,
        symbol: str,
        regime: Optional[str] = None,
    ) -> Dict[str, float]:
        """DB에서 최신 IC 값을 로드. 없으면 사전 확률(prior) 반환."""
        try:
            from config.database import db
            records = db.get_factor_ic_history(symbol=symbol, regime=regime)
            ic_map: Dict[str, float] = {}
            for rec in records:
                fname = rec.get("factor_name")
                if not fname:
                    continue
                # ic_decay_weighted 우선, 없으면 ic_30t, 없으면 prior
                ic_val = (
                    rec.get("ic_decay_weighted")
                    or rec.get("ic_30t")
                    or rec.get("ic_7t")
                )
                if ic_val is not None and rec.get("sample_count", 0) >= _MIN_SAMPLES_FOR_ANY_IC:
                    # 실제 IC와 prior를 샘플 수에 따라 혼합
                    n = rec.get("sample_count", 0)
                    blend = min(n / _MIN_SAMPLES_FOR_FULL_IC, 1.0)
                    prior = _IC_PRIORS.get(fname, 0.03)
                    ic_map[fname] = blend * float(ic_val) + (1 - blend) * prior
                else:
                    ic_map[fname] = _IC_PRIORS.get(fname, 0.03)

            # 누락된 팩터는 prior로 채움
            for fname, prior in _IC_PRIORS.items():
                if fname not in ic_map:
                    ic_map[fname] = prior

            return ic_map
        except Exception as e:
            logger.warning(f"[FactorIC] DB load failed: {e} — using priors")
            return dict(_IC_PRIORS)

    # ── IC 조정 ────────────────────────────────────────────────────────────────

    def apply_regime_adjustment(
        self,
        ic_map: Dict[str, float],
        regime: str,
    ) -> Dict[str, float]:
        """레짐별 IC 배수 적용."""
        multipliers = _REGIME_IC_MULTIPLIERS.get(regime.upper(), {})
        adjusted = {}
        for fname, ic_val in ic_map.items():
            mult = multipliers.get(fname, 1.0)
            adjusted[fname] = ic_val * mult
        return adjusted

    # ── 알파 스코어 계산 ──────────────────────────────────────────────────────

    def compute_alpha_score(
        self,
        signals: Dict[str, float],
        ic_map: Dict[str, float],
    ) -> float:
        """IC-가중 방향성 알파 스코어 계산 (-1 ~ +1).

        alpha = Σ(signal_i × IC_i) / Σ(IC_i)
        """
        numerator = 0.0
        denominator = 0.0
        for fname, signal_val in signals.items():
            ic_val = abs(ic_map.get(fname, 0.0))
            if ic_val > 0:
                numerator += signal_val * ic_val
                denominator += ic_val
        if denominator < 1e-9:
            return 0.0
        score = numerator / denominator
        return round(max(-1.0, min(1.0, float(score))), 4)

    # ── 메인 엔트리포인트 ─────────────────────────────────────────────────────

    def compute(
        self,
        state: Dict,
        regime: str = "UNKNOWN",
        symbol: str = "BTCUSDT",
    ) -> Dict:
        """신호 추출 → IC 로드 → 레짐 조정 → 알파 스코어 계산.

        Returns:
            {
                "signals": dict,        각 팩터 정규화 신호값
                "ic_weights": dict,     레짐 조정 IC 가중치
                "alpha_score": float,   IC-가중 합산 알파 스코어
                "regime": str,
                "confidence_label": str
            }
        """
        signals = self.extract_signals(state, regime=regime)
        raw_ic = self.load_ic_from_db(symbol=symbol, regime=regime)
        ic_adjusted = self.apply_regime_adjustment(raw_ic, regime)
        alpha_score = self.compute_alpha_score(signals, ic_adjusted)

        # 확신도 레이블
        abs_alpha = abs(alpha_score)
        if abs_alpha >= 0.5:
            confidence_label = "HIGH"
        elif abs_alpha >= 0.3:
            confidence_label = "MEDIUM"
        elif abs_alpha >= 0.1:
            confidence_label = "LOW"
        else:
            confidence_label = "VERY_LOW"

        result = {
            "signals": signals,
            "ic_weights": {k: round(v, 4) for k, v in ic_adjusted.items()},
            "alpha_score": alpha_score,
            "regime": regime,
            "confidence_label": confidence_label,
        }

        logger.info(
            f"[FactorIC] {symbol} regime={regime} "
            f"alpha={alpha_score:.3f} ({confidence_label}) "
            f"top_signal={max(signals, key=lambda k: abs(signals[k]))}"
        )
        return result

    # ── DB 저장 ───────────────────────────────────────────────────────────────

    def save_signal_snapshot(
        self,
        symbol: str,
        mode: str,
        result: Dict,
        decision_id: Optional[str] = None,
        final_decision: str = "HOLD",
        allocation_pct: float = 0.0,
    ) -> Optional[int]:
        """factor_signals 테이블에 스냅샷 저장. 반환값: 저장된 row id."""
        try:
            from config.database import db
            signals = result.get("signals", {})
            record = {
                "symbol": symbol,
                "mode": mode,
                "decision_id": decision_id,
                "regime": result.get("regime"),
                "funding_rate_signal": signals.get("funding_rate_signal"),
                "oi_change_signal": signals.get("oi_change_signal"),
                "liquidation_trap_signal": signals.get("liquidation_trap_signal"),
                "onchain_flow_signal": signals.get("onchain_flow_signal"),
                "narrative_sentiment_signal": signals.get("narrative_sentiment_signal"),
                "rsi_divergence_signal": signals.get("rsi_divergence_signal"),
                "macro_regime_signal": signals.get("macro_regime_signal"),
                "microstructure_signal": signals.get("microstructure_signal"),
                "ic_weights": result.get("ic_weights", {}),
                "alpha_score": result.get("alpha_score"),
                "final_decision": final_decision,
                "allocation_pct": allocation_pct,
            }
            saved = db.insert_factor_signals(record)
            return saved.get("id") if saved else None
        except Exception as e:
            logger.error(f"[FactorIC] save_signal_snapshot failed: {e}")
            return None

    # ── 요약 문자열 (Judge에게 전달) ─────────────────────────────────────────

    def format_for_judge(self, result: Dict) -> str:
        """Judge Agent에게 전달할 IC 분석 요약 문자열."""
        signals = result.get("signals", {})
        ic = result.get("ic_weights", {})
        alpha = result.get("alpha_score", 0.0)
        conf = result.get("confidence_label", "UNKNOWN")

        lines = [
            f"[FACTOR IC ANALYSIS] Alpha Score: {alpha:+.3f} ({conf})",
            f"Regime: {result.get('regime', 'UNKNOWN')}",
            "",
            "Factor Signals (IC weight | normalized signal):",
        ]
        # IC 가중치 내림차순 정렬
        sorted_factors = sorted(
            _IC_PRIORS.keys(),
            key=lambda k: ic.get(k, 0.0),
            reverse=True,
        )
        for fname in sorted_factors:
            sig = signals.get(fname, 0.0)
            ic_val = ic.get(fname, 0.0)
            direction = "↑" if sig > 0.1 else ("↓" if sig < -0.1 else "→")
            lines.append(
                f"  {fname:<32} IC={ic_val:.3f}  sig={sig:+.3f} {direction}"
            )
        lines.append("")
        lines.append(
            "Interpretation: "
            + ("Strong bullish signal" if alpha > 0.4 else
               "Moderate bullish signal" if alpha > 0.2 else
               "Neutral / conflicting" if abs(alpha) <= 0.2 else
               "Moderate bearish signal" if alpha > -0.4 else
               "Strong bearish signal")
        )
        return "\n".join(lines)


# ── Singleton ──────────────────────────────────────────────────────────────────
factor_ic_tracker = FactorICTracker()
