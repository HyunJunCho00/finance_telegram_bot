"""MicrostructureAgent — deterministic quantitative signal extractor.

왜 LLM 불필요: 입력이 이미 "[MICRO] spread=2.3bps imbalance=0.82" 형태의 정형 수치.
숫자 파싱 + 임계값 상태기계로 LLM과 동등한 정확도 달성 가능.

신호 로직 (기관급 마이크로스트럭처):
- Orderbook Imbalance: 매수벽/매도벽 불균형
  ≥ 0.7  → 강한 bid-side 압력 (short-term 상승 편향)
  ≤ -0.7 → 강한 ask-side 압력 (short-term 하락 편향)
  복합: spread도 좁으면 신뢰도 상승, 넓으면 신중

- Spread (bps): 유동성 지표
  < 1bps  → 매우 tight (고유동성, imbalance 신뢰도 HIGH)
  1-3bps  → 정상 (imbalance 신뢰도 MEDIUM)
  > 5bps  → wide (유동성 부족, imbalance 신뢰도 LOW — 배경 노이즈 가능)
  > 15bps → 위기 수준 (microstructure_breakdown)

- Slippage: 대형 주문 충격 추정
  > 0.1% → 대형 포지션 진입 비용이 높음

Confidence 공식:
  base = abs(imbalance) (0-1)
  liquidity_mult = 1.2 if spread < 1 else 1.0 if spread < 3 else 0.7 if spread < 8 else 0.4
  confidence = min(base * liquidity_mult, 1.0)
"""

import re
from loguru import logger
from processors.math_engine import calculate_z_score


class MicrostructureAgent:
    """Deterministic quantitative microstructure signal extractor."""

    # Regime thresholds (기관 표준)
    IMBALANCE_STRONG = 0.70    # abs(imbalance) > this = strong directional signal
    IMBALANCE_EXTREME = 0.85   # abs(imbalance) > this = extreme, potential spoofing risk
    SPREAD_TIGHT = 1.0         # bps — high liquidity
    SPREAD_NORMAL = 3.0        # bps — standard
    SPREAD_WIDE = 8.0          # bps — thin market
    SPREAD_CRISIS = 15.0       # bps — breakdown / flash crash risk
    SLIPPAGE_HIGH = 0.10       # % — large order impact

    def analyze(
        self,
        microstructure_context: str = "",
        mode: str = "SWING",
        micro_stats: dict = None,  # 7-day stats: {imbalance_mean, imbalance_std, spread_mean, spread_std}
    ) -> dict:
        """Deterministic quantitative microstructure analysis.

        Returns the same JSON schema as the old LLM-based agent for drop-in compatibility.
        """
        result = {
            "anomaly": "none",
            "confidence": 0.0,
            "imbalance": 0.0,
            "spread_bps": 0.0,
            "imbalance_z": None,   # Z-Score vs 7-day baseline (None if stats unavailable)
            "spread_z": None,
            "directional_bias": "NEUTRAL",
            "signal_quality": "LOW",      # LOW | MEDIUM | HIGH
            "rationale": "no microstructure data",
        }

        if not microstructure_context:
            return result

        try:
            # ── Parse fields ──────────────────────────────────────────────────
            imbalance = 0.0
            spread_bps = 0.0
            slippage_pct = 0.0

            m_imb = re.search(r"imbalance[:=]\s*([-]?[\d\.]+)", microstructure_context)
            if m_imb:
                imbalance = float(m_imb.group(1))

            m_spd = re.search(r"spread[:=]\s*([\d\.]+)\s*bps", microstructure_context, re.IGNORECASE)
            if m_spd:
                spread_bps = float(m_spd.group(1))

            m_slip = re.search(r"slippage[:=]\s*([\d\.]+)%?", microstructure_context, re.IGNORECASE)
            if m_slip:
                slippage_pct = float(m_slip.group(1))

            result["imbalance"] = round(imbalance, 4)
            result["spread_bps"] = round(spread_bps, 3)

            # ── Z-Score 계산 (동적 임계값용) ──────────────────────────────────
            imb_z = None
            spread_z = None
            if micro_stats and isinstance(micro_stats, dict):
                _imb_m   = micro_stats.get("imbalance_mean")
                _imb_s   = micro_stats.get("imbalance_std")
                _spd_m   = micro_stats.get("spread_mean")
                _spd_s   = micro_stats.get("spread_std")
                if _imb_m is not None and _imb_s and _imb_s > 0:
                    imb_z = calculate_z_score(abs(imbalance), abs(_imb_m), _imb_s)
                    result["imbalance_z"] = round(imb_z, 2)
                if _spd_m is not None and _spd_s and _spd_s > 0:
                    spread_z = calculate_z_score(spread_bps, _spd_m, _spd_s)
                    result["spread_z"] = round(spread_z, 2)

            # ── Liquidity multiplier (spread-based) ───────────────────────────
            if spread_bps == 0 or spread_bps < self.SPREAD_TIGHT:
                liq_mult = 1.2
                liq_label = "HIGH"
            elif spread_bps < self.SPREAD_NORMAL:
                liq_mult = 1.0
                liq_label = "MEDIUM"
            elif spread_bps < self.SPREAD_WIDE:
                liq_mult = 0.7
                liq_label = "LOW"
            else:
                liq_mult = 0.3
                liq_label = "VERY_LOW"

            result["signal_quality"] = liq_label

            abs_imb = abs(imbalance)

            # ── 우선 검사: Spread Z ≥ 3.0 + Imbalance Z ≥ 2.5 동시 발생 ───────
            # → '유동성 고갈에 의한 추세 전환' (단순 breakdown보다 강한 신호)
            if (spread_z is not None and spread_z >= 3.0
                    and imb_z is not None and imb_z >= 2.5):
                conf = min((imb_z + spread_z) / 10.0, 1.0)
                direction = "BID_HEAVY" if imbalance > 0 else "ASK_HEAVY"
                bias = "BULLISH" if imbalance > 0 else "BEARISH"
                if mode.upper() == "POSITION":
                    conf *= 0.6
                result["anomaly"] = "liquidity_exhaustion_breakout"
                result["confidence"] = round(conf, 3)
                result["directional_bias"] = bias
                result["rationale"] = (
                    f"Orderbook {direction}: imbalance={imbalance:.3f} Z={imb_z:.1f}, "
                    f"spread={spread_bps:.2f}bps Z={spread_z:.1f} "
                    f"— 유동성 고갈 + 강한 방향 압력 (추세 전환 가능)."
                )
                return result

            # ── Spread crisis 단독: 절대값 OR Z ≥ 3.0 (방향성 없음) ─────────────
            spread_crisis = (spread_bps > self.SPREAD_CRISIS) or (spread_z is not None and spread_z >= 3.0)
            if spread_crisis:
                result["anomaly"] = "microstructure_breakdown"
                result["confidence"] = min(spread_bps / 20.0, 1.0) if spread_bps > 0 else 0.8
                result["directional_bias"] = "NEUTRAL"
                z_note = f" Z={spread_z:.1f}" if spread_z is not None else ""
                result["rationale"] = (
                    f"Spread {spread_bps:.1f}bps{z_note} — 유동성 고갈 수준. "
                    f"Flash crash / large OTC block risk. Avoid market orders."
                )
                return result

            # ── Imbalance 게이트: Z ≥ 2.5 OR 절대값 ≥ IMBALANCE_STRONG ────────
            imb_gate = (imb_z is not None and imb_z >= 2.5) or (abs_imb >= self.IMBALANCE_STRONG)

            if imb_gate:
                z_imb_str = f" Z={imb_z:.1f}" if imb_z is not None else ""
                z_spd_str = f" SpreadZ={spread_z:.1f}" if spread_z is not None else ""

                # ── Spoofing 판단: Z-Score 활용 ────────────────────────────────
                # spread_z도 함께 높으면 '실제 유동성 고갈 + 압박' → spoofing 아님
                is_spoofed = (
                    abs_imb >= self.IMBALANCE_EXTREME
                    and spread_bps > self.SPREAD_WIDE
                    and (spread_z is None or spread_z < 3.0)
                )

                if is_spoofed:
                    result["anomaly"] = "potential_spoofing"
                    result["confidence"] = 0.4
                    result["directional_bias"] = "NEUTRAL"
                    result["rationale"] = (
                        f"Extreme imbalance ({imbalance:.2f}{z_imb_str}) with wide spread "
                        f"({spread_bps:.1f}bps{z_spd_str}) — likely spoofed wall, directional signal unreliable."
                    )
                else:
                    result["anomaly"] = "microstructure_imbalance"
                    conf = min(abs_imb * liq_mult, 1.0)

                    direction = "BID_HEAVY" if imbalance > 0 else "ASK_HEAVY"
                    bias = "BULLISH" if imbalance > 0 else "BEARISH"

                    if mode.upper() == "POSITION":
                        conf *= 0.6
                        bias_note = " (downweighted for POSITION timeframe)"
                    else:
                        bias_note = ""

                    result["confidence"] = round(conf, 3)
                    result["directional_bias"] = bias
                    result["rationale"] = (
                        f"Orderbook {direction}: imbalance={imbalance:.3f}{z_imb_str}, "
                        f"spread={spread_bps:.2f}bps{z_spd_str} (quality={liq_label}){bias_note}."
                    )
            else:
                result["directional_bias"] = "NEUTRAL"
                z_note = f" (Z={imb_z:.1f})" if imb_z is not None else ""
                result["rationale"] = (
                    f"Imbalance {imbalance:.3f}{z_note} below threshold — no directional signal. "
                    f"Spread={spread_bps:.2f}bps."
                )

            # ── High slippage addendum ─────────────────────────────────────────
            if slippage_pct > self.SLIPPAGE_HIGH:
                result["rationale"] += (
                    f" Slippage {slippage_pct:.2f}% > {self.SLIPPAGE_HIGH}% threshold — "
                    f"large order cost elevated."
                )

        except Exception as e:
            logger.warning(f"MicrostructureAgent parse error: {e}")

        return result


microstructure_agent = MicrostructureAgent()
