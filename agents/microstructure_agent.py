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

    def analyze(self, microstructure_context: str = "", mode: str = "SWING") -> dict:
        """Deterministic quantitative microstructure analysis.

        Returns the same JSON schema as the old LLM-based agent for drop-in compatibility.
        """
        result = {
            "anomaly": "none",
            "confidence": 0.0,
            "imbalance": 0.0,
            "spread_bps": 0.0,
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

            # ── Spread crisis (independent of imbalance) ───────────────────────
            if spread_bps > self.SPREAD_CRISIS:
                result["anomaly"] = "microstructure_breakdown"
                result["confidence"] = min(spread_bps / 20.0, 1.0)
                result["directional_bias"] = "NEUTRAL"
                result["rationale"] = (
                    f"Spread {spread_bps:.1f}bps — liquidity crisis level. "
                    f"Flash crash / large OTC block risk. Avoid market orders."
                )
                return result

            # ── Imbalance signal ──────────────────────────────────────────────
            abs_imb = abs(imbalance)

            if abs_imb >= self.IMBALANCE_STRONG:
                # Spoofing check: extreme imbalance on wide spread = likely spoofed
                if abs_imb >= self.IMBALANCE_EXTREME and spread_bps > self.SPREAD_WIDE:
                    result["anomaly"] = "potential_spoofing"
                    result["confidence"] = 0.4
                    result["directional_bias"] = "NEUTRAL"
                    result["rationale"] = (
                        f"Extreme imbalance ({imbalance:.2f}) with wide spread "
                        f"({spread_bps:.1f}bps) — likely spoofed wall, directional signal unreliable."
                    )
                else:
                    conf = min(abs_imb * liq_mult, 1.0)
                    direction = "BID_HEAVY" if imbalance > 0 else "ASK_HEAVY"
                    bias = "BULLISH" if imbalance > 0 else "BEARISH"

                    # SWING mode: sustained imbalance matters more
                    # POSITION mode: single snapshot less reliable for multi-week thesis
                    if mode.upper() == "POSITION":
                        conf *= 0.6
                        bias_note = " (downweighted for POSITION timeframe)"
                    else:
                        bias_note = ""

                    result["anomaly"] = "microstructure_imbalance"
                    result["confidence"] = round(conf, 3)
                    result["directional_bias"] = bias
                    result["rationale"] = (
                        f"Orderbook {direction}: imbalance={imbalance:.3f}, "
                        f"spread={spread_bps:.2f}bps (quality={liq_label}){bias_note}."
                    )
            else:
                result["directional_bias"] = "NEUTRAL"
                result["rationale"] = (
                    f"Imbalance {imbalance:.3f} below threshold — no directional signal. "
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
