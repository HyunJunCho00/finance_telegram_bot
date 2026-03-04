"""MacroOptionsAgent — quantitative options + macro signal extractor.

왜 LLM을 "조건부로" 대체 가능한가:
입력이 "[DERIBIT BTC] DVOL=65 PCR=0.72" + "[MACRO] DGS10=4.2 DXY=104.5 NASDAQ=19500" 처럼
이미 정형화된 수치이므로 복합 상태기계로 LLM과 동등한 해석 가능.

단 LLM이 필요한 케이스 (현재 코드는 fallback sentinel로 표시):
- 서술형 이벤트가 deribit_context에 섞인 경우 (e.g. "Unusual block trade 50K contracts @ IV120")
- 여러 지표가 서로 모순되는 극단적 레짐 (DVOL↑ + PCR↓ = 상반된 공포 신호)
  → 이 경우 requires_llm=True를 반환해 meta_agent가 추가 해석

신호 로직:
─────────────────────────────────────────────────────────────────────────────
1. DVOL (Deribit Vol Index) — 공포/탐욕 지표
   < 40  : 저변동성 레짐 (complacency) → breakout risk HIGH (방향 불명)
   40-60 : 정상 레짐
   60-80 : 高변동성 (방어적 헤징 중) → BEARISH 편향
   > 80  : 공황 (→ 역발상: 너무 높으면 바닥 근처일 수 있음)

2. PCR OI (Put/Call Ratio by Open Interest)
   < 0.5 : 콜 편중 (탐욕/상승 베팅 과열) → contrarian BEARISH
   0.5-0.8: 균형
   > 0.8 : 풋 편중 (헤징 과열) → 하락 위험 HIGH, 하지만 too_crowded이면 반등 가능
   > 1.2 : 극단적 헤징 → 역발상 BULLISH (세력이 이미 방어 완료)

3. IV Term Structure (Skew)
   "INVERTED" in context → 단기 IV > 장기 IV = 즉각적 공황 헤징 (BEARISH 확정)
   skew25d:
     > +5% : 콜 스큐 (상방 프리미엄) → 기관이 upside를 사고 있음 = BULLISH
     < -5% : 풋 스큐 (하방 프리미엄) → 기관이 downside를 헤징 = BEARISH

4. Macro Context
   DGS10 (미국 10년물 금리):
     > 5.0% : 위험 자산 헤드윈드 (주식/코인 → BEARISH 환경)
     < 4.0% : 금리 환경 우호적
   DXY (달러 인덱스):
     > 106  : 달러 강세 = 위험 자산 전반 압력 (BEARISH)
     < 100  : 달러 약세 = 리스크온 환경 (BULLISH)
   NASDAQ (또는 Nasdaq 변동폭):
     > 0.5% 일일 상승 → 리스크온 spillover
     < -1%  → 리스크오프

복합 레짐 분류:
  PANIC_HEDGE:      DVOL>70 + INVERTED + PCR>0.9
  COMPLACENCY:      DVOL<45 + PCR<0.6
  INSTITUTIONAL_BUY: PCR>1.1 + skew25d>3 (기관이 콜 매수 중 = upside 기대)
  MACRO_HEADWIND:   DGS10>5.0 OR DXY>106
  NORMAL:           나머지
"""

import re
from typing import Optional
from loguru import logger


class MacroOptionsAgent:
    """Quantitative macro + options signal extractor."""

    # DVOL thresholds
    DVOL_LOW      = 40.0
    DVOL_ELEVATED = 60.0
    DVOL_HIGH     = 80.0

    # PCR thresholds
    PCR_CALL_OVERLOADED = 0.50   # < this: 콜 과열
    PCR_HEDGE_START     = 0.80
    PCR_HEDGE_EXTREME   = 1.20   # > this: 역발상 BULLISH

    # Skew thresholds (25d risk reversal, %)
    SKEW_CALL_PREMIUM = 5.0
    SKEW_PUT_PREMIUM  = -5.0

    # Macro thresholds
    DGS10_HEADWIND = 5.0
    DGS10_BENIGN   = 4.0
    DXY_STRONG     = 106.0
    DXY_WEAK       = 100.0

    def analyze(
        self,
        deribit_context: str = "",
        macro_context: str = "",
        mode: str = "SWING",
        dgs10_1w_chg: float = None  # 1-week change in 10Y yield (e.g. +0.4 for 40bps)
    ) -> dict:
        result = {
            "anomaly": "none",
            "options_bias": "NEUTRAL",
            "confidence": 0.0,
            "dvol": 0.0,
            "pcr": 0.0,
            "skew25d": 0.0,
            "regime": "NORMAL",
            "requires_llm": False,   # True = meta_agent에게 추가 해석 위임 신호
            "rationale": "no options data",
            "macro_signal": "NEUTRAL",
        }

        signals = []
        option_score = 0.0
        macro_score = 0.0
        contradictions = 0  # 지표 모순 카운트 → requires_llm 트리거

        # ── 1. Parse Deribit Context ──────────────────────────────────────────
        try:
            dvol  = self._parse_float(r"DVOL=([\d\.]+)", deribit_context)
            pcr   = self._parse_float(r"PCR=?(oi)?\s*=?\s*([\d\.]+)", deribit_context, group=2)
            skew  = self._parse_float(r"skew25d=([-]?[\d\.]+)", deribit_context)

            if dvol is not None:
                result["dvol"] = dvol
            if pcr is not None:
                result["pcr"] = pcr
            if skew is not None:
                result["skew25d"] = skew

            iv_inverted = "INVERTED" in deribit_context

            # ── DVOL signal ─────────────────────────────────────────────────
            if dvol is not None:
                if dvol > self.DVOL_HIGH:
                    signals.append(f"DVOL={dvol:.0f} (PANIC level)")
                    option_score += 0.4
                    # Counter-signal: extreme fear can be buy
                    if dvol > 90:
                        signals.append(f"DVOL>{dvol:.0f} EXTREME → potential capitulation bottom")
                elif dvol > self.DVOL_ELEVATED:
                    signals.append(f"DVOL={dvol:.0f} (elevated)")
                    option_score += 0.2
                elif dvol < self.DVOL_LOW:
                    signals.append(f"DVOL={dvol:.0f} (complacency—breakout risk)")
                    option_score += 0.1

            # ── IV Term Inversion ───────────────────────────────────────────
            if iv_inverted:
                signals.append("IV term INVERTED: immediate panic hedge, BEARISH confirmed")
                result["anomaly"] = "options_panic"
                result["options_bias"] = "BEARISH"
                option_score += 0.5
            elif dvol is not None and dvol > self.DVOL_ELEVATED:
                result["options_bias"] = "BEARISH"

            # ── PCR signal ──────────────────────────────────────────────────
            if pcr is not None:
                if pcr < self.PCR_CALL_OVERLOADED:
                    signals.append(f"PCR={pcr:.2f} (call overloaded—contrarian BEARISH)")
                    if result["options_bias"] == "BULLISH":
                        contradictions += 1  # skew says bullish, PCR says contrarian-bearish
                    result["options_bias"] = "BEARISH" if result["options_bias"] == "NEUTRAL" else result["options_bias"]
                    option_score += 0.25
                elif pcr > self.PCR_HEDGE_EXTREME:
                    signals.append(f"PCR={pcr:.2f} (extreme put hedge—contrarian BULLISH)")
                    if result["options_bias"] == "BEARISH":
                        contradictions += 1
                    result["options_bias"] = "BULLISH"
                    option_score += 0.30
                elif pcr > self.PCR_HEDGE_START:
                    signals.append(f"PCR={pcr:.2f} (elevated put hedging—BEARISH bias)")
                    option_score += 0.15

            # ── 25d Skew ────────────────────────────────────────────────────
            if skew is not None:
                if skew > self.SKEW_CALL_PREMIUM:
                    signals.append(f"25d skew={skew:+.1f}% (call premium—institutional LONG demand)")
                    if result["options_bias"] != "BEARISH":
                        result["options_bias"] = "BULLISH"
                    option_score += 0.2
                elif skew < self.SKEW_PUT_PREMIUM:
                    signals.append(f"25d skew={skew:+.1f}% (put premium—downside hedge active)")
                    if result["options_bias"] != "BULLISH":
                        result["options_bias"] = "BEARISH"
                    option_score += 0.2

            # ── Contradictions → requires_llm ───────────────────────────────
            if contradictions >= 2:
                result["requires_llm"] = True
                signals.append("⚠ Contradicting options signals: meta_agent LLM recommended")

        except Exception as e:
            logger.warning(f"MacroOptionsAgent deribit parse error: {e}")

        # ── 2. Parse Macro Context ────────────────────────────────────────────
        try:
            dgs10   = self._parse_float(r"DGS10=([\d\.]+)", macro_context)
            dxy     = self._parse_float(r"DXY=([\d\.]+)", macro_context)
            nasdaq  = self._parse_float(r"NASDAQ=([\d,\.]+)", macro_context)

            macro_signal = "NEUTRAL"

            # ── Relative Rate of Change (Rate Shock) ──
            if dgs10_1w_chg is not None:
                if dgs10_1w_chg > 0.3:
                    signals.append(f"DGS10 surged {dgs10_1w_chg:+.2f}% (Rate shock—BEARISH macro)")
                    macro_signal = "BEARISH"
                    macro_score += 0.30
                elif dgs10_1w_chg < -0.3:
                    signals.append(f"DGS10 fell {dgs10_1w_chg:+.2f}% (Yields dropping—BULLISH macro)")
                    macro_signal = "BULLISH"
                    macro_score += 0.20

            if dgs10 is not None and macro_signal == "NEUTRAL":
                if dgs10 > self.DGS10_HEADWIND:
                    signals.append(f"DGS10={dgs10:.2f}% (risk headwind—BEARISH macro)")
                    macro_signal = "BEARISH"
                    macro_score += 0.25
                elif dgs10 < self.DGS10_BENIGN:
                    signals.append(f"DGS10={dgs10:.2f}% (rate-friendly—BULLISH macro)")
                    if macro_signal != "BEARISH":
                        macro_signal = "BULLISH"
                    macro_score += 0.15

            if dxy is not None:
                if dxy > self.DXY_STRONG:
                    signals.append(f"DXY={dxy:.1f} (strong USD—BEARISH risk assets)")
                    macro_signal = "BEARISH"
                    macro_score += 0.20
                elif dxy < self.DXY_WEAK:
                    signals.append(f"DXY={dxy:.1f} (weak USD—BULLISH risk assets)")
                    if macro_signal != "BEARISH":
                        macro_signal = "BULLISH"
                    macro_score += 0.15

            result["macro_signal"] = macro_signal
            # Macro BEARISH overrides NEUTRAL options bias but not BULLISH (requires both)
            if macro_signal == "BEARISH" and result["options_bias"] != "BULLISH":
                result["options_bias"] = "BEARISH"
                if result["anomaly"] == "none":
                    result["anomaly"] = "macro_divergence"

        except Exception as e:
            logger.warning(f"MacroOptionsAgent macro parse error: {e}")

        # ── 3. Regime Classifier ──────────────────────────────────────────────
        dvol_val = result["dvol"]
        pcr_val  = result["pcr"]
        inverted = "INVERTED" in deribit_context

        if dvol_val > self.DVOL_ELEVATED and inverted and pcr_val > self.PCR_HEDGE_START:
            result["regime"] = "PANIC_HEDGE"
        elif dvol_val < self.DVOL_LOW and pcr_val < self.PCR_CALL_OVERLOADED:
            result["regime"] = "COMPLACENCY"
        elif pcr_val > self.PCR_HEDGE_EXTREME and result["skew25d"] > 2:
            result["regime"] = "INSTITUTIONAL_BUY"
        elif result["macro_signal"] == "BEARISH":
            result["regime"] = "MACRO_HEADWIND"
        else:
            result["regime"] = "NORMAL"

        # ── 4. POSITION mode adjustment ───────────────────────────────────────
        # Options snapshots are less reliable for multi-week thesis
        # But macro context (DGS10, DXY) remains highly relevant
        if mode.upper() == "POSITION":
            option_score *= 0.7   # Options snapshot less reliable long-term
            # macro_score unchanged — DGS10/DXY are multi-month signals

        # ── 5. Final confidence ────────────────────────────────────────────────
        total_conf = min(option_score + macro_score, 1.0)
        result["confidence"] = round(total_conf, 3)
        result["rationale"] = " | ".join(signals) if signals else "No significant options/macro signal"

        return result

    def _parse_float(self, pattern: str, text: str, group: int = 1) -> Optional[float]:
        """Safe regex float extractor."""
        try:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return float(m.group(group).replace(",", ""))
        except Exception:
            pass
        return None


macro_options_agent = MacroOptionsAgent()
