"""LiquidityAgent — quantitative liquidation + OI divergence signal extractor.

왜 LLM 불필요:
- liquidation_context: "[LIQUIDATION] Total=$120,000,000 (Long=$90M, Short=$30M)" — 수치
- funding_context: "[OI_DIV] OI_chg=+2.3% Price_chg=-0.8% Status=DIVERGENCE | [MFI_PROXY] OUTFLOW"

신호 로직 (기관급 청산 분석):
─────────────────────────────────────────────────────────────────────────────
1. Liquidation Cascade Detection
   - Long liq > $50M in context window → 롱 청산 폭포 시작 가능성
   - Short liq > $50M → 숏 청산 폭포 (short squeeze)
   - 치명적: 한쪽이 90% 이상이면 Directional Liq (한 방향으로만 청산)

2. OI Divergence Regime (CVD 대체)
   Status=DIVERGENCE이면:
   - OI▲ + Price▼ → 롱이 쌓이는데 가격 안 오름 → fragile long / 롱 스퀴즈 위험
   - OI▼ + Price▲ → 롱 청산 중인데 가격 오름 → 숏 스퀴즈 (강한 상승 신호)

3. MFI Proxy
   - INFLOW (OI▲ + Price▲) → 신규 자금 유입, 추세 확인
   - OUTFLOW (OI▼ + Price▼) → 자금 이탈, 추세 확인 (하락 방향으로)
   - NEUTRAL → 방향성 없음

4. Funding Rate (high positive = crowded long, contrarian SHORT signal)
   - funding_rate > 0.03% per 8h (annualized ~32%) → 극단적 롱 편향
   - funding_rate < -0.01% → 극단적 숏 편향

Confidence 공식:
  liq_score = log10(max(1, liq_usd_dominant / 50_000_000)) / 2  (0-1)
  oi_score  = 0.6 if DIVERGENCE else 0.3 if ALIGNED else 0
  mfi_score = 0.2 if INFLOW/OUTFLOW (directional) else 0
  total = min(liq_score + oi_score + mfi_score, 1.0)
"""

import re
import math
from loguru import logger


class LiquidityAgent:
    """Quantitative liquidation + OI divergence signal extractor (zero LLM cost)."""

    # Liquidation thresholds (USD)
    LIQ_MINOR = 20_000_000     # $20M  — notable
    LIQ_MAJOR = 50_000_000     # $50M  — significant cascade
    LIQ_EXTREME = 150_000_000  # $150M — extreme event
    SKEW_DIRECTIONAL = 0.80    # > 80% of total liq on one side = directional cascade

    # Funding thresholds (per 8h rate)
    FUNDING_CROWDED_LONG  = 0.0003   # +0.03% / 8h → 롱 과매수
    FUNDING_CROWDED_SHORT = -0.0001  # -0.01% / 8h → 숏 과매도

    def analyze(
        self,
        cvd_context: str = "",       # OI_DIV + MFI summary (funding_context의 끝부분)
        liquidation_context: str = "",
        mode: str = "SWING",
        funding_sma: float = None,   # 7-day average funding rate
        total_oi: float = None       # Total Open Interest
    ) -> dict:
        """Quantitative liquidity signal extraction.

        cvd_context: 이 파라미터는 호환성 유지용. funding_context의 OI_DIV 부분을 전달.
        """
        result = {
            "anomaly": "none",
            "confidence": 0.0,
            "target_entry": 0.0,
            "directional_bias": "NEUTRAL",
            "liq_dominant_usd": 0.0,
            "liq_side": "none",
            "oi_status": "UNKNOWN",
            "mfi_status": "UNKNOWN",
            "rationale": "no data",
        }

        liq_score = 0.0
        oi_score = 0.0
        mfi_score = 0.0
        signals = []

        # ── 1. Liquidation Analysis ───────────────────────────────────────────
        try:
            # Parse total and breakdown
            m_total = re.search(
                r"Total=\$?([\d,\.]+)",
                liquidation_context.replace(",", "")
            )
            m_long  = re.search(r"Long=\$?([\d,\.]+)", liquidation_context.replace(",", ""))
            m_short = re.search(r"Short=\$?([\d,\.]+)", liquidation_context.replace(",", ""))

            total_usd = float(m_total.group(1)) if m_total else 0.0
            long_usd  = float(m_long.group(1))  if m_long  else 0.0
            short_usd = float(m_short.group(1)) if m_short else 0.0

            if total_oi and total_oi > 0:
                liq_minor = total_oi * 0.002
                liq_major = total_oi * 0.005
                liq_extreme = total_oi * 0.015
            else:
                liq_minor = self.LIQ_MINOR
                liq_major = self.LIQ_MAJOR
                liq_extreme = self.LIQ_EXTREME

            if total_usd > liq_minor:
                long_ratio = long_usd / max(total_usd, 1)
                short_ratio = short_usd / max(total_usd, 1)

                # Determine dominant side
                if long_ratio >= self.SKEW_DIRECTIONAL:
                    dominant = "LONG"
                    dominant_usd = long_usd
                    result["liq_side"] = "LONG_DOMINATED"
                    # Mass long liquidation = price dropped hard = potential reversal zone below
                    result["directional_bias"] = "CAUTIOUS_LONG"  # buy the dip if OI confirms
                    signals.append(
                        f"Long liq cascade ${long_usd/1e6:.1f}M ({long_ratio*100:.0f}% of total)"
                    )
                elif short_ratio >= self.SKEW_DIRECTIONAL:
                    dominant = "SHORT"
                    dominant_usd = short_usd
                    result["liq_side"] = "SHORT_DOMINATED"
                    result["directional_bias"] = "CAUTIOUS_SHORT"  # potential short squeeze top
                    signals.append(
                        f"Short squeeze liq ${short_usd/1e6:.1f}M ({short_ratio*100:.0f}% of total)"
                    )
                else:
                    dominant = "MIXED"
                    dominant_usd = total_usd
                    result["liq_side"] = "MIXED"
                    signals.append(
                        f"Mixed liq ${total_usd/1e6:.1f}M "
                        f"(Long={long_ratio*100:.0f}% / Short={short_ratio*100:.0f}%)"
                    )

                result["liq_dominant_usd"] = round(dominant_usd, 0)

                # Score: log scale capped at 1.0
                liq_score = min(math.log10(max(1, dominant_usd / liq_major + 1)) / 1.0, 0.6)

                if total_usd >= liq_extreme:
                    result["anomaly"] = "liquidation_cascade_extreme"
                elif total_usd >= liq_major:
                    result["anomaly"] = "liquidation_cascade"
                else:
                    result["anomaly"] = "liquidation_minor"

        except Exception as e:
            logger.warning(f"LiquidityAgent liq parse error: {e}")

        # ── 2. OI Divergence (replaces CVD) ──────────────────────────────────
        try:
            ctx = cvd_context  # funding_context의 OI_DIV 부분

            # OI change %
            m_oi = re.search(r"OI_chg=([+-]?[\d\.]+)%", ctx)
            m_price = re.search(r"Price_chg=([+-]?[\d\.]+)%", ctx)

            oi_chg   = float(m_oi.group(1))    if m_oi    else 0.0
            price_chg = float(m_price.group(1)) if m_price else 0.0

            # OI Divergence status
            if "Status=DIVERGENCE" in ctx:
                oi_status = "DIVERGENCE"
                oi_score = 0.5
                # Interpret direction
                if oi_chg > 0 and price_chg < 0:
                    # Rising OI + falling price = new longs adding AGAINST trend = fragile
                    signals.append(
                        f"OI DIVERGENCE: OI+{oi_chg:.2f}% w/ Price{price_chg:.2f}% "
                        f"→ Fragile long stack, squeeze risk"
                    )
                    # Strengthen bearish liq signal
                    if result["liq_side"] == "SHORT_DOMINATED":
                        result["directional_bias"] = "BEARISH"  # confirm
                elif oi_chg < 0 and price_chg > 0:
                    # Falling OI + rising price = short covering into rally = strong
                    signals.append(
                        f"OI DIVERGENCE: OI{oi_chg:.2f}% w/ Price+{price_chg:.2f}% "
                        f"→ Short covering, momentum up"
                    )
                    result["directional_bias"] = "BULLISH"
            elif "Status=ALIGNED" in ctx:
                oi_status = "ALIGNED"
                oi_score = 0.2
                if oi_chg > 0 and price_chg > 0:
                    signals.append(f"OI ALIGNED: New money entering (OI+{oi_chg:.2f}%, Price+{price_chg:.2f}%)")
                elif oi_chg < 0 and price_chg < 0:
                    signals.append(f"OI ALIGNED: Derisking (OI{oi_chg:.2f}%, Price{price_chg:.2f}%)")
            else:
                oi_status = "UNKNOWN"

            result["oi_status"] = oi_status

        except Exception as e:
            logger.warning(f"LiquidityAgent OI parse error: {e}")

        # ── 3. MFI Proxy ──────────────────────────────────────────────────────
        try:
            if "[MFI_PROXY] INFLOW" in cvd_context:
                mfi_score = 0.2
                result["mfi_status"] = "INFLOW"
                signals.append("MFI: INFLOW (OI↑ + Price↑ = new capital entering)")
                if result["directional_bias"] == "NEUTRAL":
                    result["directional_bias"] = "BULLISH"
            elif "[MFI_PROXY] OUTFLOW" in cvd_context:
                mfi_score = 0.2
                result["mfi_status"] = "OUTFLOW"
                signals.append("MFI: OUTFLOW (OI↓ + Price↓ = capital leaving)")
                if result["directional_bias"] == "NEUTRAL":
                    result["directional_bias"] = "BEARISH"
            else:
                result["mfi_status"] = "NEUTRAL"
        except Exception as e:
            logger.warning(f"LiquidityAgent MFI parse error: {e}")

        # ── 4. Funding Rate Extreme Check ─────────────────────────────────────
        try:
            m_fr = re.search(r"funding_rate[\":\s]+([-]?[\d\.eE+-]+)", cvd_context)
            if m_fr:
                fr = float(m_fr.group(1))
                crowded_long = self.FUNDING_CROWDED_LONG
                crowded_short = self.FUNDING_CROWDED_SHORT
                
                if funding_sma is not None:
                    crowded_long = funding_sma + 0.0002
                    crowded_short = funding_sma - 0.0002

                if fr > crowded_long:
                    signals.append(
                        f"Funding CROWDED_LONG ({fr*100:.4f}%/8h) → contrarian SHORT signal"
                    )
                    if result["anomaly"] == "none":
                        result["anomaly"] = "funding_extreme_long"
                elif fr < crowded_short:
                    signals.append(
                        f"Funding CROWDED_SHORT ({fr*100:.4f}%/8h) → contrarian LONG signal"
                    )
                    if result["anomaly"] == "none":
                        result["anomaly"] = "funding_extreme_short"
        except Exception:
            pass

        # ── 5. Final Confidence + Rationale ──────────────────────────────────
        total_conf = min(liq_score + oi_score + mfi_score, 1.0)
        result["confidence"] = round(total_conf, 3)
        result["rationale"] = " | ".join(signals) if signals else "No significant liquidity signal"

        # POSITION mode: single-snapshot liq events less relevant for multi-week thesis
        if mode.upper() == "POSITION" and result["confidence"] > 0:
            result["confidence"] = round(result["confidence"] * 0.7, 3)
            result["rationale"] += " [downweighted for POSITION timeframe]"

        return result


liquidity_agent = LiquidityAgent()
