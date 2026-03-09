"""Quantitative liquidation + OI divergence signal extractor."""

from __future__ import annotations

import math
import re

from loguru import logger

from processors.math_engine import calculate_z_score


class LiquidityAgent:
    """Quantitative liquidation + OI divergence signal extractor (zero LLM cost)."""

    LIQ_MINOR = 20_000_000
    LIQ_MAJOR = 50_000_000
    LIQ_EXTREME = 150_000_000
    SKEW_DIRECTIONAL = 0.80

    # Relative/liquidity shock ratios versus aggregate OI.
    LIQ_TO_OI_NOTABLE = 0.002
    LIQ_TO_OI_MAJOR = 0.005
    LIQ_TO_OI_EXTREME = 0.010

    # Robust anomaly gates.
    ROBUST_Z_NOTABLE = 2.5
    ROBUST_Z_MAJOR = 3.5
    ROBUST_Z_EXTREME = 5.0

    FUNDING_CROWDED_LONG = 0.0003
    FUNDING_CROWDED_SHORT = -0.0001

    @staticmethod
    def _safe_float(value) -> float | None:
        try:
            if value in (None, "", "N/A"):
                return None
            return float(value)
        except Exception:
            return None

    def _robust_z_score(self, current: float, median: float | None, mad: float | None) -> float | None:
        if median is None or mad is None or mad <= 0:
            return None
        robust_scale = 1.4826 * mad
        if robust_scale <= 0:
            return None
        return (current - median) / robust_scale

    def _percentile_band(self, total_usd: float, liq_stats: dict | None) -> str:
        if not liq_stats:
            return "unknown"
        p99 = self._safe_float(liq_stats.get("p99"))
        p95 = self._safe_float(liq_stats.get("p95"))
        p90 = self._safe_float(liq_stats.get("p90"))
        if p99 is not None and total_usd >= p99:
            return "p99"
        if p95 is not None and total_usd >= p95:
            return "p95"
        if p90 is not None and total_usd >= p90:
            return "p90"
        return "sub_p90"

    def analyze(
        self,
        cvd_context: str = "",
        liquidation_context: str = "",
        mode: str = "SWING",
        funding_sma: float = None,
        total_oi: float = None,
        liq_stats: dict = None,
    ) -> dict:
        result = {
            "anomaly": "none",
            "confidence": 0.0,
            "target_entry": 0.0,
            "directional_bias": "NEUTRAL",
            "liq_dominant_usd": 0.0,
            "liq_side": "none",
            "liq_z_score": None,
            "liq_robust_z": None,
            "liq_to_oi_ratio": None,
            "liq_percentile_band": "unknown",
            "oi_status": "UNKNOWN",
            "mfi_status": "UNKNOWN",
            "rationale": "no data",
        }

        liq_score = 0.0
        oi_score = 0.0
        mfi_score = 0.0
        signals: list[str] = []

        try:
            clean = liquidation_context.replace(",", "")
            m_total = re.search(r"Total=\$?([\d,\.]+)", clean)
            m_long = re.search(r"Long=\$?([\d,\.]+)", clean)
            m_short = re.search(r"Short=\$?([\d,\.]+)", clean)

            total_usd = float(m_total.group(1)) if m_total else 0.0
            long_usd = float(m_long.group(1)) if m_long else 0.0
            short_usd = float(m_short.group(1)) if m_short else 0.0
            if total_usd <= 0:
                total_usd = long_usd + short_usd

            if total_oi and total_oi > 0:
                liq_ratio = total_usd / total_oi
                liq_minor = total_oi * self.LIQ_TO_OI_NOTABLE
                liq_major = total_oi * self.LIQ_TO_OI_MAJOR
                liq_extreme = total_oi * self.LIQ_TO_OI_EXTREME
                result["liq_to_oi_ratio"] = round(liq_ratio, 6)
            else:
                liq_ratio = None
                liq_minor = self.LIQ_MINOR
                liq_major = self.LIQ_MAJOR
                liq_extreme = self.LIQ_EXTREME

            liq_z = None
            robust_z = None
            percentile_band = self._percentile_band(total_usd, liq_stats)
            result["liq_percentile_band"] = percentile_band
            if liq_stats and isinstance(liq_stats, dict):
                mean = self._safe_float(liq_stats.get("mean"))
                std = self._safe_float(liq_stats.get("std"))
                median = self._safe_float(liq_stats.get("median"))
                mad = self._safe_float(liq_stats.get("mad"))
                if mean is not None and std is not None and std > 0:
                    liq_z = calculate_z_score(total_usd, mean, std)
                    result["liq_z_score"] = round(liq_z, 2)
                robust_z = self._robust_z_score(total_usd, median, mad)
                if robust_z is not None:
                    result["liq_robust_z"] = round(robust_z, 2)

            gate_passed = any(
                [
                    robust_z is not None and robust_z >= self.ROBUST_Z_NOTABLE,
                    liq_z is not None and liq_z >= 2.0,
                    percentile_band in ("p95", "p99"),
                    liq_ratio is not None and liq_ratio >= self.LIQ_TO_OI_NOTABLE,
                    liq_z is None and robust_z is None and total_usd >= liq_minor,
                ]
            )

            if gate_passed:
                long_ratio = long_usd / max(total_usd, 1.0)
                short_ratio = short_usd / max(total_usd, 1.0)

                if long_ratio >= self.SKEW_DIRECTIONAL:
                    dominant_usd = long_usd
                    result["liq_side"] = "LONG_DOMINATED"
                    result["directional_bias"] = "CAUTIOUS_LONG"
                    signals.append(f"Long liq cascade ${long_usd/1e6:.1f}M ({long_ratio*100:.0f}% of total)")
                elif short_ratio >= self.SKEW_DIRECTIONAL:
                    dominant_usd = short_usd
                    result["liq_side"] = "SHORT_DOMINATED"
                    result["directional_bias"] = "CAUTIOUS_SHORT"
                    signals.append(f"Short squeeze liq ${short_usd/1e6:.1f}M ({short_ratio*100:.0f}% of total)")
                else:
                    dominant_usd = total_usd
                    result["liq_side"] = "MIXED"
                    signals.append(
                        f"Mixed liq ${total_usd/1e6:.1f}M (Long={long_ratio*100:.0f}% / Short={short_ratio*100:.0f}%)"
                    )
                result["liq_dominant_usd"] = round(dominant_usd, 0)

                if robust_z is not None:
                    liq_score = min(max((robust_z - self.ROBUST_Z_NOTABLE) / 4.0, 0.0), 0.6)
                    signals.append(f"Liq robustZ={robust_z:.2f}")
                elif liq_z is not None:
                    liq_score = min(max((liq_z - 2.0) / 3.0, 0.0), 0.6)
                    signals.append(f"Liq Z={liq_z:.2f}")
                elif liq_ratio is not None and liq_ratio > 0:
                    liq_score = min(max(math.log10((liq_ratio / self.LIQ_TO_OI_NOTABLE) + 1.0), 0.0), 0.6)
                    signals.append(f"Liq/OI={liq_ratio*100:.2f}%")
                else:
                    liq_score = min(math.log10(max(1, dominant_usd / max(liq_major, 1.0) + 1)) / 1.0, 0.6)

                if (
                    (robust_z is not None and robust_z >= self.ROBUST_Z_EXTREME)
                    or percentile_band == "p99"
                    or (liq_ratio is not None and liq_ratio >= self.LIQ_TO_OI_EXTREME)
                    or (robust_z is None and liq_z is None and total_usd >= liq_extreme)
                ):
                    result["anomaly"] = "liquidation_cascade_extreme"
                elif (
                    (robust_z is not None and robust_z >= self.ROBUST_Z_MAJOR)
                    or (liq_z is not None and liq_z >= 2.5)
                    or percentile_band in ("p95", "p99")
                    or (liq_ratio is not None and liq_ratio >= self.LIQ_TO_OI_MAJOR)
                    or (robust_z is None and liq_z is None and total_usd >= liq_major)
                ):
                    result["anomaly"] = "liquidation_cascade"
                else:
                    result["anomaly"] = "liquidation_minor"
        except Exception as e:
            logger.warning(f"LiquidityAgent liq parse error: {e}")

        try:
            m_oi = re.search(r"OI_chg=([+-]?[\d\.]+)%", cvd_context)
            m_price = re.search(r"Price_chg=([+-]?[\d\.]+)%", cvd_context)
            oi_chg = float(m_oi.group(1)) if m_oi else 0.0
            price_chg = float(m_price.group(1)) if m_price else 0.0

            if "Status=DIVERGENCE" in cvd_context:
                result["oi_status"] = "DIVERGENCE"
                oi_score = 0.5
                if oi_chg > 0 and price_chg < 0:
                    signals.append(f"OI DIVERGENCE: OI+{oi_chg:.2f}% w/ Price{price_chg:.2f}%")
                    if result["liq_side"] == "SHORT_DOMINATED":
                        result["directional_bias"] = "BEARISH"
                elif oi_chg < 0 and price_chg > 0:
                    signals.append(f"OI DIVERGENCE: OI{oi_chg:.2f}% w/ Price+{price_chg:.2f}%")
                    result["directional_bias"] = "BULLISH"
            elif "Status=ALIGNED" in cvd_context:
                result["oi_status"] = "ALIGNED"
                oi_score = 0.2
        except Exception as e:
            logger.warning(f"LiquidityAgent OI parse error: {e}")

        try:
            if "[MFI_PROXY] INFLOW" in cvd_context:
                mfi_score = 0.2
                result["mfi_status"] = "INFLOW"
                signals.append("MFI: INFLOW")
                if result["directional_bias"] == "NEUTRAL":
                    result["directional_bias"] = "BULLISH"
            elif "[MFI_PROXY] OUTFLOW" in cvd_context:
                mfi_score = 0.2
                result["mfi_status"] = "OUTFLOW"
                signals.append("MFI: OUTFLOW")
                if result["directional_bias"] == "NEUTRAL":
                    result["directional_bias"] = "BEARISH"
            else:
                result["mfi_status"] = "NEUTRAL"
        except Exception as e:
            logger.warning(f"LiquidityAgent MFI parse error: {e}")

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
                    signals.append(f"Funding CROWDED_LONG ({fr*100:.4f}%/8h)")
                    if result["anomaly"] == "none":
                        result["anomaly"] = "funding_extreme_long"
                elif fr < crowded_short:
                    signals.append(f"Funding CROWDED_SHORT ({fr*100:.4f}%/8h)")
                    if result["anomaly"] == "none":
                        result["anomaly"] = "funding_extreme_short"
        except Exception:
            pass

        total_conf = min(liq_score + oi_score + mfi_score, 1.0)
        result["confidence"] = round(total_conf, 3)
        result["rationale"] = " | ".join(signals) if signals else "No significant liquidity signal"

        if mode.upper() == "POSITION" and result["confidence"] > 0:
            result["confidence"] = round(result["confidence"] * 0.7, 3)
            result["rationale"] += " [downweighted for POSITION timeframe]"

        return result


liquidity_agent = LiquidityAgent()
