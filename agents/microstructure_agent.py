"""Deterministic microstructure signal extractor."""

from __future__ import annotations

import re

from loguru import logger

from processors.math_engine import calculate_z_score


class MicrostructureAgent:
    """Parse compact microstructure text into deterministic signals."""

    IMBALANCE_STRONG = 0.70
    IMBALANCE_EXTREME = 0.85
    SPREAD_TIGHT = 1.0
    SPREAD_NORMAL = 3.0
    SPREAD_WIDE = 8.0
    SPREAD_CRISIS = 15.0
    SLIPPAGE_HIGH = 0.10

    # Avoid unstable z-scores when the historical distribution is nearly flat.
    MIN_IMBALANCE_STD = 0.02
    MIN_SPREAD_STD_BPS = 0.05
    MAX_REASONABLE_Z = 12.0

    def analyze(
        self,
        microstructure_context: str = "",
        mode: str = "SWING",
        micro_stats: dict | None = None,
    ) -> dict:
        result = {
            "anomaly": "none",
            "confidence": 0.0,
            "imbalance": 0.0,
            "spread_bps": 0.0,
            "imbalance_z": None,
            "spread_z": None,
            "directional_bias": "NEUTRAL",
            "signal_quality": "LOW",
            "rationale": "no microstructure data",
        }

        if not microstructure_context:
            return result

        try:
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

            imb_z = None
            spread_z = None
            if micro_stats and isinstance(micro_stats, dict):
                imb_mean = micro_stats.get("imbalance_mean")
                imb_std = micro_stats.get("imbalance_std")
                spread_mean = micro_stats.get("spread_mean")
                spread_std = micro_stats.get("spread_std")

                if imb_mean is not None and imb_std and imb_std >= self.MIN_IMBALANCE_STD:
                    candidate = calculate_z_score(abs(imbalance), abs(imb_mean), imb_std)
                    if abs(candidate) <= self.MAX_REASONABLE_Z:
                        imb_z = candidate
                        result["imbalance_z"] = round(candidate, 2)

                if spread_mean is not None and spread_std and spread_std >= self.MIN_SPREAD_STD_BPS:
                    candidate = calculate_z_score(spread_bps, spread_mean, spread_std)
                    if abs(candidate) <= self.MAX_REASONABLE_Z:
                        spread_z = candidate
                        result["spread_z"] = round(candidate, 2)

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

            if spread_z is not None and spread_z >= 3.0 and imb_z is not None and imb_z >= 2.5:
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
                    f"spread={spread_bps:.2f}bps Z={spread_z:.1f} - liquidity exhaustion breakout risk."
                )
                return result

            spread_crisis = (spread_bps > self.SPREAD_CRISIS) or (spread_z is not None and spread_z >= 3.0)
            if spread_crisis:
                result["anomaly"] = "microstructure_breakdown"
                result["confidence"] = min(spread_bps / 20.0, 1.0) if spread_bps > 0 else 0.8
                result["directional_bias"] = "NEUTRAL"
                z_note = f" Z={spread_z:.1f}" if spread_z is not None else ""
                result["rationale"] = (
                    f"Spread {spread_bps:.1f}bps{z_note} - liquidity breakdown risk. Avoid market orders."
                )
                return result

            imb_gate = (imb_z is not None and imb_z >= 2.5) or (abs_imb >= self.IMBALANCE_STRONG)
            if imb_gate:
                z_imb_str = f" Z={imb_z:.1f}" if imb_z is not None else ""
                z_spd_str = f" SpreadZ={spread_z:.1f}" if spread_z is not None else ""
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
                        f"({spread_bps:.1f}bps{z_spd_str}) - likely spoofed wall."
                    )
                else:
                    conf = min(abs_imb * liq_mult, 1.0)
                    bias = "BULLISH" if imbalance > 0 else "BEARISH"
                    direction = "BID_HEAVY" if imbalance > 0 else "ASK_HEAVY"
                    if mode.upper() == "POSITION":
                        conf *= 0.6
                        bias_note = " (downweighted for POSITION timeframe)"
                    else:
                        bias_note = ""
                    result["anomaly"] = "microstructure_imbalance"
                    result["confidence"] = round(conf, 3)
                    result["directional_bias"] = bias
                    result["rationale"] = (
                        f"Orderbook {direction}: imbalance={imbalance:.3f}{z_imb_str}, "
                        f"spread={spread_bps:.2f}bps{z_spd_str} (quality={liq_label}){bias_note}."
                    )
            else:
                z_note = f" (Z={imb_z:.1f})" if imb_z is not None else ""
                result["rationale"] = (
                    f"Imbalance {imbalance:.3f}{z_note} below threshold - no directional signal. "
                    f"Spread={spread_bps:.2f}bps."
                )

            if slippage_pct > self.SLIPPAGE_HIGH:
                result["rationale"] += (
                    f" Slippage {slippage_pct:.2f}% > {self.SLIPPAGE_HIGH}% threshold - large order cost elevated."
                )
        except Exception as e:
            logger.warning(f"MicrostructureAgent parse error: {e}")

        return result


microstructure_agent = MicrostructureAgent()
