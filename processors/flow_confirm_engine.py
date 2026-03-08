from __future__ import annotations

from typing import Dict, Optional

import pandas as pd


class FlowConfirmEngine:
    """Deterministic flow confirmation for policy-gated entries."""

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            if value in (None, "", "N/A"):
                return default
            return float(value)
        except Exception:
            return default

    def evaluate(
        self,
        direction: str,
        raw_funding: Optional[dict] = None,
        cvd_df: Optional[pd.DataFrame] = None,
        liq_df: Optional[pd.DataFrame] = None,
    ) -> Dict:
        direction = str(direction or "HOLD").upper()
        result = {
            "confirmed": False,
            "matched_signals": [],
            "metrics": {},
            "reason": "",
        }
        if direction not in ("LONG", "SHORT"):
            result["reason"] = "No directional trade candidate."
            return result

        matched = []

        if cvd_df is not None and not cvd_df.empty and "volume_delta" in cvd_df.columns:
            tail = cvd_df.sort_values("timestamp").tail(min(5, len(cvd_df)))
            cvd_3 = self._safe_float(tail.tail(min(3, len(tail)))["volume_delta"].fillna(0).sum())
            cvd_5 = self._safe_float(tail["volume_delta"].fillna(0).sum())
            result["metrics"]["cvd_delta_3"] = round(cvd_3, 4)
            result["metrics"]["cvd_delta_5"] = round(cvd_5, 4)
            if direction == "LONG" and (cvd_3 > 0 or cvd_5 > 0):
                matched.append("CVD positive flow")
            if direction == "SHORT" and (cvd_3 < 0 or cvd_5 < 0):
                matched.append("CVD negative flow")

        if liq_df is not None and not liq_df.empty:
            tail = liq_df.sort_values("timestamp").tail(min(5, len(liq_df)))
            long_liq = self._safe_float(tail.get("long_liq_usd", pd.Series(dtype=float)).fillna(0).sum())
            short_liq = self._safe_float(tail.get("short_liq_usd", pd.Series(dtype=float)).fillna(0).sum())
            result["metrics"]["long_liq_5"] = round(long_liq, 2)
            result["metrics"]["short_liq_5"] = round(short_liq, 2)
            if direction == "LONG" and short_liq > max(long_liq * 1.1, 100000.0):
                matched.append("Short liquidation squeeze")
            if direction == "SHORT" and long_liq > max(short_liq * 1.1, 100000.0):
                matched.append("Long liquidation flush")

        if raw_funding:
            funding_rate = self._safe_float(raw_funding.get("funding_rate"))
            ls_ratio = self._safe_float(raw_funding.get("long_short_ratio"), 1.0)
            oi_total = (
                self._safe_float(raw_funding.get("oi_binance"))
                + self._safe_float(raw_funding.get("oi_bybit"))
                + self._safe_float(raw_funding.get("oi_okx"))
            )
            result["metrics"]["funding_rate"] = round(funding_rate, 6)
            result["metrics"]["long_short_ratio"] = round(ls_ratio, 4)
            result["metrics"]["oi_total"] = round(oi_total, 2)
            if direction == "LONG" and funding_rate >= -0.01 and ls_ratio >= 1.0 and oi_total > 0:
                matched.append("Funding/OI supports long")
            if direction == "SHORT" and funding_rate <= 0.01 and ls_ratio <= 1.0 and oi_total > 0:
                matched.append("Funding/OI supports short")

        result["matched_signals"] = matched
        result["confirmed"] = len(matched) > 0
        result["reason"] = ", ".join(matched) if matched else "No confirmation signal aligned."
        return result


flow_confirm_engine = FlowConfirmEngine()
