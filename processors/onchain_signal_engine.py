from __future__ import annotations

from datetime import datetime, timezone
from statistics import mean
from typing import Dict, Iterable, Optional


class OnChainSignalEngine:
    """Deterministic daily on-chain feature engineering and regime gating."""

    SYMBOL_TO_ASSET = {
        "BTCUSDT": "btc",
        "ETHUSDT": "eth",
    }

    def safe_float(self, value, default: Optional[float] = None) -> Optional[float]:
        try:
            if value in (None, "", "null"):
                return default
            return float(value)
        except Exception:
            return default

    def pct_change(self, newest: Optional[float], oldest: Optional[float]) -> Optional[float]:
        if newest is None or oldest in (None, 0):
            return None
        return ((newest - oldest) / abs(oldest)) * 100.0

    def ratio(self, numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
        if numerator is None or denominator in (None, 0):
            return None
        return numerator / denominator

    def _series_value(self, rows: list[dict], key: str, idx: int = -1) -> Optional[float]:
        if not rows:
            return None
        if abs(idx) > len(rows):
            return None
        return self.safe_float(rows[idx].get(key))

    def _rolling_mean(self, rows: list[dict], key: str, size: int) -> Optional[float]:
        if not rows:
            return None
        sample = rows[-size:]
        vals = [self.safe_float(r.get(key)) for r in sample]
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        return mean(vals)

    def _score_valuation(self, mvrv: Optional[float], price_vs_realized_pct: Optional[float]) -> tuple[float, str]:
        score = 0.0
        state = "neutral"

        if mvrv is not None:
            if mvrv < 1.0:
                score += 35
                state = "deep_value"
            elif mvrv < 1.5:
                score += 20
                state = "cheap"
            elif mvrv < 2.4:
                score += 5
                state = "neutral"
            elif mvrv < 3.1:
                score -= 20
                state = "elevated"
            else:
                score -= 38
                state = "overheated"

        if price_vs_realized_pct is not None:
            if price_vs_realized_pct < -10:
                score += 18
            elif price_vs_realized_pct < 5:
                score += 8
            elif price_vs_realized_pct > 80:
                score -= 18
            elif price_vs_realized_pct > 35:
                score -= 10

        return max(-50.0, min(50.0, score)), state

    def _score_flow(self, exchange_supply_30d_delta_pct: Optional[float]) -> tuple[float, str]:
        if exchange_supply_30d_delta_pct is None:
            return 0.0, "unknown"
        if exchange_supply_30d_delta_pct <= -3.0:
            return 35.0, "accumulation"
        if exchange_supply_30d_delta_pct <= -1.0:
            return 18.0, "improving"
        if exchange_supply_30d_delta_pct < 1.0:
            return 0.0, "neutral"
        if exchange_supply_30d_delta_pct < 3.0:
            return -18.0, "distribution"
        return -35.0, "heavy_distribution"

    def _score_activity(
        self,
        txcnt_ratio_7d_30d: Optional[float],
        newaddr_ratio_7d_30d: Optional[float],
    ) -> tuple[float, str]:
        samples = [v for v in (txcnt_ratio_7d_30d, newaddr_ratio_7d_30d) if v is not None]
        if not samples:
            return 0.0, "unknown"
        avg_ratio = mean(samples)
        if avg_ratio >= 1.15:
            return 22.0, "improving"
        if avg_ratio >= 1.03:
            return 10.0, "slightly_improving"
        if avg_ratio >= 0.95:
            return 0.0, "flat"
        if avg_ratio >= 0.85:
            return -10.0, "softening"
        return -22.0, "weakening"

    def _score_eth_structure(self, eip1559_ratio_7d_30d: Optional[float], asset: str) -> tuple[float, str]:
        if asset != "eth":
            return 0.0, "n/a"
        if eip1559_ratio_7d_30d is None:
            return 0.0, "unknown"
        if eip1559_ratio_7d_30d >= 1.10:
            return 12.0, "supportive"
        if eip1559_ratio_7d_30d >= 1.00:
            return 5.0, "neutral_positive"
        if eip1559_ratio_7d_30d >= 0.90:
            return 0.0, "neutral"
        if eip1559_ratio_7d_30d >= 0.80:
            return -5.0, "soft"
        return -12.0, "weak"

    def _determine_quality(self, as_of_date: str, available_metrics: int, total_metrics: int) -> str:
        quality = "low"
        if available_metrics >= max(3, total_metrics - 1):
            quality = "high"
        elif available_metrics >= 3:
            quality = "partial"

        try:
            dt = datetime.fromisoformat(as_of_date).replace(tzinfo=timezone.utc)
            age_hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0
            if age_hours > 36:
                return "stale"
        except Exception:
            pass
        return quality

    def build_snapshot(
        self,
        symbol: str,
        series_rows: list[dict],
        spot_price: Optional[float],
        metric_aliases: Dict[str, str],
        source: str = "coinmetrics",
    ) -> Dict:
        asset = self.SYMBOL_TO_ASSET.get(symbol, symbol.replace("USDT", "").lower())
        rows = sorted(series_rows, key=lambda r: str(r.get("time", "")))
        latest = rows[-1] if rows else {}
        as_of_date = str(latest.get("time") or datetime.now(timezone.utc).date().isoformat())[:10]

        mvrv = self._series_value(rows, metric_aliases.get("mvrv", ""))
        realized_cap_usd = self._series_value(rows, metric_aliases.get("realized_cap_usd", ""))
        current_supply = self._series_value(rows, metric_aliases.get("current_supply", ""))
        exchange_supply = self._series_value(rows, metric_aliases.get("exchange_supply", ""))
        tx_count = self._series_value(rows, metric_aliases.get("tx_count", ""))
        new_addresses = self._series_value(rows, metric_aliases.get("new_addresses", ""))
        tx_eip1559_count = self._series_value(rows, metric_aliases.get("tx_eip1559_count", ""))

        realized_price = self.ratio(realized_cap_usd, current_supply)
        price_vs_realized_pct = self.pct_change(spot_price, realized_price)

        exchange_supply_7d_delta = self.pct_change(
            self._series_value(rows, metric_aliases.get("exchange_supply", "")),
            self._series_value(rows[:-6] if len(rows) >= 7 else rows, metric_aliases.get("exchange_supply", "")),
        )
        exchange_supply_30d_delta = self.pct_change(
            self._series_value(rows, metric_aliases.get("exchange_supply", "")),
            self._series_value(rows[:-29] if len(rows) >= 30 else rows, metric_aliases.get("exchange_supply", "")),
        )

        tx_7d_avg = self._rolling_mean(rows, metric_aliases.get("tx_count", ""), 7)
        tx_30d_avg = self._rolling_mean(rows, metric_aliases.get("tx_count", ""), 30)
        tx_ratio = self.ratio(tx_7d_avg, tx_30d_avg)

        new_7d_avg = self._rolling_mean(rows, metric_aliases.get("new_addresses", ""), 7)
        new_30d_avg = self._rolling_mean(rows, metric_aliases.get("new_addresses", ""), 30)
        new_ratio = self.ratio(new_7d_avg, new_30d_avg)

        eip_7d_avg = self._rolling_mean(rows, metric_aliases.get("tx_eip1559_count", ""), 7)
        eip_30d_avg = self._rolling_mean(rows, metric_aliases.get("tx_eip1559_count", ""), 30)
        eip_ratio = self.ratio(eip_7d_avg, eip_30d_avg)

        valuation_score, valuation_state = self._score_valuation(mvrv, price_vs_realized_pct)
        flow_score, flow_state = self._score_flow(exchange_supply_30d_delta)
        activity_score, activity_state = self._score_activity(tx_ratio, new_ratio)
        eth_score, eth_state = self._score_eth_structure(eip_ratio, asset)

        bias_score = (0.45 * valuation_score) + (0.30 * flow_score) + (0.15 * activity_score) + (0.10 * eth_score)
        bias_score = max(-100.0, min(100.0, bias_score))

        if bias_score >= 35:
            risk_bias = "RISK_ON"
        elif bias_score <= -35:
            risk_bias = "RISK_OFF"
        else:
            risk_bias = "NEUTRAL"

        chase_long_blocked = valuation_state in {"elevated", "overheated"} or flow_state in {"distribution", "heavy_distribution"}
        allow_long = not (bias_score <= -55 or (valuation_state == "overheated" and flow_state in {"distribution", "heavy_distribution"}))
        allow_short = True
        if asset == "eth" and risk_bias == "RISK_ON":
            allow_short = False

        long_mult = 1.0
        short_mult = 1.0
        if bias_score >= 45:
            long_mult, short_mult = 1.15, 0.65
        elif bias_score >= 15:
            long_mult, short_mult = 1.0, 0.85
        elif bias_score <= -45:
            long_mult, short_mult = 0.5, 1.1
        elif bias_score <= -15:
            long_mult, short_mult = 0.75, 1.0

        if chase_long_blocked:
            long_mult = min(long_mult, 0.7)

        available_metrics = sum(
            1
            for value in (mvrv, realized_cap_usd, current_supply, exchange_supply, tx_count, new_addresses, tx_eip1559_count)
            if value is not None
        )
        data_quality = self._determine_quality(as_of_date, available_metrics, 7)

        raw_metrics = {
            "spot_price_usd": spot_price,
            "mvrv": mvrv,
            "realized_cap_usd": realized_cap_usd,
            "current_supply": current_supply,
            "exchange_supply": exchange_supply,
            "tx_count": tx_count,
            "new_addresses": new_addresses,
            "tx_eip1559_count": tx_eip1559_count,
            "metric_aliases": metric_aliases,
        }
        derived_features = {
            "realized_price_usd": realized_price,
            "price_vs_realized_pct": price_vs_realized_pct,
            "exchange_supply_7d_delta_pct": exchange_supply_7d_delta,
            "exchange_supply_30d_delta_pct": exchange_supply_30d_delta,
            "tx_count_7d_avg": tx_7d_avg,
            "tx_count_30d_avg": tx_30d_avg,
            "tx_count_ratio_7d_30d": tx_ratio,
            "new_addresses_7d_avg": new_7d_avg,
            "new_addresses_30d_avg": new_30d_avg,
            "new_addresses_ratio_7d_30d": new_ratio,
            "tx_eip1559_ratio_7d_30d": eip_ratio,
            "component_scores": {
                "valuation": valuation_score,
                "flow": flow_score,
                "activity": activity_score,
                "eth_structure": eth_score,
            },
        }
        regime_flags = {
            "valuation_state": valuation_state,
            "flow_state": flow_state,
            "activity_state": activity_state,
            "eth_state": eth_state,
            "allow_long": allow_long,
            "allow_short": allow_short,
            "long_size_multiplier": round(long_mult, 3),
            "short_size_multiplier": round(short_mult, 3),
            "chase_long_blocked": chase_long_blocked,
        }

        return {
            "symbol": symbol,
            "asset": asset,
            "source": source,
            "as_of_date": as_of_date,
            "raw_metrics": raw_metrics,
            "derived_features": derived_features,
            "regime_flags": regime_flags,
            "bias_score": round(bias_score, 3),
            "risk_bias": risk_bias,
            "data_quality": data_quality,
        }

    def format_context(self, snapshot: Optional[Dict]) -> str:
        if not snapshot:
            return "On-chain Context: unavailable"

        derived = snapshot.get("derived_features", {}) or {}
        flags = snapshot.get("regime_flags", {}) or {}
        raw = snapshot.get("raw_metrics", {}) or {}
        as_of_date = snapshot.get("as_of_date", "unknown")
        risk_bias = snapshot.get("risk_bias", "NEUTRAL")
        bias_score = snapshot.get("bias_score", 0)
        stale = snapshot.get("is_stale")
        stale_text = "stale" if stale else "fresh"

        parts = [
            f"On-chain ({as_of_date}, {stale_text}): bias={risk_bias} score={bias_score}.",
            f"MVRV={raw.get('mvrv')} valuation={flags.get('valuation_state')}.",
            f"Price vs realized={derived.get('price_vs_realized_pct')}%.",
            f"Exchange supply 30d delta={derived.get('exchange_supply_30d_delta_pct')}% flow={flags.get('flow_state')}.",
            f"Tx ratio 7d/30d={derived.get('tx_count_ratio_7d_30d')} activity={flags.get('activity_state')}.",
        ]

        if snapshot.get("asset") == "eth":
            parts.append(
                f"EIP1559 ratio 7d/30d={derived.get('tx_eip1559_ratio_7d_30d')} eth_state={flags.get('eth_state')}."
            )

        if flags:
            parts.append(
                "Gate: "
                f"allow_long={flags.get('allow_long')} "
                f"allow_short={flags.get('allow_short')} "
                f"long_mult={flags.get('long_size_multiplier')} "
                f"short_mult={flags.get('short_size_multiplier')} "
                f"chase_long_blocked={flags.get('chase_long_blocked')}."
            )

        return " ".join(parts)

    def build_gate(self, snapshot: Optional[Dict]) -> Dict:
        if not snapshot:
            return {
                "bias_score": 0.0,
                "risk_bias": "NEUTRAL",
                "allow_long": True,
                "allow_short": True,
                "long_size_multiplier": 0.85,
                "short_size_multiplier": 0.85,
                "chase_long_blocked": False,
                "data_quality": "missing",
            }

        flags = snapshot.get("regime_flags", {}) or {}
        gate = {
            "bias_score": float(snapshot.get("bias_score", 0.0) or 0.0),
            "risk_bias": snapshot.get("risk_bias", "NEUTRAL"),
            "allow_long": bool(flags.get("allow_long", True)),
            "allow_short": bool(flags.get("allow_short", True)),
            "long_size_multiplier": float(flags.get("long_size_multiplier", 1.0) or 1.0),
            "short_size_multiplier": float(flags.get("short_size_multiplier", 1.0) or 1.0),
            "chase_long_blocked": bool(flags.get("chase_long_blocked", False)),
            "data_quality": snapshot.get("data_quality", "unknown"),
        }

        if snapshot.get("is_stale"):
            gate["long_size_multiplier"] = min(gate["long_size_multiplier"], 0.8)
            gate["short_size_multiplier"] = min(gate["short_size_multiplier"], 0.8)
            gate["data_quality"] = "stale"
        return gate


onchain_signal_engine = OnChainSignalEngine()
