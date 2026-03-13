from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from loguru import logger

from config.settings import settings
from config.database import db
from executors.execution_repository import execution_repository
from executors.outbox_dispatcher import outbox_dispatcher
from liquidation_cascade.dataset import load_minute_panel
from liquidation_cascade.features import compute_feature_panel
from liquidation_cascade.inference import score_latest_feature_row


class CascadeWarningEngine:
    def __init__(self):
        self.artifact_dir = Path(getattr(settings, "LIQUIDATION_CASCADE_ARTIFACT_DIR", "data/models/liquidation_cascade"))
        self.lookback_days = int(getattr(settings, "LIQUIDATION_CASCADE_LOOKBACK_DAYS", 1))
        self.lookback_minutes = int(getattr(settings, "LIQUIDATION_CASCADE_LOOKBACK_MINUTES", self.lookback_days * 24 * 60))
        self.enabled = bool(getattr(settings, "ENABLE_LIQUIDATION_CASCADE_ALERTS", True))
        self.horizon_minutes = int(getattr(settings, "LIQUIDATION_CASCADE_HORIZON_MINUTES", 5))
        self.watch_prob = float(getattr(settings, "LIQUIDATION_CASCADE_WATCH_PROB", 0.45))
        self.warn_prob = float(getattr(settings, "LIQUIDATION_CASCADE_WARN_PROB", 0.60))
        self.confirm_prob = float(getattr(settings, "LIQUIDATION_CASCADE_CONFIRM_PROB", 0.75))
        self.vulnerability_floor = float(getattr(settings, "LIQUIDATION_CASCADE_VULNERABILITY_PCT", 0.90))
        self.ignition_floor = float(getattr(settings, "LIQUIDATION_CASCADE_IGNITION_PCT", 0.97))
        self.slope_floor = float(getattr(settings, "LIQUIDATION_CASCADE_SLOPE_PCT", 0.80))
        self.r2_floor = float(getattr(settings, "LIQUIDATION_CASCADE_R2_PCT", 0.70))

    def _artifact_exists(self, symbol: str, side: str) -> bool:
        stem = f"{symbol.lower()}_{side}_{self.horizon_minutes}m.metadata.json"
        return (self.artifact_dir / stem).exists()

    @staticmethod
    def _feature_row_to_payload(feature_row) -> Dict:
        return {
            "timestamp": feature_row["timestamp"].isoformat() if hasattr(feature_row["timestamp"], "isoformat") else feature_row["timestamp"],
            "symbol": str(feature_row["symbol"]),
            "side": str(feature_row["side"]),
            "event_candidate": bool(int(feature_row.get("event_candidate") or 0)),
            "vulnerability_score": float(feature_row.get("vulnerability_score") or 0.0),
            "ignition_score": float(feature_row.get("ignition_score") or 0.0),
            "ignition_ema": float(feature_row.get("ignition_ema") or 0.0),
            "ignition_slope": float(feature_row.get("ignition_slope") or 0.0),
            "ignition_r2": float(feature_row.get("ignition_r2") or 0.0),
            "vulnerability_pct_rank": float(feature_row.get("vulnerability_pct_rank") or 0.0),
            "ignition_pct_rank": float(feature_row.get("ignition_pct_rank") or 0.0),
            "slope_pct_rank": float(feature_row.get("slope_pct_rank") or 0.0),
            "r2_pct_rank": float(feature_row.get("r2_pct_rank") or 0.0),
            "feature_version": str(feature_row.get("feature_version") or "unknown"),
            "features": {
                "same_side_liq_surprise_1m": float(feature_row.get("same_side_liq_surprise_1m") or 0.0),
                "same_side_liq_surprise_5m": float(feature_row.get("same_side_liq_surprise_5m") or 0.0),
                "same_side_liq_count_surprise_1m": float(feature_row.get("same_side_liq_count_surprise_1m") or 0.0),
                "liq_imbalance_1m": float(feature_row.get("liq_imbalance_1m") or 0.0),
                "oi_level_z": float(feature_row.get("oi_level_z") or 0.0),
                "oi_change_5m_z": float(feature_row.get("oi_change_5m_z") or 0.0),
                "funding_crowding_z": float(feature_row.get("funding_crowding_z") or 0.0),
                "basis_crowding_z": float(feature_row.get("basis_crowding_z") or 0.0),
                "spread_z": float(feature_row.get("spread_z") or 0.0),
                "slippage_100k_z": float(feature_row.get("slippage_100k_z") or 0.0),
                "cvd_delta_3m_z": float(feature_row.get("cvd_delta_3m_z") or 0.0),
                "rv_5m_z": float(feature_row.get("rv_5m_z") or 0.0),
            },
        }

    def _severity(self, result: Dict) -> str | None:
        diagnostics = result.get("diagnostics") or {}
        prob = float(result.get("probability") or 0.0)
        event_candidate = int(diagnostics.get("event_candidate") or 0) == 1
        vuln = float(diagnostics.get("vulnerability_pct_rank") or 0.0)
        ign = float(diagnostics.get("ignition_pct_rank") or 0.0)
        slope = float(diagnostics.get("slope_pct_rank") or 0.0)
        r2 = float(diagnostics.get("r2_pct_rank") or 0.0)

        if not event_candidate or vuln < self.vulnerability_floor or ign < self.ignition_floor:
            return None
        if prob >= self.confirm_prob and slope >= self.slope_floor and r2 >= self.r2_floor:
            return "CONFIRM"
        if prob >= self.warn_prob and slope >= self.slope_floor:
            return "WARN"
        if prob >= self.watch_prob:
            return "WATCH"
        return None

    @staticmethod
    def _direction_text(side: str) -> str:
        return "downside cascade risk" if side == "down" else "upside squeeze risk"

    def _format_message(self, result: Dict, severity: str) -> str:
        diagnostics = result.get("diagnostics") or {}
        symbol = str(result.get("symbol") or "?")
        side = str(result.get("side") or "?")
        prob = float(result.get("probability") or 0.0)
        return (
            f"<b>[{severity}] {symbol} {self._direction_text(side)}</b>\n"
            f"- horizon: {int(result.get('horizon_minutes') or self.horizon_minutes)}m\n"
            f"- probability: {prob:.1%}\n"
            f"- vulnerability pct: {float(diagnostics.get('vulnerability_pct_rank') or 0.0):.0%}\n"
            f"- ignition pct: {float(diagnostics.get('ignition_pct_rank') or 0.0):.0%}\n"
            f"- slope pct: {float(diagnostics.get('slope_pct_rank') or 0.0):.0%}\n"
            f"- r2 pct: {float(diagnostics.get('r2_pct_rank') or 0.0):.0%}"
        )

    def _persist_prediction(self, result: Dict, severity: str | None) -> None:
        db.insert_liquidation_cascade_prediction(
            {
                "timestamp": result["timestamp"],
                "symbol": result["symbol"],
                "side": result["side"],
                "horizon_minutes": result["horizon_minutes"],
                "model_name": "liquidation_cascade_gbm",
                "model_version": result.get("model_version") or "unknown",
                "feature_version": result.get("feature_version") or "unknown",
                "probability": result["probability"],
                "predicted_label": bool(severity),
                "diagnostics": {**(result.get("diagnostics") or {}), "severity": severity},
            }
        )

    def _send_alert(self, result: Dict, severity: str) -> None:
        message = self._format_message(result, severity)
        ts_key = str(result.get("timestamp") or "")[:16]
        idempotency_key = f"telegram:liq_cascade:{result['symbol']}:{result['side']}:{severity}:{ts_key}"
        execution_repository.enqueue_outbox_event(
            "telegram_message",
            {"text": message, "parse_mode": "HTML"},
            idempotency_key=idempotency_key,
        )
        outbox_dispatcher.publish_pending(limit=20)

    def run_symbol(self, symbol: str) -> List[Dict]:
        if not self.enabled:
            return []

        panel = load_minute_panel(symbol=symbol, lookback_minutes=self.lookback_minutes)
        if panel.empty:
            return []

        results: List[Dict] = []
        for side in ("down", "up"):
            if not self._artifact_exists(symbol, side):
                continue
            try:
                feature_panel = compute_feature_panel(panel, side=side)
                if feature_panel.empty:
                    continue
                latest = feature_panel.sort_values("timestamp").tail(1).iloc[0]
                db.batch_upsert_liquidation_cascade_features([self._feature_row_to_payload(latest)])
                result = score_latest_feature_row(
                    feature_panel=feature_panel,
                    artifact_dir=self.artifact_dir,
                    symbol=symbol,
                    side=side,
                    horizon_minutes=self.horizon_minutes,
                )
                if result.get("status") != "ok":
                    continue
                severity = self._severity(result)
                self._persist_prediction(result, severity)
                if severity:
                    self._send_alert(result, severity)
                result["severity"] = severity
                results.append(result)
            except Exception as e:
                logger.error(f"Liquidation cascade alert failed for {symbol}/{side}: {e}")
        return results

    def run_all(self) -> Dict[str, List[Dict]]:
        summary: Dict[str, List[Dict]] = {}
        for symbol in settings.trading_symbols:
            rows = self.run_symbol(symbol)
            if rows:
                summary[symbol] = rows
        return summary


cascade_warning_engine = CascadeWarningEngine()



