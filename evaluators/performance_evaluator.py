from typing import Dict, Optional, List
from datetime import datetime, timedelta, timezone
from config.database import db
from config.settings import settings
from loguru import logger
import json
import math


class PerformanceEvaluator:
    def __init__(self):
        self.evaluation_hours = 24
        self.evaluation_horizon_minutes = self.evaluation_hours * 60
        self.hold_flat_threshold_pct = float(getattr(settings, "EVAL_HOLD_FLAT_THRESHOLD_PCT", 1.0))
        self.round_trip_fee_pct = float(getattr(settings, "EVAL_ROUND_TRIP_FEE_PCT", 0.10))

    def get_prediction_from_24h_ago(self) -> Optional[Dict]:
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.evaluation_hours)

            response = db.client.table("ai_reports")\
                .select("*")\
                .lte("created_at", cutoff_time.isoformat())\
                .order("created_at", desc=True)\
                .limit(1)\
                .execute()

            if response.data:
                return response.data[0]

            return None

        except Exception as e:
            logger.error(f"Error fetching 24h old prediction: {e}")
            return None

    def get_predictions_due_for_evaluation(
        self,
        after_created_at: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict]:
        """Fetch reports older than the fixed evaluation horizon, in chronological order."""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.evaluation_hours)
            query = (
                db.client.table("ai_reports")
                .select("*")
                .lte("created_at", cutoff_time.isoformat())
                .order("created_at")
                .limit(limit)
            )
            if after_created_at:
                query = query.gt("created_at", after_created_at)
            response = query.execute()
            return response.data if response.data else []
        except Exception as e:
            logger.error(f"Error fetching due predictions: {e}")
            return []

    @staticmethod
    def _parse_ts(value: str) -> datetime:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)

    def _get_first_market_row_at_or_after(self, symbol: str, ts: datetime) -> Optional[Dict]:
        try:
            response = db.client.table("market_data")\
                .select("close,timestamp")\
                .eq("symbol", symbol)\
                .gte("timestamp", ts.isoformat())\
                .order("timestamp")\
                .limit(1)\
                .execute()
            if response.data:
                row = response.data[0]
                return {
                    "close": float(row["close"]),
                    "timestamp": self._parse_ts(row["timestamp"]),
                }
            return None
        except Exception as e:
            logger.error(f"Error fetching market row after timestamp: {e}")
            return None

    def _get_close_series_between(self, symbol: str, start_time: datetime, end_time: datetime) -> List[Dict]:
        try:
            df = db.get_market_data_since(symbol, since=start_time, limit=2000)
            if df is None or df.empty:
                return []
            filtered = df[df["timestamp"] <= end_time].copy()
            if filtered.empty:
                return []
            return [
                {"timestamp": row["timestamp"].to_pydatetime(), "close": float(row["close"])}
                for _, row in filtered.iterrows()
                if row.get("close") is not None
            ]
        except Exception as e:
            logger.error(f"Error fetching close series: {e}")
            return []

    def _compute_risk_metrics(self, close_series: List[float]) -> Dict:
        """Simple realized metrics on available intraday close series."""
        if len(close_series) < 3:
            return {
                "volatility_pct": None,
                "max_drawdown_pct": None,
                "sharpe_like": None,
                "calmar_like": None,
            }

        returns = []
        for i in range(1, len(close_series)):
            prev = close_series[i - 1]
            cur = close_series[i]
            if prev == 0:
                continue
            returns.append((cur - prev) / prev)

        if len(returns) < 2:
            return {
                "volatility_pct": None,
                "max_drawdown_pct": None,
                "sharpe_like": None,
                "calmar_like": None,
            }

        mean_r = sum(returns) / len(returns)
        var = sum((r - mean_r) ** 2 for r in returns) / max(1, len(returns) - 1)
        std = math.sqrt(var)

        running_peak = close_series[0]
        max_dd = 0.0
        for p in close_series:
            if p > running_peak:
                running_peak = p
            dd = (p - running_peak) / running_peak
            if dd < max_dd:
                max_dd = dd

        total_return = (close_series[-1] - close_series[0]) / close_series[0] if close_series[0] else 0.0
        sharpe_like = (mean_r / std) * math.sqrt(len(returns)) if std > 0 else None
        calmar_like = (total_return / abs(max_dd)) if max_dd < 0 else None

        return {
            "volatility_pct": round(std * 100, 4),
            "max_drawdown_pct": round(max_dd * 100, 4),
            "sharpe_like": round(sharpe_like, 4) if sharpe_like is not None else None,
            "calmar_like": round(calmar_like, 4) if calmar_like is not None else None,
        }

    @staticmethod
    def _path_metrics(entry_price: float, path_rows: List[Dict], predicted_direction: str) -> Dict:
        if entry_price == 0 or not path_rows:
            return {"mfe_pct": None, "mae_pct": None}

        realized_path = []
        for row in path_rows:
            price = float(row["close"])
            if predicted_direction == "SHORT":
                realized_path.append(((entry_price - price) / entry_price) * 100.0)
            elif predicted_direction == "LONG":
                realized_path.append(((price - entry_price) / entry_price) * 100.0)
            else:
                realized_path.append(0.0)

        return {
            "mfe_pct": round(max(realized_path), 4) if realized_path else None,
            "mae_pct": round(min(realized_path), 4) if realized_path else None,
        }

    @staticmethod
    def _benchmark_return(entry_price: float, exit_price: float) -> Optional[float]:
        if not isinstance(entry_price, (int, float)) or not isinstance(exit_price, (int, float)) or entry_price == 0:
            return None
        return ((exit_price - entry_price) / entry_price) * 100.0

    def _realized_return(self, predicted_direction: str, benchmark_return_pct: float) -> float:
        if predicted_direction == "LONG":
            return benchmark_return_pct
        if predicted_direction == "SHORT":
            return -benchmark_return_pct
        return 0.0

    def _is_direction_correct(self, predicted_direction: str, benchmark_return_pct: float) -> bool:
        if predicted_direction == "LONG":
            return benchmark_return_pct > 0
        if predicted_direction == "SHORT":
            return benchmark_return_pct < 0
        return abs(benchmark_return_pct) < self.hold_flat_threshold_pct

    def _actual_direction(self, benchmark_return_pct: float) -> str:
        if benchmark_return_pct > 0:
            return "LONG"
        if benchmark_return_pct < 0:
            return "SHORT"
        return "HOLD"

    def _get_trade_note(self, symbol: str, prediction_time: datetime) -> str:
        trade_note = "N/A"
        try:
            exec_res = db.client.table("trade_executions")\
                .select("note")\
                .eq("symbol", symbol)\
                .gte("created_at", (prediction_time - timedelta(minutes=5)).isoformat())\
                .lte("created_at", (prediction_time + timedelta(minutes=5)).isoformat())\
                .order("created_at", desc=True)\
                .limit(1)\
                .execute()
            if exec_res.data:
                trade_note = exec_res.data[0].get("note", "N/A")
        except Exception as e:
            logger.warning(f"Could not fetch trade note for evaluation: {e}")
        return trade_note

    def _ensure_prediction_row(
        self,
        old_report: Dict,
        final_decision: Dict,
        prediction_time: datetime,
    ) -> Optional[Dict]:
        report_id = old_report.get("id")
        if not report_id:
            return None

        existing = db.get_evaluation_prediction_by_source("ai_report", int(report_id))
        if existing:
            return existing

        payload = {
            "source_type": "ai_report",
            "source_id": int(report_id),
            "ai_report_id": int(report_id),
            "prediction_time": prediction_time.isoformat(),
            "symbol": old_report.get("symbol"),
            "mode": str(old_report.get("mode", "unknown") or "unknown"),
            "decision": str(final_decision.get("decision", "HOLD")).upper(),
            "prediction_label": str(final_decision.get("decision", "HOLD")).upper(),
            "confidence": final_decision.get("confidence"),
            "entry_price": final_decision.get("entry_price"),
            "take_profit": final_decision.get("take_profit"),
            "stop_loss": final_decision.get("stop_loss"),
            "model_version": str(getattr(settings, "MODEL_JUDGE", "")),
            "prompt_version": str(getattr(settings, "PROMPT_VERSION", "judge_default_v1")),
            "rag_version": str(getattr(settings, "RAG_VERSION", "lightrag_v1")),
            "strategy_version": str(getattr(settings, "STRATEGY_VERSION", "orchestrator_v8")),
            "metadata": {
                "backfilled": True,
                "report_created_at": old_report.get("created_at"),
            },
        }
        try:
            return db.upsert_evaluation_prediction(payload)
        except Exception as e:
            logger.error(f"Failed to backfill evaluation prediction for report {report_id}: {e}")
            return None

    def evaluate_prediction(self, old_report: Dict, horizon_minutes: Optional[int] = None) -> Dict:
        try:
            symbol = old_report["symbol"]
            prediction_time = self._parse_ts(old_report["created_at"])
            target_time = prediction_time + timedelta(minutes=horizon_minutes or self.evaluation_horizon_minutes)

            final_decision = old_report["final_decision"]
            if isinstance(final_decision, str):
                final_decision = json.loads(final_decision)
            predicted_direction = str(final_decision.get("decision", "HOLD")).upper()

            entry_row = self._get_first_market_row_at_or_after(symbol, prediction_time)
            exit_row = self._get_first_market_row_at_or_after(symbol, target_time)
            if not entry_row or not exit_row:
                return {"error": "Could not calculate fixed-horizon price change"}

            entry_price = float(entry_row["close"])
            exit_price = float(exit_row["close"])
            benchmark_return_pct = self._benchmark_return(entry_price, exit_price)
            if benchmark_return_pct is None:
                return {"error": "Could not calculate benchmark return"}

            actual_direction = self._actual_direction(benchmark_return_pct)
            realized_return_pct = self._realized_return(predicted_direction, benchmark_return_pct)
            fee_pct = self.round_trip_fee_pct if predicted_direction in ("LONG", "SHORT") else 0.0
            fee_adjusted_pnl_pct = realized_return_pct - fee_pct
            excess_return_pct = fee_adjusted_pnl_pct - benchmark_return_pct
            direction_correct = self._is_direction_correct(predicted_direction, benchmark_return_pct)

            path_rows = self._get_close_series_between(symbol, prediction_time, exit_row["timestamp"])
            close_series = [entry_price] + [float(row["close"]) for row in path_rows]
            risk_metrics = self._compute_risk_metrics(close_series)
            path_metrics = self._path_metrics(entry_price, path_rows, predicted_direction)

            tp = final_decision.get("take_profit")
            sl = final_decision.get("stop_loss")
            tp_hit = None
            sl_hit = None
            if path_rows and predicted_direction in ("LONG", "SHORT"):
                prices = [float(row["close"]) for row in path_rows]
                if predicted_direction == "LONG":
                    tp_hit = bool(isinstance(tp, (int, float)) and any(price >= float(tp) for price in prices))
                    sl_hit = bool(isinstance(sl, (int, float)) and any(price <= float(sl) for price in prices))
                else:
                    tp_hit = bool(isinstance(tp, (int, float)) and any(price <= float(tp) for price in prices))
                    sl_hit = bool(isinstance(sl, (int, float)) and any(price >= float(sl) for price in prices))

            prediction_row = self._ensure_prediction_row(old_report, final_decision, prediction_time)
            prediction_id = prediction_row.get("id") if isinstance(prediction_row, dict) else None
            data_delay_minutes = round(max((exit_row["timestamp"] - target_time).total_seconds() / 60.0, 0.0), 2)
            trade_note = self._get_trade_note(symbol, prediction_time)

            evaluation = {
                "report_id": old_report.get("id"),
                "report_created_at": old_report.get("created_at"),
                "prediction_id": prediction_id,
                "symbol": symbol,
                "prediction_time": prediction_time.isoformat(),
                "target_time": target_time.isoformat(),
                "horizon_minutes": horizon_minutes or self.evaluation_horizon_minutes,
                "predicted_direction": predicted_direction,
                "actual_direction": actual_direction,
                "actual_change_pct": round(benchmark_return_pct, 4),
                "realized_return_pct": round(realized_return_pct, 4),
                "fee_adjusted_pnl_pct": round(fee_adjusted_pnl_pct, 4),
                "benchmark_return_pct": round(benchmark_return_pct, 4),
                "excess_return_pct": round(excess_return_pct, 4),
                "direction_correct": direction_correct,
                "predicted_entry": final_decision.get("entry_price", 0),
                "evaluated_entry_price": round(entry_price, 4),
                "evaluated_exit_price": round(exit_price, 4),
                "confidence": final_decision.get("confidence", 0),
                "reasoning": final_decision.get("reasoning", ""),
                "note": trade_note,
                "sample_count": len(close_series),
                "data_delay_minutes": data_delay_minutes,
                "tp_hit": tp_hit,
                "sl_hit": sl_hit,
                **path_metrics,
                **risk_metrics,
            }

            if prediction_id is not None:
                try:
                    db.upsert_evaluation_outcome({
                        "prediction_id": int(prediction_id),
                        "horizon_minutes": int(horizon_minutes or self.evaluation_horizon_minutes),
                        "evaluated_at": datetime.now(timezone.utc).isoformat(),
                        "target_time": target_time.isoformat(),
                        "entry_price": round(entry_price, 4),
                        "exit_price": round(exit_price, 4),
                        "actual_direction": actual_direction,
                        "realized_return_pct": round(realized_return_pct, 4),
                        "fee_adjusted_pnl_pct": round(fee_adjusted_pnl_pct, 4),
                        "benchmark_return_pct": round(benchmark_return_pct, 4),
                        "excess_return_pct": round(excess_return_pct, 4),
                        "correct": direction_correct,
                        "tp_hit": tp_hit,
                        "sl_hit": sl_hit,
                        "mfe_pct": path_metrics.get("mfe_pct"),
                        "mae_pct": path_metrics.get("mae_pct"),
                        "sample_count": len(close_series),
                        "data_delay_minutes": data_delay_minutes,
                        "metadata": {
                            "source_type": "ai_report",
                            "report_created_at": old_report.get("created_at"),
                            "predicted_direction": predicted_direction,
                        },
                    })
                except Exception as e:
                    logger.error(f"Failed to persist evaluation outcome for report {old_report.get('id')}: {e}")

            return evaluation

        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            return {"error": str(e)}

    def run_evaluation(self) -> Optional[Dict]:
        old_report = self.get_prediction_from_24h_ago()

        if not old_report:
            logger.info("No report from 24h ago to evaluate")
            return None

        evaluation = self.evaluate_prediction(old_report)

        logger.info(f"Evaluation result: {evaluation.get('direction_correct', False)}")

        return evaluation

    def run_evaluation_batch(self, after_created_at: Optional[str] = None, limit: int = 20) -> List[Dict]:
        due_reports = self.get_predictions_due_for_evaluation(after_created_at=after_created_at, limit=limit)
        results: List[Dict] = []
        for report in due_reports:
            evaluation = self.evaluate_prediction(report)
            results.append({
                "report_id": report.get("id"),
                "report_created_at": report.get("created_at"),
                "symbol": report.get("symbol"),
                "evaluation": evaluation,
            })
        if results:
            logger.info(f"Batch evaluation processed: {len(results)} report(s)")
        else:
            logger.info("No due reports found for batch evaluation")
        return results


performance_evaluator = PerformanceEvaluator()
