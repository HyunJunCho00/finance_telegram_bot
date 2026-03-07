from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from statistics import median
from typing import Dict, Iterable, List, Optional

from config.database import db
from loguru import logger


def _mean(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values if isinstance(v, (int, float))]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _median(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values if isinstance(v, (int, float))]
    if not vals:
        return None
    return float(median(vals))


def _utc_bounds(target_date: date) -> tuple[datetime, datetime]:
    start = datetime.combine(target_date, time.min, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return start, end


def _parse_prediction_time(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)


class EvaluationRollupService:
    def __init__(self):
        self.now_fn = lambda: datetime.now(timezone.utc)

    def _rollup_row(
        self,
        rollup_date: date,
        symbol: str,
        mode: str,
        scope: str,
        metric_name: str,
        metric_value,
        sample_size: int,
        horizon_minutes: int = 0,
        bucket_key: str = "",
        metadata: Optional[dict] = None,
    ) -> Optional[dict]:
        if metric_value is None:
            return None
        return {
            "rollup_date": rollup_date.isoformat(),
            "symbol": symbol,
            "mode": mode,
            "scope": scope,
            "horizon_minutes": horizon_minutes,
            "metric_name": metric_name,
            "metric_value": round(float(metric_value), 6),
            "sample_size": int(sample_size),
            "bucket_key": bucket_key,
            "metadata": metadata or {},
            "updated_at": self.now_fn().isoformat(),
        }

    def _system_rollups_for_group(
        self,
        rollup_date: date,
        symbol: str,
        mode: str,
        predictions: List[dict],
        outcomes: List[dict],
    ) -> List[dict]:
        rows: List[dict] = []
        total_predictions = len(predictions)
        actionable_predictions = [p for p in predictions if str(p.get("decision", "")).upper() in ("LONG", "SHORT")]
        rows.extend(
            row for row in [
                self._rollup_row(rollup_date, symbol, mode, "system", "total_predictions", total_predictions, total_predictions),
                self._rollup_row(rollup_date, symbol, mode, "system", "action_predictions", len(actionable_predictions), total_predictions),
                self._rollup_row(rollup_date, symbol, mode, "system", "hold_predictions", total_predictions - len(actionable_predictions), total_predictions),
                self._rollup_row(
                    rollup_date,
                    symbol,
                    mode,
                    "system",
                    "action_rate_pct",
                    (len(actionable_predictions) / total_predictions * 100.0) if total_predictions else None,
                    total_predictions,
                ),
                self._rollup_row(
                    rollup_date,
                    symbol,
                    mode,
                    "llm",
                    "avg_confidence_pct",
                    _mean([p.get("confidence") for p in predictions]),
                    total_predictions,
                ),
            ] if row
        )

        outcomes_by_horizon: Dict[int, List[dict]] = defaultdict(list)
        prediction_by_id = {int(p["id"]): p for p in predictions if p.get("id") is not None}
        for outcome in outcomes:
            outcomes_by_horizon[int(outcome.get("horizon_minutes", 0))].append(outcome)

        for horizon, horizon_rows in outcomes_by_horizon.items():
            sample_size = len(horizon_rows)
            if sample_size == 0:
                continue
            horizon_prediction_ids = {int(row["prediction_id"]) for row in horizon_rows if row.get("prediction_id") is not None}
            horizon_predictions = [prediction_by_id[pred_id] for pred_id in horizon_prediction_ids if pred_id in prediction_by_id]
            actionable_rows = [
                row for row in horizon_rows
                if str(prediction_by_id.get(int(row.get("prediction_id", -1)), {}).get("decision", "")).upper() in ("LONG", "SHORT")
            ]
            correct_pct = _mean([100.0 if row.get("correct") else 0.0 for row in horizon_rows])
            action_correct_pct = _mean([100.0 if row.get("correct") else 0.0 for row in actionable_rows])

            rows.extend(
                row for row in [
                    self._rollup_row(
                        rollup_date,
                        symbol,
                        mode,
                        "system",
                        "evaluated_coverage_pct",
                        (sample_size / total_predictions * 100.0) if total_predictions else None,
                        total_predictions,
                        horizon_minutes=horizon,
                    ),
                    self._rollup_row(rollup_date, symbol, mode, "system", "direction_accuracy_pct", correct_pct, sample_size, horizon_minutes=horizon),
                    self._rollup_row(rollup_date, symbol, mode, "system", "action_direction_accuracy_pct", action_correct_pct, len(actionable_rows), horizon_minutes=horizon),
                    self._rollup_row(
                        rollup_date,
                        symbol,
                        mode,
                        "system",
                        "avg_realized_return_pct",
                        _mean([row.get("realized_return_pct") for row in horizon_rows]),
                        sample_size,
                        horizon_minutes=horizon,
                    ),
                    self._rollup_row(
                        rollup_date,
                        symbol,
                        mode,
                        "system",
                        "median_realized_return_pct",
                        _median([row.get("realized_return_pct") for row in horizon_rows]),
                        sample_size,
                        horizon_minutes=horizon,
                    ),
                    self._rollup_row(
                        rollup_date,
                        symbol,
                        mode,
                        "system",
                        "avg_fee_adjusted_pnl_pct",
                        _mean([row.get("fee_adjusted_pnl_pct") for row in horizon_rows]),
                        sample_size,
                        horizon_minutes=horizon,
                    ),
                    self._rollup_row(
                        rollup_date,
                        symbol,
                        mode,
                        "system",
                        "avg_benchmark_return_pct",
                        _mean([row.get("benchmark_return_pct") for row in horizon_rows]),
                        sample_size,
                        horizon_minutes=horizon,
                    ),
                    self._rollup_row(
                        rollup_date,
                        symbol,
                        mode,
                        "system",
                        "avg_excess_return_pct",
                        _mean([row.get("excess_return_pct") for row in horizon_rows]),
                        sample_size,
                        horizon_minutes=horizon,
                    ),
                    self._rollup_row(
                        rollup_date,
                        symbol,
                        mode,
                        "system",
                        "tp_hit_rate_pct",
                        _mean([100.0 if row.get("tp_hit") else 0.0 for row in horizon_rows if row.get("tp_hit") is not None]),
                        len([row for row in horizon_rows if row.get("tp_hit") is not None]),
                        horizon_minutes=horizon,
                    ),
                    self._rollup_row(
                        rollup_date,
                        symbol,
                        mode,
                        "system",
                        "sl_hit_rate_pct",
                        _mean([100.0 if row.get("sl_hit") else 0.0 for row in horizon_rows if row.get("sl_hit") is not None]),
                        len([row for row in horizon_rows if row.get("sl_hit") is not None]),
                        horizon_minutes=horizon,
                    ),
                    self._rollup_row(
                        rollup_date,
                        symbol,
                        mode,
                        "system",
                        "avg_mfe_pct",
                        _mean([row.get("mfe_pct") for row in horizon_rows]),
                        sample_size,
                        horizon_minutes=horizon,
                    ),
                    self._rollup_row(
                        rollup_date,
                        symbol,
                        mode,
                        "system",
                        "avg_mae_pct",
                        _mean([row.get("mae_pct") for row in horizon_rows]),
                        sample_size,
                        horizon_minutes=horizon,
                    ),
                ] if row
            )

            confidence_rows = []
            for outcome in actionable_rows:
                pred = prediction_by_id.get(int(outcome.get("prediction_id", -1)))
                if pred is None:
                    continue
                conf = pred.get("confidence")
                if isinstance(conf, (int, float)):
                    confidence_rows.append((float(conf), outcome))

            if confidence_rows:
                confidence_rows.sort(key=lambda item: item[0], reverse=True)
                top_n = max(1, int(len(confidence_rows) * 0.2))
                top_bucket = confidence_rows[:top_n]
                top_accuracy = _mean([100.0 if item[1].get("correct") else 0.0 for item in top_bucket])
                rows.append(
                    self._rollup_row(
                        rollup_date,
                        symbol,
                        mode,
                        "system",
                        "direction_accuracy_pct",
                        top_accuracy,
                        len(top_bucket),
                        horizon_minutes=horizon,
                        bucket_key="high_confidence_top20pct",
                    )
                )

                probs = [min(max(item[0] / 100.0, 0.0), 1.0) for item in confidence_rows]
                labels = [1.0 if item[1].get("correct") else 0.0 for item in confidence_rows]
                brier = _mean([(prob - label) ** 2 for prob, label in zip(probs, labels)])
                rows.append(
                    self._rollup_row(
                        rollup_date,
                        symbol,
                        mode,
                        "llm",
                        "brier_score",
                        brier,
                        len(confidence_rows),
                        horizon_minutes=horizon,
                    )
                )

        return [row for row in rows if row]

    def _component_rollups_for_group(
        self,
        rollup_date: date,
        symbol: str,
        mode: str,
        component_scores: List[dict],
    ) -> List[dict]:
        grouped: Dict[tuple[str, str], List[dict]] = defaultdict(list)
        for score in component_scores:
            component_type = str(score.get("component_type", "unknown"))
            metric_name = str(score.get("metric_name", "unknown"))
            grouped[(component_type, metric_name)].append(score)

        rows: List[dict] = []
        for (component_type, metric_name), scores in grouped.items():
            rows.append(
                self._rollup_row(
                    rollup_date,
                    symbol,
                    mode,
                    component_type,
                    metric_name,
                    _mean([score.get("metric_value") for score in scores]),
                    len(scores),
                )
            )
        return [row for row in rows if row]

    def run_daily_rollup(self, target_date: Optional[str] = None, lookback_days: int = 3) -> Dict:
        now = self.now_fn()
        if target_date:
            end_date = date.fromisoformat(target_date)
        else:
            end_date = (now - timedelta(days=1)).date()

        processed_dates = []
        total_rows = 0
        for offset in range(max(lookback_days, 1)):
            rollup_date = end_date - timedelta(days=offset)
            start_dt, end_dt = _utc_bounds(rollup_date)

            predictions = db.get_evaluation_predictions(start_time=start_dt, end_time=end_dt, limit=10000)
            db.delete_evaluation_rollups_for_date(rollup_date.isoformat())
            if not predictions:
                processed_dates.append(rollup_date.isoformat())
                continue

            prediction_ids = [int(pred["id"]) for pred in predictions if pred.get("id") is not None]
            outcomes = db.get_evaluation_outcomes(prediction_ids=prediction_ids, limit=20000)
            component_scores = db.get_evaluation_component_scores(prediction_ids)

            predictions_by_group: Dict[tuple[str, str], List[dict]] = defaultdict(list)
            outcomes_by_group: Dict[tuple[str, str], List[dict]] = defaultdict(list)
            scores_by_group: Dict[tuple[str, str], List[dict]] = defaultdict(list)
            prediction_by_id = {int(pred["id"]): pred for pred in predictions if pred.get("id") is not None}

            for pred in predictions:
                group_key = (
                    str(pred.get("symbol") or "UNKNOWN"),
                    str(pred.get("mode") or "unknown"),
                )
                predictions_by_group[group_key].append(pred)

            for outcome in outcomes:
                prediction = prediction_by_id.get(int(outcome.get("prediction_id", -1)))
                if not prediction:
                    continue
                group_key = (
                    str(prediction.get("symbol") or "UNKNOWN"),
                    str(prediction.get("mode") or "unknown"),
                )
                outcomes_by_group[group_key].append(outcome)

            for score in component_scores:
                prediction = prediction_by_id.get(int(score.get("prediction_id", -1)))
                if not prediction:
                    continue
                group_key = (
                    str(prediction.get("symbol") or "UNKNOWN"),
                    str(prediction.get("mode") or "unknown"),
                )
                scores_by_group[group_key].append(score)

            rollup_rows: List[dict] = []
            for group_key, group_predictions in predictions_by_group.items():
                symbol, mode = group_key
                group_outcomes = outcomes_by_group.get(group_key, [])
                group_scores = scores_by_group.get(group_key, [])
                rollup_rows.extend(
                    self._system_rollups_for_group(rollup_date, symbol, mode, group_predictions, group_outcomes)
                )
                rollup_rows.extend(
                    self._component_rollups_for_group(rollup_date, symbol, mode, group_scores)
                )

            rollup_rows = [row for row in rollup_rows if row]
            if rollup_rows:
                db.upsert_evaluation_rollups(rollup_rows)
                total_rows += len(rollup_rows)

            processed_dates.append(rollup_date.isoformat())

        result = {
            "processed_dates": processed_dates,
            "rollup_rows": total_rows,
        }
        logger.info(f"Evaluation rollup completed: {result}")
        return result


evaluation_rollup_service = EvaluationRollupService()
