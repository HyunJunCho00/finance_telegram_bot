from typing import Dict, Optional, List
from datetime import datetime, timedelta, timezone
from config.database import db
from loguru import logger
import json
import math


class PerformanceEvaluator:
    def __init__(self):
        self.evaluation_hours = 24

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

    def _get_first_price_after(self, symbol: str, ts: datetime) -> Optional[float]:
        """Price closest after prediction time (entry proxy)."""
        try:
            response = db.client.table("market_data")\
                .select("close,timestamp")\
                .eq("symbol", symbol)\
                .gte("timestamp", ts.isoformat())\
                .order("timestamp")\
                .limit(1)\
                .execute()
            if response.data:
                return float(response.data[0]["close"])
            return None
        except Exception as e:
            logger.error(f"Error fetching first price after timestamp: {e}")
            return None

    def _get_latest_price(self, symbol: str) -> Optional[float]:
        try:
            response = db.client.table("market_data")\
                .select("close,timestamp")\
                .eq("symbol", symbol)\
                .order("timestamp", desc=True)\
                .limit(1)\
                .execute()
            if response.data:
                return float(response.data[0]["close"])
            return None
        except Exception as e:
            logger.error(f"Error fetching latest price: {e}")
            return None

    def _get_latest_market_timestamp(self, symbol: str) -> Optional[datetime]:
        try:
            response = db.client.table("market_data")\
                .select("timestamp")\
                .eq("symbol", symbol)\
                .order("timestamp", desc=True)\
                .limit(1)\
                .execute()
            if not response.data:
                return None
            return datetime.fromisoformat(response.data[0]["timestamp"].replace("Z", "+00:00"))
        except Exception as e:
            logger.error(f"Error fetching latest market timestamp: {e}")
            return None

    def get_actual_price_change(self, symbol: str, prediction_time: datetime) -> Optional[float]:
        try:
            old_price = self._get_first_price_after(symbol, prediction_time)
            new_price = self._get_latest_price(symbol)
            if old_price is None or new_price is None or old_price == 0:
                return None

            change_pct = ((new_price - old_price) / old_price) * 100
            return change_pct

        except Exception as e:
            logger.error(f"Error calculating price change: {e}")
            return None

    def _get_close_series_since(self, symbol: str, prediction_time: datetime) -> List[float]:
        try:
            response = db.client.table("market_data")\
                .select("close,timestamp")\
                .eq("symbol", symbol)\
                .gte("timestamp", prediction_time.isoformat())\
                .order("timestamp")\
                .limit(2000)\
                .execute()
            if not response.data:
                return []
            return [float(r["close"]) for r in response.data if r.get("close") is not None]
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

    def _get_data_delay_minutes(self, symbol: str) -> Optional[float]:
        latest_ts = self._get_latest_market_timestamp(symbol)
        if latest_ts is None:
            return None
        delay = (datetime.now(timezone.utc) - latest_ts).total_seconds() / 60
        return round(max(delay, 0.0), 2)

    def evaluate_prediction(self, old_report: Dict) -> Dict:
        try:
            symbol = old_report['symbol']
            prediction_time = datetime.fromisoformat(old_report['created_at'].replace('Z', '+00:00'))

            final_decision = old_report['final_decision']
            if isinstance(final_decision, str):
                final_decision = json.loads(final_decision)
            predicted_direction = final_decision.get('decision', 'HOLD')

            actual_change = self.get_actual_price_change(symbol, prediction_time)

            if actual_change is None:
                return {"error": "Could not calculate actual price change"}

            actual_direction = 'LONG' if actual_change > 0 else 'SHORT' if actual_change < 0 else 'HOLD'

            direction_correct = (
                (predicted_direction == 'LONG' and actual_direction == 'LONG') or
                (predicted_direction == 'SHORT' and actual_direction == 'SHORT') or
                (predicted_direction == 'HOLD' and abs(actual_change) < 1.0)
            )

            close_series = self._get_close_series_since(symbol, prediction_time)
            risk_metrics = self._compute_risk_metrics(close_series)

            # [VLM ENHANCEMENT] Fetch related trade execution notes for deeper self-correction
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

            evaluation = {
                "symbol": symbol,
                "prediction_time": prediction_time.isoformat(),
                "predicted_direction": predicted_direction,
                "actual_direction": actual_direction,
                "actual_change_pct": actual_change,
                "direction_correct": direction_correct,
                "predicted_entry": final_decision.get('entry_price', 0),
                "confidence": final_decision.get('confidence', 0),
                "reasoning": final_decision.get('reasoning', ''),
                "note": trade_note,  # Pass execution context (slippage, strategy) to feedback generator
                "sample_count": len(close_series),
                "data_delay_minutes": self._get_data_delay_minutes(symbol),
                **risk_metrics,
            }

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


performance_evaluator = PerformanceEvaluator()
