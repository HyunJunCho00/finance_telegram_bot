from typing import Dict, Optional
from datetime import datetime, timedelta, timezone
from config.database import db
from loguru import logger
import json


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

    def get_actual_price_change(self, symbol: str, prediction_time: datetime) -> Optional[float]:
        try:
            prediction_time_str = prediction_time.isoformat()
            current_time_str = datetime.now(timezone.utc).isoformat()

            response = db.client.table("market_data")\
                .select("close, timestamp")\
                .eq("symbol", symbol)\
                .gte("timestamp", prediction_time_str)\
                .lte("timestamp", current_time_str)\
                .order("timestamp")\
                .limit(2)\
                .execute()

            if len(response.data) >= 2:
                old_price = float(response.data[0]['close'])
                new_price = float(response.data[-1]['close'])

                change_pct = ((new_price - old_price) / old_price) * 100

                return change_pct

            return None

        except Exception as e:
            logger.error(f"Error calculating price change: {e}")
            return None

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

            evaluation = {
                "symbol": symbol,
                "prediction_time": prediction_time.isoformat(),
                "predicted_direction": predicted_direction,
                "actual_direction": actual_direction,
                "actual_change_pct": actual_change,
                "direction_correct": direction_correct,
                "predicted_entry": final_decision.get('entry_price', 0),
                "confidence": final_decision.get('confidence', 0),
                "reasoning": final_decision.get('reasoning', '')
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
