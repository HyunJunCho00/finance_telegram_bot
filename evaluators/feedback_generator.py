from typing import Dict, Optional
from agents.ai_router import ai_client
from config.database import db
from datetime import datetime, timezone
from loguru import logger


class FeedbackGenerator:
    def __init__(self):
        self._failure_system_prompt = """You are a trading performance analyst.

Analyze why a prediction was incorrect and generate actionable lessons.

Focus on:
- What pattern or signal was misread?
- What context was ignored?
- What assumptions were wrong?
- How to avoid this mistake next time?

Provide concise, specific feedback that can be used to improve future decisions."""

        self._success_system_prompt = """You are a trading performance analyst specialising in reinforcement learning.

Analyze why a prediction was CORRECT and identify the specific signals that made it work.

Focus on:
- Which signals or confluence of signals were the strongest predictors?
- What market regime or context made those signals reliable?
- What decision logic should be preserved and repeated?
- Formulate a reusable pattern rule the Judge can apply in future similar setups.

Provide concise, specific positive reinforcement that can be used to repeat this success."""

    def generate_feedback(self, evaluation: Dict) -> Optional[Dict]:
        """Generate feedback for a completed prediction — negative for failures, positive for successes."""
        if evaluation.get("direction_correct"):
            return self._generate_positive_feedback(evaluation)
        return self._generate_negative_feedback(evaluation)

    def _generate_negative_feedback(self, evaluation: Dict) -> Optional[Dict]:
        user_message = f"""Prediction Analysis:

Symbol: {evaluation.get('symbol', 'N/A')}
Predicted Direction: {evaluation.get('predicted_direction', 'N/A')}
Actual Direction: {evaluation.get('actual_direction', 'N/A')}
Actual Price Change: {evaluation.get('actual_change_pct', 0):.2f}%

Original Reasoning:
{evaluation.get('reasoning', 'N/A')[:1000]}

Execution Notes (Slippage/Strategy):
{evaluation.get('note', 'N/A')}

This prediction was INCORRECT.

Analyze what went wrong and provide:
1. Mistake Summary (2-3 sentences)
2. Root Cause (1-2 sentences)
3. Lesson Learned (specific rule to follow next time)"""

        try:
            response = ai_client.generate_response(
                system_prompt=self._failure_system_prompt,
                user_message=user_message,
                temperature=0.3,
                max_tokens=1000,
                role="self_correction",
            )
            feedback_data = {
                "symbol": evaluation.get("symbol"),
                "prediction_time": evaluation.get("prediction_time"),
                "predicted_direction": evaluation.get("predicted_direction"),
                "actual_direction": evaluation.get("actual_direction"),
                "actual_change_pct": evaluation.get("actual_change_pct"),
                "feedback_type": "negative",
                "mistake_summary": response[:500],
                "lesson_learned": response,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            db.insert_feedback(feedback_data)
            logger.info(f"Negative feedback generated for {evaluation.get('symbol')} {evaluation.get('predicted_direction')}")
            return feedback_data
        except Exception as e:
            logger.error(f"Negative feedback generation error: {e}")
            return None

    def _generate_positive_feedback(self, evaluation: Dict) -> Optional[Dict]:
        """Capture what worked so the Judge can repeat successful patterns."""
        # Only generate positive feedback for actionable predictions with meaningful moves.
        # A HOLD that happened to be "correct" (flat market) carries no useful signal.
        if evaluation.get("predicted_direction") not in ("LONG", "SHORT"):
            return None
        actual_change = evaluation.get("actual_change_pct") or 0.0
        if abs(actual_change) < 0.5:
            # Sub-0.5% move — not meaningful enough to reinforce
            return None

        user_message = f"""Prediction Analysis:

Symbol: {evaluation.get('symbol', 'N/A')}
Predicted Direction: {evaluation.get('predicted_direction', 'N/A')}
Actual Direction: {evaluation.get('actual_direction', 'N/A')}
Actual Price Change: {evaluation.get('actual_change_pct', 0):.2f}%
Realized Return (fee-adjusted): {evaluation.get('fee_adjusted_pnl_pct', 0):.2f}%
MFE: {evaluation.get('mfe_pct', 'N/A')}%   MAE: {evaluation.get('mae_pct', 'N/A')}%

Original Reasoning:
{evaluation.get('reasoning', 'N/A')[:1000]}

Execution Notes:
{evaluation.get('note', 'N/A')}

This prediction was CORRECT.

Identify what worked and provide:
1. Signal Confluence Summary (2-3 sentences — which specific signals aligned?)
2. Market Regime (1-2 sentences — what context made those signals reliable?)
3. Reusable Pattern Rule (a concrete rule the Judge should apply in similar setups)"""

        try:
            response = ai_client.generate_response(
                system_prompt=self._success_system_prompt,
                user_message=user_message,
                temperature=0.3,
                max_tokens=1000,
                role="self_correction",
            )
            feedback_data = {
                "symbol": evaluation.get("symbol"),
                "prediction_time": evaluation.get("prediction_time"),
                "predicted_direction": evaluation.get("predicted_direction"),
                "actual_direction": evaluation.get("actual_direction"),
                "actual_change_pct": evaluation.get("actual_change_pct"),
                "feedback_type": "positive",
                "mistake_summary": "",
                "lesson_learned": response,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            db.insert_feedback(feedback_data)
            logger.info(f"Positive feedback generated for {evaluation.get('symbol')} {evaluation.get('predicted_direction')}")
            return feedback_data
        except Exception as e:
            logger.error(f"Positive feedback generation error: {e}")
            return None

    def run_feedback_cycle(self) -> None:
        from evaluators.performance_evaluator import performance_evaluator
        from config.local_state import state_manager

        cursor_key = "evaluation:last_24h_report_created_at"
        last_cursor = state_manager.get_system_config(cursor_key, "")
        batch = performance_evaluator.run_evaluation_batch(after_created_at=last_cursor, limit=30)
        if not batch:
            return

        max_cursor = last_cursor
        for item in batch:
            report_created_at = item.get("report_created_at") or ""
            evaluation = item.get("evaluation", {})
            if evaluation and not evaluation.get("error"):
                self.generate_feedback(evaluation)
            if report_created_at and (not max_cursor or report_created_at > max_cursor):
                max_cursor = report_created_at

        if max_cursor and max_cursor != last_cursor:
            state_manager.set_system_config(cursor_key, max_cursor)
            logger.info(f"24h evaluation cursor advanced to {max_cursor}")


feedback_generator = FeedbackGenerator()
