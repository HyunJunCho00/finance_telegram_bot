from typing import Dict, Optional
from agents.claude_client import claude_client
from config.database import db
from datetime import datetime, timezone
from loguru import logger


class FeedbackGenerator:
    def __init__(self):
        self.system_prompt = """You are a trading performance analyst.

Analyze why a prediction was incorrect and generate actionable lessons.

Focus on:
- What pattern or signal was misread?
- What context was ignored?
- What assumptions were wrong?
- How to avoid this mistake next time?

Provide concise, specific feedback that can be used to improve future decisions."""

    def generate_feedback(self, evaluation: Dict) -> Optional[Dict]:
        if evaluation.get('direction_correct'):
            logger.info("Prediction was correct, no negative feedback needed")
            return None

        user_message = f"""Prediction Analysis:

Symbol: {evaluation.get('symbol', 'N/A')}
Predicted Direction: {evaluation.get('predicted_direction', 'N/A')}
Actual Direction: {evaluation.get('actual_direction', 'N/A')}
Actual Price Change: {evaluation.get('actual_change_pct', 0):.2f}%

Original Reasoning:
{evaluation.get('reasoning', 'N/A')[:1000]}

This prediction was INCORRECT.

Analyze what went wrong and provide:
1. Mistake Summary (2-3 sentences)
2. Root Cause (1-2 sentences)
3. Lesson Learned (specific rule to follow next time)"""

        try:
            response = claude_client.generate_response(
                system_prompt=self.system_prompt,
                user_message=user_message,
                temperature=0.3,
                max_tokens=1000,
                role="self_correction"
            )

            feedback_data = {
                "symbol": evaluation.get('symbol'),
                "prediction_time": evaluation.get('prediction_time'),
                "predicted_direction": evaluation.get('predicted_direction'),
                "actual_direction": evaluation.get('actual_direction'),
                "actual_change_pct": evaluation.get('actual_change_pct'),
                "mistake_summary": response[:500],
                "lesson_learned": response,
                "created_at": datetime.now(timezone.utc).isoformat()
            }

            db.insert_feedback(feedback_data)

            logger.info("Feedback generated and saved")

            return feedback_data

        except Exception as e:
            logger.error(f"Feedback generation error: {e}")
            return None

    def run_feedback_cycle(self) -> None:
        from evalutors.performance_evaluator import performance_evaluator

        evaluation = performance_evaluator.run_evaluation()

        if evaluation and not evaluation.get('error'):
            self.generate_feedback(evaluation)


feedback_generator = FeedbackGenerator()
