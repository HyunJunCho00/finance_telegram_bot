import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

class MetricsLogger:
    """
    Logs prediction and resolution events to JSONL files for academic/quantitative evaluation.
    These files can be later archived to GCS or parsed locally by data science scripts.
    """
    def __init__(self, log_dir="data/eval_metrics"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.predictions_file = self.log_dir / "predictions.jsonl"
        self.resolutions_file = self.log_dir / "resolutions.jsonl"

    def _append_jsonl(self, filepath: Path, data: dict):
        try:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(data) + "\n")
        except Exception as e:
            logger.error(f"Failed to write to metrics log {filepath}: {e}")

    def log_prediction(self, symbol: str, mode: str, final_decision: dict, blackboard: dict, anomalies: list):
        """
        Logs a prediction event (Judge's decision).
        """
        try:
            direction = final_decision.get("decision", "HOLD")
            if direction in ["HOLD", "CANCEL_AND_CLOSE"]:
                return  # We don't typically evaluate accuracy of HOLDs in the same aggressive way
            
            # Calculate Consensus Rate among text-producing agents (Market, Macro, Onchain)
            # Not all bots have these exactly, we look at blackboard outputs if available
            agent_votes = []
            if "market" in blackboard:
                agent_votes.append(blackboard["market"].get("decision", "HOLD"))
            if "onchain" in blackboard:
                agent_votes.append(blackboard["onchain"].get("decision", "HOLD"))
            if "macro" in blackboard:
                agent_votes.append(blackboard["macro"].get("decision", "HOLD"))
            
            # Count how many agents agreed with the Judge's final direction
            agreed = sum(1 for v in agent_votes if direction in str(v).upper())
            total = len(agent_votes) if agent_votes else 1
            consensus_rate = round((agreed / total) * 100, 2)
            
            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "PREDICTION",
                "symbol": symbol,
                "mode": mode,
                "predicted_direction": direction,
                "entry_price": final_decision.get("entry_price", 0),
                "take_profit": final_decision.get("take_profit", 0),
                "stop_loss": final_decision.get("stop_loss", 0),
                "confidence": final_decision.get("confidence", 0),
                "consensus_rate": consensus_rate,
                "anomalies_detected": anomalies,
                # Optional: tracking IDs
                "report_id": f"pred_{int(datetime.now().timestamp())}_{symbol}"
            }
            
            self._append_jsonl(self.predictions_file, event)
            logger.debug(f"Metrics Logger: Prediction appended for {symbol}")
        except Exception as e:
            logger.error(f"Metrics Logger Error (Prediction): {e}")

    def log_resolution(self, symbol: str, direction: str, outcome: str, pnl_pct: float, mistake_summary: str = ""):
        """
        Logs a resolution event when SL/TP is hit.
        """
        try:
            is_correct = "SUCCESS" in outcome
            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "RESOLUTION",
                "symbol": symbol,
                "predicted_direction": direction,
                "outcome_label": outcome,
                "is_correct": is_correct,
                "pnl_pct": pnl_pct,
                "mistake_summary": mistake_summary
            }
            
            self._append_jsonl(self.resolutions_file, event)
            logger.debug(f"Metrics Logger: Resolution appended for {symbol}")
        except Exception as e:
            logger.error(f"Metrics Logger Error (Resolution): {e}")

metrics_logger = MetricsLogger()
