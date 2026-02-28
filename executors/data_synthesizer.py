"""
Data-Synthesis Pipeline (V9)
Extracts successful Chain-of-Thought (CoT) data from the Judge Agent for future domain-specific fine-tuning.
"""
import json
import os
from datetime import datetime, timezone
from loguru import logger
from pathlib import Path

_BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = _BASE_DIR / "data" / "fine_tuning_datasets"

def initialize_dataset_dir():
    os.makedirs(DATASET_DIR, exist_ok=True)

def synthesize_training_data(report: dict):
    """If the trade was successful (or has high conviction), extract the CoT for fine-tuning."""
    initialize_dataset_dir()
    
    symbol = report.get("symbol", "UNKNOWN")
    decision = report.get("final_decision", {})
    if not isinstance(decision, dict):
        return

    # Filter: Only save LONG/SHORT decisions with high confidence (e.g. > 60%) or successful outcomes.
    # Note: Outcome analysis happens in post_mortem, so here we focus on high-conviction rationales.
    confidence = decision.get("confidence", 0)
    if confidence < 60 or decision.get("decision") == "HOLD":
        return

    # Structure for fine-tuning (e.g. Instruction-Tuning format)
    training_example = {
        "instruction": f"Analyze the following market data for {symbol} and make a trading decision in {report.get('mode', 'SWING')} mode.",
        "input": f"Market Data: {report.get('market_data', 'N/A')}\nBlackboard: {report.get('blackboard', 'N/A')}\nRegime: {report.get('market_regime', 'N/A')}",
        "output": {
            "thought": decision.get("reasoning", {}),
            "decision": decision.get("decision"),
            "allocation": decision.get("allocation_pct"),
            "entry": decision.get("entry_price")
        },
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "confidence": confidence,
            "symbol": symbol
        }
    }

    file_path = DATASET_DIR / f"synthetic_cot_{datetime.now(timezone.utc).strftime('%Y%m')}.jsonl"
    
    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(training_example) + "\n")
        logger.info(f"V9: Successfully synthesized training data for {symbol} (Confidence: {confidence}%)")
    except Exception as e:
        logger.error(f"Failed to save synthetic data: {e}")

