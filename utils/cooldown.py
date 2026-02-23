import os
import json
import time
from loguru import logger

COOLDOWN_FILE = "data/triage_cooldowns.json"
DEFAULT_COOLDOWN_SECONDS = 4 * 3600  # 4 hours

def _load_cooldowns() -> dict:
    if not os.path.exists(COOLDOWN_FILE):
        return {}
    try:
        with open(COOLDOWN_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_cooldowns(data: dict):
    os.makedirs(os.path.dirname(COOLDOWN_FILE), exist_ok=True)
    try:
        with open(COOLDOWN_FILE, "w") as f:
            json.dump(data, f)
    except Exception as e:
        logger.error(f"Failed to save cooldowns: {e}")

def is_on_cooldown(anomaly_type: str, symbol: str) -> bool:
    """Returns True if the specific anomaly for the symbol was triggered recently."""
    cooldowns = _load_cooldowns()
    key = f"{symbol}_{anomaly_type}"
    
    last_triggered = cooldowns.get(key, 0)
    if time.time() - last_triggered < DEFAULT_COOLDOWN_SECONDS:
        logger.info(f"Anomaly '{anomaly_type}' for {symbol} is on cooldown. Suppressing.")
        return True
    return False

def set_cooldown(anomaly_type: str, symbol: str):
    """Records the timestamp when an anomaly was triggered."""
    cooldowns = _load_cooldowns()
    key = f"{symbol}_{anomaly_type}"
    cooldowns[key] = time.time()
    _save_cooldowns(cooldowns)
