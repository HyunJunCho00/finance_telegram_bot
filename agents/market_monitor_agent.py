"""Market Monitor Agent — Hourly trigger evaluator.

Role: monitor_hourly → OpenRouter (free tier)

Every hour:
1. Load the current Daily Playbook from DB for each symbol+mode.
2. Compare live indicators against Playbook entry/invalidation conditions.
3. Output: NO_ACTION | WATCH | TRIGGER
   - TRIGGER: all entry conditions met → hand off to orchestrator for order execution.
   - WATCH: partial match, monitoring needed.
   - NO_ACTION: no condition met.
"""

import json
import re
from datetime import datetime, timezone
from typing import Dict, Optional

from agents.ai_router import ai_client
from config.database import db
from config.settings import settings, TradingMode
from loguru import logger


class MarketMonitorAgent:
    """Hourly monitoring agent (NO_ACTION / WATCH / TRIGGER)."""

    SYSTEM_PROMPT = """You are a Quant Trading Monitor. Your ONLY job is checking whether current market conditions satisfy the Daily Playbook's entry or invalidation conditions.

You will receive:
1. DAILY PLAYBOOK: The strategy designed this morning (entry/exit/invalidation/risk conditions).
2. LIVE INDICATORS: Current price, funding rate, OI divergence, MFI proxy, volatility.

Output STRICT JSON — no extra text:
{
  "status": "NO_ACTION" | "WATCH" | "TRIGGER",
  "symbol": "BTCUSDT",
  "mode": "swing" | "position",
  "matched_conditions": ["list of playbook conditions that are currently TRUE"],
  "unmatched_conditions": ["list of conditions not yet met"],
  "invalidated": false,
  "invalidation_reason": "" ,
  "reasoning": "one concise sentence"
}

Rules:
- TRIGGER only if ALL entry conditions from the playbook are matched AND none of the invalidation conditions are triggered.
- WATCH if >= 1 entry condition matched but not all.
- NO_ACTION if 0 conditions matched or if the playbook is invalidated.
- If playbook TTL is expired (>24h old), always output NO_ACTION.
"""

    def __init__(self):
        self.role = "monitor_hourly"

    def _load_playbook(self, symbol: str, mode: str) -> Optional[Dict]:
        """Load the current Daily Playbook from DB."""
        try:
            res = (
                db.client.table("daily_playbooks")
                .select("*")
                .eq("symbol", symbol)
                .eq("mode", mode)
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )
            if res.data:
                return res.data[0]
        except Exception as e:
            logger.warning(f"Playbook load error ({symbol}/{mode}): {e}")
        return None

    def _get_live_indicators(self, symbol: str) -> Dict:
        """Fetch minimal live indicators for trigger evaluation."""
        indicators = {}
        try:
            df = db.get_latest_market_data(symbol, limit=12)
            if not df.empty:
                indicators["price"] = float(df["close"].iloc[-1])
                p_now = indicators["price"]
                p_prev = float(df["close"].iloc[0])
                indicators["price_chg_pct_1h"] = round((p_now - p_prev) / p_prev * 100, 3) if p_prev else 0
        except Exception:
            pass
        try:
            f_df = db.get_funding_history(symbol, limit=1)
            if f_df is not None and not f_df.empty and "funding_rate" in f_df.columns:
                indicators["funding_rate"] = float(f_df["funding_rate"].iloc[-1])
        except Exception:
            pass
        try:
            res = (
                db.client.table("funding_data")
                .select("oi_binance", "oi_bybit", "oi_okx", "timestamp")
                .eq("symbol", symbol)
                .order("timestamp", desc=True)
                .limit(4)
                .execute()
            )
            if res.data and len(res.data) >= 2:
                oi_rows = res.data
                oi_series = [
                    float(r.get("oi_binance", 0) or 0)
                    + float(r.get("oi_bybit", 0) or 0)
                    + float(r.get("oi_okx", 0) or 0)
                    for r in oi_rows
                ]
                oi_now, oi_prev = oi_series[0], oi_series[-1]
                oi_chg = ((oi_now - oi_prev) / oi_prev * 100) if oi_prev else 0
                indicators["oi_chg_pct"] = round(oi_chg, 3)
                price_chg = indicators.get("price_chg_pct_1h", 0)
                indicators["oi_divergence"] = (
                    "DIVERGENCE"
                    if (oi_chg > 1.5 and price_chg < -0.5) or (oi_chg < -1.5 and price_chg > 0.5)
                    else "ALIGNED"
                )
                indicators["mfi_proxy"] = (
                    "INFLOW" if oi_chg > 0.5 and price_chg > 0
                    else "OUTFLOW" if oi_chg < -0.5 and price_chg < 0
                    else "NEUTRAL"
                )
        except Exception:
            pass
        return indicators

    def evaluate(self, symbol: str, mode: str) -> Dict:
        """Core evaluation: compare live indicators to playbook conditions."""
        playbook = self._load_playbook(symbol, mode)
        if not playbook:
            return {"status": "NO_ACTION", "symbol": symbol, "mode": mode,
                    "reasoning": "No active playbook found."}

        # TTL check
        try:
            from datetime import timedelta
            import dateutil.parser
            created_at = dateutil.parser.parse(playbook.get("created_at", ""))
            if datetime.now(timezone.utc) - created_at > timedelta(hours=24):
                return {"status": "NO_ACTION", "symbol": symbol, "mode": mode,
                        "reasoning": "Playbook TTL expired (>24h)."}
        except Exception:
            pass

        live = self._get_live_indicators(symbol)

        user_message = f"""SYMBOL: {symbol}  MODE: {mode}

DAILY PLAYBOOK (created: {playbook.get('created_at', 'unknown')}):
{json.dumps(playbook.get('playbook', {}), indent=2)}

LIVE INDICATORS (UTC: {datetime.now(timezone.utc).isoformat()}):
{json.dumps(live, indent=2)}

Evaluate now. Output strict JSON only."""

        try:
            raw = ai_client.generate_response(
                system_prompt=self.SYSTEM_PROMPT,
                user_message=user_message,
                role=self.role,
                temperature=0.1,
                max_tokens=600,
            )
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > start:
                result = json.loads(raw[start:end])
                result["symbol"] = symbol
                result["mode"] = mode
                logger.info(f"Monitor [{symbol}/{mode}]: {result.get('status')} — {result.get('reasoning', '')}")
                return result
        except Exception as e:
            logger.error(f"MarketMonitorAgent.evaluate error: {e}")

        return {"status": "NO_ACTION", "symbol": symbol, "mode": mode,
                "reasoning": "Monitor evaluation failed."}

    def summarize_current_status(self, indicators: dict) -> str:
        """Legacy: generate free-form market status summary (used by job_routine_market_status)."""
        system_prompt = (
            "You are a Market Status Monitor. Provide a concise, data-driven summary "
            "of current market indicators for a professional trader. "
            "Focus on Funding Rates, OI Divergence, MFI proxy, and breaking news. "
            "Format in clean Markdown with bullet points and emojis. Under 150 words."
        )
        user_message = (
            f"Current Market Indicators (UTC: {datetime.now().isoformat()}):\n"
            f"{json.dumps(indicators, indent=2)}\n\n"
            "Please provide:\n"
            "1. 🛡️ Quick Market Sentiment (Bullish/Bearish/Neutral)\n"
            "2. 🪙 Notable Token Anomalies (Prices, Funding, OI Divergence, MFI)\n"
            "3. 📰 Breaking News & Narrative (From Telegram Intel)\n"
        )
        try:
            return ai_client.generate_response(
                system_prompt=system_prompt,
                user_message=user_message,
                role=self.role,
                temperature=0.4,
                max_tokens=500,
            )
        except Exception as e:
            logger.error(f"MarketMonitorAgent.summarize error: {e}")
            return "Failed to generate market summary."


market_monitor_agent = MarketMonitorAgent()
