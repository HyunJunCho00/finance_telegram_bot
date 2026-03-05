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

CRITICAL: The "reasoning" field MUST be written in Korean.
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

    def _compare(self, a_val, op: str, b_val: float) -> bool:
        """Deterministic comparison utility."""
        if op == "<": return a_val < b_val
        if op == ">": return a_val > b_val
        if op == "<=": return a_val <= b_val
        if op == ">=": return a_val >= b_val
        if op == "==": return a_val == b_val
        return False

    def evaluate(self, symbol: str, mode: str) -> Dict:
        """Core evaluation: compare live indicators to playbook conditions deterministically."""
        playbook = self._load_playbook(symbol, mode)
        if not playbook:
            return {"status": "NO_ACTION", "symbol": symbol, "mode": mode,
                    "reasoning": "활성화된 플레이북이 없습니다."}

        # TTL check
        try:
            from datetime import timedelta
            import dateutil.parser
            created_at = dateutil.parser.parse(playbook.get("created_at", ""))
            if datetime.now(timezone.utc) - created_at > timedelta(hours=24):
                return {"status": "NO_ACTION", "symbol": symbol, "mode": mode,
                        "reasoning": "플레이북 유효기간(>24h)이 만료되었습니다."}
        except Exception:
            pass

        live = self._get_live_indicators(symbol)
        pb_data = playbook.get("playbook", {})
        
        # ── Deterministic Evaluation Logic ──
        matched = []
        unmatched = []
        invalidated = False
        inval_reason = ""
        
        # 1. Check Invalidation Conditions
        inval_conds = pb_data.get("invalidation_conditions", [])
        for ic in inval_conds:
            if isinstance(ic, dict):
                metric = ic.get("metric")
                op = ic.get("operator")
                val = ic.get("value")
                live_val = live.get(metric)
                if live_val is not None and op and val is not None:
                    try:
                        if self._compare(float(live_val), op, float(val)):
                            invalidated = True
                            inval_reason = f"{metric} {live_val} {op} {val}"
                            break
                    except Exception as e:
                        logger.debug(f"Invalidation parse error: {e}")

        if invalidated:
            logger.info(f"Monitor [{symbol}/{mode}]: NO_ACTION — Playbook Invalidated: {inval_reason}")
            return {"status": "NO_ACTION", "symbol": symbol, "mode": mode,
                    "reasoning": f"무효화됨: {inval_reason}"}
                    
        # 2. Check Entry Conditions
        entry_conds = pb_data.get("entry_conditions", [])
        if not entry_conds:
            return {"status": "NO_ACTION", "symbol": symbol, "mode": mode,
                    "reasoning": "분석 가능한 진입 조건이 정의되지 않았습니다."}
                    
        for ec in entry_conds:
            if isinstance(ec, dict):
                metric = ec.get("metric")
                op = ec.get("operator")
                val = ec.get("value")
                live_val = live.get(metric)
                if live_val is not None and op and val is not None:
                    try:
                        if self._compare(float(live_val), op, float(val)):
                            matched.append(f"{metric} {op} {val} (curr: {live_val})")
                        else:
                            unmatched.append(f"{metric} {op} {val} (curr: {live_val})")
                    except Exception:
                        unmatched.append(str(ec))
                else:
                    unmatched.append(f"Missing live data for metric: {metric}")
            elif isinstance(ec, str):
                unmatched.append(f"NLP condition needs manual review: {ec}")
                
        # 3. Final Output
        result_status = "NO_ACTION"
        if len(matched) > 0 and len(unmatched) == 0:
            result_status = "TRIGGER"
        elif len(matched) > 0:
            result_status = "WATCH"
            
        reasoning = "모든 결정론적 조건이 충족되었습니다." if result_status == "TRIGGER" else f"{len(unmatched)}개 조건 대기 중."
        
        result = {
            "status": result_status,
            "symbol": symbol,
            "mode": mode,
            "matched_conditions": matched,
            "unmatched_conditions": unmatched,
            "invalidated": False,
            "reasoning": reasoning
        }
        
        logger.info(f"Monitor [{symbol}/{mode}]: {result_status} — {reasoning}")
        return result

    def summarize_current_status(self, indicators: dict) -> str:
        """Legacy: generate free-form market status summary (used by job_routine_market_status)."""
        system_prompt = (
            "You are a Market Status Monitor. Provide a concise, data-driven summary "
            "of current market indicators for a professional trader. "
            "Focus on Funding Rates, OI Divergence, MFI proxy, and breaking news. "
            "Format in clean HTML tags (<b>, <i>, <code>) with bullet points and emojis. Do NOT use Markdown asterisks (**). Under 150 words. "
            "출력은 반드시 한국어로 작성하세요."
        )
        user_message = (
            f"Current Market Indicators (UTC: {datetime.now().isoformat()}):\n"
            f"{json.dumps(indicators, indent=2)}\n\n"
            "Please provide:\n"
            "1. 🛡️ 빠른 시장 심리 (강세/약세/중립)\n"
            "2. 🪙 주요 토큰 특이사항 (가격, 펀딩, OI 발산, MFI)\n"
            "3. 📰 주요 뉴스 및 내러티브 (텔레그램 인텔 기반)\n"
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
            return "시장 요약을 생성하지 못했습니다."


market_monitor_agent = MarketMonitorAgent()
