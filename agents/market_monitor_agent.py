"""Market Monitor Agent | Hourly trigger evaluator.

Role: monitor_hourly | OpenRouter (free tier)

Every hour:
1. Load the current Daily Playbook from DB for each symbol+mode.
2. Compare live indicators against Playbook entry/invalidation conditions.
3. Output: NO_ACTION | WATCH | TRIGGER
   - TRIGGER: all entry conditions met | hand off to orchestrator for order execution.
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
from processors.onchain_signal_engine import onchain_signal_engine


class MarketMonitorAgent:
    """Hourly monitoring agent (NO_ACTION / WATCH / TRIGGER)."""

    SYSTEM_PROMPT = """You are a Quant Trading Monitor. Your ONLY job is checking whether current market conditions satisfy the Daily Playbook's entry or invalidation conditions.

You will receive:
1. DAILY PLAYBOOK: The strategy designed this morning (entry/exit/invalidation/risk conditions).
2. LIVE INDICATORS: Current price, funding rate, OI divergence, MFI proxy, volatility.

Output STRICT JSON | no extra text:
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
            logger.info(f"Monitor [{symbol}/{mode}]: NO_ACTION | Playbook Invalidated: {inval_reason}")
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
        # [FEATURE-2] Soft-Trigger Logic: If matching > threshold, elevate status
        total_entry_count = len(entry_conds)
        match_count = len(matched)
        match_ratio = match_count / total_entry_count if total_entry_count > 0 else 0
        
        THRESHOLD = settings.MONITOR_SOFT_TRIGGER_THRESHOLD
        
        result_status = "NO_ACTION"
        if match_count > 0 and len(unmatched) == 0:
            result_status = "TRIGGER"
        elif match_ratio >= THRESHOLD:
            result_status = "SOFT_TRIGGER"
        elif match_count > 0:
            result_status = "WATCH"
            
        reasoning = (
            "모든 결정론적 조건이 충족되었습니다." if result_status == "TRIGGER"
            else f"{match_count}/{total_entry_count} 조건 만족 (Soft-Trigger)." if result_status == "SOFT_TRIGGER"
            else f"{len(unmatched)}개 조건 대기 중"
        )
        
        result = {
            "status": result_status,
            "symbol": symbol,
            "mode": mode,
            "matched_conditions": matched,
            "unmatched_conditions": unmatched,
            "invalidated": False,
            "reasoning": reasoning,
            # [FEATURE-1] Extra context for DB debug logging
            "live_indicators": live,
            "playbook_id": playbook.get("id"),
            "match_ratio": round(match_ratio, 2)
        }

        try:
            snapshot = db.get_latest_onchain_snapshot(symbol, max_age_hours=48)
            gate = onchain_signal_engine.build_gate(snapshot)
            playbook_direction = str(playbook.get("source_decision", "HOLD") or "HOLD").upper()
            blocked = (
                playbook_direction == "LONG" and not gate.get("allow_long", True)
            ) or (
                playbook_direction == "SHORT" and not gate.get("allow_short", True)
            )
            chase_block = playbook_direction == "LONG" and gate.get("chase_long_blocked") and result_status == "TRIGGER"

            if result_status in ["TRIGGER", "SOFT_TRIGGER"] and (blocked or chase_block):
                result["status"] = "WATCH"
                suffix = "온체인 상위 필터가 진입을 제한합니다."
                if chase_block and not blocked:
                    suffix = "온체인 상위 필터가 추격 롱을 제한합니다."
                result["reasoning"] = f"{result['reasoning']} {suffix}".strip()
            result["onchain_gate"] = gate
        except Exception as e:
            logger.warning(f"On-chain gate check failed for {symbol}/{mode}: {e}")
        
        logger.info(
            f"Monitor [{symbol}/{mode}]: {result.get('status')} ({match_count}/{total_entry_count}) | "
            f"{result.get('reasoning')}"
        )
        return result

    def summarize_current_status(self, indicators: dict) -> str:
        """Generate hourly Telegram market status with structured technical sections."""
        system_prompt = (
            "You are a market monitor for swing traders.\n"
            "Use ONLY provided data. Never invent prices or levels.\n"
            "Output in Korean and Telegram-safe HTML only (<b>, <i>, <code>, '-' bullets).\n"
            "Do not use markdown syntax.\n"
            "Required section titles:\n"
            "<b>🧭 구조/추세</b>\n"
            "<b>📐 빗각·지지/저항</b>\n"
            "<b>🧮 피보나치/변곡</b>\n"
            "<b>⚠️ 파생 리스크</b>\n"
            "<b>🧩 실행 레벨</b>\n"
            "<b>🔎 이벤트 상세</b> (conditional)\n"
            "For BTCUSDT and ETHUSDT, include concrete levels when present."
        )
        user_message = (
            f"Current Market Indicators (UTC: {datetime.now().isoformat()}):\n"
            f"{json.dumps(indicators, ensure_ascii=False, indent=2)}\n\n"
            "Rules:\n"
            "- technical_snapshot contains swing/position/events.\n"
            "- Always include BOTH swing and position core lines per symbol.\n"
            "- Mention CHoCH/MSB only if explicitly present.\n"
            "- In 실행 레벨, include nearest support, nearest resistance, and invalidation trigger.\n"
            "- Add <b>🔎 이벤트 상세</b> ONLY when events.has_event=true.\n"
            "- Keep concise and practical."
        )
        try:
            response = ai_client.generate_response(
                system_prompt=system_prompt,
                user_message=user_message,
                role=self.role,
                temperature=0.3,
                max_tokens=700,
            )
            if response and str(response).strip() and not self._is_suspiciously_truncated(response):
                return response

            logger.warning("MarketMonitorAgent.summarize returned empty/partial response; retrying with alternate route.")
            retry_resp = ai_client.generate_response(
                system_prompt=system_prompt,
                user_message=user_message,
                role="news_summarize",
                temperature=0.2,
                max_tokens=750,
            )
            if retry_resp and str(retry_resp).strip() and not self._is_suspiciously_truncated(retry_resp):
                return retry_resp

            logger.warning("MarketMonitorAgent.summarize retry failed; using deterministic fallback summary.")
            return self._build_fallback_summary(indicators)
        except Exception as e:
            logger.error(f"MarketMonitorAgent.summarize error: {e}")
            return self._build_fallback_summary(indicators)

    def _is_suspiciously_truncated(self, text: str) -> bool:
        """Heuristic guard for incomplete responses."""
        s = (text or "").strip()
        if not s:
            return True

        bad_suffixes = ("에서", "및", "또는", "그리고", ":", "-", "(", "[", "{", "/", ",")
        if s.endswith(bad_suffixes):
            return True

        if s.count("(") > s.count(")") or s.count("[") > s.count("]") or s.count("{") > s.count("}"):
            return True

        last_line = s.splitlines()[-1].strip()
        if len(last_line) <= 8 and not last_line.endswith((".", "!", "?", "…", "</b>", "</i>", "</code>")):
            return True

        return False

    def _build_fallback_summary(self, indicators: dict) -> str:
        """LLM fallback summary so hourly status never becomes empty."""
        lines = ["<b>시장 상태 요약 (Fallback)</b>"]

        symbol_rows = []
        for key, value in (indicators or {}).items():
            if key == "TELEGRAM_INTEL" or not isinstance(value, dict):
                continue
            symbol_rows.append((key, value))

        if not symbol_rows:
            return "<b>시장 상태 요약 (Fallback)</b>\n- 수집된 지표가 없어 요약할 수 없습니다."

        def _mode_views(row: dict):
            tech = row.get("technical_snapshot", {}) if isinstance(row.get("technical_snapshot"), dict) else {}
            if "swing" in tech or "position" in tech:
                return tech.get("swing", {}) or {}, tech.get("position", {}) or {}, tech.get("events", {}) or {}
            # backward compatibility
            return tech, {}, {}

        def _line_for_mode(snapshot: dict, label: str) -> str:
            if not isinstance(snapshot, dict) or not snapshot:
                return f"{label}: N/A"
            trend_map = {
                "uptrend": "상승추세",
                "downtrend": "하락추세",
                "ranging": "횡보",
                "insufficient_data": "데이터부족",
                "insufficient_pivots": "피벗부족",
            }
            pressure_map = {
                "strong_bullish": "실시간 상방압력 강함",
                "bullish": "실시간 상방압력",
                "early_bullish": "실시간 상방 선행신호",
                "strong_bearish": "실시간 하방압력 강함",
                "bearish": "실시간 하방압력",
                "early_bearish": "실시간 하방 선행신호",
                "mixed": "실시간 혼조",
            }
            tf = snapshot.get("primary_tf", "4h")
            ms = ((snapshot.get("market_structure") or {}).get(tf) or {})
            tr = snapshot.get(f"trendlines_{tf}", {}) or {}
            sw = snapshot.get(f"swing_levels_{tf}", {}) or {}
            fib = snapshot.get(f"fibonacci_{tf}", {}) or {}
            pressure = snapshot.get("realtime_pressure", {}) or {}
            trend = trend_map.get(ms.get("trend"), ms.get("trend", "N/A"))
            choch = ms.get("choch")
            msb = ms.get("msb")
            extra = []
            if choch:
                extra.append(f"CHoCH={choch}")
            if msb:
                extra.append(f"MSB={msb}")
            pressure_summary = pressure_map.get(pressure.get("summary"))
            pressure_details = pressure.get("details", []) or []
            pressure_text = pressure_summary
            if pressure_summary and pressure_details:
                pressure_text = f"{pressure_summary} ({', '.join(pressure_details[:2])})"
            return (
                f"{label}({tf}) {trend}"
                + (f" | {pressure_text}" if pressure_text else "")
                + (f" [{' | '.join(extra)}]" if extra else "")
                + f" | S/R {sw.get('nearest_support', 'N/A')}/{sw.get('nearest_resistance', 'N/A')}"
                + f" | Fib {fib.get('nearest_fib', 'N/A')}"
                + f" | Diag {tr.get('diagonal_support', 'N/A')}/{tr.get('diagonal_resistance', 'N/A')}"
            )

        lines.append("<b>🧭 구조/추세</b>")
        for symbol, row in symbol_rows:
            swing, position, _ = _mode_views(row)
            lines.append(f"- <b>{symbol}</b> { _line_for_mode(swing, 'SWING') }")
            lines.append(f"  - { _line_for_mode(position, 'POSITION') }")

        lines.append("<b>📐 빗각·지지/저항</b>")
        for symbol, row in symbol_rows:
            swing, _, _ = _mode_views(row)
            tf = swing.get("primary_tf", "4h")
            tr4h = swing.get(f"trendlines_{tf}", {}) or {}
            sw4h = swing.get(f"swing_levels_{tf}", {}) or {}
            lines.append(
                f"- <b>{symbol}</b> 빗각 S/R: "
                f"{tr4h.get('diagonal_support', 'N/A')} / {tr4h.get('diagonal_resistance', 'N/A')} | "
                f"수평 S/R: {sw4h.get('nearest_support', 'N/A')} / {sw4h.get('nearest_resistance', 'N/A')}"
            )

        lines.append("<b>🧮 피보나치/변곡</b>")
        for symbol, row in symbol_rows:
            swing, _, _ = _mode_views(row)
            tf = swing.get("primary_tf", "4h")
            fib4h = swing.get(f"fibonacci_{tf}", {}) or {}
            lines.append(
                f"- <b>{symbol}</b> fib={fib4h.get('nearest_fib', 'N/A')} "
                f"(0.5={fib4h.get('fib_500', 'N/A')}, 0.618={fib4h.get('fib_618', 'N/A')}, "
                f"0.705={fib4h.get('fib_705', 'N/A')}, 0.786={fib4h.get('fib_786', 'N/A')})"
            )

        lines.append("<b>⚠️ 파생 리스크</b>")
        for symbol, row in symbol_rows:
            price = row.get("price")
            funding = row.get("funding_rate")
            vol = row.get("volatility")
            price_txt = f"${price:,.2f}" if isinstance(price, (int, float)) else "N/A"
            funding_txt = f"{funding:+.5f}" if isinstance(funding, (int, float)) else "N/A"
            vol_txt = f"{vol:+.2f}%" if isinstance(vol, (int, float)) else "N/A"
            lines.append(f"- <b>{symbol}</b> 가격 {price_txt} | 펀딩 {funding_txt} | 변동성 {vol_txt}")

        lines.append("<b>🧩 실행 레벨</b>")
        for symbol, row in symbol_rows:
            swing, _, _ = _mode_views(row)
            tf = swing.get("primary_tf", "4h")
            sw4h = swing.get(f"swing_levels_{tf}", {}) or {}
            support = sw4h.get("nearest_support", "N/A")
            resistance = sw4h.get("nearest_resistance", "N/A")
            invalidation = support
            lines.append(f"- <b>{symbol}</b> 지지 {support} | 저항 {resistance} | 무효화 트리거 {invalidation}")

        lines.append("<b>🔎 이벤트 상세</b>")
        any_event = False
        for symbol, row in symbol_rows:
            _, _, events = _mode_views(row)
            if isinstance(events, dict) and events.get("has_event"):
                any_event = True
                items = events.get("event_items", []) or []
                if items:
                    lines.append(f"- <b>{symbol}</b> " + "; ".join(str(x) for x in items[:3]))
        if not any_event:
            lines.append("- 이번 사이클 이벤트 트리거 없음 (기본 요약만 유지)")

        intel = str((indicators or {}).get("TELEGRAM_INTEL", "") or "").strip()
        if intel and "주요 뉴스 없음" not in intel:
            brief = intel[:240] + ("..." if len(intel) > 240 else "")
            lines.append(f"- <i>뉴스 인텔 요약: {brief}</i>")

        lines.append(f"- <i>생성 시각(UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}</i>")
        return "\n".join(lines)


market_monitor_agent = MarketMonitorAgent()
