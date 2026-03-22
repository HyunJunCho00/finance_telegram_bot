from typing import Any, Dict, Optional
from datetime import datetime, timezone
from config.database import db
from config.settings import settings, TradingMode
from telegram import Bot
from io import BytesIO
import threading
from loguru import logger
import json
import base64
import re
from executors.execution_repository import execution_repository
from executors.outbox_dispatcher import outbox_dispatcher
from utils.retry import api_retry


class ReportGenerator:
    def __init__(self):
        self.bot_token = settings.TELEGRAM_BOT_TOKEN
        self.chat_id = settings.TELEGRAM_CHAT_ID

    @staticmethod
    def _get_target_chat_id() -> str:
        try:
            from config.local_state import state_manager
            return state_manager.get_telegram_chat_id(settings.TELEGRAM_CHAT_ID) or settings.TELEGRAM_CHAT_ID
        except Exception:
            return settings.TELEGRAM_CHAT_ID

    @staticmethod
    def _load_json_field(value, default):
        if value is None:
            return default
        if isinstance(value, (dict, list)):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return default
        return default

    @staticmethod
    def _escape_html(value) -> str:
        return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    @staticmethod
    def _sanitize_html_for_telegram(text: str) -> str:
        if not text:
            return text
        sanitized = str(text)
        sanitized = re.sub(r"^\s*#{1,6}\s*(.+)$", r"\n<b>\1</b>", sanitized, flags=re.MULTILINE)
        sanitized = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", sanitized)
        sanitized = re.sub(r"^\s*\*\s+", "- ", sanitized, flags=re.MULTILINE)
        sanitized = sanitized.replace("<blockquote>", "").replace("</blockquote>", "")
        return sanitized

    @staticmethod
    def _chunk_text_for_telegram(text: str, limit: int = 4000) -> list[str]:
        if len(text) <= limit:
            return [text]

        chunks: list[str] = []
        current = ""
        for line in text.splitlines(keepends=True):
            candidate = current + line
            if len(candidate) <= limit:
                current = candidate
                continue

            if current:
                chunks.append(current)
                current = ""

            while len(line) > limit:
                chunks.append(line[:limit])
                line = line[limit:]
            current = line

        if current:
            chunks.append(current)
        return chunks or [text[:limit]]

    @staticmethod
    def _fmt_price(value) -> str:
        if value in (None, 0, "0"):
            return "---"
        try:
            return f"{float(value):,.2f}"
        except Exception:
            return str(value)

    @staticmethod
    def _fmt_num(value, decimals: int = 3, suffix: str = "") -> str:
        try:
            return f"{float(value):,.{decimals}f}{suffix}"
        except Exception:
            return "N/A"

    @staticmethod
    def _compact_text(value: Any, limit: int = 180) -> str:
        text = str(value or "").strip()
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    @classmethod
    def _extract_reasoning_text(cls, value: Any) -> str:
        if value in (None, "", "N/A"):
            return ""
        if isinstance(value, dict):
            for key in ("final_logic", "technical", "narrative", "onchain", "derivatives", "experts", "counter_scenario"):
                text = cls._extract_reasoning_text(value.get(key))
                if text:
                    return text
            return ""

        text = str(value).strip()
        if not text or text == "N/A":
            return ""

        parsed = cls._load_json_field(text, None)
        if isinstance(parsed, dict):
            nested = cls._extract_reasoning_text(parsed.get("reasoning"))
            if nested:
                return nested
            for key in ("final_logic", "technical", "narrative", "onchain", "reasoning"):
                nested = cls._extract_reasoning_text(parsed.get(key))
                if nested:
                    return nested

        text = re.sub(r"\s+", " ", text).strip()
        if text.startswith("{") and '"decision"' in text:
            return ""
        return text

    @classmethod
    def _veto_summary(cls, decision: Dict[str, Any]) -> str:
        """VETO된 결정에서 사람이 읽을 수 있는 한 줄 요약을 추출."""
        policy = cls._load_json_field(decision.get("policy_checks"), {})
        reason = ""
        if isinstance(policy, dict):
            reason = str(policy.get("reason") or policy.get("message") or "").strip()

        # VETO 메시지가 reasoning 안에 있는 경우 정제
        if not reason:
            reasoning = decision.get("reasoning", {})
            raw = ""
            if isinstance(reasoning, dict):
                raw = str(reasoning.get("final_logic") or reasoning.get("counter_scenario") or "").strip()
            else:
                raw = str(reasoning or "").strip()
            # [POLICY VETO] ... { JSON } 패턴에서 앞부분만 추출
            if raw.startswith("[POLICY VETO]"):
                raw = raw[len("[POLICY VETO]"):].strip()
            # JSON 블록 이전 텍스트만 보존
            json_start = raw.find("{")
            if json_start >= 0:
                raw = raw[:json_start].strip()
            reason = raw

        return cls._compact_text(reason, 200) if reason else "Policy 조건 미충족으로 진입 보류됨."

    @classmethod
    def _summary_from_decision(cls, decision: Dict[str, Any]) -> str:
        # VETO / NO_TRADE 상태면 policy 이유를 깔끔하게 표시
        policy = cls._load_json_field(decision.get("policy_checks"), {})
        policy_status = str((policy or {}).get("status", "")).upper() if isinstance(policy, dict) else ""
        if policy_status in ("VETO", "NO_TRADE"):
            return cls._veto_summary(decision)

        reasoning = decision.get("reasoning", {})
        candidates: list[str] = []

        if isinstance(reasoning, dict):
            for key in ("final_logic", "technical", "narrative", "onchain", "derivatives", "experts"):
                text = cls._extract_reasoning_text(reasoning.get(key))
                if text:
                    candidates.append(text)
        else:
            text = cls._extract_reasoning_text(reasoning)
            if text:
                candidates.append(text)

        for candidate in candidates:
            # 500자로 확대 — 기존 320자는 문장 중간에서 잘렸음
            compact = cls._compact_text(candidate, 500)
            if compact:
                return compact
        return "핵심 논리가 아직 정리되지 않았습니다."

    @staticmethod
    def _should_render_execution_receipt(decision: Dict[str, Any], receipt: Any) -> bool:
        if not isinstance(receipt, dict) or not receipt:
            return False
        if str(receipt.get("status", "")).upper() == "SKIPPED":
            return False

        direction = str((decision or {}).get("decision", "HOLD") or "HOLD").upper()
        if direction not in {"LONG", "SHORT", "CANCEL_AND_CLOSE"}:
            return False

        note = str(receipt.get("note", "") or "").strip()
        if note in {
            "No valid trade direction",
            "Allocation is 0% (Vetoed or No Confidence)",
        }:
            return False
        return True

    @staticmethod
    def _chart_profile(mode: TradingMode) -> tuple[str, str]:
        return ("1D/4H", "12M")

    @staticmethod
    def _load_human_chart_panels(symbol: str, mode: TradingMode) -> list[dict]:
        try:
            from mcp_server.tools import mcp_tools
            result = mcp_tools.get_chart_images(symbol, lane=mode.value)
            if not isinstance(result, dict) or "charts" not in result:
                return []
            return result.get("charts", [])
        except Exception as e:
            logger.warning(f"Human chart panel load failed for {symbol}: {e}")
            return []

    @staticmethod
    def _extract_onchain_summary(onchain_context: str) -> str:
        text = str(onchain_context or "").strip()
        if not text or text == "On-chain Context: unavailable":
            return ""
        text = text.replace("On-chain Context:", "").strip()
        if len(text) > 900:
            text = text[:897] + "..."
        return text

    def _build_onchain_lines(
        self,
        symbol: str,
        snapshot: Optional[Dict] = None,
        onchain_context: str = "",
        include_header: bool = True,
    ) -> list[str]:
        snapshot = snapshot or {}
        raw = snapshot.get("raw_metrics", {}) if isinstance(snapshot, dict) else {}
        derived = snapshot.get("derived_features", {}) if isinstance(snapshot, dict) else {}
        flags = snapshot.get("regime_flags", {}) if isinstance(snapshot, dict) else {}

        lines: list[str] = []
        if include_header:
            lines.append(f"<b>ON-CHAIN SNAPSHOT | {self._escape_html(symbol)}</b>")

        if snapshot:
            freshness = "stale" if snapshot.get("is_stale") else "fresh"
            age_hours = snapshot.get("age_hours")
            freshness_text = freshness
            if age_hours is not None:
                freshness_text = f"{freshness} / {self._fmt_num(age_hours, 1)}h"

            lines.extend([
                f"As of: <code>{self._escape_html(snapshot.get('as_of_date', 'N/A'))}</code> ({freshness_text})",
                (
                    f"Bias: <b>{self._escape_html(snapshot.get('risk_bias', 'NEUTRAL'))}</b> | "
                    f"Score: <code>{self._fmt_num(snapshot.get('bias_score'), 2)}</code> | "
                    f"Quality: <code>{self._escape_html(snapshot.get('data_quality', 'unknown'))}</code>"
                ),
                (
                    f"MVRV: <code>{self._fmt_num(raw.get('mvrv'), 4)}</code> | "
                    f"Valuation: <code>{self._escape_html(flags.get('valuation_state', 'unknown'))}</code>"
                ),
                (
                    f"Active Addr 7d/30d: <code>{self._fmt_num(derived.get('active_addresses_ratio_7d_30d'), 3)}</code> | "
                    f"Exchange Supply 30d: <code>{self._fmt_num(derived.get('exchange_supply_30d_delta_pct'), 2, '%')}</code>"
                ),
                (
                    f"Flow: <code>{self._escape_html(flags.get('flow_state', 'unknown'))}</code> | "
                    f"Tx Ratio 7d/30d: <code>{self._fmt_num(derived.get('tx_count_ratio_7d_30d'), 3)}</code> | "
                    f"Activity: <code>{self._escape_html(flags.get('activity_state', 'unknown'))}</code>"
                ),
                (
                    f"Gate: long=<code>{self._escape_html(flags.get('allow_long', 'N/A'))}</code> "
                    f"short=<code>{self._escape_html(flags.get('allow_short', 'N/A'))}</code> "
                    f"chase_long_blocked=<code>{self._escape_html(flags.get('chase_long_blocked', 'N/A'))}</code>"
                ),
                (
                    f"Size Mult: long=<code>{self._fmt_num(flags.get('long_size_multiplier'), 2)}</code>x "
                    f"short=<code>{self._fmt_num(flags.get('short_size_multiplier'), 2)}</code>x"
                ),
            ])

            if snapshot.get("asset") == "btc":
                lines.append(
                    f"BTC HashRate 7d/30d: <code>{self._fmt_num(derived.get('hash_rate_ratio_7d_30d'), 3)}</code> | "
                    f"State: <code>{self._escape_html(flags.get('structure_state', 'unknown'))}</code>"
                )
        else:
            lines.append("<i>No stored on-chain snapshot.</i>")

        if onchain_context:
            lines.extend([
                "",
                "<b>LLM INPUT (ON-CHAIN CONTEXT)</b>",
                f"<code>{self._escape_html(onchain_context)}</code>",
            ])

        return lines

    def generate_report(
        self,
        symbol: str,
        market_data: Dict,
        bull_opinion: str,
        bear_opinion: str,
        risk_assessment: str,
        final_decision: Dict,
        funding_data: Dict = None,
        mode: TradingMode = TradingMode.SWING,
        onchain_context: str = "",
        onchain_snapshot: Optional[Dict] = None,
    ) -> Dict:
        report = {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "market_data": json.dumps(market_data, default=str),
            "bull_opinion": bull_opinion,
            "bear_opinion": bear_opinion,
            "risk_assessment": risk_assessment,
            "final_decision": json.dumps(final_decision),
            "onchain_context": onchain_context,
            "onchain_snapshot": onchain_snapshot or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            report_id = db.insert_ai_report(report)
            logger.info(f"Report generated for {symbol} ({mode.value}) with ID: {report_id}")
            report["report_id"] = report_id
            return report
        except Exception as e:
            logger.error(f"Report generation error: {e}")
            return {}

    def generate_report_from_snapshot(self, bundle: Dict, final_decision: Dict) -> Dict:
        bundle = bundle or {}
        agents = bundle.get("agents", {}) or {}
        symbol = str(bundle.get("symbol", "BTCUSDT") or "BTCUSDT")
        mode = TradingMode(str(bundle.get("mode", TradingMode.SWING.value)).lower())

        market_payload = ((agents.get("market_snapshot_agent") or {}).get("payload") or {})
        narrative_payload = ((agents.get("narrative_agent") or {}).get("payload") or {})
        funding_payload = ((agents.get("funding_liq_agent") or {}).get("payload") or {})
        onchain_payload = ((agents.get("onchain_agent") or {}).get("payload") or {})

        return self.generate_report(
            symbol=symbol,
            market_data=market_payload.get("market_data", market_payload),
            bull_opinion=str(narrative_payload.get("narrative_text", "") or ""),
            bear_opinion=str(funding_payload.get("liquidation_context", "") or ""),
            risk_assessment="",
            final_decision=final_decision or {},
            funding_data=funding_payload.get("raw_funding", {}) or {},
            mode=mode,
            onchain_context=str(onchain_payload.get("onchain_context", "") or ""),
            onchain_snapshot=onchain_payload.get("onchain_snapshot", {}) or {},
        )

    def format_summary_message(self, report: Dict, mode: TradingMode = TradingMode.SWING) -> str:
        decision = self._load_json_field(report.get("final_decision"), {})

        direction = str(decision.get("decision", "N/A") or "N/A").upper()
        confidence = decision.get("confidence", decision.get("win_probability_pct", 0))

        if direction == "LONG":
            header_icon, color_theme = "", "🟢 강세 (LONG)"
        elif direction == "SHORT":
            header_icon, color_theme = "🔻", "🔴 약세 (SHORT)"
        else:
            header_icon, color_theme = "⚖️", "⚪ 망 (HOLD)"

        mode_label = "SWING (스윙)" if mode == TradingMode.SWING else "POSITION (포션)"
        final_logic = self._summary_from_decision(decision)

        # HOLD / VETO 시에는 원래 결정의 진입가가 남아있어도 표시하지 않음
        is_actionable = direction in ("LONG", "SHORT")
        summary_lines = [
            f"{header_icon} <b>[ {mode_label} ] AI 분석 리포트 | {report['symbol']}</b>",
            f"🕒 <code>{report['timestamp'][:16].replace('T', ' ')} UTC</code>\n",
            f"🎯 <b>최종 결정: {color_theme}</b> (확신도: {confidence}%)",
            "<blockquote>",
            f"🔵 진입: <code>{self._fmt_price(decision.get('entry_price')) if is_actionable else '---'}</code>",
            f"🛑 손절: <code>{self._fmt_price(decision.get('stop_loss')) if is_actionable else '---'}</code>",
            f"🏁 목표: <code>{self._fmt_price(decision.get('take_profit')) if is_actionable else '---'}</code>",
            "</blockquote>\n",
            "📝 <b>Summary:</b>",
            f"<i>{final_logic}</i>\n",
        ]

        policy = self._load_json_field(decision.get("policy_checks"), {}) if isinstance(decision, dict) else {}
        if isinstance(policy, dict) and policy:
            summary_lines.append(
                f"Policy: <code>{self._escape_html(policy.get('status', 'N/A'))}</code> | "
                f"RR: <code>{self._fmt_num(policy.get('rr'), 2)}</code> | "
                f"Stop Basis: <code>{self._escape_html(policy.get('stop_basis', 'N/A'))}</code>"
            )
            flow_signals = policy.get("flow_signals", []) or []
            if flow_signals:
                summary_lines.append(f"Flow: <code>{self._escape_html(', '.join(flow_signals[:2]))}</code>")

        scenario_plan = self._load_json_field(decision.get("scenario_plan"), {}) if isinstance(decision, dict) else {}
        if direction in {"LONG", "SHORT"} and isinstance(scenario_plan, dict) and any(str(v or "").strip() for v in scenario_plan.values()):
            summary_lines.extend([
                "",
                "<b>Scenario:</b>",
                f"- Primary: <i>{self._escape_html(self._compact_text(scenario_plan.get('primary_scenario', ''), 140))}</i>",
                f"- Trigger: <i>{self._escape_html(self._compact_text(scenario_plan.get('trigger_to_enter', ''), 120))}</i>",
                f"- Abort: <i>{self._escape_html(self._compact_text(scenario_plan.get('trigger_to_abort', ''), 120))}</i>",
            ])

        scenario_summary = self._load_json_field(decision.get("scenario_plan_summary"), {}) if isinstance(decision, dict) else {}
        if direction in {"LONG", "SHORT"} and isinstance(scenario_summary, dict) and scenario_summary:
            summary_lines.append(
                "Setup: "
                f"<code>{self._fmt_price(scenario_summary.get('entry_zone_low'))}</code> ~ "
                f"<code>{self._fmt_price(scenario_summary.get('entry_zone_high'))}</code> | "
                f"Inv <code>{self._fmt_price(scenario_summary.get('invalidation'))}</code> | "
                f"TP1 <code>{self._fmt_price(scenario_summary.get('tp1'))}</code>"
            )

        split_entry_plan = decision.get("split_entry_plan", []) if isinstance(decision, dict) else []
        if direction in {"LONG", "SHORT"} and isinstance(split_entry_plan, list) and split_entry_plan:
            split_text = ", ".join(self._fmt_price(v) for v in split_entry_plan[:3])
            summary_lines.append(f"Scale-ins: <code>{self._escape_html(split_text)}</code>")

        if direction in {"LONG", "SHORT"} and decision.get("breakeven_rule"):
            summary_lines.append(f"BE Rule: <i>{self._escape_html(self._compact_text(decision.get('breakeven_rule'), 120))}</i>")

        receipt = decision.get("execution_receipt")
        receipt = receipt if self._should_render_execution_receipt(decision, receipt) else None
        if receipt and receipt.get("success"):
            summary_lines.append(f"✅ <b>자동매매 실행 완료</b> ({len(receipt.get('receipts', []))} orders)")
        elif receipt:
            summary_lines.append(f"⚠️ <b>실행 보류:</b> {receipt.get('note', 'PENDING')}")

        return "\n".join(summary_lines)

    def format_detail_message(self, report: Dict, mode: TradingMode = TradingMode.SWING) -> str:
        decision = self._load_json_field(report.get("final_decision"), {})
        reasoning = decision.get("reasoning", "N/A")
        onchain_context = str(report.get("onchain_context", "") or "").strip()

        lines = [f"🔍 <b>DEEP ANALYSIS | {report['symbol']}</b>\n"]

        if isinstance(reasoning, dict):
            reasoning_for_view = dict(reasoning)
            if not reasoning_for_view.get("onchain") or reasoning_for_view.get("onchain") == "N/A":
                fallback_onchain = self._extract_onchain_summary(onchain_context)
                if fallback_onchain:
                    reasoning_for_view["onchain"] = fallback_onchain
            mapping = {
                "technical": "📊 기술적 분석 (Technical)",
                "onchain": "🔗 심층 온체인 분석 (On-chain)",
                "derivatives": "⛓️ 파생상품 심리 (Derivatives)",
                "experts": "🤖 전문 스웜 (Expert Swarm)",
                "narrative": "🌐 시장 내러티브 (Narrative/News)",
                "counter_scenario": "🚨 최악의 시나리오 (Risk)",
            }
            mapping["onchain"] = "On-chain Summary"
            for key, title in mapping.items():
                value = reasoning_for_view.get(key)
                if value and value != "N/A":
                    safe_value = self._escape_html(value)
                    if len(safe_value) > 1500:
                        safe_value = safe_value[:1497] + "..."
                    safe_value = safe_value.replace(". ", ".\n\n")
                    lines.append(f"<b>{title}:</b>\n<i>{safe_value}</i>\n")
        else:
            lines.append(f"<i>{self._escape_html(reasoning)}</i>")

        scenario_plan = self._load_json_field(decision.get("scenario_plan"), {}) if isinstance(decision, dict) else {}
        if isinstance(scenario_plan, dict) and any(str(v or "").strip() for v in scenario_plan.values()):
            lines.append("<b>Scenario Plan:</b>")
            scenario_mapping = [
                ("primary_scenario", "Primary"),
                ("alternate_scenario", "Alternate"),
                ("trigger_to_enter", "Trigger"),
                ("trigger_to_abort", "Abort"),
                ("partial_tp_plan", "Partial TP"),
                ("stop_to_be_rule", "Stop to BE"),
            ]
            for key, label in scenario_mapping:
                value = scenario_plan.get(key)
                if value:
                    lines.append(f"- <b>{label}:</b> <i>{self._escape_html(value)}</i>")
            lines.append("")

        scenario_summary = self._load_json_field(decision.get("scenario_plan_summary"), {}) if isinstance(decision, dict) else {}
        if isinstance(scenario_summary, dict) and scenario_summary:
            lines.append("<b>Execution Setup:</b>")
            lines.append(f"- Trigger: <code>{self._escape_html(scenario_summary.get('trigger', 'N/A'))}</code>")
            lines.append(
                f"- Entry Zone: <code>{self._fmt_price(scenario_summary.get('entry_zone_low'))}</code> ~ "
                f"<code>{self._fmt_price(scenario_summary.get('entry_zone_high'))}</code>"
            )
            lines.append(
                f"- Invalidation: <code>{self._fmt_price(scenario_summary.get('invalidation'))}</code> | "
                f"TP1: <code>{self._fmt_price(scenario_summary.get('tp1'))}</code> | "
                f"TP2: <code>{self._fmt_price(scenario_summary.get('tp2'))}</code>"
            )
            split_entry_plan = decision.get("split_entry_plan", []) if isinstance(decision, dict) else []
            if isinstance(split_entry_plan, list) and split_entry_plan:
                split_text = ", ".join(self._fmt_price(v) for v in split_entry_plan[:3])
                lines.append(f"- Scale-ins: <code>{self._escape_html(split_text)}</code>")
            if decision.get("breakeven_rule"):
                lines.append(f"- BE Rule: <i>{self._escape_html(decision.get('breakeven_rule'))}</i>")
            if decision.get("risk_manager_note"):
                lines.append(f"- CRO Note: <i>{self._escape_html(decision.get('risk_manager_note'))}</i>")

        return "\n".join(lines)

    def format_onchain_message(self, symbol: str, snapshot: Optional[Dict]) -> str:
        if not snapshot:
            return f"<b>ON-CHAIN SNAPSHOT | {self._escape_html(symbol)}</b>\n<i>No snapshot available.</i>"
        return "\n".join(self._build_onchain_lines(symbol=symbol, snapshot=snapshot, onchain_context="", include_header=True))

    @staticmethod
    def _build_reply_markup(report_id: Any):
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        if report_id in (None, ""):
            return None
        keyboard = [[InlineKeyboardButton("🔍 View Deep Analysis", callback_data=f"detail_{report_id}")]]
        return InlineKeyboardMarkup(keyboard)

    @staticmethod
    def _last_scheduled_report_key() -> str:
        return "last_scheduled_report_payload"

    def _save_report_replay_payload(self, payload: Dict[str, Any]) -> None:
        context = str(payload.get("context", "") or "")
        if context != "scheduled_daily":
            return
        try:
            from config.local_state import state_manager
            state_manager.set_system_config(
                self._last_scheduled_report_key(),
                json.dumps(payload, ensure_ascii=False),
            )
        except Exception as e:
            logger.warning(f"Failed to persist scheduled report payload: {e}")

    def get_last_scheduled_report_payload(self) -> Dict[str, Any]:
        try:
            from config.local_state import state_manager
            raw = state_manager.get_system_config(self._last_scheduled_report_key(), "")
            if not raw:
                return {}
            payload = json.loads(raw)
            return payload if isinstance(payload, dict) else {}
        except Exception as e:
            logger.warning(f"Failed to load scheduled report payload: {e}")
            return {}

    def _build_notification_payload(
        self,
        report: Dict,
        chart_bytes: Optional[bytes],
        mode: TradingMode,
        notification_context: str,
    ) -> Dict[str, Any]:
        summary_text = self._sanitize_html_for_telegram(self.format_summary_message(report, mode))
        report_id = report.get("report_id") or report.get("id")
        payload: Dict[str, Any] = {
            "context": notification_context,
            "symbol": report.get("symbol"),
            "mode": mode.value,
            "report_id": report_id,
            "messages": [],
        }

        chart_panels = []
        if not chart_bytes:
            logger.info(f"Telegram notify: loading human chart panels for {report['symbol']} ({mode.value})")
            chart_panels = self._load_human_chart_panels(report["symbol"], mode)

        if chart_panels:
            logger.info(f"Telegram notify: sending {len(chart_panels)} chart panels for {report['symbol']}")
            total = len(chart_panels)
            for idx, panel in enumerate(chart_panels, start=1):
                timeframe = str(panel.get("timeframe", "-")).upper()
                payload["messages"].append({
                    "type": "photo",
                    "photo_b64": panel.get("chart_base64", ""),
                    "filename": f"{report['symbol']}_{timeframe}.png",
                    "caption": (
                        f"📊 <b>{report['symbol']} {mode.value.upper()} Chart</b>\n"
                        f"Timeframe: <code>{timeframe}</code>\n"
                        f"Panel: <code>{idx}/{total}</code>"
                    ),
                    "parse_mode": "HTML",
                    "with_button": False,
                })

            if len(summary_text) <= 1024:
                payload["messages"].append({
                    "type": "text",
                    "text": summary_text,
                    "parse_mode": "HTML",
                    "with_button": True,
                })
            else:
                panel_label, lookback_label = self._chart_profile(mode)
                payload["messages"].append({
                    "type": "text",
                    "text": (
                        f"<b>{report['symbol']} {mode.value.upper()} Chart Profile</b>\n"
                        f"Panels: <code>{panel_label}</code>\n"
                        f"Lookback: <code>{lookback_label}</code>"
                    ),
                    "parse_mode": "HTML",
                    "with_button": False,
                })
                payload["messages"].append({
                    "type": "text",
                    "text": summary_text,
                    "parse_mode": "HTML",
                    "with_button": True,
                })
        elif chart_bytes:
            logger.info(f"Telegram notify: using inline chart bytes for {report['symbol']} ({len(chart_bytes)} bytes)")
            photo_b64 = base64.b64encode(chart_bytes).decode("ascii")
            if len(summary_text) <= 1024:
                payload["messages"].append({
                    "type": "photo",
                    "photo_b64": photo_b64,
                    "filename": f"{report['symbol']}_chart.png",
                    "caption": summary_text,
                    "parse_mode": "HTML",
                    "with_button": True,
                })
            else:
                panel_label, lookback_label = self._chart_profile(mode)
                payload["messages"].append({
                    "type": "photo",
                    "photo_b64": photo_b64,
                    "filename": f"{report['symbol']}_chart.png",
                    "caption": f"{report['symbol']} Analysis",
                    "parse_mode": None,
                    "with_button": False,
                })
                payload["messages"].append({
                    "type": "text",
                    "text": (
                        f"<b>{report['symbol']} {mode.value.upper()} Chart Profile</b>\n"
                        f"Panels: <code>{panel_label}</code>\n"
                        f"Lookback: <code>{lookback_label}</code>"
                    ),
                    "parse_mode": "HTML",
                    "with_button": False,
                })
                payload["messages"].append({
                    "type": "text",
                    "text": summary_text,
                    "parse_mode": "HTML",
                    "with_button": True,
                })
        else:
            payload["messages"].append({
                "type": "text",
                "text": summary_text,
                "parse_mode": "HTML",
                "with_button": True,
            })

        return payload

    async def _send_payload(self, bot: Bot, chat_id: str, payload: Dict[str, Any]) -> None:
        report_id = payload.get("report_id")
        for message in payload.get("messages", []):
            reply_markup = self._build_reply_markup(report_id) if message.get("with_button") else None

            if message.get("type") == "photo":
                photo_bytes = base64.b64decode(str(message.get("photo_b64") or ""))
                caption = str(message.get("caption") or "")
                parse_mode = message.get("parse_mode")
                try:
                    photo = BytesIO(photo_bytes)
                    photo.name = str(message.get("filename") or "report.png")
                    if parse_mode:
                        await bot.send_photo(
                            chat_id=chat_id,
                            photo=photo,
                            caption=caption,
                            parse_mode=parse_mode,
                            reply_markup=reply_markup,
                        )
                    else:
                        await bot.send_photo(
                            chat_id=chat_id,
                            photo=photo,
                            caption=caption,
                            reply_markup=reply_markup,
                        )
                except Exception:
                    plain_caption = re.sub(r"<[^>]+>", "", caption)
                    retry_photo = BytesIO(photo_bytes)
                    retry_photo.name = str(message.get("filename") or "report.png")
                    caption_head = plain_caption[:1024]
                    caption_tail = plain_caption[1024:]
                    await bot.send_photo(
                        chat_id=chat_id,
                        photo=retry_photo,
                        caption=caption_head,
                        reply_markup=reply_markup,
                    )
                    if caption_tail:
                        tail_chunks = self._chunk_text_for_telegram(caption_tail)
                        for idx, chunk in enumerate(tail_chunks):
                            await bot.send_message(
                                chat_id=chat_id,
                                text=chunk,
                                reply_markup=reply_markup if idx == len(tail_chunks) - 1 else None,
                            )
                continue

            text = str(message.get("text") or "")
            parse_mode = message.get("parse_mode")
            if parse_mode:
                try:
                    await bot.send_message(
                        chat_id=chat_id,
                        text=text,
                        parse_mode=parse_mode,
                        reply_markup=reply_markup,
                    )
                    continue
                except Exception:
                    text = re.sub(r"<[^>]+>", "", text)

            chunks = self._chunk_text_for_telegram(text)
            for idx, chunk in enumerate(chunks):
                await bot.send_message(
                    chat_id=chat_id,
                    text=chunk,
                    reply_markup=reply_markup if idx == len(chunks) - 1 else None,
                )

    async def replay_last_scheduled_report(self, chat_id: str) -> bool:
        payload = self.get_last_scheduled_report_payload()
        if not payload or not payload.get("messages"):
            return False
        bot = Bot(token=self.bot_token)
        await self._send_payload(bot, chat_id, payload)
        return True

    @api_retry(max_attempts=3, delay_seconds=10)
    def send_telegram_notification(
        self,
        report: Dict,
        chart_bytes: Optional[bytes] = None,
        mode: TradingMode = TradingMode.SWING,
        notification_context: str = "analysis",
    ) -> None:
        try:
            chat_id = self._get_target_chat_id()
            logger.info(
                f"Telegram notify start: symbol={report.get('symbol')} mode={mode.value} "
                f"report_id={report.get('report_id') or report.get('id')} "
                f"chart_bytes={'yes' if chart_bytes else 'no'} context={notification_context}"
            )
            payload = self._build_notification_payload(report, chart_bytes, mode, notification_context)
            self._save_report_replay_payload(payload)
            symbol_identity = str(payload.get("symbol") or report.get("symbol") or "UNKNOWN")
            mode_identity = str(mode.value or "unknown")
            base_identity = (
                payload.get("report_id")
                or report.get("id")
                or report.get("timestamp")
                or datetime.now(timezone.utc).isoformat()
            )
            report_identity = f"{symbol_identity}:{mode_identity}:{base_identity}"
            enqueue_result = execution_repository.enqueue_outbox_event(
                "telegram_payload",
                {
                    "chat_id": chat_id,
                    "payload": payload,
                },
                idempotency_key=f"telegram_payload:{report_identity}:{notification_context}",
            )
            if enqueue_result.get("deduped"):
                logger.warning(
                    f"Telegram payload deduped for {report.get('symbol')} "
                    f"context={notification_context} report_identity={report_identity}"
                )
            dispatch_result = outbox_dispatcher.publish_pending(limit=20)
            if dispatch_result.get("failed"):
                logger.warning(
                    f"Telegram summary notification dispatch had failures: {dispatch_result.get('errors')}"
                )
            else:
                logger.info(
                    f"Telegram summary dispatch result: processed={dispatch_result.get('processed', 0)} "
                    f"published={dispatch_result.get('published', 0)} deduped={bool(enqueue_result.get('deduped'))}"
                )
            logger.info(f"Telegram summary notification queued (ID: {payload.get('report_id')})")
        except Exception as e:
            logger.error(f"Telegram notification error: {e}")

    def notify(
        self,
        report: Dict,
        chart_bytes: Optional[bytes] = None,
        mode: TradingMode = TradingMode.SWING,
        notification_context: str = "analysis",
    ) -> None:
        threading.Thread(
            target=lambda: self.send_telegram_notification(report, chart_bytes, mode, notification_context),
            daemon=True,
        ).start()


report_generator = ReportGenerator()
