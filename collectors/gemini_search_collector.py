"""Gemini Search Search API collector for market narrative context.

Provides the "WHY" behind market moves that pure technical data cannot.
Uses Gemini Search sonar-pro model for deep web search with citations.

Two search modes:
1. search_market_narrative(symbol): Per-symbol daily unified narrative (SWING + POSITION in one response)
   - Hard cap: one Gemini Search API call per symbol per UTC day
   - Injects live market state: Fear&Greed, funding rate, 10Y yield, DXY
2. search_targeted(entity, entity_type, context): Graph-triggered BTC/ETH-centric search

Cost: ~$5/month on Gemini Search Pro API plan (200 requests/day included)
"""

import requests
from typing import Dict, List, Optional, Tuple
from config.settings import settings, TradingMode
from config.database import db
from datetime import datetime, timezone, timedelta
from loguru import logger
import json
import os

class GeminiSearchCollector:
    BASE_URL = "https://api.gemini_search.ai/chat/completions"

    # 심볼당 하루 최대 API 호출 횟수 (하드 제한)
    MAX_CALLS_PER_DAY = 2

    # 인메모리 일일 캐시: Supabase가 죽어도 같은 UTC 날짜에 중복 API 호출 방지
    # { symbol: {"date": date, "result": dict} }
    _mem_cache: dict = {}

    # Full symbol → coin name mapping (not just BTC/ETH)
    COIN_NAME_MAP = {
        "BTC": "Bitcoin", "ETH": "Ethereum", "SOL": "Solana",
        "BNB": "BNB", "XRP": "XRP", "ADA": "Cardano",
        "AVAX": "Avalanche", "DOT": "Polkadot", "LINK": "Chainlink",
        "MATIC": "Polygon", "SUI": "Sui", "APT": "Aptos",
        "DOGE": "Dogecoin", "SHIB": "Shiba Inu", "UNI": "Uniswap",
        "ATOM": "Cosmos", "NEAR": "NEAR Protocol", "ARB": "Arbitrum",
        "OP": "Optimism", "INJ": "Injective", "TIA": "Celestia",
        "SEI": "Sei", "JUP": "Jupiter", "TON": "Toncoin",
        "WIF": "Dogwifhat", "PEPE": "Pepe", "FTM": "Fantom",
    }

    def __init__(self):
        self.api_key = settings.PERPLEXITY_API_KEY
        self.default_model = settings.PERPLEXITY_MODEL_NARRATIVE

    def _get_coin_name(self, symbol: str) -> Tuple[str, str]:
        """BTCUSDT → ('Bitcoin', 'BTC'),  SOLUSDT → ('Solana', 'SOL')"""
        base = symbol.upper().replace("USDT", "").replace("BUSD", "").replace("USDC", "")
        coin_name = self.COIN_NAME_MAP.get(base, base)
        return coin_name, base

    def _call_api(self, prompt: str, max_tokens: int = 1000, model: Optional[str] = None) -> Tuple[str, List[str]]:
        from agents.ai_router import ai_client
        _system = (
            "You are an external evidence engine for a crypto trading policy system. "
            "Use your Google Search access to verify and collect facts. "
            "Return ONLY a valid JSON object. No markdown, no prose, no code fences."
        )
        content = ai_client.generate_with_grounding(
            system_prompt=_system,
            user_message=prompt,
            max_tokens=max_tokens,
            temperature=0.1,
        )
        if not content:
            raise Exception("Gemini Grounding returned empty response")
        return content, []

    def _safe_load_json(self, value):
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    def _load_latest_playbook(self, symbol: str, mode: str) -> Optional[Dict]:
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
            return res.data[0] if res.data else None
        except Exception:
            return None

    def _extract_price_condition(self, conditions, preferred_ops: Optional[List[str]] = None) -> str:
        preferred_ops = preferred_ops or [">", ">=", "<", "<="]
        if not isinstance(conditions, list):
            return ""

        def _rank(op: str) -> int:
            try:
                return preferred_ops.index(op)
            except ValueError:
                return len(preferred_ops)

        price_conditions = []
        for cond in conditions:
            if not isinstance(cond, dict):
                continue
            if str(cond.get("metric", "")).lower() != "price":
                continue
            op = str(cond.get("operator", "")).strip()
            value = cond.get("value")
            try:
                price_conditions.append((_rank(op), op, float(value)))
            except Exception:
                continue

        if not price_conditions:
            return ""

        _, op, value = sorted(price_conditions, key=lambda item: (item[0], item[2]))[0]
        return f"price {op} {value:,.2f}"

    def _gather_market_state(self, symbol: str) -> Dict:
        """Pull live market indicators from DB for mode-aware prompt injection.

        Returns a dict with available fields  any field may be absent if DB
        has no data yet (cold start, API failures). Callers must handle None.
        """
        state: Dict = {}
        try:
            from processors.gcs_parquet import gcs_parquet_store

            # Fear & Greed Index
            fg = gcs_parquet_store.get_latest_row("fear_greed")
            if fg:
                state["fear_greed"] = f"{fg.get('value', 'N/A')} ({fg.get('classification', 'N/A')})"

            # Macro indicators
            macro = gcs_parquet_store.get_latest_row("macro")
            if macro:
                for state_key, db_key in [
                    ("dgs10", "dgs10"),
                    ("dxy", "dxy"),
                    ("nasdaq", "nasdaq"),
                    ("ust_spread", "ust_2s10s_spread"),
                ]:
                    if macro.get(db_key) is not None:
                        state[state_key] = macro[db_key]

            # Current price + 30d return from local parquet
            ohlcv_df = gcs_parquet_store.load_ohlcv("1m", symbol, months_back=1)
            if not ohlcv_df.empty and "close" in ohlcv_df.columns:
                state["current_price"] = float(ohlcv_df["close"].iloc[-1])
                cutoff_30d = datetime.now(timezone.utc) - timedelta(days=31)
                early = ohlcv_df[ohlcv_df["timestamp"] >= cutoff_30d]
                if not early.empty:
                    price_30d = float(early["close"].iloc[0])
                    if price_30d > 0:
                        state["return_30d"] = round(
                            (state["current_price"] / price_30d - 1) * 100, 1
                        )

            # Funding rate + OI + positioning
            try:
                funding_df = gcs_parquet_store.get_funding_history_parquet(symbol, limit=4)
                if not funding_df.empty and "funding_rate" in funding_df.columns:
                    state["funding_rate"] = float(funding_df["funding_rate"].iloc[-1])
                    latest = funding_df.iloc[-1]
                    ls_ratio = latest.get("long_short_ratio")
                    if ls_ratio is not None:
                        state["long_short_ratio"] = float(ls_ratio)

                    if len(funding_df) >= 2:
                        latest_oi = sum(float(latest.get(k, 0) or 0) for k in ("oi_binance", "oi_bybit", "oi_okx"))
                        prev = funding_df.iloc[0]
                        prev_oi = sum(float(prev.get(k, 0) or 0) for k in ("oi_binance", "oi_bybit", "oi_okx"))
                        if prev_oi:
                            state["oi_change_pct"] = round(((latest_oi - prev_oi) / prev_oi) * 100, 2)

                    funding_rate = float(latest.get("funding_rate", state.get("funding_rate", 0.0)) or 0.0)
                    long_short_ratio = float(latest.get("long_short_ratio", 1.0) or 1.0)
                    if funding_rate > 0.03 and long_short_ratio > 1.05:
                        state["positioning_risk"] = "long_crowded"
                    elif funding_rate < -0.02 and long_short_ratio < 0.95:
                        state["positioning_risk"] = "short_crowded"
                    else:
                        state["positioning_risk"] = "balanced"
            except Exception:
                pass

            try:
                latest_report = db.get_latest_report(symbol=symbol)
                decision = self._safe_load_json((latest_report or {}).get("final_decision"))
                if decision:
                    direction = str(decision.get("decision", "HOLD") or "HOLD").upper()
                    state["policy_candidate_bias"] = direction if direction in ("LONG", "SHORT") else "NEUTRAL"
                    policy_checks = self._safe_load_json(decision.get("policy_checks"))
                    if isinstance(policy_checks, dict) and policy_checks:
                        if policy_checks.get("flow_confirmed") is True:
                            state["flow_state"] = "confirmed"
                        elif policy_checks.get("flow_signals"):
                            state["flow_state"] = "mixed"
                        elif state.get("oi_change_pct") is not None:
                            state["flow_state"] = "weak" if abs(float(state["oi_change_pct"])) < 1.0 else "mixed"
                    if not state.get("flow_state"):
                        state["flow_state"] = "unknown"
            except Exception:
                pass

            try:
                for lane in ("swing", "position"):
                    playbook_row = self._load_latest_playbook(symbol, lane)
                    if not playbook_row:
                        continue
                    playbook = playbook_row.get("playbook", {})
                    if isinstance(playbook, str):
                        playbook = self._safe_load_json(playbook)
                    if not isinstance(playbook, dict):
                        continue
                    entry_hint = self._extract_price_condition(
                        playbook.get("entry_conditions", []),
                        preferred_ops=[">", ">=", "<", "<="],
                    )
                    invalid_hint = self._extract_price_condition(
                        playbook.get("invalidation_conditions", []),
                        preferred_ops=["<", "<=", ">", ">="],
                    )
                    if entry_hint:
                        state[f"{lane}_entry_trigger"] = entry_hint
                    if invalid_hint:
                        state[f"{lane}_invalidation_level"] = invalid_hint
            except Exception:
                pass

            if not state.get("policy_candidate_bias"):
                state["policy_candidate_bias"] = "NEUTRAL"
            if not state.get("flow_state"):
                state["flow_state"] = "unknown"

        except Exception as e:
            logger.warning(f"Market state gather failed for {symbol}: {e}")

        return state

    def _build_swing_prompt(self, coin_name: str, base: str, state: Dict) -> str:
        """Prompt for SWING trading mode (holding: days to 2 weeks).

        Focuses on: 1-7 day catalysts, upcoming macro events, funding extremes,
        ETF weekly flows, exchange inflow/outflow trends.
        """
        now_utc = datetime.now(timezone.utc)
        date_today = now_utc.strftime("%Y-%m-%d")
        date_7d_ago = (now_utc - timedelta(days=7)).strftime("%Y-%m-%d")

        lines = [
            f"Today is {date_today} (UTC). Analyzing {coin_name} ({base}) for SWING TRADE (holding: days to 2 weeks).",
            "",
            "Current market indicators:",
        ]
        if state.get("current_price"):
            ret = (
                f" ({state['return_30d']:+.1f}% 30d)"
                if state.get("return_30d") is not None
                else ""
            )
            lines.append(f"- {base} price: ${state['current_price']:,.0f}{ret}")
        if state.get("fear_greed"):
            lines.append(f"- Fear & Greed: {state['fear_greed']}")
        if state.get("funding_rate") is not None:
            fr = state["funding_rate"]
            note = (
                " → overleveraged longs, squeeze risk" if fr > 0.05
                else " → heavy shorts, short squeeze risk" if fr < -0.03
                else ""
            )
            lines.append(f"- Funding rate: {fr:+.4f}% per 8h{note}")
        if state.get("dgs10") is not None:
            lines.append(f"- US 10Y yield: {state['dgs10']:.2f}%")
        if state.get("dxy") is not None:
            lines.append(f"- DXY: {state['dxy']:.1f}")
        if state.get("ust_spread") is not None:
            lines.append(f"- 2s10s spread: {state['ust_spread']:+.2f}%")

        lines += [
            "",
            f"Search for events in the LAST 8 HOURS that directly affect {coin_name} price.",
            "Prioritize price action, news, and institutional flows. Every item must include its exact date/time.",
            "",
            "Return this exact JSON:",
            "{",
            f'  "summary": "2-3 sentences: what specifically happened to {coin_name} in the last 8 hours that matters for a swing trade",',
            '  "sentiment": "bullish" or "bearish" or "neutral",',
            '  "bullish_factors": ["YYYY-MM-DD HH:mm: catalyst that supports buying"],',
            '  "bearish_factors": ["YYYY-MM-DD HH:mm: risk or catalyst that supports selling"],',
            '  "upcoming_events": ["YYYY-MM-DD: scheduled macro/crypto event in next 7 days"],',
            f'  "reasoning": "1 sentence: the single highest-probability near-term driver for the next 8-24 hours",',
            f'  "macro_context": "how rates/DXY/risk appetite is interacting with {coin_name} right now"',
            "}",
            "",
            "Search priorities (in order):",
            f"1. HIGH AUTHORITY VERIFICATION: Prioritize reports from {', '.join(settings.TRUSTED_NEWS_DOMAINS[:8])}.",
            "2. Upcoming macro events (next 7 days): Fed speakers, CPI, NFP, futures/options expiry dates",
            "3. BTC ETF net flows (last 7 days): trend from BlackRock IBIT, Fidelity FBTC, ARK 21Shares",
            f"3. Exchange flows: {coin_name} inflows to exchanges (sell pressure) or outflows (accumulation)",
            "4. Funding rate extremes: whether current positioning creates liquidation cascade risk",
            "5. Key price levels: specific support/resistance levels cited by analysts with precise prices",
            "",
            f"Do NOT include: events before {date_7d_ago}, generic 'crypto is volatile' statements, "
            f"altcoin news not directly affecting {coin_name} price.",
        ]
        return "\n".join(lines)

    def _build_position_prompt(self, coin_name: str, base: str, state: Dict) -> str:
        """Prompt for POSITION trading mode (holding: 2-8 weeks).

        Focuses on: cycle position, structural macro regime, ETF cumulative flows,
        on-chain cycle indicators (NUPL/MVRV), institutional positioning.
        """
        now_utc = datetime.now(timezone.utc)
        date_today = now_utc.strftime("%Y-%m-%d")
        date_30d_ago = (now_utc - timedelta(days=30)).strftime("%Y-%m-%d")

        lines = [
            f"Today is {date_today} (UTC). Analyzing {coin_name} ({base}) for POSITION TRADE (holding: 2-8 weeks).",
            "",
            "Current market indicators:",
        ]
        if state.get("current_price"):
            ret = (
                f" ({state['return_30d']:+.1f}% over last 30 days)"
                if state.get("return_30d") is not None
                else ""
            )
            lines.append(f"- {base} price: ${state['current_price']:,.0f}{ret}")
        if state.get("fear_greed"):
            lines.append(
                f"- Fear & Greed: {state['fear_greed']} "
                "(index below 20 has historically been strong accumulation zones)"
            )
        if state.get("dgs10") is not None:
            lines.append(f"- US 10Y yield: {state['dgs10']:.2f}%")
        if state.get("dxy") is not None:
            lines.append(f"- DXY: {state['dxy']:.1f}")
        if state.get("ust_spread") is not None:
            lines.append(
                f"- 2s10s spread: {state['ust_spread']:+.2f}% "
                "(positive = curve steepening = risk-on potential)"
            )
        if state.get("nasdaq") is not None:
            lines.append(f"- Nasdaq: {state['nasdaq']:,.0f}")

        lines += [
            "",
            f"Search for structural developments from {date_30d_ago} to {date_today} "
            f"that affect the {coin_name} medium-term investment thesis.",
            "",
            "Return this exact JSON:",
            "{",
            f'  "summary": "2-3 sentences: where is {coin_name} in its market cycle and what is the dominant regime",',
            '  "sentiment": "bullish" or "bearish" or "neutral",',
            '  "cycle_position": "one of: early_accumulation | late_accumulation | distribution | capitulation | recovery  with 1-sentence evidence",',
            '  "bullish_factors": ["YYYY-MM-DD: structural factor supporting medium-term bullish thesis"],',
            '  "bearish_factors": ["structural risk that could extend the current bearish phase or trigger further decline"],',
            f'  "reasoning": "1 sentence: the single most important factor for {coin_name} direction over the next 2-8 weeks",',
            '  "macro_context": "macro regime: rates trajectory, dollar trend, real yield impact on risk assets"',
            "}",
            "",
            "Search priorities (in order):",
            f"1. {coin_name} ETF cumulative flows (last 30 days): net trend, institutional appetite, AUM change",
            "2. Fed policy trajectory: rate cut/hike probability shift, inflation trend, FOMC minutes tone",
            "3. On-chain cycle indicators: NUPL, MVRV ratio, long-term holder behavior from Glassnode/CryptoQuant",
            "4. Institutional positioning: corporate treasury moves, sovereign wealth fund BTC allocation",
            "5. Regulatory regime: pending legislation or enforcement actions affecting the crypto investment thesis",
            "",
            f"Do NOT include: day-to-day price moves, short-term funding spikes, "
            f"events before {date_30d_ago}, altcoin news unrelated to {coin_name}.",
        ]
        return "\n".join(lines)

    def _build_unified_prompt(self, coin_name: str, base: str, state: Dict) -> str:
        """Single daily prompt aligned to the deterministic trading policy."""
        now_utc = datetime.now(timezone.utc)
        date_today = now_utc.strftime("%Y-%m-%d")
        date_8h_ago = (now_utc - timedelta(hours=8)).strftime("%Y-%m-%d %H:%M")
        date_7d_ago = (now_utc - timedelta(days=7)).strftime("%Y-%m-%d")
        date_30d_ago = (now_utc - timedelta(days=30)).strftime("%Y-%m-%d")

        lines = [
            f"Today is {date_today} (UTC). Analyze {coin_name} ({base}) with TWO horizons in one report.",
            "",
            "Current market indicators:",
        ]
        if state.get("current_price"):
            ret = (
                f" ({state['return_30d']:+.1f}% 30d)"
                if state.get("return_30d") is not None
                else ""
            )
            lines.append(f"- {base} price: ${state['current_price']:,.0f}{ret}")
        if state.get("fear_greed"):
            lines.append(f"- Fear & Greed: {state['fear_greed']}")
        if state.get("funding_rate") is not None:
            lines.append(f"- Funding rate: {state['funding_rate']:+.4f}% per 8h")
        if state.get("dgs10") is not None:
            lines.append(f"- US 10Y yield: {state['dgs10']:.2f}%")
        if state.get("dxy") is not None:
            lines.append(f"- DXY: {state['dxy']:.1f}")
        if state.get("ust_spread") is not None:
            lines.append(f"- 2s10s spread: {state['ust_spread']:+.2f}%")
        if state.get("nasdaq") is not None:
            lines.append(f"- Nasdaq: {state['nasdaq']:,.0f}")
        policy_hint_lines = []
        if state.get("policy_candidate_bias"):
            policy_hint_lines.append(f"- Policy candidate bias: {state['policy_candidate_bias']}")
        if state.get("flow_state"):
            flow_line = f"- Flow state: {state['flow_state']}"
            if state.get("oi_change_pct") is not None:
                flow_line += f" (OI change {state['oi_change_pct']:+.2f}%)"
            policy_hint_lines.append(flow_line)
        if state.get("positioning_risk"):
            policy_hint_lines.append(f"- Positioning risk: {state['positioning_risk']}")
        if state.get("swing_entry_trigger"):
            policy_hint_lines.append(f"- Swing reference trigger: {state['swing_entry_trigger']}")
        if state.get("swing_invalidation_level"):
            policy_hint_lines.append(f"- Swing invalidation reference: {state['swing_invalidation_level']}")
        if state.get("position_entry_trigger"):
            policy_hint_lines.append(f"- Position reference trigger: {state['position_entry_trigger']}")
        if state.get("position_invalidation_level"):
            policy_hint_lines.append(f"- Position invalidation reference: {state['position_invalidation_level']}")

        lines += [
            "",
            "Minimal internal policy context (relevance hint only, not ground truth):",
        ]
        if policy_hint_lines:
            lines.extend(policy_hint_lines[:7])
        else:
            lines.append("- No internal policy context available.")

        lines += [
            "",
            "Search instructions:",
            "You are NOT making a trade recommendation.",
            "You are an external evidence engine for a deterministic crypto trading policy.",
            "Your job is to identify only factual, policy-relevant external events that may strengthen, weaken, or invalidate an existing technical/flow setup.",
            f"1) SWING horizon: last 8 hours to next 7 days (window starts {date_8h_ago} UTC).",
            f"2) POSITION horizon: structural context from {date_30d_ago} to {date_today}.",
            f"3) Every factor/event must include concrete dates. Ignore events earlier than {date_30d_ago}.",
            "4) Prefer high-authority reporting and official sources. Avoid speculation, social chatter, and generic sentiment commentary.",
            "5) Distinguish whether the move is driven by a short-lived headline or a structural regime shift.",
            "6) Focus on events that matter for policy questions such as: should a trend-following setup be trusted, delayed, downsized, or invalidated?",
            "",
            "Return strict JSON only with this schema:",
            "{",
            f'  "symbol": "{base}",',
            '  "summary": "2-3 sentence policy-oriented summary of the most relevant external facts",',
            '  "sentiment": "bullish" | "bearish" | "neutral",',
            '  "reasoning": "single most important external driver that currently matters for policy decisions",',
            '  "macro_context": "rates/dxy/risk regime impact now, only if directly relevant",',
            '  "policy_supporting_factors": ["YYYY-MM-DD HH:mm or YYYY-MM-DD: verified fact that supports trusting an existing LONG/SHORT setup"],',
            '  "policy_risks": ["YYYY-MM-DD HH:mm or YYYY-MM-DD: verified fact that argues for caution, delay, smaller size, or lower conviction"],',
            '  "invalidation_risks": ["YYYY-MM-DD HH:mm or YYYY-MM-DD: verified fact that could invalidate the thesis or sharply reverse positioning"],',
            '  "scheduled_binary_events": ["YYYY-MM-DD HH:mm or YYYY-MM-DD: upcoming event that can materially move BTC/ETH"],',
            '  "headline_vs_structural": "headline | structural | mixed, with one short sentence of explanation",',
            '  "watchitems": ["policy-external but important emerging variable to monitor"],',
            '  "swing_view": {',
            '    "summary": "1-2 sentence short-term external evidence summary for the next 8-24h",',
            '    "sentiment": "bullish" | "bearish" | "neutral",',
            '    "trigger_support": ["YYYY-MM-DD HH:mm: short-term fact that supports trusting the current setup"],',
            '    "trigger_risks": ["YYYY-MM-DD HH:mm: short-term fact that weakens the setup or increases squeeze/fade risk"],',
            '    "invalidation_risks": ["YYYY-MM-DD HH:mm: short-term fact that could break the setup"],',
            '    "scheduled_events": ["YYYY-MM-DD HH:mm or YYYY-MM-DD: event in next 7 days"],',
            '    "key_levels": ["support/resistance with exact prices"]',
            '  },',
            '  "position_view": {',
            '    "summary": "1-2 sentence medium-term external evidence summary for the next 2-8 weeks",',
            '    "sentiment": "bullish" | "bearish" | "neutral",',
            '    "cycle_position": "early_accumulation | late_accumulation | distribution | capitulation | recovery",',
            '    "structural_support": ["YYYY-MM-DD: structural fact that supports the medium-term thesis"],',
            '    "structural_risks": ["YYYY-MM-DD: structural fact that weakens the medium-term thesis"],',
            '    "invalidation_risks": ["YYYY-MM-DD: structural fact that could invalidate the medium-term thesis"],',
            '    "institutional_flows": ["ETF/corporate flow facts with dates"],',
            '    "regime_risks": ["macro/regulatory risk with dates"],',
            '    "watchitems": ["emerging medium-term variable not yet part of the core policy"]',
            '  }',
            "}",
            "",
            "Output rules:",
            "- No trade recommendation, no target position, no leverage guidance.",
            "- Do not tell the user to buy, sell, long, short, hold, or wait.",
            "- If evidence is weak, say so explicitly instead of filling with generic commentary.",
            "- Keep lists short and high signal. Do not repeat the same fact in multiple sections unless needed.",
            "Source priority: high-authority financial media + official domains, then crypto data providers.",
            f"Do NOT include: generic statements, unrelated altcoin noise, or events before {date_7d_ago} for swing factors.",
        ]
        return "\n".join(lines)

    def _ensure_list(self, value) -> List[str]:
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    def _ensure_dict(self, value) -> Dict:
        return value if isinstance(value, dict) else {}

    def _normalize_narrative_result(self, result: Dict, symbol: str, status: str = "ok") -> Dict:
        normalized = dict(result or {})
        normalized["symbol"] = symbol
        normalized["status"] = status

        normalized["summary"] = str(normalized.get("summary", "") or "").strip()
        normalized["sentiment"] = str(normalized.get("sentiment", "neutral") or "neutral").lower()
        normalized["reasoning"] = str(normalized.get("reasoning", "") or "").strip()
        normalized["macro_context"] = str(normalized.get("macro_context", "") or "").strip()
        normalized["headline_vs_structural"] = str(
            normalized.get("headline_vs_structural", "") or ""
        ).strip()

        normalized["policy_supporting_factors"] = self._ensure_list(
            normalized.get("policy_supporting_factors", normalized.get("bullish_factors", []))
        )
        normalized["policy_risks"] = self._ensure_list(
            normalized.get("policy_risks", normalized.get("bearish_factors", []))
        )
        normalized["invalidation_risks"] = self._ensure_list(normalized.get("invalidation_risks", []))
        normalized["scheduled_binary_events"] = self._ensure_list(
            normalized.get("scheduled_binary_events", normalized.get("upcoming_events", []))
        )
        normalized["watchitems"] = self._ensure_list(normalized.get("watchitems", []))

        swing_view = self._ensure_dict(normalized.get("swing_view", {}))
        normalized["swing_view"] = {
            "summary": str(swing_view.get("summary", "") or "").strip(),
            "sentiment": str(swing_view.get("sentiment", normalized["sentiment"]) or normalized["sentiment"]).lower(),
            "trigger_support": self._ensure_list(
                swing_view.get("trigger_support", swing_view.get("bullish_factors", []))
            ),
            "trigger_risks": self._ensure_list(
                swing_view.get("trigger_risks", swing_view.get("bearish_factors", []))
            ),
            "invalidation_risks": self._ensure_list(swing_view.get("invalidation_risks", [])),
            "scheduled_events": self._ensure_list(
                swing_view.get("scheduled_events", swing_view.get("upcoming_events", []))
            ),
            "key_levels": self._ensure_list(swing_view.get("key_levels", [])),
        }

        position_view = self._ensure_dict(normalized.get("position_view", {}))
        normalized["position_view"] = {
            "summary": str(position_view.get("summary", "") or "").strip(),
            "sentiment": str(position_view.get("sentiment", normalized["sentiment"]) or normalized["sentiment"]).lower(),
            "cycle_position": str(
                position_view.get("cycle_position", normalized.get("cycle_position", "")) or ""
            ).strip(),
            "structural_support": self._ensure_list(
                position_view.get("structural_support", position_view.get("bullish_factors", []))
            ),
            "structural_risks": self._ensure_list(
                position_view.get("structural_risks", position_view.get("bearish_factors", []))
            ),
            "invalidation_risks": self._ensure_list(position_view.get("invalidation_risks", [])),
            "institutional_flows": self._ensure_list(position_view.get("institutional_flows", [])),
            "regime_risks": self._ensure_list(position_view.get("regime_risks", [])),
            "watchitems": self._ensure_list(position_view.get("watchitems", [])),
        }

        normalized["cycle_position"] = normalized["position_view"].get("cycle_position", "")
        normalized["upcoming_events"] = normalized["scheduled_binary_events"]
        normalized["bullish_factors"] = normalized["policy_supporting_factors"]
        normalized["bearish_factors"] = normalized["policy_risks"]
        normalized["key_events"] = (
            normalized["policy_supporting_factors"][:2]
            + normalized["policy_risks"][:2]
            + normalized["scheduled_binary_events"][:2]
        )
        return normalized

    def search_market_narrative(self, symbol: str = "BTC", is_emergency: bool = False) -> Dict:
        """Daily narrative search (one API call per symbol per UTC day)."""
        import os
        _job = os.environ.get("JOB_NAME", "").strip()
        if _job != "daily_precision":
            logger.warning(
                f"[Gemini Search] BLOCKED: search_market_narrative은 daily_precision 잡에서만 허용 "
                f"(현재 JOB_NAME={_job!r})"
            )
            return self._empty_result(symbol)

        if not self.api_key:
            logger.warning("PERPLEXITY_API_KEY not set, skipping narrative search")
            result = self._empty_result(symbol)
            self.persist_narrative(result, symbol)
            return result

        coin_name, base = self._get_coin_name(symbol)
        today = datetime.now(timezone.utc).date()

        # 1차 캐시: 인메모리 (Supabase 무관 — 프로세스 수명 동안 유효)
        mem = GeminiSearchCollector._mem_cache.get(symbol)
        if mem and mem.get("date") == today:
            logger.info(f"Gemini Search [{symbol}]: in-memory cache hit — skipping API call")
            return mem["result"]

        # 2차 캐시: Supabase (프로세스 재시작 후 복구용)
        try:
            last_n = db.get_latest_narrative_data(symbol)
            if last_n:
                last_ts = datetime.fromisoformat(last_n['timestamp'].replace('Z', '+00:00'))
                if last_ts.date() == today:
                    logger.info(f"Gemini Search [{symbol}]: already fetched today (UTC), using cached narrative.")
                    cached_result = last_n.get('raw_payload', {})
                    if not cached_result:
                        cached_result = self._empty_result(symbol)
                        cached_result.update(last_n)
                    GeminiSearchCollector._mem_cache[symbol] = {"date": today, "result": cached_result}
                    return cached_result
        except Exception as e:
            logger.warning(f"Narrative cache check failed (Supabase): {e}")

        # 하드 일일 한도 체크 — 캐시 레이어 모두 통과한 경우에만 도달
        if not self._acquire_daily_slot(symbol):
            return self._empty_result(symbol)

        # Emergency uses cheaper target model; normal run uses narrative model.
        use_model = settings.PERPLEXITY_MODEL_TARGETED if is_emergency else settings.PERPLEXITY_MODEL_NARRATIVE
        state = self._gather_market_state(symbol)
        prompt = self._build_unified_prompt(coin_name, base, state)
        logger.info(f"Gemini Search [{coin_name}]: unified SWING+POSITION daily search")

        try:
            content, citations = self._call_api(prompt, model=use_model)
            result = self._parse_response(content, symbol)
            if citations:
                result["sources"] = citations[:5]
            self.persist_narrative(result, symbol)
            GeminiSearchCollector._mem_cache[symbol] = {"date": today, "result": result}
            logger.info(
                f"Gemini Search narrative [{coin_name}]: "
                f"sentiment={result.get('sentiment', '?')} (unified)"
            )
            return result

        except requests.exceptions.HTTPError as e:
            logger.error(f"Gemini Search API HTTP error: {e}")
        except requests.exceptions.Timeout:
            logger.error("Gemini Search API timeout")
        except Exception as e:
            logger.error(f"Gemini Search collector error: {e}")

        result = self._empty_result(symbol)
        self.persist_narrative(result, symbol)
        return result

    # ── 통합 BTC+ETH 1회 호출 ──────────────────────────────────────────────────
    _UNIFIED_GCS_PATH = "fallback/narrative/unified/latest.json"
    _unified_mem_cache: dict = {}  # {"date": date, "btc": {...}, "eth": {...}}

    def _extract_and_validate_json(self, content: str) -> Optional[Dict]:
        """LLM 응답에서 JSON 추출 + btc/eth 필드 검증."""
        import re as _re
        if not content:
            return None
        text = content.strip()

        # 마크다운 펜스 제거
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.rsplit("```", 1)[0].strip()

        # { } 범위 추출
        start = text.find("{")
        end = text.rfind("}") + 1
        if start < 0 or end <= start:
            return None

        candidate = text[start:end]

        # Grounding citation 마커 제거: [1], [2], [1][2] 등
        candidate = _re.sub(r'\[\d+\]', '', candidate)

        try:
            raw = json.loads(candidate)
        except json.JSONDecodeError:
            # 2차 시도: JSON 문자열 값 내 개행/탭 정리
            candidate = _re.sub(r'[\x00-\x1f](?=[^"]*"(?:[^"]*"[^"]*")*[^"]*$)', ' ', candidate)
            try:
                raw = json.loads(candidate)
            except json.JSONDecodeError:
                return None

        # 최소 구조 검증
        if not isinstance(raw, dict):
            return None
        if "btc" not in raw or "eth" not in raw:
            return None
        for key in ("btc", "eth"):
            sec = raw.get(key, {})
            if not isinstance(sec, dict):
                return None
            # 필수 필드 기본값 보장
            sec.setdefault("summary", "")
            sec.setdefault("sentiment", "neutral")
            sec.setdefault("reasoning", "")
            sec.setdefault("macro_context", "")
            sec.setdefault("policy_supporting_factors", [])
            sec.setdefault("policy_risks", [])
            sec.setdefault("invalidation_risks", [])
            sec.setdefault("scheduled_binary_events", [])
            sec.setdefault("headline_vs_structural", "")
            sec.setdefault("swing_view", {"summary": "", "sentiment": "neutral", "scheduled_events": []})
            # sentiment 값 정규화
            if sec["sentiment"] not in ("bullish", "bearish", "neutral"):
                sec["sentiment"] = "neutral"
            # list 필드 타입 보장
            for f in ("policy_supporting_factors", "policy_risks", "invalidation_risks",
                      "scheduled_binary_events"):
                if not isinstance(sec[f], list):
                    sec[f] = [str(sec[f])] if sec[f] else []

        return raw

    def _collect_news_context(self, hours: int = 4, max_articles: int = 20) -> str:
        """최근 RSS 기사 + Telegram 메시지를 텍스트로 수집."""
        lines = []

        # RSS 기사 (SQLite)
        try:
            from collectors.crypto_news_collector import collector as news_collector
            articles = news_collector.fetch_news(
                categories=["bitcoin", "ethereum", "macro", "institutional"],
                limit=max_articles,
            )
            if articles:
                lines.append("=== RECENT NEWS ARTICLES ===")
                for a in articles:
                    lines.append(f"[{a['source']}] {a['title']}")
                    if a.get("description"):
                        lines.append(f"  {a['description'][:300]}")
        except Exception as e:
            logger.debug(f"[Narrative] RSS collect failed: {e}")

        # Telegram 메시지 (GCS parquet)
        try:
            from processors.gcs_parquet import gcs_parquet_store
            msgs = gcs_parquet_store.get_recent_telegram_messages_parquet(hours=hours) or []
            if msgs:
                lines.append("\n=== RECENT TELEGRAM SIGNALS ===")
                for m in msgs[:20]:
                    ch = m.get("channel", "")
                    txt = str(m.get("text", ""))[:200]
                    if txt:
                        lines.append(f"[{ch}] {txt}")
        except Exception as e:
            logger.debug(f"[Narrative] Telegram collect failed: {e}")

        return "\n".join(lines)

    def _build_unified_multi_prompt(self, btc_state: Dict, eth_state: Dict, collected_context: str = "") -> str:
        """BTC+ETH 동시 분석 프롬프트 — 하루 1회 호출."""
        now_utc = datetime.now(timezone.utc)
        date_today = now_utc.strftime("%Y-%m-%d")
        date_8h_ago = (now_utc - timedelta(hours=8)).strftime("%Y-%m-%d %H:%M")
        date_7d_ago = (now_utc - timedelta(days=7)).strftime("%Y-%m-%d")
        date_30d_ago = (now_utc - timedelta(days=30)).strftime("%Y-%m-%d")

        def _state_lines(base: str, state: Dict) -> list:
            lines = [f"[{base}]"]
            if state.get("current_price"):
                ret = f" ({state['return_30d']:+.1f}% 30d)" if state.get("return_30d") is not None else ""
                lines.append(f"- Price: ${state['current_price']:,.0f}{ret}")
            if state.get("funding_rate") is not None:
                lines.append(f"- Funding: {state['funding_rate']:+.4f}% per 8h")
            if state.get("flow_state"):
                lines.append(f"- Flow: {state['flow_state']}")
            return lines

        shared = []
        if btc_state.get("fear_greed"):
            shared.append(f"- Fear & Greed: {btc_state['fear_greed']}")
        if btc_state.get("dgs10") is not None:
            shared.append(f"- US 10Y yield: {btc_state['dgs10']:.2f}%")
        if btc_state.get("dxy") is not None:
            shared.append(f"- DXY: {btc_state['dxy']:.1f}")
        if btc_state.get("ust_spread") is not None:
            shared.append(f"- 2s10s spread: {btc_state['ust_spread']:+.2f}%")
        if btc_state.get("nasdaq") is not None:
            shared.append(f"- Nasdaq: {btc_state['nasdaq']:,.0f}")

        lines = [
            f"Today is {date_today} (UTC).",
            "Analyze BOTH Bitcoin (BTC) and Ethereum (ETH) in ONE response.",
            "",
            "Shared macro indicators:",
        ] + (shared or ["- No macro data available."]) + [
            "",
            "Asset-specific indicators:",
        ] + _state_lines("BTC", btc_state) + _state_lines("ETH", eth_state) + [
            "",
            "Search instructions:",
            "You are an external evidence engine for a deterministic crypto trading policy.",
            f"1) SWING horizon: last 8 hours to next 7 days (window starts {date_8h_ago} UTC).",
            f"2) POSITION horizon: structural context from {date_30d_ago} to {date_today}.",
            f"3) Every factor must include concrete dates. Ignore events earlier than {date_30d_ago}.",
            "4) Prefer high-authority sources. Avoid speculation and generic sentiment.",
            "5) Distinguish short-lived headlines from structural regime shifts.",
            "6) For ETH-specific: Pectra upgrade, staking flows, DeFi TVL, L2 metrics.",
            "7) For BTC-specific: ETF flows, institutional custody, miner activity, strategic reserve.",
            "",
            "OUTPUT FORMAT — CRITICAL:",
            "- Return ONLY a single JSON object. Zero prose before or after.",
            "- Do NOT use markdown code fences (no ```json).",
            "- All string values must be non-empty. Use 'N/A' if unknown.",
            '- "sentiment" must be exactly one of: "bullish", "bearish", "neutral".',
            "- All list fields must be JSON arrays (even if empty: []).",
            "",
            "EXACT SCHEMA (copy this structure):",
            '{',
            '  "btc": {',
            '    "symbol": "BTC",',
            '    "summary": "string",',
            '    "sentiment": "bullish",',
            '    "reasoning": "string",',
            '    "macro_context": "string",',
            '    "policy_supporting_factors": ["YYYY-MM-DD: fact"],',
            '    "policy_risks": ["YYYY-MM-DD: fact"],',
            '    "invalidation_risks": ["YYYY-MM-DD: fact"],',
            '    "scheduled_binary_events": ["YYYY-MM-DD HH:mm: event"],',
            '    "headline_vs_structural": "structural",',
            '    "swing_view": {',
            '      "summary": "string",',
            '      "sentiment": "bullish",',
            '      "scheduled_events": ["YYYY-MM-DD HH:mm: event"]',
            '    }',
            '  },',
            '  "eth": {',
            '    "symbol": "ETH",',
            '    "summary": "string",',
            '    "sentiment": "neutral",',
            '    "reasoning": "string",',
            '    "macro_context": "string",',
            '    "policy_supporting_factors": [],',
            '    "policy_risks": [],',
            '    "invalidation_risks": [],',
            '    "scheduled_binary_events": [],',
            '    "headline_vs_structural": "headline",',
            '    "swing_view": {',
            '      "summary": "string",',
            '      "sentiment": "neutral",',
            '      "scheduled_events": []',
            '    }',
            '  }',
            '}',
        ]

        if collected_context:
            lines = [
                "The following news and signals have already been collected for you:",
                "",
                collected_context,
                "",
                "Use the above as your primary source. Use Google Search to fill gaps "
                "or verify recency of key events.",
                "",
            ] + lines

        return "\n".join(lines)

    def search_market_narrative_unified(self) -> Dict:
        """BTC+ETH 통합 1회 호출 — GCS에 캐시, 당일 재호출 시 캐시 반환.

        Returns: {"btc": narrative_dict, "eth": narrative_dict}
        """
        import os
        _job = os.environ.get("JOB_NAME", "").strip()
        if _job != "daily_precision":
            logger.warning(f"[Gemini Search Unified] BLOCKED: JOB_NAME={_job!r}")
            return {"btc": self._empty_result("BTCUSDT"), "eth": self._empty_result("ETHUSDT")}

        today = datetime.now(timezone.utc).date()

        # 인메모리 캐시
        mem = GeminiSearchCollector._unified_mem_cache
        if mem.get("date") == today:
            logger.info("[Gemini Search Unified] in-memory cache hit")
            return {"btc": mem["btc"], "eth": mem["eth"]}

        # GCS 캐시
        try:
            from processors.gcs_parquet import gcs_parquet_store
            cached = gcs_parquet_store.download_json(self._UNIFIED_GCS_PATH)
            if cached and cached.get("date") == today.isoformat():
                logger.info("[Gemini Search Unified] GCS cache hit")
                GeminiSearchCollector._unified_mem_cache = {
                    "date": today, "btc": cached["btc"], "eth": cached["eth"]
                }
                return {"btc": cached["btc"], "eth": cached["eth"]}
        except Exception:
            pass

        # 일일 슬롯 (1회만)
        if not self._acquire_daily_slot("UNIFIED"):
            return {"btc": self._empty_result("BTCUSDT"), "eth": self._empty_result("ETHUSDT")}

        btc_state = self._gather_market_state("BTCUSDT")
        eth_state = self._gather_market_state("ETHUSDT")
        collected_context = self._collect_news_context()
        prompt = self._build_unified_multi_prompt(btc_state, eth_state, collected_context)
        logger.info("[Narrative] Gemini+Grounding으로 BTC+ETH 통합 분석 시작")

        try:
            from agents.ai_router import ai_client
            _system = (
                "You are an external evidence engine for a crypto trading policy system. "
                "Use the provided news articles and your Google Search access. "
                "Return ONLY a valid JSON object. No markdown, no prose, no code fences."
            )
            content = ai_client.generate_with_grounding(
                system_prompt=_system,
                user_message=prompt,
                max_tokens=8000,
                temperature=0.1,
            )
            if not content:
                raise ValueError("Empty response")

            raw = self._extract_and_validate_json(content)
            if not raw:
                # 1회 재시도 — 전체 프롬프트 유지, JSON 전용 지시만 추가
                logger.warning(f"[Narrative] JSON 파싱 실패 (응답 앞 500자: {content[:500]!r}), JSON-only 재시도")
                retry = ai_client.generate_with_grounding(
                    system_prompt=(
                        "You are a JSON-only response system. "
                        "Return ONLY a valid JSON object matching the schema. "
                        "No markdown, no prose, no code fences, no explanation."
                    ),
                    user_message=prompt,
                    max_tokens=8000,
                    temperature=0.0,
                )
                raw = self._extract_and_validate_json(retry or "")
            if not raw:
                raise ValueError("JSON 파싱 2회 모두 실패")

            btc_result = self._normalize_narrative_result(raw.get("btc", {}), "BTCUSDT")
            eth_result = self._normalize_narrative_result(raw.get("eth", {}), "ETHUSDT")

            self.persist_narrative(btc_result, "BTCUSDT")
            self.persist_narrative(eth_result, "ETHUSDT")
            GeminiSearchCollector._mem_cache["BTCUSDT"] = {"date": today, "result": btc_result}
            GeminiSearchCollector._mem_cache["ETHUSDT"] = {"date": today, "result": eth_result}
            GeminiSearchCollector._unified_mem_cache = {"date": today, "btc": btc_result, "eth": eth_result}
            try:
                from processors.gcs_parquet import gcs_parquet_store
                gcs_parquet_store.upload_json(self._UNIFIED_GCS_PATH, {
                    "date": today.isoformat(), "btc": btc_result, "eth": eth_result
                })
            except Exception:
                pass
            logger.info(f"[Narrative] 완료 BTC={btc_result.get('sentiment')} ETH={eth_result.get('sentiment')}")
            return {"btc": btc_result, "eth": eth_result}

        except Exception as e:
            logger.error(f"[Narrative] Gemini grounding error: {e}")
            # Gemini 완전 실패 시 stale GCS cache 폴백 (날짜 무관)
            try:
                from processors.gcs_parquet import gcs_parquet_store
                cached = gcs_parquet_store.download_json(self._UNIFIED_GCS_PATH)
                if cached and cached.get("btc") and cached.get("eth"):
                    logger.warning(f"[Narrative] Gemini failed — using stale GCS cache (date={cached.get('date')})")
                    return {"btc": cached["btc"], "eth": cached["eth"]}
            except Exception:
                pass
            return {"btc": self._empty_result("BTCUSDT"), "eth": self._empty_result("ETHUSDT")}

    def _check_tavily_budget(self, daily_limit: int = 33) -> bool:
        """Check if we have remaining daily budget for Tavily.
        
        [V12.2] Updated limit to 33/day (1000/month).
        """
        usage_file = "data/tavily_usage.json"
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        usage = {"date": today, "count": 0}
        if os.path.exists(usage_file):
            try:
                with open(usage_file, "r") as f:
                    usage = json.load(f)
            except Exception:
                pass
        
        if usage.get("date") != today:
            usage = {"date": today, "count": 0}
            
        if usage.get("count", 0) >= daily_limit:
            return False
            
        # Increment and save
        usage["count"] = usage.get("count", 0) + 1
        os.makedirs(os.path.dirname(usage_file), exist_ok=True)
        with open(usage_file, "w") as f:
            json.dump(usage, f)
        return True

    def search_targeted(
        self,
        entity: str,
        entity_type: str = "institution",
        context: str = "",
        force_gemini_search: bool = False,
        search_depth: str = "basic"
    ) -> Dict:
        """Graph-triggered targeted search.
        
        Strategy: Use Tavily as default (routine triangulation) with configurable depth.
        Budget Guard: Max 30 Tavily searches per day to stay within free tier.
        """
        # [V12.2] Prioritize Tavily (budget: 33/day). Reserve Gemini Search for Reports only.
        if settings.TAVILY_API_KEY and not force_gemini_search:
            from .tavily_collector import tavily_collector
            if self._check_tavily_budget(daily_limit=33):
                logger.info(f"Using Tavily ({search_depth}) for targeted search: [{entity}]")
                try:
                    # [Alpha Choice] For targeted entity updates, use compat wrapper
                    result = tavily_collector.search_targeted_compat(entity, context, search_depth=search_depth)
                    if result.get("status") == "ok":
                        return result
                except Exception as e:
                    logger.error(f"Tavily search error for [{entity}]: {e}")

        # [V12.3] Secondary Fallback: Serper (Google SERP) - High precision for links
        if settings.SERPER_API_KEY and not force_gemini_search:
            from .serper_collector import serper_collector
            logger.info(f"Using Serper for targeted link verification: [{entity}]")
            try:
                serper_data = serper_collector.verify_news(f"{entity} {context}")
                if serper_data.get("status") == "ok" and serper_data.get("results"):
                    # Convert Serper format to targeted schema
                    organic = serper_data["results"]
                    return {
                        "entity": entity,
                        "status": "ok",
                        "summary": f"Google Search Results for {entity}: " + organic[0].get("snippet", ""),
                        "confidence_score": 70,
                        "btc_eth_impact": "Verification via Google SERP.",
                        "key_facts": [f"{r.get('title')}: {r.get('link')}" for r in organic[:3]],
                        "market_relevance": "Detected via Google Search sniping.",
                        "sources": [r.get("link") for r in organic[:5]],
                        "trust_score": 75
                    }
            except Exception as e:
                logger.error(f"Serper search error for [{entity}]: {e}")

        # Final Fallback to Gemini Search (Reserve for High Importance or if forced)
        if not self.api_key:
            logger.warning("PERPLEXITY_API_KEY not set, skipping targeted search")
            return self._empty_targeted_result(entity)

        now_utc = datetime.now(timezone.utc)
        date_today = now_utc.strftime("%Y-%m-%d")

        # BTC/ETH-centric entity type focus
        type_focus = {
            "institution": (
                f"Recent activity of {entity} related to Bitcoin or Ethereum: "
                "purchases, custody announcements, ETF filings, fund allocation, public statements. "
                f"Verify using {', '.join(settings.TRUSTED_NEWS_DOMAINS[:5])}."
            ),
            "regulator": (
                f"Recent actions by {entity} affecting Bitcoin, Ethereum, or crypto markets: "
                "rulings, enforcement actions, statements, pending legislation. "
                "Prioritize official government domains (.gov) and tier-1 financial media."
            ),
            "exchange": (
                f"Recent {entity} exchange data affecting BTC and ETH: "
                "reserve changes, significant inflow/outflow, unusual volume, regulatory issues. "
                "Cross-reference with on-chain data providers if possible."
            ),
            "macro_event": (
                f"Details of '{entity}' and its expected impact on Bitcoin and Ethereum "
                "as risk assets: market reaction, rate expectations, dollar impact. "
                "Prioritize Bloomberg, Reuters, and FT for macro context."
            ),
            "narrative": (
                f"Current strength and evidence of '{entity}' as a crypto market narrative: "
                "capital inflows, institutional backing, BTC/ETH price correlation. "
                f"Check if {', '.join(settings.TRUSTED_NEWS_DOMAINS[5:10])} are reporting this."
            ),
        }.get(
            entity_type,
            f"Recent developments related to {entity} and its impact on Bitcoin or Ethereum price. Prioritize high-authority financial news.",
        )

        context_line = f"\nAdditional context from market signals: {context}" if context else ""

        prompt = f"""Today is {date_today} (UTC).
{type_focus}.{context_line}

Search for factual, specific information from the last 7 days.
Focus ONLY on implications for Bitcoin (BTC) or Ethereum (ETH) price and positioning.

Return this exact JSON:
{{
  "summary": "2-3 sentences: what is {entity} and what has specifically happened recently",
  "confidence_score": 0-100,
  "btc_eth_impact": "1-2 sentences: how does this directly affect BTC or ETH price, positioning, or investment thesis",
  "key_facts": [
    "YYYY-MM-DD: specific fact with date",
    "YYYY-MM-DD: specific fact",
    "YYYY-MM-DD: specific fact"
  ],
  "market_relevance": "1 sentence: why {entity} matters for BTC/ETH right now"
}}

Be specific and factual. If {entity} has no clear BTC/ETH relevance, state that explicitly in btc_eth_impact."""

        try:
            content, citations = self._call_api(prompt, max_tokens=600, model=settings.PERPLEXITY_MODEL_TARGETED)
            result = self._parse_targeted_response(content, entity)
            if citations:
                result["sources"] = citations[:5]
            
            # Calculate trust_score: Hybrid of AI confidence and source authority
            sources = result.get("sources", [])
            trusted_count = 0
            for s in sources:
                if any(domain in s for domain in settings.TRUSTED_NEWS_DOMAINS):
                    trusted_count += 1
            
            ai_conf = result.get("confidence_score", 50)
            try:
                ai_conf = int(ai_conf)
            except:
                ai_conf = 50
            
            source_score = min(60, trusted_count * 15)
            result["trust_score"] = int((ai_conf * 0.4) + source_score)

            logger.info(
                f"Gemini Search targeted [{entity}/{entity_type}]: "
                f"Trust {result['trust_score']}% | {len(result.get('key_facts', []))} facts"
            )
            return result

        except requests.exceptions.Timeout:
            logger.error(f"Gemini Search targeted search timeout for [{entity}]")
            return self._empty_targeted_result(entity)
        except Exception as e:
            logger.error(f"Gemini Search targeted search error [{entity}]: {e}")
            return self._empty_targeted_result(entity)

    def _parse_targeted_response(self, content: str, entity: str) -> Dict:
        """Parse JSON from targeted search response."""
        try:
            text = content.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines)
            result = json.loads(text)
            result["entity"] = entity
            result["status"] = "ok"
            return result
        except json.JSONDecodeError:
            return {
                "entity": entity,
                "status": "raw",
                "summary": content[:400],
                "btc_eth_impact": content[:200],
                "key_facts": [],
                "market_relevance": "",
            }

    def _empty_targeted_result(self, entity: str) -> Dict:
        return {
            "entity": entity,
            "status": "unavailable",
            "summary": f"No information found for {entity}.",
            "confidence_score": 0,
            "btc_eth_impact": "",
            "key_facts": [],
            "market_relevance": "low",
            "sources": [],
        }

    def format_targeted_for_agents(self, result: Dict) -> str:
        """겟 색 결과를 에이전트용 텍스트로 포맷."""
        if result.get("status") == "unavailable":
            return f"[SEARCH] {result.get('entity', '?')}: No data"
        lines = [
            f"[SEARCH] {result.get('entity', '?')}",
            f"Summary: {result.get('summary', '')}",
        ]
        if result.get("btc_eth_impact"):
            lines.append(f"BTC/ETH Impact: {result.get('btc_eth_impact')}")
        for fact in result.get("key_facts", [])[:3]:
            lines.append(f"  · {fact}")
        if result.get("market_relevance"):
            lines.append(f"Relevance: {result.get('market_relevance')}")
        return "\n".join(lines)

    def _parse_response(self, content: str, symbol: str) -> Dict:
        """Parse JSON from Gemini Search response, handling markdown code blocks."""
        try:
            text = content.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines)
            result = json.loads(text)
            return self._normalize_narrative_result(result, symbol, status="ok")
        except json.JSONDecodeError:
            logger.warning("Failed to parse Gemini Search JSON, using raw text")
            raw = {
                "symbol": symbol,
                "summary": content[:500],
                "sentiment": "neutral",
                "reasoning": content[:300],
                "macro_context": "",
                "policy_supporting_factors": [],
                "policy_risks": [],
                "invalidation_risks": [],
                "scheduled_binary_events": [],
                "headline_vs_structural": "unknown - raw text fallback",
                "watchitems": [],
                "swing_view": {},
                "position_view": {},
            }
            return self._normalize_narrative_result(raw, symbol, status="raw")

    def _empty_result(self, symbol: str) -> Dict:
        empty = {
            "symbol": symbol,
            "summary": "Market narrative unavailable.",
            "sentiment": "neutral",
            "reasoning": "No narrative data available.",
            "macro_context": "",
            "policy_supporting_factors": [],
            "policy_risks": [],
            "invalidation_risks": [],
            "scheduled_binary_events": [],
            "headline_vs_structural": "",
            "watchitems": [],
            "swing_view": {},
            "position_view": {},
        }
        return self._normalize_narrative_result(empty, symbol, status="unavailable")

    def _acquire_daily_slot(self, symbol: str) -> bool:
        """파일 기반 하드 일일 카운터. Supabase·인메모리 무관하게 MAX_CALLS_PER_DAY 절대 제한.

        Returns True(호출 허용) / False(한도 초과 차단).
        파일 I/O 오류 시 fail-open (허용) — 단, 로그 남김.
        """
        from pathlib import Path
        today = datetime.now(timezone.utc).date().isoformat()
        path = Path("cache/gemini_search_daily.json")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data: dict = json.loads(path.read_text()) if path.exists() else {}
            # 오늘 날짜 키만 유지 (묵은 키 자동 정리)
            data = {k: v for k, v in data.items() if k.endswith(today)}
            key = f"{symbol}_{today}"
            count = int(data.get(key, 0))
            if count >= self.MAX_CALLS_PER_DAY:
                logger.warning(
                    f"[Gemini Search] HARD LIMIT BLOCKED: {symbol} already {count}/{self.MAX_CALLS_PER_DAY} calls today"
                )
                return False
            data[key] = count + 1
            path.write_text(json.dumps(data))
            logger.info(f"[Gemini Search] daily slot acquired: {symbol} {count + 1}/{self.MAX_CALLS_PER_DAY}")
            return True
        except Exception as e:
            logger.warning(f"[Gemini Search] daily slot file error ({e}) — fail-open")
            return True  # 파일 오류는 차단 안 함 (Supabase 캐시가 백업)

    def persist_narrative(self, narrative: Dict, symbol: str) -> None:
        """Persist narrative result into PostgreSQL for 4-hour report traceability."""
        try:
            now = datetime.now(timezone.utc).replace(second=0, microsecond=0).isoformat()
            payload = {
                "timestamp": now,
                "symbol": symbol,
                "source": "gemini_search",
                "status": narrative.get("status", "unknown"),
                "sentiment": narrative.get("sentiment", "neutral"),
                "summary": narrative.get("summary", ""),
                "reasoning": narrative.get("reasoning", ""),
                "key_events": narrative.get("key_events", []),
                "bullish_factors": narrative.get("bullish_factors", []),
                "bearish_factors": narrative.get("bearish_factors", []),
                "macro_context": narrative.get("macro_context", ""),
                "sources": narrative.get("sources", []),
                "raw_payload": narrative,  # full dict incl. cycle_position, upcoming_events
            }
            db.upsert_narrative_data(payload)
        except Exception as e:
            logger.error(f"Failed to persist narrative for {symbol}: {e}")

    def format_for_agents(self, narrative: Dict) -> str:
        """Format narrative data as compact text for agent consumption."""
        if narrative.get("status") == "unavailable":
            return "Market Narrative: Unavailable"

        lines = [
            f"[NARRATIVE] {narrative.get('symbol', '?')} | Policy Bias: {narrative.get('sentiment', '?').upper()}",
            f"Summary: {narrative.get('summary', 'N/A')}",
        ]

        reasoning = narrative.get("reasoning", "")
        if reasoning:
            lines.append(f"Policy Driver: {reasoning}")

        macro = narrative.get("macro_context", "")
        if macro:
            lines.append(f"Macro: {macro}")

        support = narrative.get("policy_supporting_factors", [])
        if support:
            lines.append(f"Policy Support: {' | '.join(support[:3])}")

        risks = narrative.get("policy_risks", [])
        if risks:
            lines.append(f"Policy Risks: {' | '.join(risks[:3])}")

        invalidation = narrative.get("invalidation_risks", [])
        if invalidation:
            lines.append(f"Invalidation Risks: {' | '.join(invalidation[:3])}")

        events = narrative.get("scheduled_binary_events", [])
        if events:
            lines.append(f"Scheduled Events: {' | '.join(events[:3])}")

        structure = narrative.get("headline_vs_structural", "")
        if structure:
            lines.append(f"Regime Type: {structure}")

        watchitems = narrative.get("watchitems", [])
        if watchitems:
            lines.append(f"Watchitems: {' | '.join(watchitems[:2])}")

        swing_view = narrative.get("swing_view", {})
        if isinstance(swing_view, dict) and swing_view:
            lines.append(f"[SWING] {swing_view.get('summary', '')}")
            sw_support = swing_view.get("trigger_support", [])
            if sw_support:
                lines.append(f"[SWING Support] {' | '.join(sw_support[:2])}")
            sw_risks = swing_view.get("trigger_risks", [])
            if sw_risks:
                lines.append(f"[SWING Risks] {' | '.join(sw_risks[:2])}")
            sw_invalid = swing_view.get("invalidation_risks", [])
            if sw_invalid:
                lines.append(f"[SWING Invalid] {' | '.join(sw_invalid[:2])}")
            sw_events = swing_view.get("scheduled_events", [])
            if sw_events:
                lines.append(f"[SWING Events] {' | '.join(sw_events[:2])}")
            key_levels = swing_view.get("key_levels", [])
            if key_levels:
                lines.append(f"[SWING Levels] {' | '.join(key_levels[:2])}")

        position_view = narrative.get("position_view", {})
        if isinstance(position_view, dict) and position_view:
            cyc = position_view.get("cycle_position", "")
            pos_summary = position_view.get("summary", "")
            if cyc:
                lines.append(f"[POSITION] cycle={cyc} | {pos_summary}")
            else:
                lines.append(f"[POSITION] {pos_summary}")
            pos_support = position_view.get("structural_support", [])
            if pos_support:
                lines.append(f"[POSITION Support] {' | '.join(pos_support[:2])}")
            pos_risks = position_view.get("structural_risks", [])
            if pos_risks:
                lines.append(f"[POSITION Risks] {' | '.join(pos_risks[:2])}")
            pos_invalid = position_view.get("invalidation_risks", [])
            if pos_invalid:
                lines.append(f"[POSITION Invalid] {' | '.join(pos_invalid[:2])}")
            inst_flows = position_view.get("institutional_flows", [])
            if inst_flows:
                lines.append(f"[POSITION Flows] {' | '.join(inst_flows[:2])}")
            pos_watch = position_view.get("watchitems", [])
            if pos_watch:
                lines.append(f"[POSITION Watch] {' | '.join(pos_watch[:2])}")

        return "\n".join(lines)


gemini_search_collector = GeminiSearchCollector()
