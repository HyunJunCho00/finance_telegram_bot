"""Perplexity Search API collector for market narrative context.

Provides the "WHY" behind market moves that pure technical data cannot.
Uses Perplexity sonar-pro model for deep web search with citations.

Two search modes:
1. search_market_narrative(symbol): Per-symbol trading context (runs on analysis schedule)
   - Mode-aware: SWING (1-7 day catalysts) vs POSITION (2-8 week cycle)
   - Injects live market state: Fear&Greed, funding rate, 10Y yield, DXY
2. search_targeted(entity, entity_type, context): Graph-triggered BTC/ETH-centric search

Cost: ~$5/month on Perplexity Pro API plan (200 requests/day included)
"""

import requests
from typing import Dict, List, Optional, Tuple
from config.settings import settings, TradingMode
from config.database import db
from datetime import datetime, timezone, timedelta
from loguru import logger
import json


class PerplexityCollector:
    BASE_URL = "https://api.perplexity.ai/chat/completions"

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
        self.model = "sonar-pro"

    def _get_coin_name(self, symbol: str) -> Tuple[str, str]:
        """BTCUSDT → ('Bitcoin', 'BTC'),  SOLUSDT → ('Solana', 'SOL')"""
        base = symbol.upper().replace("USDT", "").replace("BUSD", "").replace("USDC", "")
        coin_name = self.COIN_NAME_MAP.get(base, base)
        return coin_name, base

    def _call_api(self, prompt: str, max_tokens: int = 1000) -> Tuple[str, List[str]]:
        """공통 API 호출. (content, citations) 반환."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional crypto market analyst. Respond in valid JSON only. No markdown.",
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "return_citations": True,
        }
        response = requests.post(self.BASE_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        citations = data.get("citations", [])
        return content, citations

    def _gather_market_state(self, symbol: str) -> Dict:
        """Pull live market indicators from DB for mode-aware prompt injection.

        Returns a dict with available fields — any field may be absent if DB
        has no data yet (cold start, API failures). Callers must handle None.
        """
        state: Dict = {}
        try:
            # Fear & Greed Index (daily, from alternative.me)
            fg = db.get_latest_fear_greed()
            if fg:
                state["fear_greed"] = f"{fg.get('value', 'N/A')} ({fg.get('label', 'N/A')})"

            # Macro indicators (FRED + yfinance snapshot, 4h cadence)
            macro = db.get_latest_macro_data()
            if macro:
                for state_key, db_key in [
                    ("dgs10", "dgs10"),
                    ("dxy", "dxy"),
                    ("nasdaq", "nasdaq"),
                    ("ust_spread", "ust_2s10s_spread"),
                ]:
                    if macro.get(db_key) is not None:
                        state[state_key] = macro[db_key]

            # Current price (1 row — lightweight)
            latest_df = db.get_latest_market_data(symbol, limit=1)
            if not latest_df.empty and "close" in latest_df.columns:
                state["current_price"] = float(latest_df["close"].iloc[-1])

                # 30d return: fetch earliest close within last 31 days (2 rows total)
                cutoff_30d = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
                try:
                    res = (
                        db.client.table("market_data")
                        .select("close")
                        .eq("symbol", symbol)
                        .eq("exchange", "binance")
                        .gte("timestamp", cutoff_30d)
                        .order("timestamp")
                        .limit(1)
                        .execute()
                    )
                    if res.data:
                        price_30d = float(res.data[0]["close"])
                        if price_30d > 0:
                            state["return_30d"] = round(
                                (state["current_price"] / price_30d - 1) * 100, 1
                            )
                except Exception:
                    pass

            # Latest funding rate (1 row)
            try:
                funding_df = db.get_funding_history(symbol, limit=1)
                if not funding_df.empty and "funding_rate" in funding_df.columns:
                    state["funding_rate"] = float(funding_df["funding_rate"].iloc[-1])
            except Exception:
                pass

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
            f"Search for events from {date_7d_ago} to {date_today} that directly affect {coin_name} price.",
            "Prioritize the last 48 hours. Every item must include its exact date.",
            "",
            "Return this exact JSON:",
            "{",
            f'  "summary": "2-3 sentences: what specifically happened to {coin_name} in the last 7 days that matters for a swing trade",',
            '  "sentiment": "bullish" or "bearish" or "neutral",',
            '  "bullish_factors": ["YYYY-MM-DD: catalyst that supports buying — expected price impact"],',
            '  "bearish_factors": ["YYYY-MM-DD: risk or catalyst that supports selling or staying flat"],',
            '  "upcoming_events": ["YYYY-MM-DD: scheduled macro/crypto event in next 7 days that could move price"],',
            f'  "reasoning": "1 sentence: the single highest-probability near-term driver for the next 1-7 days",',
            f'  "macro_context": "how rates/DXY/risk appetite is interacting with {coin_name} right now"',
            "}",
            "",
            "Search priorities (in order):",
            "1. Upcoming macro events (next 7 days): Fed speakers, CPI, NFP, futures/options expiry dates",
            "2. BTC ETF net flows (last 7 days): trend from BlackRock IBIT, Fidelity FBTC, ARK 21Shares",
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
            '  "cycle_position": "one of: early_accumulation | late_accumulation | distribution | capitulation | recovery — with 1-sentence evidence",',
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

    def search_market_narrative(self, symbol: str = "BTC") -> Dict:
        """Mode-aware market narrative search. Runs on each analysis cycle.

        SWING mode: 1-7 day catalysts, upcoming macro events, funding extremes
        POSITION mode: 2-8 week cycle position, structural regime shifts

        Injects live market state (Fear&Greed, price, funding, macro) into the
        prompt so Perplexity can contextualise results against current conditions.

        Returns structured context compatible with format_for_agents() and persist_narrative().
        """
        if not self.api_key:
            logger.warning("PERPLEXITY_API_KEY not set, skipping narrative search")
            result = self._empty_result(symbol)
            self.persist_narrative(result, symbol)
            return result

        coin_name, base = self._get_coin_name(symbol)
        mode = settings.trading_mode

        # Inject live market state into prompt
        state = self._gather_market_state(symbol)

        if mode == TradingMode.POSITION:
            prompt = self._build_position_prompt(coin_name, base, state)
            logger.info(f"Perplexity [{coin_name}]: POSITION mode search (2-8 week horizon)")
        else:
            prompt = self._build_swing_prompt(coin_name, base, state)
            logger.info(f"Perplexity [{coin_name}]: SWING mode search (1-7 day horizon)")

        try:
            content, citations = self._call_api(prompt)
            result = self._parse_response(content, symbol)
            if citations:
                result["sources"] = citations[:5]
            self.persist_narrative(result, symbol)
            logger.info(
                f"Perplexity narrative [{coin_name}]: "
                f"sentiment={result.get('sentiment', '?')} mode={mode.value}"
            )
            return result

        except requests.exceptions.HTTPError as e:
            logger.error(f"Perplexity API HTTP error: {e}")
        except requests.exceptions.Timeout:
            logger.error("Perplexity API timeout")
        except Exception as e:
            logger.error(f"Perplexity collector error: {e}")

        result = self._empty_result(symbol)
        self.persist_narrative(result, symbol)
        return result

    def search_targeted(
        self,
        entity: str,
        entity_type: str = "institution",
        context: str = "",
    ) -> Dict:
        """Graph-triggered targeted search for BTC/ETH market signal context.

        Called when the graph analyzer detects significant entity spikes or
        cross-channel corroboration that warrants deeper investigation.

        All searches are framed through the lens of BTC/ETH price impact.

        Args:
            entity:      Entity detected in the graph
                         (e.g., "BlackRock", "Federal Reserve", "BTC ETF flows")
            entity_type: BTC/ETH-centric signal type — one of:
                         "institution"  → fund/corporate activity affecting BTC/ETH
                         "regulator"    → policy action affecting crypto markets
                         "exchange"     → exchange-specific BTC/ETH flow data
                         "macro_event"  → macro event impact on crypto risk appetite
                         "narrative"    → market narrative affecting BTC/ETH thesis
            context:     Additional context from graph signals (optional)

        Returns:
            {"entity", "summary", "btc_eth_impact", "key_facts", "market_relevance", "sources", "status"}
        """
        if not self.api_key:
            logger.warning("PERPLEXITY_API_KEY not set, skipping targeted search")
            return self._empty_targeted_result(entity)

        now_utc = datetime.now(timezone.utc)
        date_today = now_utc.strftime("%Y-%m-%d")

        # BTC/ETH-centric entity type focus
        type_focus = {
            "institution": (
                f"Recent activity of {entity} related to Bitcoin or Ethereum: "
                "purchases, custody announcements, ETF filings, fund allocation, public statements"
            ),
            "regulator": (
                f"Recent actions by {entity} affecting Bitcoin, Ethereum, or crypto markets: "
                "rulings, enforcement actions, statements, pending legislation"
            ),
            "exchange": (
                f"Recent {entity} exchange data affecting BTC and ETH: "
                "reserve changes, significant inflow/outflow, unusual volume, regulatory issues"
            ),
            "macro_event": (
                f"Details of '{entity}' and its expected impact on Bitcoin and Ethereum "
                "as risk assets: market reaction, rate expectations, dollar impact"
            ),
            "narrative": (
                f"Current strength and evidence of '{entity}' as a crypto market narrative: "
                "capital inflows, institutional backing, BTC/ETH price correlation"
            ),
        }.get(
            entity_type,
            f"Recent developments related to {entity} and its impact on Bitcoin or Ethereum price",
        )

        context_line = f"\nAdditional context from market signals: {context}" if context else ""

        prompt = f"""Today is {date_today} (UTC).
{type_focus}.{context_line}

Search for factual, specific information from the last 7 days.
Focus ONLY on implications for Bitcoin (BTC) or Ethereum (ETH) price and positioning.

Return this exact JSON:
{{
  "summary": "2-3 sentences: what is {entity} and what has specifically happened recently",
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
            content, citations = self._call_api(prompt, max_tokens=600)
            result = self._parse_targeted_response(content, entity)
            if citations:
                result["sources"] = citations[:5]
            logger.info(
                f"Perplexity targeted [{entity}/{entity_type}]: "
                f"{len(result.get('key_facts', []))} facts found"
            )
            return result

        except requests.exceptions.Timeout:
            logger.error(f"Perplexity targeted search timeout for [{entity}]")
            return self._empty_targeted_result(entity)
        except Exception as e:
            logger.error(f"Perplexity targeted search error [{entity}]: {e}")
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
            "btc_eth_impact": "",
            "key_facts": [],
            "market_relevance": "",
            "sources": [],
        }

    def format_targeted_for_agents(self, result: Dict) -> str:
        """타겟 검색 결과를 에이전트용 텍스트로 포맷."""
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
        """Parse JSON from Perplexity response, handling markdown code blocks."""
        try:
            text = content.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines)
            result = json.loads(text)
            result["symbol"] = symbol
            result["status"] = "ok"
            return result
        except json.JSONDecodeError:
            logger.warning("Failed to parse Perplexity JSON, using raw text")
            return {
                "symbol": symbol,
                "status": "raw",
                "summary": content[:500],
                "sentiment": "neutral",
                "bullish_factors": [],
                "bearish_factors": [],
                "reasoning": content[:300],
                "key_events": [],
                "macro_context": "",
            }

    def _empty_result(self, symbol: str) -> Dict:
        return {
            "symbol": symbol,
            "status": "unavailable",
            "summary": "Market narrative unavailable.",
            "sentiment": "neutral",
            "bullish_factors": [],
            "bearish_factors": [],
            "reasoning": "No narrative data available.",
            "key_events": [],
            "macro_context": "",
        }

    def persist_narrative(self, narrative: Dict, symbol: str) -> None:
        """Persist narrative result into PostgreSQL for 4-hour report traceability."""
        try:
            now = datetime.now(timezone.utc).replace(second=0, microsecond=0).isoformat()
            payload = {
                "timestamp": now,
                "symbol": symbol,
                "source": "perplexity",
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
            f"[NARRATIVE] {narrative.get('symbol', '?')} | Sentiment: {narrative.get('sentiment', '?').upper()}",
            f"Summary: {narrative.get('summary', 'N/A')}",
        ]

        # POSITION mode: show cycle position
        if narrative.get("cycle_position"):
            lines.append(f"Cycle: {narrative.get('cycle_position')}")

        # SWING mode: show upcoming events
        upcoming = narrative.get("upcoming_events", [])
        if upcoming:
            lines.append(f"Upcoming: {' | '.join(upcoming[:3])}")

        bull = narrative.get("bullish_factors", [])
        if bull:
            lines.append(f"Bullish: {' | '.join(bull[:3])}")

        bear = narrative.get("bearish_factors", [])
        if bear:
            lines.append(f"Bearish: {' | '.join(bear[:3])}")

        reasoning = narrative.get("reasoning", "")
        if reasoning:
            lines.append(f"Why: {reasoning}")

        macro = narrative.get("macro_context", "")
        if macro:
            lines.append(f"Macro: {macro}")

        return "\n".join(lines)


perplexity_collector = PerplexityCollector()
