"""Perplexity Search API collector for market narrative context.

Provides the "WHY" behind market moves that pure technical data cannot.
Uses Perplexity sonar-pro model for deep web search with citations.

Output format:
- 3-line summary of key market events
- Bullish/Bearish narrative classification
- Current price movement reasoning

Cost: ~$5/month on Perplexity Pro API plan (200 requests/day included)
"""

import requests
from typing import Dict, Optional
from config.settings import settings
from config.database import db
from datetime import datetime, timezone
from loguru import logger
import json


class PerplexityCollector:
    BASE_URL = "https://api.perplexity.ai/chat/completions"

    def __init__(self):
        self.api_key = settings.PERPLEXITY_API_KEY
        self.model = "sonar-pro"  # Best for deep search with citations

    def search_market_narrative(self, symbol: str = "BTC") -> Dict:
        """Search for the current market narrative around a coin.

        Returns structured context:
        - summary: 3-line key events summary
        - sentiment: 'bullish' | 'bearish' | 'neutral'
        - bullish_factors: list of positive catalysts
        - bearish_factors: list of negative catalysts
        - reasoning: why the price is moving this way right now
        """
        if not self.api_key:
            logger.warning("PERPLEXITY_API_KEY not set, skipping narrative search")
            result = self._empty_result(symbol)
            self.persist_narrative(result, symbol)
            return result

        coin_name = "Bitcoin" if symbol.upper().startswith("BTC") else "Ethereum"

        from datetime import timedelta
        now_utc = datetime.now(timezone.utc)
        date_today = now_utc.strftime("%Y-%m-%d")
        date_3d_ago = (now_utc - timedelta(days=3)).strftime("%Y-%m-%d")

        prompt = f"""Today's date is {date_today} (UTC). Analyze the {coin_name} ({symbol}) market from {date_3d_ago} to {date_today}.

Prioritize the most recent events (last 72 hours). Every factor must include its specific date.

Provide your analysis in this EXACT JSON format (no markdown, pure JSON):
{{
  "summary": "3 sentences summarizing the most important events/news affecting {symbol} price from {date_3d_ago} to {date_today}",
  "sentiment": "bullish" or "bearish" or "neutral",
  "bullish_factors": ["YYYY-MM-DD: factor1", "YYYY-MM-DD: factor2", "YYYY-MM-DD: factor3"],
  "bearish_factors": ["YYYY-MM-DD: factor1", "YYYY-MM-DD: factor2", "YYYY-MM-DD: factor3"],
  "reasoning": "1-2 sentences explaining WHY the price is moving as of {date_today}",
  "key_events": ["YYYY-MM-DD: event1", "YYYY-MM-DD: event2", "YYYY-MM-DD: event3"],
  "macro_context": "macro/regulatory context relevant to {date_today}"
}}

Focus on (date-anchored, most recent first):
- Regulatory news (SEC, ETF approvals/rejections, legal cases)
- Whale movements and large exchange inflows/outflows
- Macro events (Fed decisions, CPI/PPI releases, geopolitical)
- Protocol updates, exchange listings, or technical milestones
- Market structure shifts (liquidations, funding rate flips, OI changes)

IMPORTANT: Do NOT report events before {date_3d_ago}. Be factual. Do not speculate or fabricate dates."""

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional crypto market analyst. Provide factual, citation-backed analysis. Always respond in valid JSON format only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.1,
                "return_citations": True,
            }

            response = requests.post(
                self.BASE_URL,
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()

            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Parse JSON from response
            result = self._parse_response(content, symbol)

            # Add citations if available
            citations = data.get("citations", [])
            if citations:
                result["sources"] = citations[:5]

            self.persist_narrative(result, symbol)
            logger.info(f"Perplexity narrative for {symbol}: sentiment={result.get('sentiment', '?')}")
            return result

        except requests.exceptions.HTTPError as e:
            logger.error(f"Perplexity API HTTP error: {e}")
            result = self._empty_result(symbol)
            self.persist_narrative(result, symbol)
            return result
        except requests.exceptions.Timeout:
            logger.error("Perplexity API timeout")
            result = self._empty_result(symbol)
            self.persist_narrative(result, symbol)
            return result
        except Exception as e:
            logger.error(f"Perplexity collector error: {e}")
            result = self._empty_result(symbol)
            self.persist_narrative(result, symbol)
            return result

    def _parse_response(self, content: str, symbol: str) -> Dict:
        """Parse JSON from Perplexity response, handling markdown code blocks."""
        try:
            # Strip markdown code blocks if present
            text = content.strip()
            if text.startswith("```"):
                # Remove ```json and closing ```
                lines = text.split('\n')
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = '\n'.join(lines)

            result = json.loads(text)
            result["symbol"] = symbol
            result["status"] = "ok"
            return result
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse Perplexity JSON, using raw text")
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
                "raw_payload": narrative,
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

        return '\n'.join(lines)


perplexity_collector = PerplexityCollector()
