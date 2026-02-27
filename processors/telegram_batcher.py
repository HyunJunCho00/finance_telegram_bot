"""Telegram message batcher for LightRAG graph ingestion.

Groups raw Telegram messages by BTC/ETH signal category and time window,
then uses specialized LLM prompts to synthesize market intelligence
for LightRAG graph node extraction.

4-tier BTC/ETH signal architecture:
- BTC_ETH_ONCHAIN:     Pure on-chain BTC/ETH intelligence (highest signal density)
- SMART_MONEY_FLOW:    Large capital movements into/out of BTC & ETH
- MARKET_INTELLIGENCE: Structural market events affecting BTC/ETH thesis (Asia focus)
- BREAKING_FILTER:     High-noise news — filtered for BTC/ETH relevance before ingest

Design principle: all prompts frame signals through "how does this affect BTC or ETH?"
Unrelated altcoin noise is explicitly discarded at the category level.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional

from config.settings import settings
from config.database import db
from agents.claude_client import claude_client
from processors.light_rag import light_rag

logger = logging.getLogger(__name__)


class TelegramBatcher:
    """
    Groups raw Telegram messages by BTC/ETH signal category and time window,
    then uses specialized LLM prompts to synthesize market intelligence
    for LightRAG graph ingestion.
    """

    # BTC/ETH signal-centric channel grouping
    # Each tier has a different signal value and extraction strategy
    CATEGORIES = {
        # Tier 1: Pure BTC/ETH on-chain intelligence — highest signal for positioning
        "BTC_ETH_ONCHAIN": ["CryptoQuant", "Glassnode", "Lookonchain"],

        # Tier 2: Large capital movements — whale/institutional flows into/out of BTC & ETH
        "SMART_MONEY_FLOW": ["Whale_Alert", "Arkham_Alerter", "DeFi_Million"],

        # Tier 3: Market structure — regulatory, protocol events, Asia institutional activity
        "MARKET_INTELLIGENCE": ["Wu_Blockchain", "Unfolded", "PeckShield"],

        # Tier 4: Noisy news stream — filter for BTC/ETH relevance before ingesting
        "BREAKING_FILTER": [
            "WalterBloomberg", "Tree_News", "Watcher_Guru",
            "Cointelegraph", "Binance_Announcements",
        ],
    }

    PROMPTS = {
        "BTC_ETH_ONCHAIN": """You are a Lead On-Chain Analyst specializing in Bitcoin and Ethereum.
Input: On-chain data, analytics, and flow reports from CryptoQuant, Glassnode, and Lookonchain.
Task: Extract BTC and ETH on-chain signals that indicate institutional positioning or capital flows.

Focus ONLY on:
- BTC and ETH exchange inflows/outflows (net: accumulation if outflow, distribution if inflow to exchanges)
- Long-term holder (LTH) / short-term holder (STH) behavior for BTC and ETH
- Miner behavior: BTC miner selling pressure or accumulation into the market
- ETH staking flows, validator activity, or liquid staking metrics
- Whale wallet clusters (>1000 BTC or >10000 ETH) accumulating or distributing

Discard: altcoin flows, DeFi TVL unrelated to ETH/BTC, stablecoin metrics unless tied to BTC/ETH buying pressure.

Output: A single dense paragraph. Start with "ON-CHAIN SIGNAL: [ACCUMULATION/DISTRIBUTION/NEUTRAL]" followed by specific evidence with amounts. Suitable for Knowledge Graph entity extraction.""",

        "SMART_MONEY_FLOW": """You are a Smart Money Flow Analyst. Your only job is tracking large capital into and out of Bitcoin and Ethereum.
Input: Whale transfer alerts, labeled wallet tracking, and large DeFi movements.
Task: Determine the NET capital flow direction for BTC and ETH.

Focus ONLY on:
- BTC and ETH transfers to/from major exchanges (Binance, Coinbase, Kraken) — inflows = sell pressure
- Labeled wallet activity: known funds (Jump Crypto, a16z, Grayscale, BlackRock) moving BTC or ETH
- Stablecoin (USDT/USDC) large minting or movements → potential BTC/ETH buying powder
- Cross-chain bridges moving large ETH amounts

Discard: altcoin whale moves unrelated to ETH/BTC, NFT transactions, transfers under $10M equivalent.

Output: A single dense paragraph. Start with "FLOW SIGNAL: [BUY_PRESSURE/SELL_PRESSURE/NEUTRAL]" then list the top 2-3 specific movements with amounts and destination. Suitable for Knowledge Graph entity extraction.""",

        "MARKET_INTELLIGENCE": """You are a Crypto Market Intelligence Analyst (Asia-Pacific focus).
Input: Investigative reports, regulatory news, protocol analysis, and institutional market commentary.
Task: Identify structural changes that affect the BTC and ETH investment thesis.

Focus ONLY on:
- Regulatory developments: SEC, CFTC, global policy directly affecting BTC/ETH or Bitcoin ETFs
- Institutional moves: corporate treasury BTC purchases, fund allocations, ETF inflow/outflow trends
- Protocol-level events: Ethereum upgrades, Bitcoin protocol changes, major fork proposals
- Security incidents: exchange hacks or protocol exploits where attacker holds or dumps BTC/ETH
- Asia market intelligence: Korean/Japanese/HK institutional flows into BTC/ETH

Discard: altcoin regulatory issues unless they set precedent for BTC/ETH, DeFi exploits unrelated to ETH ecosystem.

Output: A single dense paragraph. Start with "MARKET STRUCTURE: [BULLISH/BEARISH/NEUTRAL] for BTC/ETH" then explain the 1-2 most important structural developments. Suitable for Knowledge Graph entity extraction.""",

        "BREAKING_FILTER": """You are a News Filter for a Bitcoin and Ethereum trader.
Input: Mixed news headlines — macro, crypto, regulation, markets.
Task: DISCARD everything not directly relevant to Bitcoin or Ethereum price. Then synthesize what remains.

Keep ONLY news where:
- The headline explicitly mentions Bitcoin, BTC, Ethereum, ETH, crypto ETF, or spot crypto regulation
- The macro event directly affects risk appetite (Fed decision, CPI release, major equity crash/rally >2%)
- The event changes the probability of BTC/ETH institutional adoption

If NOTHING in the input is relevant to BTC or ETH price, respond with exactly:
NO_BTC_ETH_SIGNAL

If relevant content exists, output a single dense paragraph starting with "NEWS SIGNAL: [BULLISH/BEARISH/NEUTRAL]" then list only the BTC/ETH-relevant events in order of price importance. Suitable for Knowledge Graph entity extraction.""",
    }

    # BREAKING_FILTER: if LLM returns this sentinel, skip LightRAG ingest entirely
    NO_SIGNAL_SENTINEL = "NO_BTC_ETH_SIGNAL"

    def __init__(self):
        self.worker_model = settings.MODEL_RAG_EXTRACTION

    def get_category(self, channel_name: str) -> str:
        for cat, channels in self.CATEGORIES.items():
            if channel_name in channels:
                return cat
        return "MARKET_INTELLIGENCE"  # Default: treat unknowns as general market intel

    def process_and_ingest(self, lookback_hours: int = 4):
        """
        Main entry point:
        1. Fetch recent messages from DB.
        2. Group by BTC/ETH signal category.
        3. Synthesize with category-specific BTC/ETH-focused prompt.
        4. Ingest to LightRAG (BREAKING_FILTER skipped if no BTC/ETH signal found).
        """
        logger.info(f"TelegramBatcher: Starting batch for last {lookback_hours}h...")

        cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        try:
            res = (
                db.client.table("telegram_messages")
                .select("*")
                .gte("timestamp", cutoff.isoformat())
                .execute()
            )
            messages = res.data or []
        except Exception as e:
            logger.error(f"Failed to fetch telegram messages: {e}")
            return

        if not messages:
            logger.info("No new telegram messages to process.")
            return

        # Group by category — drop empty text
        grouped: Dict[str, List[str]] = {cat: [] for cat in self.PROMPTS.keys()}
        for msg in messages:
            cat = self.get_category(msg.get("channel", ""))
            text = msg.get("text", "").strip()
            if text:
                grouped[cat].append(text)

        total = sum(len(v) for v in grouped.values())
        logger.info(
            f"TelegramBatcher: {total} messages grouped → "
            + " | ".join(f"{cat}:{len(v)}" for cat, v in grouped.items())
        )

        # Process each category in chunks
        CHUNK_SIZE = 150
        for cat, content_list in grouped.items():
            if not content_list:
                continue

            chunks = [
                content_list[i : i + CHUNK_SIZE]
                for i in range(0, len(content_list), CHUNK_SIZE)
            ]
            logger.info(
                f"Synthesizing {len(content_list)} messages for [{cat}] "
                f"in {len(chunks)} chunk(s)..."
            )

            for chunk_idx, chunk in enumerate(chunks):
                full_text = "\n---\n".join(chunk)
                try:
                    summary = claude_client.generate_response(
                        system_prompt=self.PROMPTS[cat],
                        user_message=(
                            f"Messages from last {lookback_hours} hours "
                            f"(Part {chunk_idx + 1}/{len(chunks)}):\n\n{full_text}"
                        ),
                        temperature=0.2,
                        max_tokens=800,
                        role="rag_extraction",
                    )

                    if not summary or len(summary) < 30:
                        continue

                    # BREAKING_FILTER: skip if LLM found no BTC/ETH signal
                    if cat == "BREAKING_FILTER" and self.NO_SIGNAL_SENTINEL in summary:
                        logger.info(
                            f"[BREAKING_FILTER] Chunk {chunk_idx + 1}: "
                            "No BTC/ETH signal found — skipping LightRAG ingest"
                        )
                        continue

                    logger.info(
                        f"Ingesting [{cat}] chunk {chunk_idx + 1}/{len(chunks)} to LightRAG."
                    )
                    light_rag.ingest_message(
                        text=summary,
                        channel=f"BATCHED_{cat}",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )

                except Exception as e:
                    logger.error(
                        f"Failed to synthesize [{cat}] chunk {chunk_idx + 1}: {e}"
                    )


telegram_batcher = TelegramBatcher()
