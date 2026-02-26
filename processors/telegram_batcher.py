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
    Groups raw Telegram messages by category and time window, 
    then uses specialized Gemini prompts to synthesize macro narratives 
    for LightRAG ingestion.
    """

    CATEGORIES = {
        "FLASH_NEWS": ["WalterBloomberg", "Tree_News", "Unfolded", "Watcher_Guru", "Binance_Announcements"],
        "ON_CHAIN": ["Whale_Alert", "Lookonchain", "Arkham_Alerter", "PeckShield", "DeFi_Million", "CryptoQuant"],
        "MACRO_ANALYSIS": ["Cointelegraph", "Wu_Blockchain", "Glassnode"]
    }

    PROMPTS = {
        "FLASH_NEWS": """You are a Senior Market News Editor. 
Input: A collection of short headlines from the last hour.
Task: Synthesize these into a 3-5 point 'Market Pulse' report. 
- Focus only on hard facts, macroeconomic data, and major institutional moves.
- Combine duplicate or related news.
- Keep it extremely high-density.
- Format: A single cohesive paragraph suitable for Knowledge Graph extraction.""",

        "ON_CHAIN": """You are a Lead On-Chain Analyst.
Input: Whale movements, exchange inflows/outflows, and smart money alerts.
Task: Summarize the 'Net Capital Flow' for the last hour.
- Determine if the overall sentiment is accumulation or distribution.
- Identify the primary assets involved (e.g., BTC, ETH, Stablecoins).
- Mention any significant 'Whale clusters' or unusual protocol activities.
- Format: A single cohesive paragraph suitable for Knowledge Graph extraction.""",

        "MACRO_ANALYSIS": """You are a Chief Investment Officer (Macro).
Input: Long-form analysis reports, deep dives, and sentiment updates.
Task: Extract the 'Narrative Shift'.
- What is the prevailing market mood?
- Are there new systemic risks or long-term structural changes discussed?
- Filter out noise and generic opinions; keep the core logic.
- Format: A single cohesive paragraph suitable for Knowledge Graph extraction."""
    }

    def __init__(self):
        self.worker_model = "gemini-3.1-flash"

    def get_category(self, channel_name: str) -> str:
        for cat, channels in self.CATEGORIES.items():
            if channel_name in channels:
                return cat
        return "MACRO_ANALYSIS" # Default

    def process_and_ingest(self, lookback_hours: int = 4):
        """
        Main entry point: 
        1. Fetch unprocessed messages from DB.
        2. Group by category and 1H window.
        3. Synthesize summaries.
        4. Ingest to LightRAG.
        """
        logger.info(f"TelegramBatcher: Starting batch processing for last {lookback_hours} hours...")
        
        # 1. Fetch raw messages
        # Ideally, we'd have a 'processed' flag in DB, but for now we look at lookback
        cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        try:
            res = db.client.table("telegram_messages").select("*").gte("timestamp", cutoff.isoformat()).execute()
            messages = res.data or []
        except Exception as e:
            logger.error(f"Failed to fetch telegram messages: {e}")
            return

        if not messages:
            logger.info("No new telegram messages to process.")
            return

        # 2. Group by category
        grouped = {cat: [] for cat in self.PROMPTS.keys()}
        for msg in messages:
            cat = self.get_category(msg.get("channel", ""))
            grouped[cat].append(msg.get("text", ""))

        # 3. Process each category
        for cat, content_list in grouped.items():
            if not content_list:
                continue

            # Process in chunks of 150 to avoid exceeding Gemini context window safely
            # and to ensure all messages are ingested even if volume is very high.
            CHUNK_SIZE = 150
            chunks = [content_list[i:i + CHUNK_SIZE] for i in range(0, len(content_list), CHUNK_SIZE)]
            
            logger.info(f"Synthesizing {len(content_list)} messages for category {cat} in {len(chunks)} chunks...")
            
            for chunk_idx, chunk in enumerate(chunks):
                full_text = "\n---\n".join(chunk) 
                
                try:
                    summary = claude_client.generate_response(
                        system_prompt=self.PROMPTS[cat],
                        user_message=f"Messages from last {lookback_hours} hours (Part {chunk_idx+1}/{len(chunks)}):\n\n{full_text}",
                        temperature=0.2,
                        max_tokens=800, # Increased slightly to accommodate larger combined reports
                        role="rag_extraction" 
                    )
                    
                    if summary and len(summary) > 50:
                        # 4. Ingest to LightRAG
                        logger.info(f"Ingesting synthesized {cat} narrative (Chunk {chunk_idx+1}) to LightRAG.")
                        light_rag.ingest_message(
                            text=summary,
                            channel=f"BATCHED_{cat}",
                            timestamp=datetime.now(timezone.utc).isoformat()
                        )
                except Exception as e:
                    logger.error(f"Failed to synthesize category {cat} chunk {chunk_idx+1}: {e}")

telegram_batcher = TelegramBatcher()
