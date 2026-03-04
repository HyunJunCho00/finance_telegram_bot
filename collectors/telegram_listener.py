import os
import stat
import asyncio
import json
import logging
import base64
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

from telethon import TelegramClient, events
from telethon.errors import FloodWaitError
from config.settings import settings
from config.database import db
from agents.ai_router import ai_client
from processors.light_rag import light_rag
from utils.text_sanitizer import clean_telegram_text
from config.local_state import state_manager

logger = logging.getLogger(__name__)

# --- Configuration & Constants ---
VISUAL_CHANNELS = {"CryptoQuant", "Glassnode", "Lookonchain"}
PRIORITY_CHANNELS = {"WalterBloomberg", "Tree_News", "Binance_Announcements", "Whale_Alert"}

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SESSION_DIR = os.path.join(_PROJECT_ROOT, 'data')
SESSION_PATH = os.path.join(_SESSION_DIR, 'trading_session')

_SESSION_SECRET_ID = "TELEGRAM_SESSION_FILE"
_GCS_SESSION_OBJECT = "telegram/session/trading_session.session"

# Alpha-focused Source Credibility (0.0 to 1.0)
SOURCE_CREDIBILITY = {
    "Whale_Alert": 1.0,
    "Arkham_Alerter": 1.0,
    "CryptoQuant": 0.95,
    "Glassnode": 0.95,
    "Lookonchain": 0.9,
    "Wu_Blockchain": 0.85,
    "Tree_News": 0.85,
    "WalterBloomberg": 0.8,
    "Unfolded": 0.8,
    "PeckShield": 0.8,
    "Watcher_Guru": 0.7,
    "Cointelegraph": 0.6,
}

# Keywords that indicate a chart caption carries actionable on-chain signal.
# Used in _extract_chart_analysis() to skip low-value images (logos, banners, etc.)
SIGNAL_KEYWORDS = {
    "btc", "bitcoin", "eth", "ethereum",
    "exchange", "inflow", "outflow", "flow",
    "whale", "holder", "miner", "staking",
    "nupl", "mvrv", "sopr", "nvt", "puell",
    "oi", "funding", "liquidat", "leverage",
    "price", "volume", "reserve", "supply",
}

# Routine NOISE Patterns to ignore for LLM Extraction (High-confidence spam/ads)
NOISE_KEYWORDS = {
    "referral", "sign up", "join now", "exclusive offer", "maintenance",
    "advertising", "sponsored", "trading competition", "giveaway", "discount code"
}

ALPHA_EXTRACTION_PROMPT = """You are a Senior Crypto Alpha Strategist.
Your goal is to extract structured intelligence from real-time global news and Telegram alerts.

TASK:
1. FILTER: Discard routine noise (periodic price updates, exchange maintenance, ads).
2. SIGNAL: Identify impactful news (Crypto-specific, Macro-financial, or Geopolitical events).
3. EXTRACT: For each signal, identify key Entities, Actions, and the Logic for why it matters for crypto markets (e.g., War -> Risk-off -> BTC/ETH impact).

EXTRACTION RULES:
- Format: [ENTITY] | [RELATION] | [TARGET] | [IMPACT_LOGIC]
- Focus on anything that shifts Market Sentiment or Liquidity.

SOURCE CONTEXT:
Channels: {sources}
Overall Signal: {signal_type}

Output should be a list of dense factual triplets, one per line. If no significant signal exists, return "NONE".
"""

VLM_SYSTEM_PROMPT = (
    "You are a crypto on-chain chart analyst embedded in an automated trading system. "
    "Your job: extract exactly three fields from a chart image, anchored by its caption. "
    "Accuracy is critical — your output feeds directly into trading decisions. "
    "Never invent data. Only report values that appear as explicit printed text annotations on the chart itself."
)

VLM_USER_TEMPLATE = """\
Channel: {channel}
Caption: "{caption}"

Study the chart using the caption as your anchor for what metric is being shown.
Return EXACTLY three lines in this format:
LABELS: <text values printed as annotations ON the chart — NONE if absent>
TREND: <BULLISH | BEARISH | NEUTRAL>
MISMATCH: <If the chart contradicts the caption, one sentence. NONE if aligned.>
"""

# --- Session Security & Persistence ---

def _ensure_session_security():
    os.makedirs(_SESSION_DIR, exist_ok=True)
    session_file = SESSION_PATH + '.session'
    if not os.path.exists(session_file):
        if not _download_session_from_secret_manager(session_file):
            _download_session_from_gcs(session_file)

    for path in [session_file, SESSION_PATH + '.session-journal']:
        if os.path.exists(path) and os.name != 'nt':
            try:
                os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
            except Exception as e:
                logger.warning(f"Could not set session file permissions: {e}")

def _download_session_from_secret_manager(local_path: str) -> bool:
    project_id = settings.PROJECT_ID
    if not project_id or os.getenv("USE_SECRET_MANAGER", "false").lower() != "true":
        return False
    try:
        from google.cloud import secretmanager
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{_SESSION_SECRET_ID}/versions/latest"
        response = client.access_secret_version(request={"name": name})
        raw = base64.b64decode(response.payload.data)
        try:
            import zlib
            session_bytes = zlib.decompress(raw)
        except:
            session_bytes = raw
        with open(local_path, 'wb') as f:
            f.write(session_bytes)
        logger.info("Session downloaded from Secret Manager")
        return True
    except:
        return False

def _download_session_from_gcs(local_path: str) -> bool:
    bucket_name = settings.GCS_ARCHIVE_BUCKET
    if not bucket_name: return False
    try:
        import zlib
        from google.cloud import storage
        client = storage.Client(project=settings.PROJECT_ID or None)
        blob = client.bucket(bucket_name).blob(_GCS_SESSION_OBJECT)
        if not blob.exists(): return False
        session_bytes = zlib.decompress(blob.download_as_bytes())
        with open(local_path, 'wb') as f:
            f.write(session_bytes)
        logger.info("Session downloaded from GCS")
        return True
    except:
        return False

def upload_session_to_cloud():
    session_file = SESSION_PATH + '.session'
    if not os.path.exists(session_file): return
    with open(session_file, 'rb') as f:
        session_bytes = f.read()

    # Upload to GCS
    bucket_name = settings.GCS_ARCHIVE_BUCKET
    if bucket_name:
        try:
            import zlib
            from google.cloud import storage
            compressed = zlib.compress(session_bytes, level=9)
            client = storage.Client(project=settings.PROJECT_ID or None)
            blob = client.bucket(bucket_name).blob(_GCS_SESSION_OBJECT)
            blob.upload_from_string(compressed, content_type="application/octet-stream")
            logger.info("Session backed up to GCS")
        except Exception as e:
            logger.warning(f"GCS backup failed: {e}")

    # Upload to Secret Manager
    project_id = settings.PROJECT_ID
    if project_id and os.getenv("USE_SECRET_MANAGER", "false").lower() == "true":
        try:
            import zlib
            from google.cloud import secretmanager
            client = secretmanager.SecretManagerServiceClient()
            parent = f"projects/{project_id}/secrets/{_SESSION_SECRET_ID}"
            compressed = zlib.compress(session_bytes, level=9)
            encoded = base64.b64encode(compressed)
            client.add_secret_version(request={"parent": parent, "payload": {"data": encoded}})
            logger.info("Session backed up to Secret Manager")
        except:
            pass

class TelegramListener:
    """
    Unified Telegram Agent (V13.2)
    - Persistent real-time listener (Telethon)
    - Historical backfill on startup
    - Micro-batching Alpha Extraction (Gemini 3.0 Flash)
    - VLM Chart Analysis
    - Session cloud persistence (GCS/SM)
    """

    def __init__(self):
        self.api_id = settings.TELEGRAM_API_ID
        self.api_hash = settings.TELEGRAM_API_HASH
        self.client: Optional[TelegramClient] = None
        self._message_buffer: List[Dict] = []
        self._triggered_buffer: List[Dict] = [] # Messages that passed triage
        self._buffer_lock = asyncio.Lock()
        self._running = False
        self.channels = {
            "WalterBloomberg": "WalterBloomberg",
            "Tree_News": "TreeNewsFeed",
            "Cointelegraph": "cointelegraph",
            "Wu_Blockchain": "wublockchainenglish",
            "Binance_Announcements": "binance_announcements",
            "Whale_Alert": "whale_alert_io",
            "PeckShield": "peckshield",
            "Arkham_Alerter": "ArkhamAlertBot",
            "DeFi_Million": "DeFiMillionz",
            "CryptoQuant": "cryptoquant_official",
            "Glassnode": "glassnode",
            "Unfolded": "unfolded",
            "Lookonchain": "lookonchainchannel",
            "Watcher_Guru": "WatcherGuru"
        }
        # Pre-compute reverse map for lightning-fast lookups in real-time handler
        self._username_to_key = {v.lower(): k for k, v in self.channels.items()}

    async def start(self):
        if not self.api_id or not self.api_hash:
            logger.error("Telegram API credentials missing.")
            return

        _ensure_session_security()
        self.client = TelegramClient(SESSION_PATH, int(self.api_id), self.api_hash)
        
        @self.client.on(events.NewMessage)
        async def handler(event):
            try:
                chat = await event.get_chat()
                # 1. Normalize identifiers (username or title)
                username = getattr(chat, 'username', '')
                username_low = username.lower() if username else ""
                title = getattr(chat, 'title', '')
                
                # 2. Match with our target channels
                sender_key = self._username_to_key.get(username_low)
                if not sender_key:
                    # Fallback for channels without usernames or title matches
                    for k, v in self.channels.items():
                        if v == title:
                            sender_key = k
                            break
                
                if not sender_key:
                    logger.debug(f"Telegram: Spurious message from non-target: {username or title}")
                    return # Not a target channel

                # 3. Real-time logging (Proof of delivery)
                logger.info(f"⚡ REAL-TIME: Received message from [{sender_key}] (User: {username}, Title: {title})")
                await self._process_single_message(event.message, sender_key)
            except Exception as e:
                logger.error(f"Handler error: {e}")

        await self.client.start()
        
        # [V13.8] Auth validation
        if not await self.client.is_user_authorized():
            logger.error("❌ Telegram Client is NOT authorized. Real-time alpha will NOT work.")
            logger.error("Please run the listener manually once to log in (data/trading_session.session).")
            # We don't stop the thread, but it won't receive messages.
        else:
            logger.info("✅ Telegram Client authorized and connected.")

        self._running = True
        logger.info("Unified Telegram Agent (V13.2) started.")

        # 1. Historical Backfill (Resume from where we left off)
        asyncio.create_task(self.run_backfill(hours=1))

        # 2. Start Micro-batch loop
        asyncio.create_task(self._batch_processor_loop())
        
        await self.client.run_until_disconnected()

    async def run_backfill(self, hours: int = 1):
        """Pull missed messages since last run."""
        logger.info(f"Starting historical backfill (last {hours}h)...")
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        for channel_name, username in self.channels.items():
            try:
                entity = await self.client.get_entity(username)
                min_id = self._get_max_id(channel_name)
                
                async for message in self.client.iter_messages(entity, limit=100, min_id=min_id):
                    if message.date.replace(tzinfo=timezone.utc) < cutoff_time and min_id == 0:
                        break
                    await self._process_single_message(message, channel_name)
            except Exception as e:
                logger.error(f"Backfill error for {channel_name}: {e}")
        logger.info("Backfill completed.")

    def _get_max_id(self, channel: str) -> int:
        res = db.client.table("telegram_messages").select("message_id").eq("channel", channel).order("message_id", desc=True).limit(1).execute()
        return res.data[0]["message_id"] if res.data else 0

    async def _process_single_message(self, message, sender_name):
        text = message.message or ""
        clean_text = clean_telegram_text(text)
        
        if sender_name in VISUAL_CHANNELS and getattr(message, 'photo', None):
            analysis = await self._extract_chart_analysis(message, sender_name, clean_text)
            if analysis:
                clean_text = f"{clean_text}\n[CHART] {analysis}" if clean_text else f"[CHART] {analysis}"

        if not clean_text: return
        
        msg_payload = {
            "source": sender_name, 
            "text": clean_text, 
            "message_id": message.id,
            "timestamp": message.date.isoformat() if hasattr(message, 'date') and message.date else datetime.now(timezone.utc).isoformat()
        }
        
        # 1. Immediate Persistence (V13.8 Refactor)
        # [FIX] Wrap sync DB call to prevent loop blocking
        await asyncio.to_thread(
            db.upsert_telegram_message,
            {
                "channel": sender_name, 
                "text": clean_text, 
                "message_id": message.id,
                "timestamp": msg_payload["timestamp"]
            }
        )

        # 2. IMMEDIATE TRIAGE (Zero-cost Cloudflare)
        # [FIX] Wrap sync Triage call (which has web requests) to prevent loop blocking
        is_trigger = await asyncio.to_thread(light_rag.triage_message, clean_text)
        
        if is_trigger:
            async with self._buffer_lock:
                self._triggered_buffer.append(msg_payload)
            
            # Priority Trigger: If it's a critical source, flush extraction immediately
            if sender_name in PRIORITY_CHANNELS:
                logger.info(f"🚀 PRIORITY EXTRACTION: Immediate trigger for [{sender_name}]")
                asyncio.create_task(self._process_triggered_now())
        else:
            # Still buffer for "Routine Alpha" if we want, but usually junk is just junk.
            # We'll skip adding junk messages to any extraction buffer to save AI costs.
            pass

    async def _process_triggered_now(self):
        """Flush triggered messages for extraction immediately."""
        async with self._buffer_lock:
            if not self._triggered_buffer: return
            batch, self._triggered_buffer = self._triggered_buffer, []
        await self._process_batch(batch)

    async def _batch_processor_loop(self):
        last_flush = datetime.now(timezone.utc)
        while self._running:
            await asyncio.sleep(10) # High-resolution state check
            
            is_panic = state_manager.is_panic_mode()
            threshold = 30 if is_panic else 300 # 30s in Panic, 5m in Routine
            
            elapsed = (datetime.now(timezone.utc) - last_flush).total_seconds()
            if elapsed < threshold:
                continue

            async with self._buffer_lock:
                if not self._triggered_buffer: 
                    last_flush = datetime.now(timezone.utc)
                    continue
                batch, self._triggered_buffer = self._triggered_buffer, []
            
            logger.info(f"Flushing Triggered Telegram batch ({len(batch)} messages, Mode: {'PANIC' if is_panic else 'ROUTINE'})")
            await self._process_batch(batch)
            last_flush = datetime.now(timezone.utc)

    async def _process_batch(self, batch: List[Dict]):
        # [V14.1 Update] Batch now only contains messages that already passed Cloudflare triage.
        # We perform one final junk filter before hitting Groq/Gemini extraction.
        filtered_batch = []
        seen_texts = set()
        
        for msg in batch:
            text_low = msg['text'].lower()
            if text_low in seen_texts: continue
            seen_texts.add(text_low)
            
            spam_hit = next((nkw for nkw in NOISE_KEYWORDS if nkw in text_low), None)
            if spam_hit:
                logger.debug(f"Junk Filter: Skipped triggered msg from {msg.get('source')} due to keyword [{spam_hit}]")
                continue
                
            filtered_batch.append(msg)

        if not filtered_batch:
            return

        sources = list(set([m['source'] for m in filtered_batch]))
        full_text = "\n---\n".join([f"[{m['source']}]: {m['text']}" for m in filtered_batch])
        try:
            extraction = await asyncio.to_thread(
                ai_client.generate_response,
                system_prompt=ALPHA_EXTRACTION_PROMPT.format(sources=", ".join(sources), signal_type="REAL-TIME_ALPHA"),
                user_message=f"Current market alerts:\n\n{full_text}",
                role="rag_extraction", temperature=0.1
            )
            if extraction and len(extraction) > 20:
                # [FIX] Wrap sync ingest call to prevent loop blocking
                await asyncio.to_thread(
                    light_rag.ingest_message,
                    text=extraction, 
                    channel="REALTIME_LISTENER", 
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
        except Exception as e:
            logger.error(f"Batch extraction error: {e}")

    async def _extract_chart_analysis(self, message, channel_name: str, caption: str) -> str:
        try:
            if len(caption) > 30 and not any(kw in caption.lower() for kw in SIGNAL_KEYWORDS): return ""
            image_bytes = await self.client.download_media(message.photo, file=bytes)
            if not image_bytes or len(image_bytes) < 2000: return ""
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            result = await asyncio.to_thread(
                ai_client.generate_response,
                system_prompt=VLM_SYSTEM_PROMPT,
                user_message=VLM_USER_TEMPLATE.format(channel=channel_name, caption=caption[:400]),
                max_tokens=120, temperature=0.0, chart_image_b64=image_b64, role="vlm_telegram_chart"
            )
            if result and "TREND:" in result:
                lines = [ln.strip() for ln in result.splitlines() if ln.strip().startswith(("LABELS:", "TREND:", "MISMATCH:"))]
                return " | ".join(lines) if len(lines) >= 2 else ""
        except: pass
        return ""

telegram_listener = TelegramListener()
