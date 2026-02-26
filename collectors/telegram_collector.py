import os
import stat
import base64
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
from config.settings import settings
from config.database import db
from loguru import logger
from utils.text_sanitizer import clean_telegram_text

# 세션 파일 경로: data/ 디렉토리에 저장 (프로젝트 루트 노출 방지)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SESSION_DIR = os.path.join(_PROJECT_ROOT, 'data')
SESSION_PATH = os.path.join(_SESSION_DIR, 'trading_session')

# Secret Manager secret ID for session file
_SESSION_SECRET_ID = "TELEGRAM_SESSION_FILE"

# GCS path for session backup (inside GCS_ARCHIVE_BUCKET)
_GCS_SESSION_OBJECT = "telegram/session/trading_session.session"


def _ensure_session_security():
    """Ensure session file exists with secure permissions (owner-only: 600).
    If Secret Manager is configured, download session from there on cold start.
    """
    os.makedirs(_SESSION_DIR, exist_ok=True)

    # ── Session Download on cold start (Secret Manager → GCS fallback) ──
    session_file = SESSION_PATH + '.session'
    if not os.path.exists(session_file):
        if not _download_session_from_secret_manager(session_file):
            _download_session_from_gcs(session_file)

    # ── File permissions (Linux VM) ──
    for path in [session_file, SESSION_PATH + '.session-journal']:
        if os.path.exists(path) and os.name != 'nt':
            try:
                current_mode = os.stat(path).st_mode
                desired_mode = stat.S_IRUSR | stat.S_IWUSR  # 0o600
                if current_mode & 0o777 != desired_mode:
                    os.chmod(path, desired_mode)
                    logger.info(f"Session file permissions secured (chmod 600): {path}")
            except Exception as e:
                logger.warning(f"Could not set session file permissions: {e}")


def _download_session_from_secret_manager(local_path: str) -> bool:
    """Download session file from Secret Manager (base64-encoded). Returns True on success."""
    project_id = settings.PROJECT_ID
    if not project_id or os.getenv("USE_SECRET_MANAGER", "false").lower() != "true":
        logger.debug("Secret Manager not configured — skipping SM download")
        return False

    try:
        from google.cloud import secretmanager
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{_SESSION_SECRET_ID}/versions/latest"
        response = client.access_secret_version(request={"name": name})
        raw = base64.b64decode(response.payload.data)
        # Try zlib decompression (new format). Fall back to raw bytes (legacy format).
        try:
            import zlib
            session_bytes = zlib.decompress(raw)
        except Exception:
            session_bytes = raw

        with open(local_path, 'wb') as f:
            f.write(session_bytes)
        logger.info(f"✅ Session downloaded from Secret Manager ({_SESSION_SECRET_ID})")
        return True
    except Exception as e:
        # Secret may not exist yet (first run) — this is expected
        logger.info(f"Session not found in Secret Manager (will try GCS fallback): {e}")
        return False


def _download_session_from_gcs(local_path: str) -> bool:
    """Download session file from GCS bucket. Returns True on success."""
    bucket_name = settings.GCS_ARCHIVE_BUCKET
    if not bucket_name:
        return False
    try:
        import zlib
        from google.cloud import storage
        client = storage.Client(project=settings.PROJECT_ID or None)
        blob = client.bucket(bucket_name).blob(_GCS_SESSION_OBJECT)
        if not blob.exists():
            logger.info("Session not found in GCS (first run or bucket empty)")
            return False
        compressed = blob.download_as_bytes()
        session_bytes = zlib.decompress(compressed)
        with open(local_path, 'wb') as f:
            f.write(session_bytes)
        logger.info(f"✅ Session downloaded from GCS ({_GCS_SESSION_OBJECT}, {len(session_bytes):,} bytes)")
        return True
    except Exception as e:
        logger.warning(f"Session GCS download failed: {e}")
        return False


def _upload_session_to_gcs(session_bytes: bytes) -> bool:
    """Upload session file to GCS (no size limit, zlib-compressed raw bytes).
    Returns True on success."""
    bucket_name = settings.GCS_ARCHIVE_BUCKET
    if not bucket_name:
        return False
    try:
        import zlib
        from google.cloud import storage
        compressed = zlib.compress(session_bytes, level=9)
        client = storage.Client(project=settings.PROJECT_ID or None)
        blob = client.bucket(bucket_name).blob(_GCS_SESSION_OBJECT)
        blob.upload_from_string(compressed, content_type="application/octet-stream")
        logger.info(f"✅ Session uploaded to GCS ({_GCS_SESSION_OBJECT}, {len(session_bytes):,} bytes uncompressed)")
        return True
    except Exception as e:
        logger.warning(f"Session GCS upload failed: {e}")
        return False


def upload_session_to_secret_manager():
    """Upload session file to GCS (primary) and Secret Manager (best-effort).
    Call on shutdown or after successful Telegram auth."""
    project_id = settings.PROJECT_ID
    use_sm = os.getenv("USE_SECRET_MANAGER", "false").lower() == "true"

    session_file = SESSION_PATH + '.session'
    if not os.path.exists(session_file):
        return

    with open(session_file, 'rb') as f:
        session_bytes = f.read()

    # ── Primary: GCS (no size limit) ──
    gcs_ok = _upload_session_to_gcs(session_bytes)

    # ── Delete local session files after successful GCS upload ──
    # Session only needs to exist on disk while Telethon is running.
    # After upload, wipe local copies so they don't persist on the VM.
    if gcs_ok:
        for path in [session_file, session_file + '-journal', session_file + '-wal', session_file + '-shm']:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    logger.info(f"Local session file deleted (backed up to GCS): {path}")
                except Exception as e:
                    logger.warning(f"Could not delete local session file {path}: {e}")
    else:
        logger.warning("GCS upload failed — keeping local session file as fallback")

    # ── Secondary: Secret Manager (best-effort, may fail if >64KB) ──
    if not project_id or not use_sm:
        return
    try:
        import zlib
        from google.cloud import secretmanager
        client = secretmanager.SecretManagerServiceClient()
        parent = f"projects/{project_id}/secrets/{_SESSION_SECRET_ID}"

        compressed = zlib.compress(session_bytes, level=9)
        encoded = base64.b64encode(compressed)

        # Ensure secret exists (create if not)
        try:
            client.get_secret(request={"name": parent})
        except Exception:
            client.create_secret(request={
                "parent": f"projects/{project_id}",
                "secret_id": _SESSION_SECRET_ID,
                "secret": {"replication": {"automatic": {}}},
            })
            logger.info(f"Created secret: {_SESSION_SECRET_ID}")

        client.add_secret_version(
            request={"parent": parent, "payload": {"data": encoded}}
        )
        logger.info(f"✅ Session uploaded to Secret Manager ({_SESSION_SECRET_ID})")
    except Exception as e:
        logger.warning(f"Session Secret Manager upload failed (non-critical, GCS backup exists): {e}")


class TelegramCollector:
    def __init__(self):
        # [FIX Cold Start] Lazy init — Telethon client is NOT created at import time.
        # This prevents Telegram session errors from crashing the entire process.
        self._client = None
        self._init_failed = False

        self.channels = {
            # 기존 채널 및 크립토 주요 채널
            "WalterBloomberg": "WalterBloomberg",
            "Tree_News": "TreeNewsFeed",
            "Cointelegraph": "cointelegraph",
            "Wu_Blockchain": "wublockchainenglish",
            "Binance_Announcements": "binance_announcements",
            "Whale_Alert": "whale_alert_io",
            "PeckShield": "peckshield",
            
            # 신규 추가 (Arkham 봇 및 분석 채널)
            "Arkham_Alerter": "ArkhamAlertBot",
            "DeFi_Million": "DeFiMillionz",
            "CryptoQuant": "cryptoquant_official",
            "Glassnode": "glassnode",

            # 2026 베스트 스마트 머니 + 매크로 (유저 인증 팩트 체크 완료)
            "Unfolded": "unfolded",
            "Lookonchain": "lookonchainchannel",
            "Watcher_Guru": "WatcherGuru"
        }

    @property
    def client(self):
        """Lazy-init Telethon client. If init failed before, skip to avoid repeated errors."""
        if self._client is None and not self._init_failed:
            try:
                from telethon import TelegramClient
                _ensure_session_security()
                self._client = TelegramClient(
                    SESSION_PATH,
                    int(settings.TELEGRAM_API_ID),
                    settings.TELEGRAM_API_HASH
                )
                logger.info("Telethon client initialized successfully")
            except Exception as e:
                self._init_failed = True
                logger.error(f"Telethon client init failed (will skip Telegram collection): {e}")
        return self._client

    def _get_channel_max_message_id(self, channel_name: str) -> int:
        """Query DB for the latest message_id for a channel. Returns 0 if none found."""
        try:
            res = (
                db.client.table("telegram_messages")
                .select("message_id")
                .eq("channel", channel_name)
                .order("message_id", desc=True)
                .limit(1)
                .execute()
            )
            if res.data:
                return res.data[0]["message_id"]
        except Exception as e:
            logger.warning(f"Could not fetch max message_id for {channel_name}: {e}")
        return 0

    async def fetch_recent_messages(self, hours: Optional[int] = 4) -> List[Dict]:
        """Fetch messages newer than what's already in DB (resume) or within hours window.

        Resume logic (per channel):
          - If DB has messages for this channel → fetch only message_id > max_id (Telethon min_id)
          - If DB is empty for this channel → fall back to hours cutoff (or all history if hours=None)
        This avoids re-downloading the full history on every run.
        """
        if self.client is None:
            logger.warning("Telegram client unavailable — skipping message fetch")
            return []

        messages = []
        cutoff_time = None
        if hours and hours > 0:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        try:
            from telethon.errors import FloodWaitError
            import asyncio

            async with self.client:
                for channel_name, channel_username in self.channels.items():
                    try:
                        entity = await self.client.get_entity(channel_username)

                        # ── Resume: find where we left off ──
                        min_id = self._get_channel_max_message_id(channel_name)
                        if min_id > 0:
                            print(f"\n[{channel_name}] Resuming from message_id > {min_id:,}...", flush=True)
                        else:
                            print(f"\n[{channel_name}] Starting fetch (no prior data)...", flush=True)

                        count = 0

                        async for message in self.client.iter_messages(
                            entity,
                            limit=None,
                            offset_date=datetime.now(timezone.utc),
                            min_id=min_id,   # Telethon stops when it hits this ID (exclusive)
                            wait_time=1,     # Proactive rate limiting
                        ):
                            # Secondary time-based cutoff (used when DB is empty + hours set)
                            if min_id == 0 and cutoff_time and message.date.replace(tzinfo=timezone.utc) < cutoff_time:
                                break

                            count += 1
                            if count % 100 == 0:
                                print(f"\r  [{channel_name}] Downloaded: {count:,} new messages...", end="", flush=True)

                            if message.message:
                                cleaned_text = clean_telegram_text(message.message)
                                # Skip messages that are completely empty after sanitization
                                if not cleaned_text:
                                    continue
                                messages.append({
                                    'channel': channel_name,
                                    'message_id': message.id,
                                    'text': cleaned_text[:5000],
                                    'views': message.views or 0,
                                    'forwards': message.forwards or 0,
                                    'timestamp': message.date.isoformat(),
                                    'created_at': datetime.now(timezone.utc).isoformat()
                                })

                            # Save every 2000 messages to prevent OOM
                            if len(messages) >= 2000:
                                print(f"\r  [{channel_name}] Saving batch of 2000 to GCS + DB...", end="", flush=True)
                                self.save_to_gcs(messages)
                                self.save_to_database(messages)
                                messages = []

                    except FloodWaitError as e:
                        wait = e.seconds + 5
                        print(f"\n  [{channel_name}] FloodWait: sleeping {wait}s as required by Telegram...", flush=True)
                        await asyncio.sleep(wait)
                        continue
                    except Exception as e:
                        logger.error(f"Error fetching from {channel_name}: {e}")
                        continue

                    if count == 0 and min_id > 0:
                        print(f"\r  [{channel_name}] Already up to date (no new messages).          \n", end="", flush=True)
                    else:
                        print(f"\r  [{channel_name}] Done. New messages: {count:,}.          \n", end="", flush=True)

            upload_session_to_secret_manager()
        except Exception as e:
            logger.error(f"Telegram session error: {e}")
            self._init_failed = True
            self._client = None

        return messages

    def save_to_gcs(self, messages: List[Dict]) -> None:
        """Save a batch of messages to GCS as monthly Parquet files.

        Layout: gs://bucket/telegram/{channel}/{YYYY-MM}.parquet
        Deduplication is done on (channel, message_id).
        """
        if not messages:
            return
        try:
            from processors.gcs_parquet import gcs_parquet_store
            if not gcs_parquet_store.enabled:
                return

            import pandas as pd
            df = pd.DataFrame(messages)
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df['month'] = df['timestamp'].dt.strftime('%Y-%m')

            for (channel, month), group in df.groupby(['channel', 'month']):
                path = f"telegram/{channel}/{month}.parquet"
                gcs_parquet_store._merge_upload_parquet(
                    path,
                    group.drop(columns=['month']),
                    dedup_cols=['channel', 'message_id']
                )
            logger.info(f"GCS: saved {len(messages)} telegram messages")
        except Exception as e:
            logger.error(f"GCS telegram save error: {e}")

    def save_to_database(self, messages: List[Dict]) -> None:
        if not messages:
            return
        try:
            # Bulk upsert: 2000개를 1번 HTTP 요청으로 처리 (개별 루프 대비 100배 이상 빠름)
            db.client.table("telegram_messages").upsert(
                messages, on_conflict="channel,message_id"
            ).execute()
            logger.info(f"Saved {len(messages)} telegram messages to DB")
        except Exception as e:
            logger.error(f"Database save error: {e}")

    async def run_async(self, hours: int = 4) -> None:
        messages = await self.fetch_recent_messages(hours)
        self.save_to_database(messages)

    def run(self, hours: int = 4) -> None:
        if self._init_failed:
            logger.debug("Telegram collection skipped (previous init failure)")
            return
        try:
            import asyncio
            asyncio.run(self.run_async(hours))
        except Exception as e:
            logger.error(f"Telegram collector run error: {e}")


telegram_collector = TelegramCollector()
