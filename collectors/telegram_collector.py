import os
import stat
import base64
from datetime import datetime, timedelta, timezone
from typing import List, Dict
from config.settings import settings
from config.database import db
from loguru import logger

# 세션 파일 경로: data/ 디렉토리에 저장 (프로젝트 루트 노출 방지)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SESSION_DIR = os.path.join(_PROJECT_ROOT, 'data')
SESSION_PATH = os.path.join(_SESSION_DIR, 'trading_session')

# Secret Manager secret ID for session file
_SESSION_SECRET_ID = "TELEGRAM_SESSION_FILE"


def _ensure_session_security():
    """Ensure session file exists with secure permissions (owner-only: 600).
    If Secret Manager is configured, download session from there on cold start.
    """
    os.makedirs(_SESSION_DIR, exist_ok=True)

    # ── Secret Manager Download (cold start) ──
    session_file = SESSION_PATH + '.session'
    if not os.path.exists(session_file):
        _download_session_from_secret_manager(session_file)

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


def _download_session_from_secret_manager(local_path: str):
    """Download session file from Secret Manager (base64-encoded)."""
    project_id = settings.PROJECT_ID
    if not project_id or os.getenv("USE_SECRET_MANAGER", "false").lower() != "true":
        logger.debug("Secret Manager not configured — using local-only session")
        return

    try:
        from google.cloud import secretmanager
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{_SESSION_SECRET_ID}/versions/latest"
        response = client.access_secret_version(request={"name": name})
        session_bytes = base64.b64decode(response.payload.data)

        with open(local_path, 'wb') as f:
            f.write(session_bytes)
        logger.info(f"✅ Session downloaded from Secret Manager ({_SESSION_SECRET_ID})")
    except Exception as e:
        # Secret may not exist yet (first run) — this is expected
        logger.info(f"Session not found in Secret Manager (will create after first auth): {e}")


def upload_session_to_secret_manager():
    """Upload session file to Secret Manager as a new version (base64-encoded).
    Call on shutdown or after successful Telegram auth."""
    project_id = settings.PROJECT_ID
    if not project_id or os.getenv("USE_SECRET_MANAGER", "false").lower() != "true":
        return

    session_file = SESSION_PATH + '.session'
    if not os.path.exists(session_file):
        return

    try:
        from google.cloud import secretmanager
        client = secretmanager.SecretManagerServiceClient()
        parent = f"projects/{project_id}/secrets/{_SESSION_SECRET_ID}"

        # Read and base64-encode
        with open(session_file, 'rb') as f:
            session_bytes = f.read()
        encoded = base64.b64encode(session_bytes)

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

        # Add new version
        client.add_secret_version(
            request={"parent": parent, "payload": {"data": encoded}}
        )
        logger.info(f"✅ Session uploaded to Secret Manager ({_SESSION_SECRET_ID}, {len(session_bytes)} bytes)")
    except Exception as e:
        logger.warning(f"Session upload to Secret Manager failed: {e}")


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

    async def fetch_recent_messages(self, hours: int = 4) -> List[Dict]:
        if self.client is None:
            logger.warning("Telegram client unavailable — skipping message fetch")
            return []

        messages = []
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        try:
            async with self.client:
                for channel_name, channel_username in self.channels.items():
                    try:
                        entity = await self.client.get_entity(channel_username)

                        async for message in self.client.iter_messages(
                            entity,
                            limit=100,
                            offset_date=datetime.now(timezone.utc)
                        ):
                            if message.date.replace(tzinfo=timezone.utc) < cutoff_time:
                                break

                            if message.message:
                                messages.append({
                                    'channel': channel_name,
                                    'message_id': message.id,
                                    'text': message.message[:5000],
                                    'views': message.views or 0,
                                    'forwards': message.forwards or 0,
                                    'timestamp': message.date.isoformat(),
                                    'created_at': datetime.now(timezone.utc).isoformat()
                                })

                    except Exception as e:
                        logger.error(f"Error fetching from {channel_name}: {e}")
                        continue

            # Upload session to Secret Manager after successful use (captures any auth updates)
            upload_session_to_secret_manager()
        except Exception as e:
            logger.error(f"Telegram session error: {e}")
            # Mark as failed so we don't retry session auth every cycle
            self._init_failed = True
            self._client = None

        return messages

    def save_to_database(self, messages: List[Dict]) -> None:
        if messages:
            try:
                for msg in messages:
                    db.insert_telegram_message(msg)
                logger.info(f"Saved {len(messages)} telegram messages")
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
