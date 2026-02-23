import os
from telethon import TelegramClient
from datetime import datetime, timedelta, timezone
from typing import List, Dict
from config.settings import settings
from config.database import db
from loguru import logger

# 세션 파일 경로: 프로젝트 루트 고정 (VM 어느 위치에서 실행해도 동일)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSION_PATH = os.path.join(_PROJECT_ROOT, 'trading_session')


class TelegramCollector:
    def __init__(self):
        self.client = TelegramClient(
            SESSION_PATH,
            int(settings.TELEGRAM_API_ID),
            settings.TELEGRAM_API_HASH
        )

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
            "Arkham_Alerter": "ArkhamAlertBot",    # ID: 5254703353
            "DeFi_Million": "DeFiMillionz",
            "CryptoQuant": "cryptoquant_official",
            "Unfolded": "unfolded",
            "Glassnode": "glassnode"
        }

    async def fetch_recent_messages(self, hours: int = 4) -> List[Dict]:
        messages = []
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

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
        import asyncio
        asyncio.run(self.run_async(hours))


telegram_collector = TelegramCollector()
