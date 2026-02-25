import asyncio
import os
import sys

# Add parent directory to python path to allow importing config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from telethon import TelegramClient
from config.settings import settings
from collectors.telegram_collector import _ensure_session_security, SESSION_PATH

async def main():
    print("🔹 시크릿 매니저에서 세션을 불러오고 텔레그램에 연결 중입니다...")
    # 시크릿 매니저에서 세션 파일 다운로드
    os.environ["USE_SECRET_MANAGER"] = "true"
    _ensure_session_security()
    
    client = TelegramClient(SESSION_PATH, int(settings.TELEGRAM_API_ID), settings.TELEGRAM_API_HASH)
    await client.connect()
    
    if not await client.is_user_authorized():
        print("❌ 세션이 만료되었거나 인증되지 않았습니다.")
        return
        
    print("\n✅ 사용자님이 입장해 계신 채널 목록 (코드에 넣을 아이디 추출):\n")
    print(f"{'채널 이름':<40} | {'영어 아이디 (코드에 넣을 값)':<30}")
    print("-" * 75)
    
    async for dialog in client.iter_dialogs(limit=200):
        # 그룹이나 채널만 필터링
        if dialog.is_channel or dialog.is_group:
            entity = dialog.entity
            username = getattr(entity, 'username', None)
            name = dialog.name[:38] + ".." if len(dialog.name) > 40 else dialog.name
            
            if username:
                print(f"{name:<40} | {username:<30} (O)")
            else:
                print(f"{name:<40} | (아이디 없음 - 비공개 채널) (X)")
                
    await client.disconnect()
    print("\n✅ 조회 완료!")

if __name__ == "__main__":
    asyncio.run(main())
