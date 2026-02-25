import asyncio
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_PROJECT_ROOT, '.env'))

api_id = os.environ.get("TELEGRAM_API_ID")
api_hash = os.environ.get("TELEGRAM_API_HASH")

if not api_id or not api_hash:
    print("❌ .env 파일에서 TELEGRAM_API_ID 또는 TELEGRAM_API_HASH를 찾을 수 없습니다.")
    sys.exit(1)

from telethon import TelegramClient

_SESSION_DIR = os.path.join(_PROJECT_ROOT, 'data')
os.makedirs(_SESSION_DIR, exist_ok=True)
SESSION_PATH = os.path.join(_SESSION_DIR, 'trading_session')

async def main():
    print("🔹 텔레그램에 연결 중입니다...")
    
    # 세션 파일이 없으면 터미널에서 전화번호와 인증번호를 물어봅니다.
    client = TelegramClient(SESSION_PATH, int(api_id), api_hash)
    
    # client.start() will automatically handle the interactive login prompt if needed
    await client.start()
    
    if not await client.is_user_authorized():
        print("❌ 로그인이 필요합니다. 프롬프트에 따라 인증을 진행해주세요.")
        return
        
    print("\n✅ 사용자님이 입장해 계신 채널 목록 (코드에 넣을 아이디 추출):\n")
    print(f"{'채널 이름':<40} | {'영어 아이디 (코드에 넣을 값)':<30}")
    print("-" * 75)
    
    async for dialog in client.iter_dialogs(limit=200):
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
