import asyncio
import os
import sys

# Add parent directory to python path to allow importing config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from telethon import TelegramClient
from config.settings import settings

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SESSION_DIR = os.path.join(_PROJECT_ROOT, 'data')
SESSION_PATH = os.path.join(_SESSION_DIR, 'trading_session')
_SESSION_SECRET_ID = "TELEGRAM_SESSION_FILE"

def _ensure_session_local():
    """Download session securely without importing database modules"""
    import base64
    from google.cloud import secretmanager
    
    os.makedirs(_SESSION_DIR, exist_ok=True)
    
    try:
        client = secretmanager.SecretManagerServiceClient()
        project_id = os.environ.get("PROJECT_ID", "tj-trading-384306")
        name = f"projects/{project_id}/secrets/{_SESSION_SECRET_ID}/versions/latest"
        response = client.access_secret_version(request={"name": name})
        session_bytes = base64.b64decode(response.payload.data)
        
        with open(SESSION_PATH, 'wb') as f:
            f.write(session_bytes)
        
        # Set exact file permissions to 600
        import stat
        os.chmod(SESSION_PATH, stat.S_IRUSR | stat.S_IWUSR)
        print("✅ Session downloaded from Secret Manager successfully.")
    except Exception as e:
        print(f"⚠️ Could not download session from Secret Manager: {e}")
        print("⚠️ Trying to use local session file if it exists...")

async def main():
    print("🔹 시크릿 매니저에서 세션을 불러오고 텔레그램에 연결 중입니다...")
    _ensure_session_local()
    
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
