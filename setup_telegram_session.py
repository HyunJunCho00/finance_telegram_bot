"""
Telegram 세션 최초 생성 + Secret Manager 업로드 스크립트.

사용법 (VM에서 1회만 실행):
    python setup_telegram_session.py

동작:
    1. .env에서 TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_PHONE 읽기
    2. Telethon으로 인증 (인증코드 직접 입력)
    3. data/trading_session.session 생성
    4. chmod 600 적용
    5. Secret Manager에 base64 인코딩 후 업로드

이후 scheduler.py가 콜드 스타트 시 Secret Manager에서 자동 다운로드합니다.
"""

import os
import sys
import asyncio

# 프로젝트 루트를 Python path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import settings
from loguru import logger


async def main():
    from telethon import TelegramClient

    session_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(session_dir, exist_ok=True)
    session_path = os.path.join(session_dir, 'trading_session')

    api_id = int(settings.TELEGRAM_API_ID)
    api_hash = settings.TELEGRAM_API_HASH
    phone = settings.TELEGRAM_PHONE

    print(f"\n{'='*50}")
    print(f"  Telegram 세션 생성")
    print(f"{'='*50}")
    print(f"  API ID:  {'*' * (len(str(api_id)) - 3) + str(api_id)[-3:]}")
    print(f"  Phone:   {'*' * (len(phone) - 4) + phone[-4:]}")
    print(f"  Session: {session_path}.session")
    print(f"{'='*50}\n")

    client = TelegramClient(session_path, api_id, api_hash)

    await client.start(phone=phone)
    # ↑ 여기서 인증코드 입력을 요청합니다 (터미널에서 입력)

    me = await client.get_me()
    print(f"\n✅ 인증 성공!")
    print(f"   계정: {me.first_name} (인증 완료)")
    print(f"   세션 파일: {session_path}.session")

    await client.disconnect()

    # chmod 600 (Linux/Mac)
    session_file = session_path + '.session'
    if os.name != 'nt':
        import stat
        os.chmod(session_file, stat.S_IRUSR | stat.S_IWUSR)
        print(f"   퍼미션: chmod 600 ✅")

    # Secret Manager 업로드
    print(f"\n{'='*50}")
    print(f"  Secret Manager 업로드")
    print(f"{'='*50}\n")

    try:
        from collectors.telegram_collector import upload_session_to_secret_manager
        upload_session_to_secret_manager()
        print("✅ Secret Manager 업로드 완료!")
        print(f"   시크릿 이름: TELEGRAM_SESSION_FILE")
        print(f"   이후 콜드 스타트 시 자동 다운로드됩니다.")
    except Exception as e:
        print(f"⚠️  Secret Manager 업로드 실패: {e}")
        print(f"   .env에 USE_SECRET_MANAGER=true, PROJECT_ID 확인하세요.")
        print(f"   세션 파일은 로컬에 안전하게 저장되어 있습니다.")

    print(f"\n{'='*50}")
    print(f"  완료! 이제 scheduler.py를 시작하세요.")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    asyncio.run(main())
