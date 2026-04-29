"""
Telegram 세션 재인증 스크립트.
기존 세션이 AuthKeyDuplicatedError로 무효화됐을 때 실행.
VM에서: docker run --rm -it -v /opt/app/data:/app/data --env-file /opt/app/.env finance-bot:latest python scripts/reauth_telegram.py
"""
import os
import asyncio
from telethon import TelegramClient
from collectors.telegram_listener import SESSION_PATH, upload_session_to_cloud

SESSION_FILE = SESSION_PATH + '.session'


async def main():
    api_id = os.environ.get("TELEGRAM_API_ID", "").strip()
    api_hash = os.environ.get("TELEGRAM_API_HASH", "").strip()

    if not api_id or not api_hash:
        from config.settings import settings
        api_id = settings.TELEGRAM_API_ID
        api_hash = settings.TELEGRAM_API_HASH

    if not api_id or not api_hash:
        print("ERROR: TELEGRAM_API_ID / TELEGRAM_API_HASH 없음. 환경변수 확인 필요.")
        return

    if os.path.exists(SESSION_FILE):
        os.remove(SESSION_FILE)
        print(f"기존 세션 삭제: {SESSION_FILE}")

    print(f"새 세션 생성 중... (api_id={api_id})")
    client = TelegramClient(SESSION_PATH, int(api_id), api_hash)

    await client.start()  # 전화번호 + 인증코드 입력 프롬프트 나옴
    me = await client.get_me()
    print(f"인증 성공: {me.first_name} (@{me.username})")

    await client.disconnect()

    print("클라우드에 세션 업로드 중...")
    upload_session_to_cloud()
    print("완료. tgbot-shared-listener 재시작 가능.")


if __name__ == "__main__":
    asyncio.run(main())
