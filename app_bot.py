"""Telegram UI Application."""
import sys
from loguru import logger

def main():
    logger.info("Starting Telegram Bot Application (app_bot.py)")
    try:
        from bot.telegram_bot import trading_bot
        trading_bot.run()
    except Exception as e:
        logger.error(f"Telegram bot crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
