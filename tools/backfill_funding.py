"""One-time backfill tool: Binance historical funding rates → GCS Parquet.

Usage:
    python tools/backfill_funding.py                       # BTC+ETH from 2021-01-01
    python tools/backfill_funding.py --dry-run             # 확인만 (GCS 저장 안 함)
    python tools/backfill_funding.py --start 2022-01-01   # 시작일 지정
    python tools/backfill_funding.py --symbol BTC/USDT    # 특정 심볼만

Binance 데이터 가용 범위:
  - Funding rate : BTCUSDT ~2019-09-10 / ETHUSDT ~2021-05-12 부터 전량
  - Open Interest: API 최근 30일만 제공 (장기 이력 없음)

실행 후 orchstrator가 GCS Parquet merge를 통해 차트에 반영됨.
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collectors.funding_collector import funding_collector
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Backfill Binance funding history to GCS")
    parser.add_argument("--start", default="2021-01-01",
                        help="Start date (YYYY-MM-DD), default: 2021-01-01")
    parser.add_argument("--symbol", default=None,
                        help="Single symbol e.g. BTC/USDT (default: all configured symbols)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch and log only, do not write to GCS")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Binance Funding Rate Historical Backfill")
    logger.info(f"  Start date : {args.start}")
    logger.info(f"  Symbol     : {args.symbol or 'ALL'}")
    logger.info(f"  Dry run    : {args.dry_run}")
    logger.info("=" * 60)

    if args.symbol:
        result = funding_collector.backfill_funding_history(
            args.symbol, start_date=args.start, dry_run=args.dry_run
        )
        logger.info(f"Result: {result}")
    else:
        results = funding_collector.backfill_all_symbols(
            start_date=args.start, dry_run=args.dry_run
        )
        for sym, r in results.items():
            logger.info(f"{sym}: {r}")

    logger.info("Backfill complete.")


if __name__ == "__main__":
    main()
