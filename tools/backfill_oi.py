"""Historical OI + Funding backfill: data.binance.vision → GCS Parquet.

데이터 소스:
  OI       : data.binance.vision daily metrics zip (5분 해상도, 2019-09~)
  Funding  : data.binance.vision monthly fundingRate zip (8시간 해상도, 2019-09~)

GCS 출력 경로 (live collector와 동일):
  funding/{symbol}/{YYYY-MM}.parquet

스키마 (live collector funding_data 테이블 호환):
  symbol, timestamp (UTC), funding_rate, open_interest_value

중복 처리:
  _merge_upload_parquet (keep="last") 가 알아서 처리.
  live collector가 이미 쌓은 데이터가 있어도 안전하게 merge됨.

Usage:
    python tools/backfill_oi.py                        # BTC+ETH 2020-01-01 ~ 오늘
    python tools/backfill_oi.py --dry-run              # GCS 저장 없이 파싱만 확인
    python tools/backfill_oi.py --start 2022-01-01    # 시작일 지정
    python tools/backfill_oi.py --symbol BTCUSDT       # 특정 심볼만

참고:
  BTCUSDT : 2019-09-25 부터 데이터 존재
  ETHUSDT : 2020-12-10 부터 데이터 존재 (그 이전 날짜는 404 → 자동 스킵)
  --start 2020-01-01 이어도 실제 파일 없는 날짜는 조용히 스킵됨.
"""

import argparse
import os
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from processors.gcs_parquet import gcs_parquet_store

# ── 상수 ──────────────────────────────────────────────────────────────────────
DEFAULT_START = "2020-01-01"
OI_BASE_URL   = "https://data.binance.vision/data/futures/um/daily/metrics"
FUND_BASE_URL = "https://data.binance.vision/data/futures/um/monthly/fundingRate"
REQUEST_DELAY = 0.05   # 초 — Binance Data Vision rate limit 방지


# ── 다운로드 ──────────────────────────────────────────────────────────────────
def _download(url: str, dest: Path) -> bool:
    """URL → dest 다운로드. 이미 캐시된 파일은 스킵. 404는 False 반환."""
    if dest.exists():
        return True
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            dest.write_bytes(r.content)
            return True
        if r.status_code == 404:
            return False          # 해당 날짜 파일 없음 — 정상적인 경우
        logger.warning(f"HTTP {r.status_code}: {url}")
    except requests.RequestException as e:
        logger.warning(f"Download error ({dest.name}): {e}")
    return False


# ── 타임스탬프 파싱 ────────────────────────────────────────────────────────────
def _parse_ts(series: pd.Series) -> pd.Series:
    """ms / us / s 정수 또는 문자열 타임스탬프 → UTC datetime."""
    numeric = pd.to_numeric(series, errors="coerce")
    result  = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")

    mask_num = numeric.notna()
    if mask_num.any():
        ts = numeric[mask_num].copy()
        ts[ts > 1e15] /= 1000      # microseconds → milliseconds
        ts[ts < 1e10] *= 1000      # seconds → milliseconds
        result[mask_num] = pd.to_datetime(ts, unit="ms", utc=True, errors="coerce")

    mask_str = ~mask_num & series.notna()
    if mask_str.any():
        result[mask_str] = pd.to_datetime(
            series[mask_str], utc=True, errors="coerce"
        )
    return result


# ── OI zip 파싱 ───────────────────────────────────────────────────────────────
def _read_oi_zip(path: Path) -> pd.DataFrame:
    """
    daily metrics zip → DataFrame[timestamp, open_interest_value]

    파일 구조:
      헤더 있는 경우: create_time, symbol, sum_open_interest, sum_open_interest_value, ...
      헤더 없는 경우: col0=time, col3=sum_open_interest_value (위치 기반 fallback)
    """
    try:
        with zipfile.ZipFile(path) as z:
            csv_name = z.namelist()[0]

            # 1차 시도: 헤더 이름으로 파싱
            with z.open(csv_name) as f:
                df = pd.read_csv(f)
            df.columns = [c.strip() for c in df.columns]

            if "create_time" in df.columns and "sum_open_interest_value" in df.columns:
                df = df[["create_time", "sum_open_interest_value"]].copy()
            else:
                # 2차 시도: 헤더 없음 → 위치 기반
                with z.open(csv_name) as f:
                    df = pd.read_csv(f, header=None)
                # 첫 행이 텍스트 헤더인 경우 제거
                try:
                    pd.to_numeric(str(df.iloc[0, 0]))
                except ValueError:
                    df = df.iloc[1:].reset_index(drop=True)
                df = df.iloc[:, [0, 3]].copy()  # col0=time, col3=oi_value

            df.columns = ["timestamp", "open_interest_value"]
            df["open_interest_value"] = pd.to_numeric(
                df["open_interest_value"], errors="coerce"
            )
            df["timestamp"] = _parse_ts(df["timestamp"])
            return df.dropna().reset_index(drop=True)

    except Exception as e:
        logger.debug(f"OI parse error ({path.name}): {e}")
        return pd.DataFrame()


# ── Funding zip 파싱 ──────────────────────────────────────────────────────────
def _read_funding_zip(path: Path) -> pd.DataFrame:
    """
    monthly fundingRate zip → DataFrame[timestamp, funding_rate]

    파일 구조:
      헤더 있는 경우: calc_time, funding_interval, last_funding_rate
      헤더 없는 경우: col0=time, col2=rate (위치 기반 fallback)
    """
    try:
        with zipfile.ZipFile(path) as z:
            csv_name = z.namelist()[0]

            # 1차 시도: 헤더 이름으로 파싱
            with z.open(csv_name) as f:
                df = pd.read_csv(f)
            df.columns = [c.strip().lower() for c in df.columns]

            if "calc_time" in df.columns and "last_funding_rate" in df.columns:
                df = df[["calc_time", "last_funding_rate"]].copy()
            else:
                # 2차 시도: 헤더 없음 → 위치 기반
                with z.open(csv_name) as f:
                    df = pd.read_csv(f, header=None)
                try:
                    pd.to_numeric(str(df.iloc[0, 0]))
                except ValueError:
                    df = df.iloc[1:].reset_index(drop=True)
                df = df.iloc[:, [0, 2]].copy()  # col0=time, col2=rate

            df.columns = ["timestamp", "funding_rate"]
            df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
            df["timestamp"] = _parse_ts(df["timestamp"])
            return df.dropna().reset_index(drop=True)

    except Exception as e:
        logger.debug(f"Funding parse error ({path.name}): {e}")
        return pd.DataFrame()


# ── 심볼별 처리 ───────────────────────────────────────────────────────────────
def process_symbol(
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
    cache_dir: Path,
    dry_run: bool,
) -> dict:
    logger.info(f"\n{'='*55}")
    logger.info(f"  {symbol}  |  {start_dt.date()} → {end_dt.date()}")
    logger.info(f"{'='*55}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. OI: 일별 metrics zip 다운로드 + 파싱 ──────────────────────────────
    logger.info("[1/3] Downloading OI metrics (daily zips)...")
    oi_dfs = []
    days = [
        start_dt + timedelta(days=i)
        for i in range((end_dt - start_dt).days + 1)
    ]

    for day in tqdm(days, desc=f"OI {symbol}", unit="day"):
        date_str = day.strftime("%Y-%m-%d")
        fname    = f"{symbol}-metrics-{date_str}.zip"
        dest     = cache_dir / fname
        url      = f"{OI_BASE_URL}/{symbol}/{fname}"

        if _download(url, dest):
            df = _read_oi_zip(dest)
            if not df.empty:
                oi_dfs.append(df)
        time.sleep(REQUEST_DELAY)

    if not oi_dfs:
        logger.warning(f"No OI data found for {symbol}. 선물 런칭 이전 날짜일 수 있음.")
        return {"symbol": symbol, "status": "no_oi_data"}

    full_oi = (
        pd.concat(oi_dfs, ignore_index=True)
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    logger.info(f"  OI rows collected: {len(full_oi):,}")

    # ── 2. Funding: 월별 zip 다운로드 + 파싱 ─────────────────────────────────
    logger.info("[2/3] Downloading Funding rates (monthly zips)...")
    fund_dfs = []
    curr = start_dt.replace(day=1)
    while curr <= end_dt:
        fname = f"{symbol}-fundingRate-{curr.year}-{curr.month:02d}.zip"
        dest  = cache_dir / fname
        url   = f"{FUND_BASE_URL}/{symbol}/{fname}"

        if _download(url, dest):
            df = _read_funding_zip(dest)
            if not df.empty:
                fund_dfs.append(df)
        curr += relativedelta(months=1)
        time.sleep(REQUEST_DELAY)

    logger.info(f"  Funding months collected: {len(fund_dfs)}")

    # ── 3. Merge: OI 타임스탬프 기준으로 funding_rate forward-fill ───────────
    logger.info("[3/3] Merging OI + Funding...")

    if fund_dfs:
        full_fund = (
            pd.concat(fund_dfs, ignore_index=True)
            .drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        # OI 타임스탬프 기준, 가장 가까운 이전 funding_rate를 붙임
        merged = pd.merge_asof(
            full_oi,
            full_fund,
            on="timestamp",
            direction="backward",
        )
    else:
        merged = full_oi.copy()
        merged["funding_rate"] = 0.0

    merged["funding_rate"]       = merged["funding_rate"].fillna(0.0)
    merged["open_interest_value"] = merged["open_interest_value"].fillna(0.0)
    merged["symbol"]             = symbol

    # live collector 스키마에 맞춤
    merged = merged[["symbol", "timestamp", "funding_rate", "open_interest_value"]].copy()

    # 날짜 범위 재확인 (타임스탬프 파싱 오류로 인한 이상값 제거)
    start_ts = pd.Timestamp(start_dt)
    end_ts   = pd.Timestamp(end_dt) + timedelta(days=1)
    merged   = merged[(merged["timestamp"] >= start_ts) & (merged["timestamp"] < end_ts)]

    logger.info(f"  Final rows to write: {len(merged):,}")

    # ── 4. GCS에 월 단위로 업로드 ────────────────────────────────────────────
    merged["_month"] = merged["timestamp"].dt.strftime("%Y-%m")
    total_rows   = 0
    months_done  = 0

    for month, group in merged.groupby("_month"):
        path = f"funding/{symbol}/{month}.parquet"
        g    = group.drop(columns=["_month"]).copy()
        rows = len(g)

        if dry_run:
            logger.info(f"  [DRY RUN] {path}  ←  {rows:,} rows")
        else:
            if not gcs_parquet_store.enabled:
                logger.error("GCS 비활성화 — .env에서 ENABLE_GCS_ARCHIVE=true 확인")
                return {"symbol": symbol, "status": "gcs_disabled"}

            size = gcs_parquet_store._merge_upload_parquet(
                path, g, dedup_cols=["timestamp"]
            )
            logger.info(f"  ✅ {path}  ←  {rows:,} rows  ({size:,} bytes)")
            months_done += 1

        total_rows += rows

    return {
        "symbol":     symbol,
        "oi_rows":    len(full_oi),
        "total_rows": total_rows,
        "months":     months_done if not dry_run else merged["_month"].nunique(),
        "dry_run":    dry_run,
        "status":     "ok",
    }


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Backfill historical OI + Funding from data.binance.vision → GCS Parquet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--start", default=DEFAULT_START,
        help=f"시작일 YYYY-MM-DD (기본값: {DEFAULT_START})"
    )
    parser.add_argument(
        "--end", default=None,
        help="종료일 YYYY-MM-DD (기본값: 오늘)"
    )
    parser.add_argument(
        "--symbol", default=None,
        help="특정 심볼만 처리 (예: BTCUSDT). 기본값: settings.TRADING_SYMBOLS 전체"
    )
    parser.add_argument(
        "--cache-dir", default="cache/oi_backfill",
        help="다운로드 zip 캐시 경로 (기본값: cache/oi_backfill)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="파싱까지만 수행, GCS 업로드 안 함"
    )
    args = parser.parse_args()

    start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt   = (
        datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.end
        else datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    )

    if start_dt >= end_dt:
        logger.error(f"start ({start_dt.date()}) >= end ({end_dt.date()})")
        sys.exit(1)

    symbols   = [args.symbol] if args.symbol else settings.trading_symbols
    cache_dir = Path(args.cache_dir)

    logger.info("=" * 55)
    logger.info("  OI + Funding Historical Backfill")
    logger.info(f"  Range    : {start_dt.date()} → {end_dt.date()}")
    logger.info(f"  Symbols  : {symbols}")
    logger.info(f"  Cache    : {cache_dir}")
    logger.info(f"  Dry run  : {args.dry_run}")
    logger.info(f"  GCS      : {'enabled' if gcs_parquet_store.enabled else '⚠ DISABLED'}")
    logger.info("=" * 55)

    results = {}
    for symbol in symbols:
        results[symbol] = process_symbol(
            symbol, start_dt, end_dt, cache_dir, args.dry_run
        )

    logger.info("\n── 최종 결과 ──────────────────────────────────────")
    for sym, r in results.items():
        logger.info(f"  {sym}: {r}")
    logger.info("Backfill complete.")


if __name__ == "__main__":
    main()
