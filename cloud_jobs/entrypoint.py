"""Cloud Run Job entrypoint.

# /app/cloud_jobs/ 에서 실행될 때 /app 을 path에 추가
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

실행 방법:
  JOB_NAME=etf_flow python cloud_jobs/entrypoint.py

지원 잡:
  ── 데이터 수집 ──────────────────────────────────
  etf_flow          → ETF 자금 흐름 수집 (일 1회)
  stablecoin        → USDT/USDC 공급량 수집 (일 1회)
  coinglass         → Binance LSR + OI 수집 (4시간)
  dune              → Dune Analytics 온체인 수집 (15분)

  ── 분석/RAG ─────────────────────────────────────
  telegram_batch    → Telegram → LightRAG ingest (1시간)
  crypto_news       → 외부 뉴스 → LightRAG ingest (1시간)
  market_status     → 시장 지표 요약 + Telegram 발송 (1시간)
  snapshot_narrative → SWING 스냅샷 갱신 (30분)
  hourly_monitor    → 시간별 플레이북 모니터 (1시간)
  daily_precision   → 일일 정밀 분석 (일 1회, JOB_SYMBOL 필요)

  ── 평가 ─────────────────────────────────────────
  evaluation        → 에피소드 메모리 평가 (1시간)
  evaluation_rollup → 일별 평가 롤업 (일 1회)
  evaluation_24h    → 24시간 피드백 사이클 (일 1회)
"""
import os
import sys

from loguru import logger

JOB_NAME = os.environ.get("JOB_NAME", "").strip().replace("-", "_")  # market-status → market_status
JOB_SYMBOL = os.environ.get("JOB_SYMBOL", "").strip().upper()  # daily_precision용

if not JOB_NAME:
    logger.error("JOB_NAME 환경변수가 설정되지 않았습니다")
    sys.exit(1)

logger.info(f"[cloud-run] job={JOB_NAME} symbol={JOB_SYMBOL or 'N/A'} 시작")


def _bootstrap_parquet():
    """Brain 잡 실행 전 GCS → 로컬 parquet 캐시 동기화."""
    try:
        from processors.gcs_parquet import gcs_parquet_store
        from config.settings import settings

        if not gcs_parquet_store.enabled:
            logger.warning("[bootstrap] GCS 비활성화 — parquet 없이 실행")
            return

        for symbol in settings.trading_symbols:
            for tf, months_back in [
                ("4h", settings.SWING_HISTORY_MONTHS),
                ("1d", settings.SWING_HISTORY_MONTHS),
                ("1w", settings.SWING_HISTORY_MONTHS),
            ]:
                paths = gcs_parquet_store._build_ohlcv_paths(tf, symbol, months_back)
                missing = [p for p in paths
                           if not gcs_parquet_store._local_cache_path(p).exists()]
                if missing:
                    logger.info(f"[bootstrap] {symbol} {tf}: {len(missing)}개 파일 다운로드")
                for p in missing:
                    try:
                        gcs_parquet_store._download_parquet(p)
                    except Exception as e:
                        logger.warning(f"[bootstrap] 다운로드 실패 {p}: {e}")

        logger.info("[bootstrap] parquet 캐시 준비 완료")
    except Exception as e:
        logger.warning(f"[bootstrap] 실패 (non-fatal, 빈 캐시로 실행): {e}")


def _precision_prepare():
    """Daily precision 사전 준비: coinmetrics + macro 수집."""
    try:
        from config.settings import settings
        from collectors.coinmetrics_collector import coinmetrics_collector
        from collectors.macro_collector import macro_collector
        if settings.COINMETRICS_ENABLED:
            coinmetrics_collector.run()
        macro_collector.run()
    except Exception as e:
        logger.warning(f"[precision_prepare] 실패 (non-fatal): {e}")


try:
    # ── 데이터 수집 잡 ─────────────────────────────────────────────────────────

    if JOB_NAME == "etf_flow":
        from collectors.etf_flow_collector import etf_flow_collector
        etf_flow_collector.run()

    elif JOB_NAME == "stablecoin":
        from collectors.stablecoin_collector import stablecoin_collector
        stablecoin_collector.run()

    elif JOB_NAME == "coinglass":
        from collectors.coinglass_collector import coinglass_collector
        coinglass_collector.run()

    elif JOB_NAME == "dune":
        from collectors.dune_collector import dune_collector
        if dune_collector is None:
            logger.warning("Dune collector 비활성화 (API 키 없음) — 잡 스킵")
        else:
            dune_collector.run_due_queries(limit=200, offset=0)

    # ── 분석/RAG 잡 ────────────────────────────────────────────────────────────

    elif JOB_NAME == "telegram_batch":
        from processors.telegram_batcher import telegram_batcher
        from processors.light_rag import light_rag
        telegram_batcher.process_and_ingest(lookback_hours=1)
        light_rag.run_triangulation_worker(limit=3)

    elif JOB_NAME == "crypto_news":
        from collectors.crypto_news_collector import collector as news_collector
        news_collector.fetch_and_ingest()

    elif JOB_NAME == "market_status":
        from cloud_jobs.market_status import run_market_status
        run_market_status()

    elif JOB_NAME == "snapshot_narrative":
        _bootstrap_parquet()
        from config.settings import settings, TradingMode
        from executors.orchestrator import orchestrator
        for symbol in settings.trading_symbols:
            try:
                orchestrator.refresh_snapshot_with_mode(
                    symbol, TradingMode.SWING,
                    allow_perplexity=True,
                    include_meta=False,
                    include_vlm=False,
                )
            except Exception as e:
                logger.error(f"snapshot_narrative error for {symbol}: {e}")

    elif JOB_NAME == "hourly_monitor":
        _bootstrap_parquet()
        from executors.orchestrator import orchestrator
        orchestrator.run_hourly_monitor()

    elif JOB_NAME == "daily_precision":
        if not JOB_SYMBOL:
            logger.error("daily_precision은 JOB_SYMBOL 환경변수가 필요합니다 (예: BTCUSDT)")
            sys.exit(1)
        _bootstrap_parquet()
        _precision_prepare()
        from config.settings import TradingMode
        from executors.orchestrator import orchestrator
        orchestrator.run_daily_playbook_for_symbol(JOB_SYMBOL, TradingMode.SWING)

    # ── 평가 잡 ────────────────────────────────────────────────────────────────

    elif JOB_NAME == "evaluation":
        from executors.evaluator_daemon import EvaluatorDaemon
        EvaluatorDaemon().evaluate_recent_trades()

    elif JOB_NAME == "evaluation_rollup":
        from evaluators.evaluation_rollup import evaluation_rollup_service
        result = evaluation_rollup_service.run_daily_rollup(lookback_days=3)
        logger.info(f"Rollup 결과: {result}")

    elif JOB_NAME == "evaluation_24h":
        from evaluators.feedback_generator import feedback_generator
        feedback_generator.run_feedback_cycle()

    else:
        logger.error(f"알 수 없는 JOB_NAME: {JOB_NAME!r}")
        sys.exit(1)

    logger.info(f"[cloud-run] job={JOB_NAME} 완료")

except Exception as e:
    logger.exception(f"[cloud-run] job={JOB_NAME} 실패: {e}")
    sys.exit(1)
