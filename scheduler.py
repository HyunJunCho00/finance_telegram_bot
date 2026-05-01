# -*- coding: utf-8 -*-
# [PERF] uvloop replaces CPython asyncio event loop with a C-based implementation.
# Improves async I/O throughput 2-4x. Linux/macOS only — degrades silently on Windows.
try:
    import uvloop
    uvloop.install()
except ImportError:
    pass  # Windows dev environment — no-op

from apscheduler.triggers.cron import CronTrigger
from collectors.price_collector import collector
from collectors.funding_collector import funding_collector
from collectors.volatility_monitor import volatility_monitor
from collectors.dune_collector import dune_collector
from collectors.microstructure_collector import microstructure_collector
from collectors.macro_collector import macro_collector
from collectors.deribit_collector import deribit_collector
from collectors.fear_greed_collector import fear_greed_collector
from collectors.crypto_news_collector import collector as news_collector
from collectors.coinmetrics_collector import coinmetrics_collector
from collectors.etf_flow_collector import etf_flow_collector
from collectors.stablecoin_collector import stablecoin_collector
from collectors.coinglass_collector import coinglass_collector
from executors.orchestrator import orchestrator
from evaluators.feedback_generator import feedback_generator
from evaluators.evaluation_rollup import evaluation_rollup_service
from processors.light_rag import light_rag
from processors.gcs_archive import gcs_archive_exporter
from processors.math_engine import math_engine
from processors.gcs_parquet import gcs_parquet_store
from agents.market_monitor_agent import market_monitor_agent
from tools.evaluate_market_status_pressure import run_evaluation as run_pressure_signal_evaluation
from config import scheduler_config
from config.settings import settings, TradingMode
from config.database import db
from executors.order_manager import execution_desk
from executors.evaluator_daemon import EvaluatorDaemon
from executors.paper_exchange import paper_engine
from executors.cascade_warning_engine import cascade_warning_engine
from loguru import logger
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import base64
import re
import time as _time
from datetime import datetime, timezone, timedelta
from processors.market_snapshot_builder import build_mode_technical_snapshot, detect_technical_events
from processors.playbook_service import get_recent_market_regime, get_latest_playbook_snapshot
from utils.text_sanitizer import looks_english_dominant
from utils.math_utils import project_trendline_price, pct_change, distance_pct

# ── Prometheus 메트릭 ──────────────────────────────────────────────────────────
try:
    from prometheus_client import Counter, Histogram, start_http_server as _prom_start
    SCHED_JOB_DURATION = Histogram(
        "scheduler_job_duration_seconds",
        "스케줄러 Job 실행 시간",
        ["job"],
        buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1800],
    )
    SCHED_JOB_RESULTS = Counter(
        "scheduler_job_results_total",
        "스케줄러 Job 실행 결과",
        ["job", "result"],  # result: success / error
    )
    _SCHED_PROM = True
except Exception:
    _SCHED_PROM = False

_daily_precision_prepare_lock = threading.Lock()
_daily_precision_prepare_bucket = ""

# ── 프로세스 분리 모드 플래그 ────────────────────────────────────────────────
# True  → execution_main.py를 별도 프로세스로 실행 중
#          scheduler.py는 execution job(1min_execution, 8h_funding)을 등록하지 않음
# False → 기존 단일 프로세스 방식 (기본값, 변경 불필요)
EXECUTION_PROCESS_SEPARATE: bool = True


def _daily_precision_schedule_hours_utc() -> list[int]:
    raw = str(getattr(settings, "DAILY_PRECISION_HOURS_UTC", "") or "").strip()
    parsed: list[int] = []
    if raw:
        for chunk in raw.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                hour = int(chunk)
            except Exception:
                continue
            if 0 <= hour <= 23:
                parsed.append(hour)
    if parsed:
        return sorted(set(parsed))
    return [int(getattr(settings, "DAILY_PRECISION_HOUR_UTC", 0))]


def _daily_precision_schedule_hours_expr_utc() -> str:
    return ",".join(str(hour) for hour in _daily_precision_schedule_hours_utc())


def _daily_precision_label_utc() -> str:
    minute = int(getattr(settings, "DAILY_PRECISION_MINUTE_UTC", 30))
    return ", ".join(f"{hour:02d}:{minute:02d}" for hour in _daily_precision_schedule_hours_utc())


def _daily_precision_schedule_slots_utc(offset_minutes: int = 0) -> list[tuple[int, int]]:
    minute = int(getattr(settings, "DAILY_PRECISION_MINUTE_UTC", 30))
    slots: list[tuple[int, int]] = []
    for hour in _daily_precision_schedule_hours_utc():
        total_minutes = ((hour * 60) + minute + int(offset_minutes)) % (24 * 60)
        slots.append((total_minutes // 60, total_minutes % 60))
    return sorted(set(slots))


def _daily_precision_symbol_label_utc(symbol_index: int) -> str:
    gap = max(0, int(getattr(settings, "DAILY_PRECISION_SYMBOL_GAP_MINUTES", 10)))
    slots = _daily_precision_schedule_slots_utc(offset_minutes=symbol_index * gap)
    return ", ".join(f"{hour:02d}:{minute:02d}" for hour, minute in slots)


def _daily_precision_all_labels_utc() -> str:
    labels = []
    for idx, symbol in enumerate(settings.trading_symbols):
        labels.append(f"{symbol}={_daily_precision_symbol_label_utc(idx)}")
    return " | ".join(labels) if labels else _daily_precision_label_utc()


def _daily_precision_protection_window_minutes() -> int:
    return max(0, int(getattr(settings, "DAILY_PRECISION_PROTECTION_MINUTES", 20)))


def _is_daily_precision_protection_window(now_utc: datetime | None = None) -> bool:
    now = now_utc or datetime.now(timezone.utc)
    protection_minutes = _daily_precision_protection_window_minutes()
    base_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    slot_offsets = [
        idx * max(0, int(getattr(settings, "DAILY_PRECISION_SYMBOL_GAP_MINUTES", 10)))
        for idx, _ in enumerate(settings.trading_symbols)
    ] or [0]
    for day_offset in (-1, 0, 1):
        day_anchor = base_day + timedelta(days=day_offset)
        for offset in slot_offsets:
            for hour, minute in _daily_precision_schedule_slots_utc(offset_minutes=offset):
                scheduled = day_anchor.replace(hour=hour, minute=minute)
                window_start = scheduled - timedelta(minutes=protection_minutes // 2)
                window_end = scheduled + timedelta(minutes=protection_minutes)
                if window_start <= now <= window_end:
                    return True
    return False


def _should_defer_heavy_job(job_name: str) -> bool:
    if orchestrator.is_daily_precision_running():
        logger.info(f"{job_name} skipped (daily precision currently running)")
        return True
    if _is_daily_precision_protection_window():
        logger.info(
            f"{job_name} skipped (inside daily precision protection window around {_daily_precision_all_labels_utc()} UTC)"
        )
        return True
    return False


def _refresh_snapshots_for_modes(
    modes: list[TradingMode],
    *,
    allow_perplexity: bool,
    label: str,
) -> None:
    from config.local_state import state_manager

    if not state_manager.is_analysis_enabled():
        logger.info(f"{label} skipped (analysis disabled)")
        return

    for symbol in settings.trading_symbols:
        for mode in modes:
            try:
                logger.info(
                    f"{label}: refreshing snapshot for {symbol}/{mode.value} "
                    f"(perplexity={'on' if allow_perplexity else 'off'})"
                )
                orchestrator.refresh_snapshot_with_mode(
                    symbol,
                    mode,
                    allow_perplexity=allow_perplexity,
                    include_meta=False,
                    include_vlm=False,
                )
            except Exception as e:
                logger.error(f"{label} error for {symbol}/{mode.value}: {e}")


def job_snapshot_refresh_fast():
    """Warm hot-path snapshots for Swing lane without slow web narrative calls."""
    if _should_defer_heavy_job("Snapshot fast refresh"):
        return
    _refresh_snapshots_for_modes(
        [TradingMode.SWING],
        allow_perplexity=False,
        label="Snapshot fast refresh",
    )


def job_snapshot_refresh_narrative():
    """Refresh narrative/RAG-rich snapshots for Swing lane."""
    if _should_defer_heavy_job("Snapshot narrative refresh"):
        return
    _refresh_snapshots_for_modes(
        [TradingMode.SWING],
        allow_perplexity=True,
        label="Snapshot narrative refresh",
    )






def job_1min_tick():
    tasks = {
        "price":          collector.run,
        "funding":        funding_collector.run,
        "microstructure": microstructure_collector.run,
        "volatility":     volatility_monitor.run,
    }
    pool = ThreadPoolExecutor(max_workers=4)
    futures = {pool.submit(fn): name for name, fn in tasks.items()}
    try:
        for future in as_completed(futures, timeout=55):
            name = futures[future]
            try:
                future.result(timeout=1)
            except Exception as e:
                logger.error(f"{name} collection error: {e}")
    except TimeoutError:
        for future, name in futures.items():
            if not future.done():
                logger.error(f"{name} collection timed out after 55s")
    finally:
        pool.shutdown(wait=False)

def job_1min_execution():
    """V5: Process Orders, V7: Check Margin Calls + TP/SL"""
    try:
        execution_desk.process_intents()
        
        if settings.PAPER_TRADING_MODE:
            prices = {}
            # [FIX HIGH-13] Reuse existing authenticated CCXT instance
            from executors.trade_executor import trade_executor
            ex = trade_executor.binance
            for symbol in settings.trading_symbols_slash:
                try:
                    t = ex.fetch_ticker(symbol)
                    # Map back to canonical format (BTC/USDT -> BTCUSDT)
                    canonical = symbol.replace('/', '')
                    prices[canonical] = float(t['last'])
                except Exception:
                    pass
            paper_engine.check_liquidations(prices)
            # [FIX CRITICAL-2] Actually call check_tp_sl  - was implemented but never invoked
            paper_engine.check_tp_sl(prices)
            
    except Exception as e:
        logger.error(f"1-minute execution job error: {e}")


def job_liquidation_cascade_monitor():
    """Score liquidation cascade risk and emit Telegram alerts when thresholds are met."""
    try:
        result = cascade_warning_engine.run_all()
        if result:
            logger.info(f"Liquidation cascade monitor alerts: {list(result.keys())}")
    except Exception as e:
        logger.error(f"Liquidation cascade monitor job error: {e}")


def job_15min_dune():
    """Collect cadence-aware Dune snapshots and persist to DB."""
    if dune_collector is None:
        return
    try:
        dune_collector.run_due_queries(limit=200, offset=0)
    except Exception as e:
        logger.error(f"15-minute Dune job error: {e}")


def job_1hour_deribit():
    """Collect Deribit options data: DVOL, PCR, IV Term Structure, 25d Skew."""
    try:
        deribit_collector.run()
    except Exception as e:
        logger.error(f"Deribit collection job error: {e}")

def job_8hour_funding_fee():
    """V8: Simulate funding fee deduction every 8 hours."""
    try:
        from executors.paper_exchange import paper_engine
        from executors.trade_executor import trade_executor
        
        rates = {}
        prices = {}
        positions = paper_engine.get_open_positions()
        if not positions: return
            
        for pos in positions:
            symbol = pos["symbol"]
            if symbol not in rates:
                try:
                    f_info = trade_executor.binance.fetch_funding_rate(symbol)
                    rates[symbol] = float(f_info['fundingRate'])
                    t = trade_executor.binance.fetch_ticker(symbol)
                    prices[symbol] = float(t['last'])
                except Exception as e:
                    logger.warning(f"Failed to fetch funding rate for {symbol}: {e}")
                    
        paper_engine.apply_funding_fees(rates, prices)
    except Exception as e:
        logger.error(f"8-hour funding fee job error: {e}")


def job_daily_fear_greed():
    """Collect Crypto Fear & Greed Index (alternative.me, daily)."""
    try:
        fear_greed_collector.run()
    except Exception as e:
        logger.error(f"Fear & Greed collection job error: {e}")


def job_daily_coinmetrics():
    """Collect daily Coin Metrics snapshots used for regime and risk gating."""
    try:
        if not settings.COINMETRICS_ENABLED:
            logger.info("Coin Metrics job skipped (disabled)")
            return
        coinmetrics_collector.run()
    except Exception as e:
        logger.error(f"Coin Metrics daily job error: {e}")


def job_daily_etf_flow():
    """Scrape Farside Investors for daily BTC/ETH ETF net flow data."""
    try:
        etf_flow_collector.run()
    except Exception as e:
        logger.error(f"ETF flow collection job error: {e}")


def job_daily_stablecoin():
    """Fetch USDT/USDC circulating supply from DefiLlama (free)."""
    try:
        stablecoin_collector.run()
    except Exception as e:
        logger.error(f"Stablecoin collection job error: {e}")


def job_4hour_coinglass():
    """Fetch Binance Futures LSR + OI (public API, no key required)."""
    try:
        coinglass_collector.run()
    except Exception as e:
        logger.error(f"Binance positioning collection job error: {e}")


def job_routine_market_status():
    """V13.3: Routine Market Status check (Free-First) with Multi-Coin & Telegram Intel."""
    try:
        logger.info("Running routine market status check (Free-First)")

        def _parse_reference_source(source_tag: str) -> tuple[str, str]:
            raw = str(source_tag or "").strip()
            if raw.startswith("[") and raw.endswith("]"):
                raw = raw[1:-1].strip()
            if " - " in raw:
                source_name, source_ref = raw.split(" - ", 1)
                return source_name.strip() or "Unknown", source_ref.strip()
            return raw or "Unknown", ""

        def _build_reference_message(selected_news: list[dict]) -> str:
            lines = []
            for idx, item in enumerate(selected_news[:6], start=1):
                headline = str(item.get("headline") or f"뉴스 {idx}").strip()
                lines.append(f"{idx}. {headline}")

                raw_sources = item.get("sources", [])
                if not isinstance(raw_sources, list):
                    raw_sources = [raw_sources]

                seen_sources = set()
                for raw_source in raw_sources:
                    source_name, source_ref = _parse_reference_source(raw_source)
                    dedupe_key = (source_name, source_ref)
                    if dedupe_key in seen_sources:
                        continue
                    seen_sources.add(dedupe_key)

                    if source_ref:
                        lines.append(f"{source_name} : {source_ref}")
                    else:
                        lines.append(source_name)

                if idx < min(len(selected_news), 6):
                    lines.append("")

            return "\n".join(lines).strip()

        def _get_target_chat_id() -> str:
            try:
                from config.local_state import state_manager
                return state_manager.get_telegram_chat_id(settings.TELEGRAM_CHAT_ID) or settings.TELEGRAM_CHAT_ID
            except Exception:
                return settings.TELEGRAM_CHAT_ID

        indicators = {}
        for symbol in settings.trading_symbols:
            indicators[symbol] = {}
            # Price
            try:
                df = db.get_latest_market_data(symbol, limit=1)
                if not df.empty and "close" in df.columns:
                    indicators[symbol]["price"] = float(df["close"].iloc[-1])
            except Exception:
                pass

            # Funding Rate
            try:
                f_df = gcs_parquet_store.get_funding_history_parquet(symbol, limit=1)
                if not f_df.empty and "funding_rate" in f_df.columns:
                    indicators[symbol]["funding_rate"] = float(f_df["funding_rate"].iloc[-1])
            except Exception:
                pass

            # Volatility
            indicators[symbol]["volatility"] = volatility_monitor.calculate_price_change(symbol)
            swing_snapshot = build_mode_technical_snapshot(symbol, TradingMode.SWING)
            latest_regime = get_recent_market_regime(symbol)
            indicators[symbol]["market_regime"] = latest_regime
            indicators[symbol]["technical_snapshot"] = {
                "swing": swing_snapshot,
                "realtime_pressure": swing_snapshot.get("realtime_pressure") or {},
                "events": detect_technical_events(
                    symbol=symbol,
                    swing=swing_snapshot,
                    position=swing_snapshot,  # POSITION 모드 제거됨 — swing으로 대체
                    funding=indicators[symbol].get("funding_rate"),
                    volatility=indicators[symbol].get("volatility"),
                    regime=latest_regime,
                ),
            }
            indicators[symbol]["playbook_snapshot"] = get_latest_playbook_snapshot(symbol)
            try:
                onchain = gcs_parquet_store.get_latest_row("onchain", symbol)
                if onchain:
                    indicators[symbol]["onchain_snapshot"] = {
                        "risk_bias": onchain.get("risk_bias"),
                        "bias_score": onchain.get("bias_score"),
                        "regime_flags": onchain.get("regime_flags", {}),
                        "is_stale": onchain.get("is_stale"),
                    }
            except Exception:
                pass
            try:
                db.insert_market_status_event({
                    "symbol": symbol,
                    "regime": latest_regime,
                    "price": indicators[symbol].get("price"),
                    "funding_rate": indicators[symbol].get("funding_rate"),
                    "volatility": indicators[symbol].get("volatility"),
                    "technical_snapshot": indicators[symbol]["technical_snapshot"],
                })
            except Exception as e:
                logger.warning(f"market_status_events insert skipped for {symbol}: {e}")
                
        # News Intel (Last 1 hour): Telegram + external crypto news synthesized by LLM
        telegram_intel = "최근 1시간 내 주요 뉴스 없음"
        _news_synthesis_timeout_s = 90  # hard cap: skip news if AI calls hang
        final_payload: dict = {}  # populated by synthesis thread when successful
        try:
            from agents.ai_router import ai_client
            import time as _time

            tg_messages = gcs_parquet_store.get_recent_telegram_messages_parquet(hours=1) or []
            tg_items = []
            for msg in tg_messages[:20]:
                tg_items.append({
                    "source_type": "telegram",
                    "source": msg.get("channel", "telegram"),
                    "text": str(msg.get("text", "")),
                    "timestamp": msg.get("timestamp") or msg.get("created_at", ""),
                })

            ext_raw = news_collector.fetch_news(
                categories=["bitcoin", "ethereum", "macro", "trading", "institutional", "defi"],
                limit=4,
                lang="en",
            ) or []
            ext_items = []
            seen_links = set()
            for a in ext_raw:
                link = a.get("link", "")
                if not link or link in seen_links:
                    continue
                seen_links.add(link)
                ext_items.append({
                    "source_type": "external",
                    "source": a.get("source", "unknown"),
                    "title": str(a.get("title", "")),
                    "description": str(a.get("description", "")),
                    "url": link,
                })
                if len(ext_items) >= 12:
                    break

            if tg_items or ext_items:
                logger.info(
                    f"Routine news synthesis inputs: telegram={len(tg_items)} external={len(ext_items)}"
                )

                # ── AI synthesis in a thread with hard timeout so a hung provider
                #    doesn't block market summary delivery ──────────────────────
                _synthesis_result: dict = {"intel": None, "payload": None}

                def _run_news_synthesis() -> None:
                    def _extract_json_object(raw: str) -> dict:
                        if not raw:
                            return {}
                        raw = raw.strip()
                        try:
                            obj = json.loads(raw)
                            return obj if isinstance(obj, dict) else {}
                        except Exception:
                            pass
                        start = raw.find("{")
                        if start < 0:
                            return {}
                        depth = 0
                        in_string = False
                        escape = False
                        for i in range(start, len(raw)):
                            ch = raw[i]
                            if escape:
                                escape = False
                                continue
                            if ch == "\\":
                                escape = True
                                continue
                            if ch == '"':
                                in_string = not in_string
                                continue
                            if in_string:
                                continue
                            if ch == "{":
                                depth += 1
                            elif ch == "}":
                                depth -= 1
                                if depth == 0:
                                    block = raw[start:i + 1]
                                    try:
                                        obj = json.loads(block)
                                        return obj if isinstance(obj, dict) else {}
                                    except Exception:
                                        return {}
                        return {}

                    _payload = {
                        "telegram_messages": tg_items,
                        "external_news": ext_items,
                        "utc_now": datetime.now(timezone.utc).isoformat(),
                    }
                    cluster_prompt = (
                        "Select high-impact crypto news from the input and merge duplicates.\n"
                        "Return STRICT JSON only with this schema:\n"
                        "{\n"
                        "  \"selected\": [\n"
                        "    {\n"
                        "      \"headline\": \"short title\",\n"
                        "      \"claim\": \"single factual claim\",\n"
                        "      \"sources\": [\"[source - url_or_telegram]\"],\n"
                        "      \"impact\": 1-5,\n"
                        "      \"why\": \"one short reason\"\n"
                        "    }\n"
                        "  ]\n"
                        "}\n"
                        "Rules:\n"
                        "- Keep at most 6 items.\n"
                        "- Merge duplicate events and union their sources.\n"
                        "- Use only provided evidence. No fabrication."
                    )
                    cluster_raw = ai_client.generate_response(
                        system_prompt="You are a strict JSON market-news selector.",
                        user_message=f"{cluster_prompt}\n\nINPUT_JSON:\n{json.dumps(_payload, ensure_ascii=False)}",
                        temperature=0.1,
                        max_tokens=800,
                        role="news_cluster",
                    ) or ""

                    cluster_obj = _extract_json_object(cluster_raw)
                    selected_items = cluster_obj.get("selected", []) if isinstance(cluster_obj, dict) else []
                    if not isinstance(selected_items, list):
                        selected_items = []
                    if not selected_items:
                        selected_items = (ext_items[:4] + tg_items[:2])[:6]

                    normalized_items = []
                    for item in selected_items[:6]:
                        if not isinstance(item, dict):
                            continue
                        headline = str(
                            item.get("headline")
                            or item.get("title")
                            or item.get("source")
                            or "Untitled"
                        ).strip()
                        claim = str(
                            item.get("claim")
                            or item.get("description")
                            or item.get("text")
                            or headline
                        ).strip()
                        why = str(item.get("why") or "").strip()
                        impact = item.get("impact", 3)
                        try:
                            impact = int(impact)
                        except Exception:
                            impact = 3
                        sources = item.get("sources", [])
                        if not isinstance(sources, list) or not sources:
                            source_name = str(item.get("source", "unknown")).strip() or "unknown"
                            source_ref = "telegram"
                            if item.get("url"):
                                source_ref = str(item.get("url", "")).strip()
                            sources = [f"[{source_name} - {source_ref}]"]
                        else:
                            sources = [str(src).strip() for src in sources if str(src).strip()]
                        normalized_items.append({
                            "headline": headline,
                            "claim": claim,
                            "impact": max(1, min(5, impact)),
                            "why": why,
                            "sources": sources[:4],
                        })

                    if not normalized_items:
                        normalized_items = [{
                            "headline": "최근 1시간 주요 뉴스 없음",
                            "claim": "유의미한 신규 이벤트 확인되 않았습니다.",
                            "impact": 1,
                            "why": "",
                            "sources": [],
                        }]

                    _time.sleep(1.2)

                    _final_payload = {
                        "selected_news": normalized_items[:6],
                        "utc_now": datetime.now(timezone.utc).isoformat(),
                    }
                    final_prompt = (
                        "Write a concise Korean market briefing based ONLY on selected_news.\n"
                        "Output plain text only, under 220 words.\n"
                        "Summarize the top 6 items as short numbered lines.\n"
                        "Do NOT include raw URLs or long source tags in the summary body.\n"
                        "Keep each line focused on event + likely market implication.\n"
                        "If signals conflict, mention the conflict clearly.\n"
                        "The final sentence MUST end with a full stop."
                    )
                    _intel = ai_client.generate_response(
                        system_prompt="You are a crypto market briefing writer. No markdown fences.",
                        user_message=f"{final_prompt}\n\nINPUT_JSON:\n{json.dumps(_final_payload, ensure_ascii=False)}",
                        temperature=0.2,
                        max_tokens=600,
                        role="news_brief_final",
                    ) or ""

                    bad_ending = ("에서", "및", "또는", "-", ":", "(", "[", "{", "/", ",")
                    if not _intel.strip() or _intel.strip().endswith(bad_ending):
                        logger.warning("news_brief_final returned empty/partial text, retrying once.")
                        _time.sleep(1.0)
                        _intel = ai_client.generate_response(
                            system_prompt="You are a crypto market briefing writer. No markdown fences.",
                            user_message=f"{final_prompt}\n\nINPUT_JSON:\n{json.dumps(_final_payload, ensure_ascii=False)}",
                            temperature=0.1,
                            max_tokens=600,
                            role="news_brief_final",
                        ) or "최근 1시간 내 주요 뉴스 없음"
                    if looks_english_dominant(_intel):
                        logger.warning("news_brief_final returned English-dominant output, rewriting into Korean.")
                        korean_rewrite_prompt = (
                            "Rewrite the briefing into natural Korean.\n"
                            "Output plain text only.\n"
                            "Keep numbered lines.\n"
                            "Do not include English lead-in sentences.\n"
                            "Preserve only factual meaning from the source text."
                        )
                        rewritten = ai_client.generate_response(
                            system_prompt="You are a Korean crypto market editor. Output Korean only.",
                            user_message=(
                                f"{korean_rewrite_prompt}\n\n"
                                f"SOURCE_TEXT:\n{_intel}\n\n"
                                f"REFERENCE_JSON:\n{json.dumps(_final_payload, ensure_ascii=False)}"
                            ),
                            temperature=0.1,
                            max_tokens=600,
                            role="news_summarize",
                        ) or _intel
                        if rewritten.strip():
                            _intel = rewritten

                    _synthesis_result["intel"] = _intel
                    _synthesis_result["payload"] = _final_payload

                import concurrent.futures as _cf_news
                try:
                    with _cf_news.ThreadPoolExecutor(max_workers=1) as _news_pool:
                        _news_fut = _news_pool.submit(_run_news_synthesis)
                        _news_fut.result(timeout=_news_synthesis_timeout_s)
                    if _synthesis_result["intel"]:
                        telegram_intel = _synthesis_result["intel"]
                        final_payload = _synthesis_result["payload"]
                except _cf_news.TimeoutError:
                    logger.warning(
                        f"News synthesis timed out after {_news_synthesis_timeout_s}s "
                        "— skipping news briefing, market summary will still be sent"
                    )
                except Exception as _se:
                    logger.warning(f"News synthesis thread error: {_se}")
            else:
                telegram_intel = "최근 1시간 내 주요 뉴스 없음"
        except Exception as e:
            logger.warning(f"Failed to synthesize market news intel: {e}")
            
        indicators["TELEGRAM_INTEL"] = telegram_intel

        # [FEATURE-3] Split reports for better readability
        from executors.execution_repository import execution_repository
        from executors.outbox_dispatcher import outbox_dispatcher
        import hashlib

        target_chat_id = _get_target_chat_id()

        # idempotency_key에 UTC hour를 포함 → 매 시간마다 새로운 key로 재발송 보장
        _now_utc = datetime.now(timezone.utc)
        _hour_bucket = _now_utc.strftime("%Y%m%d%H")  # e.g. "2026031421"

        # Message 1: News Briefing (only if news exists)
        if telegram_intel and "주요 뉴스 없음" not in telegram_intel:
            news_header = "<b>📰 최근 1시간 뉴스 브리핑 (Synthesized)</b>"
            try:
                execution_repository.enqueue_outbox_event(
                    "telegram_message",
                    {"chat_id": target_chat_id, "text": f"{news_header}\n\n{telegram_intel}", "parse_mode": "HTML"},
                    idempotency_key=f"telegram:routine_news:{_hour_bucket}:"
                    + hashlib.sha256(f"{news_header}\n\n{telegram_intel}".encode("utf-8")).hexdigest()[:16],
                )
            except Exception as e:
                logger.warning(f"Routine news briefing enqueue failed: {e}")

            try:
                import html
                refs_text = _build_reference_message(final_payload.get("selected_news", [])[:6])
                refs_header = "<b> - 최근 1시간 뉴스 참고 링크</b>"
                execution_repository.enqueue_outbox_event(
                    "telegram_message",
                    {
                        "chat_id": target_chat_id,
                        "text": f"{refs_header}\n\n{html.escape(refs_text)}",
                        "parse_mode": "HTML",
                    },
                    idempotency_key=f"telegram:routine_news_refs:{_hour_bucket}:"
                    + hashlib.sha256(f"{refs_header}\n\n{refs_text}".encode("utf-8")).hexdigest()[:16],
                )
            except Exception as e:
                logger.warning(f"Routine news reference enqueue failed: {e}")

        # Message 2: Market Status Summary — 심볼별로 분리 발송 (4096자 초과 방지)
        for symbol in settings.trading_symbols:
            symbol_indicators = {
                symbol: indicators.get(symbol, {}),
                "TELEGRAM_INTEL": indicators.get("TELEGRAM_INTEL"),
            }
            market_header = f"<b>📊 {symbol} 시장 지표 업데이트</b>"
            summary = None
            try:
                summary = market_monitor_agent.summarize_current_status(symbol_indicators)
                logger.success(f"Market Summary Generated [{symbol}]:\n{summary}")
            except Exception as e:
                logger.warning(f"Market summary generation failed [{symbol}]: {e}")

            # AI 실패 시 최소 지표 fallback — 항상 뭔가는 보냄
            if not summary:
                ind = indicators.get(symbol, {})
                price = ind.get("price")
                funding = ind.get("funding_rate")
                vol = ind.get("volatility")
                regime = ind.get("market_regime", "UNKNOWN")
                summary = (
                    f"가격: {price:.2f} USDT\n" if isinstance(price, float) else ""
                ) + (
                    f"펀딩비: {funding:.4%}\n" if isinstance(funding, float) else ""
                ) + (
                    f"변동성(24h): {vol:.2f}%\n" if isinstance(vol, float) else ""
                ) + f"레짐: {regime}"

            try:
                execution_repository.enqueue_outbox_event(
                    "telegram_message",
                    {"chat_id": target_chat_id, "text": f"{market_header}\n\n{summary}", "parse_mode": "HTML"},
                    idempotency_key=f"telegram:routine_market_summary:{symbol}:{_hour_bucket}:"
                    + hashlib.sha256(f"{market_header}\n\n{summary}".encode("utf-8")).hexdigest()[:16],
                )
            except Exception as e:
                logger.warning(f"Routine market status enqueue failed [{symbol}]: {e}")

    except Exception as e:
        logger.error(f"Routine market status job error: {e}")
    finally:
        # 큐에 쌓인 메시지는 중간 실패와 무관하게 항상 발송 시도
        try:
            from executors.outbox_dispatcher import outbox_dispatcher as _dispatcher
            result = _dispatcher.publish_pending(limit=20)
            if result.get("published", 0) or result.get("failed", 0):
                logger.info(f"Outbox flush: published={result.get('published', 0)} failed={result.get('failed', 0)}")
        except Exception as e:
            logger.warning(f"Outbox flush (finally) failed: {e}")


def job_hourly_swing_charts():
    """Hourly Swing + Position chart push for BTC/ETH to Telegram."""
    try:
        from bot.telegram_bot import trading_bot
        from config.local_state import state_manager
        if not trading_bot:
            logger.warning("Hourly swing chart skipped: trading_bot unavailable")
            return

        from mcp_server.tools import mcp_tools
        from executors.outbox_dispatcher import outbox_dispatcher as _dispatcher
        target_chat_id = state_manager.get_telegram_chat_id(settings.TELEGRAM_CHAT_ID) or settings.TELEGRAM_CHAT_ID

        target_symbols = [s for s in settings.trading_symbols if s in ("BTCUSDT", "ETHUSDT")]
        if not target_symbols:
            target_symbols = settings.trading_symbols[:2]

        lane_profiles = [
            ("swing", settings.SWING_HISTORY_MONTHS, "SWING", ["4h", "1d", "1w"]),
        ]

        for symbol in target_symbols:
            for lane, lookback_months, lane_label, allowed_timeframes in lane_profiles:
                try:
                    result = mcp_tools.get_chart_images(symbol, lane=lane)
                    if not isinstance(result, dict) or "charts" not in result:
                        logger.warning(
                            f"Hourly {lane} chart failed for {symbol}: "
                            f"{result.get('error') if isinstance(result, dict) else 'unknown error'}"
                        )
                        continue

                    all_charts = result.get("charts", []) or []
                    charts = [
                        chart for chart in all_charts
                        if str(chart.get("timeframe", "")).lower() in allowed_timeframes
                    ]
                    if all_charts and not charts:
                        generated_tfs = [c.get("timeframe") for c in all_charts]
                        logger.warning(
                            f"Hourly {lane} chart for {symbol}: generated {generated_tfs} "
                            f"but none match allowed {allowed_timeframes} — skipping send"
                        )
                    total = len(charts)
                    for idx, chart in enumerate(charts, start=1):
                        chart_bytes = base64.b64decode(chart["chart_base64"])
                        tf = str(chart.get("timeframe", "4h")).upper()
                        caption = (
                            f" - <b>{symbol} {lane_label} 차트 (정기 1시간)</b>\n"
                            f"Lane: <code>{lane}</code>\n"
                            f"Timeframe: <code>{tf}</code>\n"
                            f"Panel: <code>{idx}/{total}</code>\n"
                            f"Lookback: <code>{lookback_months}M</code>"
                        )
                        _dispatcher._run_async(trading_bot.send_photo, target_chat_id, chart_bytes, caption)
                except Exception as e:
                    logger.warning(f"Hourly {lane} chart send failed for {symbol}: {e}")
    except Exception as e:
        logger.error(f"Hourly swing charts job error: {e}")


def job_24hour_evaluation():
    try:
        logger.info("Running 24-hour evaluation job")
        feedback_generator.run_feedback_cycle()
    except Exception as e:
        logger.error(f"24-hour evaluation job error: {e}")


def job_daily_evaluation_rollup():
    try:
        logger.info("Running daily evaluation rollup job")
        result = evaluation_rollup_service.run_daily_rollup(lookback_days=3)
        logger.info(f"Daily evaluation rollup result: {result}")
    except Exception as e:
        logger.error(f"Daily evaluation rollup job error: {e}")

def job_1hour_evaluation():
    """V6: Self-Healing RAG evaluation of completed trades."""
    try:
        logger.info("Running 1-hour episodic memory evaluation")
        daemon = EvaluatorDaemon()
        daemon.evaluate_recent_trades()
    except Exception as e:
        logger.error(f"1-hour evaluation job error: {e}")

def job_1hour_telegram():
    """Batch stored Telegram messages into LightRAG every 1 hour (Real-time listener handles collection)."""
    _t0 = _time.perf_counter()
    try:
        if _should_defer_heavy_job("1-hour Telegram job"):
            return
        logger.info("Running 1-hour Telegram batching job")

        # 1. Synthesize and ingest to LightRAG
        from processors.telegram_batcher import telegram_batcher
        telegram_batcher.process_and_ingest(lookback_hours=1)

        # 3. Truth Engine: Triangulate corroborated claims via Perplexity (Web)
        from config.local_state import state_manager
        if state_manager.is_analysis_enabled():
            light_rag.run_triangulation_worker(limit=3)
        else:
            logger.info("Triangulation worker skipped (AI analysis disabled)")
        if _SCHED_PROM:
            SCHED_JOB_RESULTS.labels(job="telegram_batch", result="success").inc()
    except Exception as e:
        logger.error(f"1-hour Telegram job error: {e}")
        if _SCHED_PROM:
            SCHED_JOB_RESULTS.labels(job="telegram_batch", result="error").inc()
    finally:
        if _SCHED_PROM:
            SCHED_JOB_DURATION.labels(job="telegram_batch").observe(_time.perf_counter() - _t0)

def job_1hour_crypto_news():
    """Fetch Free Crypto News API and ingest to LightRAG every 1 hour."""
    try:
        if _should_defer_heavy_job("1-hour Crypto News API job"):
            return
        logger.info("Running 1-hour Crypto News API fetch job")
        news_collector.fetch_and_ingest()
    except Exception as e:
        logger.error(f"1-hour Crypto News API job error: {e}")


def job_outbox_drain():
    """Safety-net: drain any queued outbox messages that weren't published by their originating job."""
    try:
        from executors.outbox_dispatcher import outbox_dispatcher as _dispatcher
        result = _dispatcher.publish_pending(limit=50)
        if result.get("published", 0) or result.get("failed", 0):
            logger.info(f"Outbox drain: published={result.get('published', 0)} failed={result.get('failed', 0)}")
    except Exception as e:
        logger.error(f"Outbox drain job error: {e}")


def job_pressure_signal_evaluation():
    """Backfill forward-return evaluation into market_status_events.technical_snapshot.evaluation."""
    try:
        logger.info("Running realtime pressure signal evaluation")
        result = run_pressure_signal_evaluation(limit=200, hours=72, dry_run=False)
        logger.info(f"Realtime pressure evaluation result: {result}")
    except Exception as e:
        logger.error(f"Realtime pressure evaluation job error: {e}")


def job_daily_refresh_higher_tf_cache():
    """1m 로컬 parquet → 4h/1d/1w 재계산. 매일 02:00 UTC 실행.

    최근 2~6개월치 1m 데이터를 리샘플링해 4h(월별)/1d/1w(연별) 파일을 갱신.
    차트 요청 시 1m 전체를 메모리에 올리지 않아도 되므로 메모리 할당 문제 해결.
    """
    try:
        symbols = list(settings.trading_symbols)   # e.g. ["BTCUSDT", "ETHUSDT"]
        for symbol in symbols:
            logger.info(f"[refresh_higher_tf] start {symbol}")
            gcs_parquet_store.refresh_higher_tf_cache(symbol)
        logger.info("[refresh_higher_tf] all symbols done")
    except Exception as e:
        logger.error(f"job_daily_refresh_higher_tf_cache error: {e}")


def job_daily_archive_to_gcs():
    """Archive expiring rows to GCS Parquet and verify manifests."""
    try:
        logger.info("Running daily GCS archive job")
        archive_result = gcs_archive_exporter.run_daily_archive()
        logger.info(f"GCS Parquet archive: {archive_result}")
    except Exception as e:
        logger.error(f"Daily GCS archive job error: {e}")


def job_daily_safe_cleanup():
    """Delete only rows covered by verified archive manifests."""
    try:
        logger.info("Running daily safe cleanup job")
        result = gcs_archive_exporter.run_safe_cleanup(limit=1000)
        logger.info(f"Safe cleanup result: {result}")

        from executors.agent_state_store import agent_state_store as _ass
        _ass.purge_expired()

        graph_days = settings.RETENTION_GRAPH_DAYS
        if graph_days > 0:
            light_rag.cleanup_old(days=graph_days)
        stats = light_rag.get_stats()
        logger.info(f"LightRAG stats: {stats}")
    except Exception as e:
        logger.error(f"Daily safe cleanup job error: {e}")

    # Time-based Supabase cleanup (GCS archive 여부와 무관하게 항상 실행)
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=settings.RETENTION_TELEGRAM_DAYS)).isoformat()
        r = db.client.table("telegram_messages").delete().lt("created_at", cutoff).execute()
        logger.info(f"Telegram messages cleanup: deleted rows older than {settings.RETENTION_TELEGRAM_DAYS}d")
    except Exception as e:
        logger.warning(f"telegram_messages cleanup failed: {e}")

    try:
        market_status_days = int(getattr(settings, "RETENTION_MARKET_STATUS_DAYS", 30))
        cutoff = (datetime.now(timezone.utc) - timedelta(days=market_status_days)).isoformat()
        db.client.table("market_status_events").delete().lt("created_at", cutoff).execute()
        logger.info(f"market_status_events cleanup: deleted rows older than {market_status_days}d")
    except Exception as e:
        logger.warning(f"market_status_events cleanup failed: {e}")

    # Retention-based cleanup (GCS archive 없이도 항상 실행)
    try:
        result = db.cleanup_old_data()
        logger.info(f"Retention cleanup result: {result}")
    except Exception as e:
        logger.warning(f"Retention cleanup failed: {e}")





def _daily_precision_prep_bucket(now_utc: datetime | None = None) -> str:
    now = now_utc or datetime.now(timezone.utc)
    candidates = []
    base_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    base_minute = int(getattr(settings, "DAILY_PRECISION_MINUTE_UTC", 30))
    for day_offset in (-1, 0, 1):
        day_anchor = base_day + timedelta(days=day_offset)
        for hour in _daily_precision_schedule_hours_utc():
            candidates.append(day_anchor.replace(hour=hour, minute=base_minute))
    latest = max((item for item in candidates if item <= now), default=None)
    if latest is None:
        latest = min(candidates) if candidates else now
    return latest.isoformat(timespec="minutes")


def job_daily_precision_prepare_shared() -> None:
    global _daily_precision_prepare_bucket

    bucket = _daily_precision_prep_bucket()
    with _daily_precision_prepare_lock:
        if bucket == _daily_precision_prepare_bucket:
            logger.info(f"Daily precision shared prep already completed for bucket={bucket}")
            return
        logger.info(f"Running daily precision shared prep for bucket={bucket}")
        job_daily_coinmetrics()
        macro_collector.run()
        _daily_precision_prepare_bucket = bucket


def job_daily_precision():
    """Daily UTC serial: BTC POSITION -> ETH POSITION.
    Runs high-quality analysis once per symbol and persists dual-lane playbooks.
    """
    _t0 = _time.perf_counter()
    try:
        from config.local_state import state_manager
        if not state_manager.is_analysis_enabled():
            logger.info("Daily precision skipped (analysis disabled)")
            return
        job_daily_precision_prepare_shared()
        orchestrator.run_daily_playbook()
        if _SCHED_PROM:
            SCHED_JOB_RESULTS.labels(job="daily_precision", result="success").inc()
    except Exception as e:
        logger.error(f"Daily precision job error: {e}")
        if _SCHED_PROM:
            SCHED_JOB_RESULTS.labels(job="daily_precision", result="error").inc()
    finally:
        if _SCHED_PROM:
            SCHED_JOB_DURATION.labels(job="daily_precision").observe(_time.perf_counter() - _t0)


def job_daily_precision_symbol(symbol: str):
    """Daily UTC single-symbol precision runner."""
    import concurrent.futures
    _HARD_TIMEOUT_S = 90 * 60  # 90분 하드 타임아웃

    try:
        from config.local_state import state_manager
        normalized_symbol = str(symbol or "").upper()
        if not state_manager.is_analysis_enabled():
            logger.info(f"Daily precision skipped for {normalized_symbol} (analysis disabled)")
            return
        job_daily_precision_prepare_shared()

        def _run():
            return orchestrator.run_daily_playbook_for_symbol(normalized_symbol, TradingMode.SWING)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run)
            try:
                result = future.result(timeout=_HARD_TIMEOUT_S)
                logger.info(f"Daily precision result for {normalized_symbol}: {result}")
            except concurrent.futures.TimeoutError:
                logger.error(
                    f"Daily precision TIMEOUT for {normalized_symbol} after {_HARD_TIMEOUT_S // 60}min — "
                    f"force-resetting precision lock"
                )
                # _daily_precision_active_count 카운터 리셋
                orchestrator._exit_daily_precision_run()
                # _analysis_locks의 실제 Lock 객체도 제거 (백그라운드 스레드가 계속 점유 중이므로
                # pop으로 새 Lock을 쓰도록 해야 다음 실행이 SKIPPED_LOCK에 걸리지 않음)
                orchestrator.force_release_analysis_lock(normalized_symbol, TradingMode.SWING)
    except Exception as e:
        logger.error(f"Daily precision job error for {symbol}: {e}")


def job_hourly_monitor():
    """Hourly: evaluate all symbol/mode pairs against Daily Playbook.
    Outputs NO_ACTION / WATCH / TRIGGER.
    TRIGGER  - run analysis + allow order execution (capped at 2/day/symbol).
    """
    _t0 = _time.perf_counter()
    try:
        from config.local_state import state_manager
        if not state_manager.is_analysis_enabled():
            logger.info("Hourly monitor skipped (analysis disabled)")
            return
        orchestrator.run_hourly_monitor()
        if _SCHED_PROM:
            SCHED_JOB_RESULTS.labels(job="hourly_monitor", result="success").inc()
    except Exception as e:
        logger.error(f"Hourly monitor job error: {e}")
        if _SCHED_PROM:
            SCHED_JOB_RESULTS.labels(job="hourly_monitor", result="error").inc()
    finally:
        if _SCHED_PROM:
            SCHED_JOB_DURATION.labels(job="hourly_monitor").observe(_time.perf_counter() - _t0)

    # SpotMode.POSITION: 현물 포지션 thesis invalidation 체크
    try:
        from executors.spot_orchestrator import SPOT_MODE_ENABLED, SPOT_MODE, SpotMode as _SpotMode, spot_orchestrator as _so
        if SPOT_MODE_ENABLED and SPOT_MODE == _SpotMode.POSITION:
            for sym in settings.trading_symbols:
                _so.check_position_invalidation(sym)
    except Exception as e:
        logger.error(f"Spot invalidation check error: {e}")


def main():
    mode = settings.trading_mode

    if _SCHED_PROM:
        try:
            _prom_start(9090, addr="127.0.0.1")
            logger.info("Prometheus metrics server started on 127.0.0.1:9090")
        except Exception as e:
            logger.warning(f"Prometheus server failed to start: {e}")

    logger.info(f"Starting Trading System (mode={mode.value})")
    logger.info(
        f"  Primary analysis cadence (UTC): daily_precision={_daily_precision_all_labels_utc()} | "
        "hourly_monitor=hh:15 | market_status=hh:20"
    )
    logger.info(f"  Timeframes: {settings.analysis_timeframes}")
    logger.info(f"  Chart timeframe: {settings.chart_timeframe}")
    logger.info(f"  Candle limit: {settings.candle_limit}")
    logger.info(f"  Data lookback: {settings.data_lookback_hours}h")
    logger.info(f"  Chart for VLM: {settings.should_use_chart}")
    logger.info(f"  Symbols: {', '.join(settings.trading_symbols)}")
    logger.info(f"  AI: Gemini Judge/VLM (Project A/B) + Cerebras (meta/risk) + Groq (news/rag) + OpenRouter (monitor)")
    logger.info(f"  Data: Global OI + OI Divergence + MFI Proxy + Liquidations + Coin Metrics + Perplexity + LightRAG")
    logger.info(f"  Dune: {'enabled' if dune_collector else 'disabled'}")
    logger.info(f"  LightRAG: Neo4j {'connected' if settings.NEO4J_URI else 'in-memory'} + "
                f"Milvus {'connected' if settings.MILVUS_URI else 'in-memory'}")

    # [PERF] Start real-time price feed — eliminates REST price fetch at order fire time
    try:
        from collectors.ws_price_feed import ws_price_feed
        ws_price_feed.start()
        logger.info("WS price feed started (zero-latency price reads for order execution)")
    except Exception as e:
        logger.warning(f"WS price feed unavailable (REST fallback active): {e}")

    # [PERF] Start User Data Stream — real-time fill confirmation (replaces outbox polling)
    try:
        from collectors.ws_user_stream import ws_user_stream
        ws_user_stream.start()
        logger.info("WS user stream started (real-time fill confirmation)")
    except Exception as e:
        logger.warning(f"WS user stream unavailable (outbox fallback active): {e}")

    # Start WebSocket collector for liquidation + whale data
    try:
        from collectors.websocket_collector import websocket_collector
        websocket_collector.start_background()
        logger.info("WebSocket collector started (liquidation + whale trades)")
    except Exception as e:
        logger.warning(f"WebSocket collector unavailable: {e}")

    # [FIX] WebSocket thread health check  - every 5 minutes
    def job_5m_ws_health_check():
        try:
            from collectors.websocket_collector import websocket_collector
            if hasattr(websocket_collector, '_thread'):
                if not websocket_collector._thread.is_alive():
                    logger.warning("WebSocket thread died  - restarting...")
                    websocket_collector.start_background()
        except Exception as e:
            logger.error(f"WS health check error: {e}")

    scheduler_config.scheduler.add_job(
        job_5m_ws_health_check,
        'interval',
        minutes=5,
        id='job_5m_ws_health',
        max_instances=1
    )

    import os
    # [FIX Cold Start] Run Telegram bot in a daemon thread (unless disabled by env var)
    if os.environ.get("DISABLE_TELEGRAM_BOT") != "true":
        def _run_telegram_bot():
            try:
                from bot.telegram_bot import trading_bot
                trading_bot.run()  # blocking within this thread
            except Exception as e:
                logger.error(f"Telegram bot crashed: {e}")
                logger.info("Trading system continues WITHOUT Telegram bot commands.")

        bot_thread = threading.Thread(target=_run_telegram_bot, name="telegram-bot", daemon=True)
        bot_thread.start()
        logger.info("Telegram bot started in background thread.")
    else:
        logger.info("Telegram bot starting skipped (DISABLE_TELEGRAM_BOT=true).")
        # 더미 스레드를 생성하여 is_alive() 체크에서 터지지 않도록 방어
        bot_thread = threading.Thread(target=lambda: None)

    # [V13.1] Persistent Telegram Listener (Real-Time Alpha Ingestion)
    if os.environ.get("DISABLE_TELEGRAM_LISTENER") != "true":
        def _run_telegram_listener():
            try:
                from collectors.telegram_listener import telegram_listener
                import asyncio
                asyncio.run(telegram_listener.start())
            except Exception as e:
                logger.error(f"Telegram listener crashed: {e}")

        listener_thread = threading.Thread(target=_run_telegram_listener, name="telegram-listener", daemon=True)
        listener_thread.start()
        logger.info("Real-time Telegram Listener (Alpha V13.1) started.")
    else:
        logger.info("Telegram Listener starting skipped (DISABLE_TELEGRAM_LISTENER=true).")
        listener_thread = threading.Thread(target=lambda: None)

    role = os.environ.get("SCHEDULER_ROLE", "all").lower()
    logger.info(f"Scheduler starting with ROLE: {role.upper()}")

    # ==========================================
    # DATA JOBS: 데이터 수집 및 인프라 (가격, 온체인, 뉴스)
    # ==========================================
    if role in ("all", "data"):
        # 1-minute tick: price, funding, microstructure, volatility
        scheduler_config.scheduler.add_job(
            job_1min_tick,
            'interval',
            minutes=1,
            id='job_1min_tick',
            max_instances=1
        )
        
        scheduler_config.scheduler.add_job(
            job_15min_dune,
            'interval',
            minutes=15,
            id='job_15min_dune',
            max_instances=1
        )

        scheduler_config.scheduler.add_job(
            job_1hour_deribit,
            CronTrigger(minute=0),
            id='job_1hour_deribit',
            max_instances=1
        )

        scheduler_config.scheduler.add_job(
            job_daily_fear_greed,
            CronTrigger(hour=0, minute=15),
            id='job_daily_fear_greed',
            max_instances=1
        )

        scheduler_config.scheduler.add_job(
            job_daily_coinmetrics,
            CronTrigger(hour=3, minute=12),
            id='job_daily_coinmetrics',
            max_instances=1
        )

        scheduler_config.scheduler.add_job(
            job_daily_etf_flow,
            CronTrigger(hour=6, minute=0),
            id='job_daily_etf_flow',
            max_instances=1
        )

        scheduler_config.scheduler.add_job(
            job_daily_stablecoin,
            CronTrigger(hour=6, minute=15),
            id='job_daily_stablecoin',
            max_instances=1
        )

        scheduler_config.scheduler.add_job(
            job_4hour_coinglass,
            CronTrigger(hour='0,4,8,12,16,20', minute=30),
            id='job_4hour_coinglass',
            max_instances=1
        )

        scheduler_config.scheduler.add_job(
            job_1hour_crypto_news,
            CronTrigger(minute=10),
            id='job_1hour_crypto_news',
            max_instances=1
        )

        scheduler_config.scheduler.add_job(
            job_daily_refresh_higher_tf_cache,
            CronTrigger(hour=2, minute=0),
            id='job_daily_refresh_higher_tf_cache',
            max_instances=1
        )

        scheduler_config.scheduler.add_job(
            job_daily_archive_to_gcs,
            CronTrigger(hour=1, minute=0),
            id='job_daily_archive_to_gcs',
            max_instances=1
        )

        scheduler_config.scheduler.add_job(
            job_daily_safe_cleanup,
            CronTrigger(hour=1, minute=20),
            id='job_daily_safe_cleanup',
            max_instances=1
        )


    # ==========================================
    # BRAIN JOBS: AI 판단, 주문 실행, 텔레그램 리포트 관리
    # ==========================================
    if role in ("all", "brain"):
        if not EXECUTION_PROCESS_SEPARATE:
            scheduler_config.scheduler.add_job(
                job_1min_execution,
                'interval',
                minutes=1,
                id='job_1min_execution',
                max_instances=1
            )

        scheduler_config.scheduler.add_job(
            job_liquidation_cascade_monitor,
            'interval',
            minutes=1,
            id='job_liquidation_cascade_monitor',
            max_instances=1
        )

        if settings.PAPER_TRADING_MODE and not EXECUTION_PROCESS_SEPARATE:
            scheduler_config.scheduler.add_job(
                job_8hour_funding_fee,
                CronTrigger(hour='0,8,16', minute=0),
                id='job_8hour_funding_fee',
                max_instances=1
            )

        scheduler_config.scheduler.add_job(
            job_1hour_telegram,
            CronTrigger(minute=5),
            id='job_1hour_telegram',
            max_instances=1
        )

        scheduler_config.scheduler.add_job(
            job_outbox_drain,
            'interval',
            minutes=5,
            id='job_outbox_drain',
            max_instances=1,
        )

        scheduler_config.scheduler.add_job(
            job_pressure_signal_evaluation,
            CronTrigger(minute='*/15'),
            id='job_pressure_signal_evaluation',
            max_instances=1,
        )

        daily_gap_minutes = max(0, int(getattr(settings, "DAILY_PRECISION_SYMBOL_GAP_MINUTES", 10)))
        for symbol_index, symbol in enumerate(settings.trading_symbols):
            slot_offset = symbol_index * daily_gap_minutes
            for hour, minute in _daily_precision_schedule_slots_utc(offset_minutes=slot_offset):
                scheduler_config.scheduler.add_job(
                    lambda _symbol=symbol: job_daily_precision_symbol(_symbol),
                    CronTrigger(hour=hour, minute=minute),
                    id=f"job_daily_precision_{symbol.lower()}_{hour:02d}{minute:02d}",
                    max_instances=1,
                )

        scheduler_config.scheduler.add_job(
            job_snapshot_refresh_fast,
            CronTrigger(minute='*/15'),
            id='job_snapshot_refresh_fast',
            max_instances=1,
        )

        scheduler_config.scheduler.add_job(
            job_snapshot_refresh_narrative,
            CronTrigger(minute='2,32'),
            id='job_snapshot_refresh_narrative',
            max_instances=1,
        )

        scheduler_config.scheduler.add_job(
            job_hourly_monitor,
            CronTrigger(minute=15),
            id='job_hourly_monitor',
            max_instances=1,
        )

        scheduler_config.scheduler.add_job(
            job_routine_market_status,
            CronTrigger(minute=20),
            id='job_market_status',
            max_instances=1,
        )

        scheduler_config.scheduler.add_job(
            job_hourly_swing_charts,
            CronTrigger(minute=22),
            id='job_hourly_swing_charts',
            max_instances=1,
        )

        scheduler_config.scheduler.add_job(
            job_24hour_evaluation,
            CronTrigger(hour=0, minute=30),
            id='job_24hour_evaluation',
            max_instances=1
        )

        scheduler_config.scheduler.add_job(
            job_daily_evaluation_rollup,
            CronTrigger(hour=0, minute=40),
            id='job_daily_evaluation_rollup',
            max_instances=1
        )
        
        scheduler_config.scheduler.add_job(
            job_1hour_evaluation,
            CronTrigger(minute=45),
            id='job_1hour_evaluation',
            max_instances=1
        )

        try:
            from executors.spot_orchestrator import (
                SPOT_MODE_ENABLED as _SPOT_ENABLED,
                SPOT_MODE as _SPOT_MODE,
                SPOT_POSITION_ANALYSIS_HOUR_UTC as _SPOT_HOUR,
                SpotMode as _SpotMode,
            )
            if _SPOT_ENABLED and _SPOT_MODE == _SpotMode.POSITION:
                def job_spot_position_daily():
                    try:
                        from executors.spot_orchestrator import spot_orchestrator
                        for sym in settings.trading_symbols:
                            spot_orchestrator.run_position_analysis(sym)
                    except Exception as e:
                        logger.error(f"Spot position daily analysis error: {e}")

                scheduler_config.scheduler.add_job(
                    job_spot_position_daily,
                    CronTrigger(hour=_SPOT_HOUR, minute=30),
                    id='job_spot_position_daily',
                    max_instances=1,
                )
                logger.info(f"Spot position daily job registered at {_SPOT_HOUR:02d}:30 UTC")
        except Exception as _spot_sched_err:
            logger.warning(f"Spot scheduler setup skipped: {_spot_sched_err}")

    scheduler_config.scheduler.start()
    logger.info("Scheduler started.")
    logger.info(
        f"Cadence(UTC): daily_precision={_daily_precision_all_labels_utc()} | snapshot_fast=hh:*/5 | "
        "snapshot_narrative=hh:02,32 | telegram_batch=hh:05 | crypto_news=hh:10 | hourly_monitor=hh:15 | market_status=hh:20 | daily_rollup=00:40 | "
        "hourly_eval=hh:45 | gcs_archive=01:00 | safe_cleanup=01:20 | higher_tf_refresh=02:00"
    )

    # [FIX Cold Start] Run initial data collection immediately so first analysis has data
    logger.info("Running initial data collection (cold start bootstrap)...")
    def _startup_telegram_catchup():
        """Handles initial RAG synthesis for the last 24h.
        Raw collection is handled by the telegram_listener thread's backfill task.
        """
        import time
        # Increased delay to 30s to let the Listener finish its metadata-only backfill
        time.sleep(30)
        
        from processors.telegram_batcher import telegram_batcher
        # Reduced to 6h for synthesized catch-up to avoid heavy LLM load on cold start
        telegram_batcher.process_and_ingest(lookback_hours=6)

    def _bootstrap_missing_parquet_cache():
        """스타트업 시 로컬 캐시에 없는 심볼의 parquet 파일을 GCS에서 다운로드."""
        try:
            from processors.gcs_parquet import gcs_parquet_store
            if not gcs_parquet_store.enabled:
                logger.info("GCS disabled — parquet cache bootstrap skipped")
                return
            for symbol in settings.trading_symbols:
                for tf, months_back in [("4h", settings.SWING_HISTORY_MONTHS), ("1d", settings.SWING_HISTORY_MONTHS), ("1w", settings.SWING_HISTORY_MONTHS)]:
                    paths = gcs_parquet_store._build_ohlcv_paths(tf, symbol, months_back)
                    missing = [p for p in paths if not gcs_parquet_store._local_cache_path(p).exists()]
                    if missing:
                        logger.info(f"Bootstrapping {len(missing)} missing {tf} parquet files for {symbol}...")
                        for p in missing:
                            gcs_parquet_store._download_parquet(p)
        except Exception as e:
            logger.warning(f"Parquet cache bootstrap failed (non-fatal): {e}")

    _initial_collectors = [
        ("Price + Funding + Microstructure", lambda: (collector.run(), funding_collector.run(), microstructure_collector.run())),
        ("Volatility", lambda: volatility_monitor.run()),
        ("Deribit", lambda: deribit_collector.run()),
        ("Fear & Greed", lambda: fear_greed_collector.run()),
        ("Coin Metrics", lambda: job_daily_coinmetrics()),
        ("Parquet cache bootstrap", _bootstrap_missing_parquet_cache),
        ("Snapshot prewarm", lambda: job_snapshot_refresh_fast()),
        ("Pressure signal evaluation", lambda: job_pressure_signal_evaluation()),
        ("Telegram catch-up (24h)", _startup_telegram_catchup),
    ]
    for name, fn in _initial_collectors:
        try:
            fn()
            logger.info(f"  [OK] {name} collected")
        except Exception as e:
            logger.warning(f"  [WARN] {name} collection failed (non-fatal): {e}")

    # Main thread: keep alive + graceful shutdown
    try:
        import time
        while True:
            time.sleep(60)
            # Optional: restart bot thread if it dies (only if not disabled)
            if os.environ.get("DISABLE_TELEGRAM_BOT") != "true":
                if not bot_thread.is_alive():
                    logger.warning("Telegram bot thread died  - restarting in 30s...")
                    time.sleep(30)
                    bot_thread = threading.Thread(target=_run_telegram_bot, name="telegram-bot", daemon=True)
                    bot_thread.start()

            if os.environ.get("DISABLE_TELEGRAM_LISTENER") != "true":
                if not listener_thread.is_alive():
                    logger.warning("Telegram listener thread died  - restarting in 30s...")
                    time.sleep(30)
                    listener_thread = threading.Thread(target=_run_telegram_listener, name="telegram-listener", daemon=True)
                    listener_thread.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down...")
        # Upload Telegram session to Secret Manager before exit
        try:
            from collectors.telegram_listener import upload_session_to_cloud
            upload_session_to_cloud()
        except Exception:
            pass
        try:
            from collectors.websocket_collector import websocket_collector
            websocket_collector.stop()
        except Exception:
            pass
        scheduler_config.scheduler.shutdown()
        sys.exit(0)


if __name__ == "__main__":
    main()







