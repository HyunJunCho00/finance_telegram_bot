"""Market status job — scheduler.py와 Cloud Run 양쪽에서 공유되는 로직."""
from __future__ import annotations

import hashlib
import json
import html as _html
from datetime import datetime, timezone

from loguru import logger

from config.database import db
from config.settings import settings, TradingMode
from processors.gcs_parquet import gcs_parquet_store
from collectors.volatility_monitor import volatility_monitor
from collectors.crypto_news_collector import collector as news_collector
from processors.market_snapshot_builder import build_mode_technical_snapshot, detect_technical_events
from processors.playbook_service import get_recent_market_regime, get_latest_playbook_snapshot
from agents.market_monitor_agent import market_monitor_agent
from utils.text_sanitizer import looks_english_dominant
from utils.json_utils import extract_json_object


# ── 레퍼런스 포맷 헬퍼 ──────────────────────────────────────────────────────────

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
        seen_sources: set[tuple[str, str]] = set()
        for raw_source in raw_sources:
            source_name, source_ref = _parse_reference_source(raw_source)
            dedupe_key = (source_name, source_ref)
            if dedupe_key in seen_sources:
                continue
            seen_sources.add(dedupe_key)
            lines.append(f"{source_name} : {source_ref}" if source_ref else source_name)
        if idx < min(len(selected_news), 6):
            lines.append("")
    return "\n".join(lines).strip()


def _get_target_chat_id() -> str:
    try:
        from config.local_state import state_manager
        return state_manager.get_telegram_chat_id(settings.TELEGRAM_CHAT_ID) or settings.TELEGRAM_CHAT_ID
    except Exception:
        return settings.TELEGRAM_CHAT_ID


# ── 심볼별 지표 수집 ────────────────────────────────────────────────────────────

def _collect_symbol_indicators(symbol: str) -> dict:
    """Collect price, funding, volatility, and technical context for one symbol."""
    ind: dict = {}
    try:
        df = db.get_latest_market_data(symbol, limit=1)
        if not df.empty and "close" in df.columns:
            ind["price"] = float(df["close"].iloc[-1])
    except Exception:
        pass
    try:
        f_df = gcs_parquet_store.get_funding_history_parquet(symbol, limit=1)
        if not f_df.empty and "funding_rate" in f_df.columns:
            ind["funding_rate"] = float(f_df["funding_rate"].iloc[-1])
    except Exception:
        pass
    ind["volatility"] = volatility_monitor.calculate_price_change(symbol)
    swing_snapshot = build_mode_technical_snapshot(symbol, TradingMode.SWING)
    latest_regime = get_recent_market_regime(symbol)
    ind["market_regime"] = latest_regime
    ind["technical_snapshot"] = {
        "swing": swing_snapshot,
        "realtime_pressure": swing_snapshot.get("realtime_pressure") or {},
        "events": detect_technical_events(
            symbol=symbol,
            swing=swing_snapshot,
            position=swing_snapshot,
            funding=ind.get("funding_rate"),
            volatility=ind.get("volatility"),
            regime=latest_regime,
        ),
    }
    ind["playbook_snapshot"] = get_latest_playbook_snapshot(symbol)
    try:
        onchain = gcs_parquet_store.get_latest_row("onchain", symbol)
        if onchain:
            ind["onchain_snapshot"] = {
                "risk_bias": onchain.get("risk_bias"),
                "bias_score": onchain.get("bias_score"),
                "regime_flags": onchain.get("regime_flags", {}),
                "is_stale": onchain.get("is_stale"),
            }
    except Exception:
        pass
    event = {
        "symbol": symbol,
        "regime": latest_regime,
        "price": ind.get("price"),
        "funding_rate": ind.get("funding_rate"),
        "volatility": ind.get("volatility"),
        "technical_snapshot": ind["technical_snapshot"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    result = None
    try:
        result = db.insert_market_status_event(event)
    except Exception as e:
        logger.warning(f"market_status_events insert skipped for {symbol}: {e}")

    if not result:
        # CB OPEN 또는 Supabase 장애 → GCS timeseries에 저장
        try:
            import pandas as pd
            df = pd.DataFrame([event])
            df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
            gcs_parquet_store.write_timeseries_to_local(
                "market_status_events", symbol, df, ["created_at", "symbol"]
            )
        except Exception as gcs_e:
            logger.warning(f"[GCS Fallback] market_status_events write failed ({symbol}): {gcs_e}")
    return ind


# ── 뉴스 AI 합성 ────────────────────────────────────────────────────────────────

def _synthesize_news_intel(
    tg_items: list[dict],
    ext_items: list[dict],
    timeout_s: int = 90,
) -> tuple[str, dict]:
    """2-stage AI news synthesis: cluster → Korean briefing."""
    if not tg_items and not ext_items:
        return "최근 1시간 내 주요 뉴스 없음", {}

    from agents.ai_router import ai_client
    import time as _t
    import concurrent.futures as _cf

    _synthesis_result: dict = {"intel": None, "payload": None}

    def _run() -> None:
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
            "      \"already_priced_in\": true/false,\n"
            "      \"why\": \"one short reason\"\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "Rules:\n"
            "- Keep at most 6 items.\n"
            "- Merge duplicate events and union their sources.\n"
            "- Use only provided evidence. No fabrication.\n"
            "- already_priced_in=true if the event is widely known, repeated, or already reflected in recent price action.\n"
            "- already_priced_in=false if the event is a genuine surprise or structural first-time development."
        )
        cluster_raw = ai_client.generate_response(
            system_prompt="You are a strict JSON market-news selector.",
            user_message=f"{cluster_prompt}\n\nINPUT_JSON:\n{json.dumps(_payload, ensure_ascii=False)}",
            temperature=0.1,
            max_tokens=800,
            role="news_cluster",
        ) or ""

        cluster_obj = extract_json_object(cluster_raw)
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
                item.get("headline") or item.get("title") or item.get("source") or "Untitled"
            ).strip()
            claim = str(
                item.get("claim") or item.get("description") or item.get("text") or headline
            ).strip()
            why = str(item.get("why") or "").strip()
            already_priced_in = bool(item.get("already_priced_in", False))
            impact = item.get("impact", 3)
            try:
                impact = int(impact)
            except Exception:
                impact = 3
            sources = item.get("sources", [])
            if not isinstance(sources, list) or not sources:
                source_name = str(item.get("source", "unknown")).strip() or "unknown"
                source_ref = str(item.get("url", "")).strip() or "telegram"
                sources = [f"[{source_name} - {source_ref}]"]
            else:
                sources = [str(src).strip() for src in sources if str(src).strip()]
            normalized_items.append({
                "headline": headline,
                "claim": claim,
                "impact": max(1, min(5, impact)),
                "already_priced_in": already_priced_in,
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

        _t.sleep(1.2)

        _now = datetime.now(timezone.utc)
        _final_payload = {
            "selected_news": normalized_items[:6],
            "utc_now": _now.isoformat(),
        }
        # Sort: already_priced_in=False first, then by impact desc
        display_items = sorted(
            normalized_items[:6],
            key=lambda x: (x.get("already_priced_in", False), -x.get("impact", 3)),
        )
        final_prompt = (
            "Write a concise Korean market briefing based ONLY on selected_news.\n"
            "Output plain text only.\n"
            "Write numbered lines for items with impact >= 3 only (skip lower impact).\n"
            "Items where already_priced_in=true should be labeled '(선반영)' at the end of the line.\n"
            "Each line MUST be under 60 Korean characters.\n"
            "Keep each line focused on event + likely market implication.\n"
            "Do NOT include raw URLs or source tags.\n"
            "If signals conflict, mention the conflict clearly.\n"
            "The final line MUST end with a full stop."
        )
        _display_payload = {
            "selected_news": display_items,
            "utc_now": _final_payload["utc_now"],
        }
        _intel = ai_client.generate_response(
            system_prompt="You are a crypto market briefing writer. No markdown fences.",
            user_message=f"{final_prompt}\n\nINPUT_JSON:\n{json.dumps(_display_payload, ensure_ascii=False)}",
            temperature=0.2,
            max_tokens=900,
            role="news_brief_final",
        ) or ""

        bad_ending = ("에서", "및", "또는", "-", ":", "(", "[", "{", "/", ",")
        if not _intel.strip() or _intel.strip().endswith(bad_ending):
            logger.warning("news_brief_final returned empty/partial text, retrying once.")
            _t.sleep(1.0)
            _intel = ai_client.generate_response(
                system_prompt="You are a crypto market briefing writer. No markdown fences.",
                user_message=f"{final_prompt}\n\nINPUT_JSON:\n{json.dumps(_display_payload, ensure_ascii=False)}",
                temperature=0.1,
                max_tokens=900,
                role="news_brief_final",
            ) or "최근 1시간 내 주요 뉴스 없음"

        if looks_english_dominant(_intel):
            logger.warning("news_brief_final returned English-dominant output, rewriting into Korean.")
            rewritten = ai_client.generate_response(
                system_prompt="You are a Korean crypto market editor. Output Korean only.",
                user_message=(
                    "Rewrite the briefing into natural Korean.\n"
                    "Output plain text only.\nKeep numbered lines.\n"
                    "Do not include English lead-in sentences.\n"
                    "Preserve only factual meaning from the source text.\n\n"
                    f"SOURCE_TEXT:\n{_intel}\n\n"
                    f"REFERENCE_JSON:\n{json.dumps(_display_payload, ensure_ascii=False)}"
                ),
                temperature=0.1,
                max_tokens=900,
                role="news_summarize",
            ) or _intel
            if rewritten.strip():
                _intel = rewritten

        # 3순위: impact 예측값 로깅 (calibration 데이터 축적)
        try:
            db.log_news_impact_predictions(normalized_items[:6], _now)
        except Exception as _log_err:
            logger.warning(f"news_impact_predictions logging skipped: {_log_err}")

        _synthesis_result["intel"] = _intel
        _synthesis_result["payload"] = _final_payload

    try:
        with _cf.ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(_run).result(timeout=timeout_s)
        if _synthesis_result["intel"]:
            return _synthesis_result["intel"], _synthesis_result["payload"] or {}
    except _cf.TimeoutError:
        logger.warning(
            f"News synthesis timed out after {timeout_s}s "
            "— skipping news briefing, market summary will still be sent"
        )
    except Exception as e:
        logger.warning(f"News synthesis error: {e}")

    return "최근 1시간 내 주요 뉴스 없음", {}


# ── 메인 잡 함수 ────────────────────────────────────────────────────────────────

def run_market_status() -> None:
    """V13.3: Routine Market Status check. scheduler.py와 Cloud Run 양쪽에서 호출."""
    try:
        logger.info("Running routine market status check (Free-First)")

        indicators: dict = {}
        for symbol in settings.trading_symbols:
            indicators[symbol] = _collect_symbol_indicators(symbol)

        tg_messages = gcs_parquet_store.get_recent_telegram_messages_parquet(hours=1) or []
        tg_items = [
            {
                "source_type": "telegram",
                "source": msg.get("channel", "telegram"),
                "text": str(msg.get("text", "")),
                "timestamp": msg.get("timestamp") or msg.get("created_at", ""),
            }
            for msg in tg_messages[:20]
        ]

        ext_raw = news_collector.fetch_news(
            categories=["bitcoin", "ethereum", "macro", "trading", "institutional", "defi"],
            limit=10,
            lang="en",
        ) or []
        seen_links: set[str] = set()
        ext_items: list[dict] = []
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
            logger.info(f"Routine news synthesis inputs: telegram={len(tg_items)} external={len(ext_items)}")

        telegram_intel, final_payload = _synthesize_news_intel(tg_items, ext_items, timeout_s=90)
        indicators["TELEGRAM_INTEL"] = telegram_intel

        from executors.execution_repository import execution_repository

        target_chat_id = _get_target_chat_id()
        _now_utc = datetime.now(timezone.utc)
        _hour_bucket = _now_utc.strftime("%Y%m%d%H")

        if telegram_intel and "주요 뉴스 없음" not in telegram_intel:
            news_header = "<b>📰 최근 1시간 뉴스 브리핑 (Synthesized)</b>"
            try:
                execution_repository.enqueue_outbox_event(
                    "telegram_message",
                    {"chat_id": target_chat_id, "text": f"{news_header}\n\n{telegram_intel}", "parse_mode": "HTML"},
                    idempotency_key=f"telegram:routine_news:{_hour_bucket}:"
                    + hashlib.sha256(f"{news_header}\n\n{telegram_intel}".encode()).hexdigest()[:16],
                )
            except Exception as e:
                logger.warning(f"Routine news briefing enqueue failed: {e}")

            try:
                refs_text = _build_reference_message(final_payload.get("selected_news", [])[:6])
                refs_header = "<b> - 최근 1시간 뉴스 참고 링크</b>"
                execution_repository.enqueue_outbox_event(
                    "telegram_message",
                    {
                        "chat_id": target_chat_id,
                        "text": f"{refs_header}\n\n{_html.escape(refs_text)}",
                        "parse_mode": "HTML",
                    },
                    idempotency_key=f"telegram:routine_news_refs:{_hour_bucket}:"
                    + hashlib.sha256(f"{refs_header}\n\n{refs_text}".encode()).hexdigest()[:16],
                )
            except Exception as e:
                logger.warning(f"Routine news reference enqueue failed: {e}")

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

            if not summary:
                ind = indicators.get(symbol, {})
                price = ind.get("price")
                funding = ind.get("funding_rate")
                vol = ind.get("volatility")
                regime = ind.get("market_regime", "UNKNOWN")
                summary = (
                    (f"가격: {price:.2f} USDT\n" if isinstance(price, float) else "")
                    + (f"펀딩비: {funding:.4%}\n" if isinstance(funding, float) else "")
                    + (f"변동성(24h): {vol:.2f}%\n" if isinstance(vol, float) else "")
                    + f"레짐: {regime}"
                )

            try:
                execution_repository.enqueue_outbox_event(
                    "telegram_message",
                    {"chat_id": target_chat_id, "text": f"{market_header}\n\n{summary}", "parse_mode": "HTML"},
                    idempotency_key=f"telegram:routine_market_summary:{symbol}:{_hour_bucket}:"
                    + hashlib.sha256(f"{market_header}\n\n{summary}".encode()).hexdigest()[:16],
                )
            except Exception as e:
                logger.warning(f"Routine market status enqueue failed [{symbol}]: {e}")

    except Exception as e:
        logger.error(f"Routine market status job error: {e}")
    finally:
        try:
            from executors.outbox_dispatcher import outbox_dispatcher as _dispatcher
            result = _dispatcher.publish_pending(limit=20)
            if result.get("published", 0) or result.get("failed", 0):
                logger.info(f"Outbox flush: published={result.get('published', 0)} failed={result.get('failed', 0)}")
        except Exception as e:
            logger.warning(f"Outbox flush (finally) failed: {e}")
