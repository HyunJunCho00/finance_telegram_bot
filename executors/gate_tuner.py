"""
Confluence Gate Auto-Tuner
==========================
모의매매 결과(tp_hit / sl_hit)를 분석해 confluence_gate_threshold를 자동 조정.

조정 규칙:
- 최근 N회 진입(LONG/SHORT) 중 win_rate > 60% + 샘플 >= MIN_SAMPLE → threshold -0.5
- win_rate < 40% + 샘플 >= MIN_SAMPLE                               → threshold +0.5
- 진입 횟수 < MIN_SAMPLE AND HOLD rate > 85%                        → threshold -0.5 (과도한 HOLD 완화)
- threshold 범위: [MIN_THRESHOLD, MAX_THRESHOLD]
"""

from datetime import datetime, timezone, timedelta
from typing import Optional
from loguru import logger
from config.database import db
from config.local_state import state_manager

MIN_THRESHOLD = 1.5
MAX_THRESHOLD = 4.0
MIN_SAMPLE = 5       # 조정 결정에 필요한 최소 진입 횟수
LOOKBACK_DAYS = 14   # 분석 기간
HOLD_RATE_FLOOR = 0.85  # 이 이상 HOLD 비율이면 threshold 내림


def _fetch_recent_reports(symbol: str, days: int = LOOKBACK_DAYS) -> list:
    """ai_reports 테이블에서 최근 평가 완료된 리포트 조회."""
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        resp = (
            db.client.table("ai_reports")
            .select("final_decision, swing_evaluation, position_evaluation")
            .eq("symbol", symbol)
            .gte("created_at", cutoff)
            .order("created_at", desc=True)
            .limit(60)
            .execute()
        )
        return resp.data or []
    except Exception as e:
        logger.warning(f"GateTuner: failed to fetch reports for {symbol}: {e}")
        return []


def _parse_decision(row: dict) -> Optional[str]:
    import json
    fd = row.get("final_decision")
    if isinstance(fd, str):
        try:
            fd = json.loads(fd)
        except Exception:
            return None
    if isinstance(fd, dict):
        return str(fd.get("decision", "")).upper()
    return None


def _parse_outcome(row: dict) -> Optional[str]:
    """swing_evaluation 또는 position_evaluation에서 tp_hit/sl_hit 추출."""
    import json
    for field in ("swing_evaluation", "position_evaluation"):
        raw = row.get(field)
        if not raw:
            continue
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                continue
        if not isinstance(raw, dict):
            continue
        if raw.get("tp_hit") is True:
            return "WIN"
        if raw.get("sl_hit") is True:
            return "LOSS"
    return None  # 아직 평가 안 됐거나 TP/SL 미도달


def analyze(symbol: str) -> dict:
    """최근 리포트를 분석해 통계 반환."""
    rows = _fetch_recent_reports(symbol)
    if not rows:
        return {"symbol": symbol, "total": 0, "entries": 0, "wins": 0, "losses": 0,
                "win_rate": None, "hold_rate": None, "current_threshold": state_manager.get_confluence_gate_threshold()}

    total = len(rows)
    holds = sum(1 for r in rows if _parse_decision(r) in ("HOLD", None))
    entries = total - holds

    wins = sum(1 for r in rows if _parse_decision(r) in ("LONG", "SHORT") and _parse_outcome(r) == "WIN")
    losses = sum(1 for r in rows if _parse_decision(r) in ("LONG", "SHORT") and _parse_outcome(r) == "LOSS")

    win_rate = (wins / (wins + losses)) if (wins + losses) > 0 else None
    hold_rate = (holds / total) if total > 0 else None

    return {
        "symbol": symbol,
        "total": total,
        "entries": entries,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "hold_rate": hold_rate,
        "current_threshold": state_manager.get_confluence_gate_threshold(),
    }


def run(symbol: str) -> dict:
    """분석 후 필요하면 threshold 조정. 변경 없으면 None 반환."""
    stats = analyze(symbol)
    current = stats["current_threshold"]
    win_rate = stats["win_rate"]
    hold_rate = stats["hold_rate"]
    evaluated = stats["wins"] + stats["losses"]  # TP/SL 확정된 것만

    reason = None
    new_threshold = current

    if evaluated >= MIN_SAMPLE and win_rate is not None:
        if win_rate > 0.60:
            new_threshold = round(current - 0.5, 1)
            reason = f"win_rate={win_rate:.0%} > 60% over {evaluated} trades → 완화"
        elif win_rate < 0.40:
            new_threshold = round(current + 0.5, 1)
            reason = f"win_rate={win_rate:.0%} < 40% over {evaluated} trades → 강화"
    elif evaluated < MIN_SAMPLE and hold_rate is not None and hold_rate > HOLD_RATE_FLOOR:
        new_threshold = round(current - 0.5, 1)
        reason = f"hold_rate={hold_rate:.0%} > {HOLD_RATE_FLOOR:.0%}, 진입 부족 → 완화"

    new_threshold = max(MIN_THRESHOLD, min(MAX_THRESHOLD, new_threshold))

    if reason and new_threshold != current:
        state_manager.set_confluence_gate_threshold(new_threshold, reason=reason)
        logger.info(f"[GateTuner] {symbol}: {current} → {new_threshold} ({reason})")
        stats["adjusted"] = True
        stats["new_threshold"] = new_threshold
        stats["reason"] = reason
    else:
        stats["adjusted"] = False
        stats["reason"] = "변경 없음"

    return stats


gate_tuner = object.__new__(type("GateTuner", (), {"run": staticmethod(run), "analyze": staticmethod(analyze)}))
