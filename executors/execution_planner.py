# -*- coding: utf-8 -*-
"""Layer 5: Execution Planner.

Institutional quant 방식:
  사이즈를 결정한 후 "언제, 어떻게, 얼마에 분할 진입하는가"를 계획한다.
  All-in 진입이 아니라 분할 진입(DCA)으로 실행 비용을 줄이고
  진입 타이밍의 불확실성을 헤지한다.

플랜 구성:
  - 2~3회 분할 진입 (각 진입 가격/비중/타이밍)
  - 1~2회 분할 청산 (부분 이익 실현)
  - 무효화 가격 (이 가격이면 전량 취소)
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple
from loguru import logger


# 진입 스타일별 파라미터
_STYLE_PARAMS = {
    "PASSIVE_MAKER": {
        "n_entries": 2,
        "entry_offsets": [-0.2, -0.5],    # ATR 배수로 현재가 아래에 지정가
        "entry_weights": [0.6, 0.4],        # 첫 진입 60%, 두번째 40%
        "tp_levels": [2.0, 3.3],            # R 배수 (SL 대비)
        "tp_weights": [0.5, 0.5],           # 각 TP에서 청산 비중
        "description": "저변동 지정가 — 수수료 절약",
    },
    "SMART_DCA": {
        "n_entries": 3,
        "entry_offsets": [0.0, -0.5, -1.0], # 즉시 + 지지선 2회
        "entry_weights": [0.4, 0.35, 0.25],
        "tp_levels": [2.0, 3.3],
        "tp_weights": [0.5, 0.5],
        "description": "지지선 분할 — 평균 단가 개선",
    },
    "MOMENTUM_SNIPER": {
        "n_entries": 1,
        "entry_offsets": [0.0],             # 즉시 시장가
        "entry_weights": [1.0],
        "tp_levels": [2.0, 4.0],
        "tp_weights": [0.6, 0.4],
        "description": "브레이크아웃 즉시 진입 — 모멘텀 포착",
    },
    "CASINO_EXIT": {
        "n_entries": 1,
        "entry_offsets": [0.0],
        "entry_weights": [1.0],
        "tp_levels": [1.5],
        "tp_weights": [1.0],
        "description": "패닉 탈출 — 즉시 시장가",
    },
}

# 무효화 배수 (SL 대비 조기 무효화 비율)
_INVALIDATION_BUFFER = 1.1  # SL 방향으로 10% 더 빡빡하게


class ExecutionPlanner:
    """분할 진입/청산 계획 생성기."""

    # ── 메인 플래닝 ──────────────────────────────────────────────────────────

    def plan(
        self,
        symbol: str,
        direction: str,              # LONG / SHORT
        total_size_usd: float,
        execution_style: str,        # PASSIVE_MAKER / SMART_DCA / MOMENTUM_SNIPER / CASINO_EXIT
        current_price: float,
        atr_anchor: Optional[Dict] = None,
        market_data: Optional[Dict] = None,
        approved_allocation_pct: float = 40.0,
    ) -> Dict:
        """
        분할 진입/청산 플랜 생성.

        Returns:
            {
                "entries": [
                    {"price": float, "size_usd": float, "weight_pct": float,
                     "delay_minutes": int, "type": "limit"|"market"},
                    ...
                ],
                "exits": [
                    {"price": float, "weight_pct": float, "r_multiple": float},
                    ...
                ],
                "invalidation_price": float,
                "total_size_usd": float,
                "style": str,
                "rationale": str
            }
        """
        direction = direction.upper()
        style = execution_style.upper()
        params = _STYLE_PARAMS.get(style, _STYLE_PARAMS["SMART_DCA"])

        # ATR 기반 가격 단위 계산
        atr = self._get_atr(atr_anchor, current_price)
        sl_price, tp1_price, tp2_price = self._get_key_prices(
            atr_anchor=atr_anchor,
            market_data=market_data,
            current_price=current_price,
            direction=direction,
            atr=atr,
        )

        # 진입 플랜 계산
        entries = self._compute_entries(
            direction=direction,
            current_price=current_price,
            atr=atr,
            total_size_usd=total_size_usd,
            params=params,
        )

        # 청산 플랜 계산
        exits = self._compute_exits(
            direction=direction,
            entry_ref_price=entries[0]["price"] if entries else current_price,
            sl_price=sl_price,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            params=params,
        )

        # 무효화 가격
        invalidation = self._compute_invalidation(
            sl_price=sl_price,
            direction=direction,
            atr=atr,
        )

        # SL까지의 거리
        sl_dist_pct = (
            abs(current_price - sl_price) / current_price * 100
            if current_price > 0 else 0.0
        )

        rationale = (
            f"{style} 플랜: {len(entries)}회 분할 진입, "
            f"{len(exits)}회 분할 청산. "
            f"ATR={atr:.2f} SL거리={sl_dist_pct:.2f}%. "
            f"{params['description']}"
        )

        logger.info(
            f"[ExecPlan] {symbol} {direction} ${total_size_usd:.0f} "
            f"style={style} entries={len(entries)} atr={atr:.2f}"
        )

        return {
            "entries": entries,
            "exits": exits,
            "invalidation_price": round(float(invalidation), 2),
            "stop_loss_price": round(float(sl_price), 2),
            "total_size_usd": round(float(total_size_usd), 2),
            "style": style,
            "atr_used": round(float(atr), 4),
            "rationale": rationale,
        }

    # ── ATR 계산 ─────────────────────────────────────────────────────────────

    def _get_atr(
        self,
        atr_anchor: Optional[Dict],
        current_price: float,
    ) -> float:
        """ATR anchor에서 ATR 값 추출. 없으면 현재가의 2% 추정."""
        if atr_anchor and atr_anchor.get("atr_14_4h"):
            try:
                return float(atr_anchor["atr_14_4h"])
            except (ValueError, TypeError):
                pass
        # 폴백: 현재가의 2%
        return current_price * 0.02

    # ── 핵심 가격 계산 ────────────────────────────────────────────────────────

    def _get_key_prices(
        self,
        atr_anchor: Optional[Dict],
        market_data: Optional[Dict],
        current_price: float,
        direction: str,
        atr: float,
    ) -> Tuple[float, float, float]:
        """SL, TP1, TP2 가격 계산.

        우선순위:
        1. atr_anchor의 suggested 값
        2. market_data의 confluence zone / fibonacci
        3. ATR 기반 폴백
        """
        sl_price = tp1_price = tp2_price = 0.0

        # atr_anchor 우선
        if atr_anchor:
            sl_pct = float(atr_anchor.get("suggested_sl_pct", 0) or 0)
            tp_pct = float(atr_anchor.get("suggested_tp_pct", 0) or 0)
            if sl_pct > 0 and tp_pct > 0:
                if direction == "LONG":
                    sl_price = current_price * (1 - sl_pct / 100)
                    tp1_price = current_price * (1 + tp_pct / 100)
                    tp2_price = current_price * (1 + tp_pct / 100 * 1.65)
                else:
                    sl_price = current_price * (1 + sl_pct / 100)
                    tp1_price = current_price * (1 - tp_pct / 100)
                    tp2_price = current_price * (1 - tp_pct / 100 * 1.65)

        # 폴백: ATR 기반 (SWING 기준: SL=1.5×ATR, TP=3.3×ATR)
        if sl_price == 0:
            sl_mult = 1.5
            tp1_mult = 3.0
            tp2_mult = 5.0
            if direction == "LONG":
                sl_price = current_price - sl_mult * atr
                tp1_price = current_price + tp1_mult * atr
                tp2_price = current_price + tp2_mult * atr
            else:
                sl_price = current_price + sl_mult * atr
                tp1_price = current_price - tp1_mult * atr
                tp2_price = current_price - tp2_mult * atr

        return sl_price, tp1_price, tp2_price

    # ── 진입 플랜 ─────────────────────────────────────────────────────────────

    def _compute_entries(
        self,
        direction: str,
        current_price: float,
        atr: float,
        total_size_usd: float,
        params: Dict,
    ) -> List[Dict]:
        """분할 진입 가격/비중/타이밍 계산."""
        entries = []
        n = params["n_entries"]
        offsets = params["entry_offsets"][:n]
        weights = params["entry_weights"][:n]
        delay_per_entry = 30  # 분 단위

        for i, (offset_atr, weight) in enumerate(zip(offsets, weights)):
            # 진입가: 현재가에서 ATR 배수만큼 이격 (LONG = 아래, SHORT = 위)
            if direction == "LONG":
                entry_price = current_price + offset_atr * atr  # offset 음수면 아래
            else:
                entry_price = current_price - offset_atr * atr  # offset 음수면 위

            entry_price = max(entry_price, 1.0)
            size_usd = total_size_usd * weight
            entry_type = "market" if (i == 0 and offset_atr == 0.0) else "limit"
            delay_min = i * delay_per_entry

            entries.append({
                "index": i + 1,
                "price": round(float(entry_price), 2),
                "size_usd": round(float(size_usd), 2),
                "weight_pct": round(weight * 100, 1),
                "delay_minutes": delay_min,
                "type": entry_type,
                "offset_atr": offset_atr,
            })

        return entries

    # ── 청산 플랜 ─────────────────────────────────────────────────────────────

    def _compute_exits(
        self,
        direction: str,
        entry_ref_price: float,
        sl_price: float,
        tp1_price: float,
        tp2_price: float,
        params: Dict,
    ) -> List[Dict]:
        """분할 청산 가격/비중 계산."""
        exits = []
        sl_dist = abs(entry_ref_price - sl_price)
        tp_levels = params["tp_levels"]
        tp_weights = params["tp_weights"]

        tp_prices = [tp1_price, tp2_price]

        for i, (tp_price, weight) in enumerate(zip(tp_prices[:len(tp_levels)], tp_weights)):
            if tp_price <= 0:
                continue
            r_multiple = (
                abs(tp_price - entry_ref_price) / sl_dist if sl_dist > 1e-9 else 0.0
            )
            exits.append({
                "index": i + 1,
                "price": round(float(tp_price), 2),
                "weight_pct": round(weight * 100, 1),
                "r_multiple": round(float(r_multiple), 2),
                "type": "limit",
            })

        return exits

    # ── 무효화 가격 ────────────────────────────────────────────────────────────

    def _compute_invalidation(
        self,
        sl_price: float,
        direction: str,
        atr: float,
    ) -> float:
        """무효화 가격: SL보다 약간 더 보수적 (10% 더)."""
        if direction == "LONG":
            return sl_price - atr * 0.1
        else:
            return sl_price + atr * 0.1

    # ── 플랜 요약 (Judge/Report용) ────────────────────────────────────────────

    def format_plan(self, plan: Dict) -> str:
        """실행 플랜 요약 문자열."""
        if not plan:
            return "[ExecPlan] No plan generated"

        lines = [
            f"[EXECUTION PLAN] Style: {plan.get('style', 'N/A')}",
            f"Total Size: ${plan.get('total_size_usd', 0):,.0f}  "
            f"ATR: {plan.get('atr_used', 0):.2f}",
            "",
            "Entry Plan:",
        ]
        for e in plan.get("entries", []):
            lines.append(
                f"  #{e['index']} {e['type'].upper():<6} @ {e['price']:,.2f}  "
                f"{e['weight_pct']:.0f}% (${e['size_usd']:,.0f})  "
                f"+{e['delay_minutes']}min"
            )

        lines.append("\nExit Plan:")
        for ex in plan.get("exits", []):
            lines.append(
                f"  #{ex['index']} TP @ {ex['price']:,.2f}  "
                f"{ex['weight_pct']:.0f}%  R={ex['r_multiple']:.1f}"
            )

        lines.append(
            f"\nInvalidation: {plan.get('invalidation_price', 0):,.2f}  "
            f"SL: {plan.get('stop_loss_price', 0):,.2f}"
        )
        lines.append(f"\n{plan.get('rationale', '')}")
        return "\n".join(lines)


# ── Singleton ──────────────────────────────────────────────────────────────────
execution_planner = ExecutionPlanner()
