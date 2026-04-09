# -*- coding: utf-8 -*-
"""Layer 3: Risk Budget Controller + Stress Test.

Institutional quant 방식:
  리스크를 예산처럼 관리한다.
  - 레짐별로 사용 가능한 리스크 예산을 할당
  - 진입 전 역사적 시나리오 스트레스 테스트 실행
  - 예산 초과 시 사이즈 축소 또는 VETO

핵심 철학:
  "얼마나 벌 수 있나?"가 아니라
  "최악의 경우 얼마를 잃을 수 있나?"를 먼저 계산한다.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple
from loguru import logger


# ── 레짐별 리스크 예산 (0~100점) ──────────────────────────────────────────────
_REGIME_RISK_BUDGET: Dict[str, float] = {
    "BULL_MOMENTUM":       100.0,   # 추세장 = 풀 배분 가능
    "SIDEWAYS_ACCUMULATION": 70.0,  # 축적 구간 = 보수적
    "RANGE_BOUND":          60.0,   # 레인지 = 중립
    "BEAR_MOMENTUM":        35.0,   # 하락추세 = 엄격 제한
    "VOLATILITY_PANIC":     15.0,   # 패닉 = 거의 불가
    "UNKNOWN":              50.0,   # 불명 = 중간
}

# ── 역사적 스트레스 시나리오 ───────────────────────────────────────────────────
# (BTC_drop_pct, ETH_drop_pct) — 각 자산별 최대 낙폭 (%)
_STRESS_SCENARIOS: Dict[str, Dict[str, float]] = {
    "covid_crash_2020_03": {
        "BTCUSDT": -40.0,
        "ETHUSDT": -55.0,
        "description": "COVID 팬데믹 충격 (2020년 3월 -40%)",
    },
    "china_ban_2021_05": {
        "BTCUSDT": -50.0,
        "ETHUSDT": -58.0,
        "description": "중국 채굴 금지 (2021년 5월 -50%)",
    },
    "luna_collapse_2022": {
        "BTCUSDT": -29.0,
        "ETHUSDT": -35.0,
        "description": "LUNA/UST 붕괴 (2022년 5월)",
    },
    "ftx_collapse_2022": {
        "BTCUSDT": -25.0,
        "ETHUSDT": -30.0,
        "description": "FTX 파산 충격 (2022년 11월)",
    },
    "rate_shock_2022": {
        "BTCUSDT": -30.0,
        "ETHUSDT": -38.0,
        "description": "연준 급격 금리인상 (2022년 연간)",
    },
    "etf_rejection_flash": {
        "BTCUSDT": -15.0,
        "ETHUSDT": -18.0,
        "description": "ETF 승인 거절 플래시 (단기 이벤트)",
    },
}

# ── 포지션 리스크 비용 계산 기준 ──────────────────────────────────────────────
# 포지션 1% = 리스크 예산 1점 (레버리지 × 상관관계 보정 포함)
_MAX_TOLERABLE_DRAWDOWN_PCT = 15.0   # 단일 시나리오 최대 허용 낙폭 (계좌 대비 %)
_MAX_SINGLE_STRESS_LOSS_PCT = 20.0   # 스트레스 최악 시나리오 한도


class RiskBudgetController:
    """레짐 기반 리스크 예산 관리 + 스트레스 테스트."""

    # ── 예산 조회 ─────────────────────────────────────────────────────────────

    def get_available_budget(self, regime: str) -> float:
        """현재 레짐에서 사용 가능한 리스크 예산 (0~100)."""
        budget = _REGIME_RISK_BUDGET.get(regime.upper(), _REGIME_RISK_BUDGET["UNKNOWN"])
        logger.debug(f"[RiskBudget] regime={regime} budget={budget}")
        return budget

    # ── 포지션 비용 계산 ──────────────────────────────────────────────────────

    def calculate_position_cost(
        self,
        symbol: str,
        allocation_pct: float,   # 0~100
        leverage: float,
        correlation_with_other: float = 0.0,  # 이미 보유한 다른 자산과의 상관계수
        other_allocation_pct: float = 0.0,    # 이미 보유한 다른 자산의 배분 %
    ) -> float:
        """이 포지션이 리스크 예산에서 소모하는 점수 계산.

        기본: allocation_pct × leverage
        상관관계 보정: 이미 같은 방향 포지션이 있으면 추가 비용
        """
        base_cost = allocation_pct * leverage
        # 상관관계 추가 비용: ρ × 다른_포지션_비용의 일부
        corr_cost = correlation_with_other * other_allocation_pct * leverage * 0.5
        total_cost = base_cost + corr_cost
        return round(total_cost, 2)

    # ── 스트레스 테스트 ───────────────────────────────────────────────────────

    def run_stress_test(
        self,
        positions: List[Dict],
        account_balance_usd: float = 2000.0,
    ) -> Dict:
        """
        포지션 목록에 대해 역사적 시나리오 스트레스 테스트 실행.

        Args:
            positions: [
                {"symbol": "BTCUSDT", "allocation_pct": 40, "leverage": 2, "direction": "LONG"},
                {"symbol": "ETHUSDT", "allocation_pct": 20, "leverage": 1, "direction": "LONG"},
            ]
            account_balance_usd: 계좌 총 잔액

        Returns:
            {
                "passed": bool,
                "worst_scenario": str,
                "worst_loss_pct": float,
                "scenario_results": dict,
                "max_drawdown_pct": float,
                "recommendation": str
            }
        """
        scenario_results: Dict[str, float] = {}
        worst_loss = 0.0
        worst_scenario = "none"

        for scenario_name, scenario_data in _STRESS_SCENARIOS.items():
            total_loss_pct = 0.0
            for pos in positions:
                symbol = pos.get("symbol", "BTCUSDT")
                alloc = float(pos.get("allocation_pct", 0)) / 100.0  # 0~1
                lev = float(pos.get("leverage", 1))
                direction = str(pos.get("direction", "LONG")).upper()

                price_shock = float(scenario_data.get(symbol, -20.0))

                if direction == "LONG":
                    pos_loss = alloc * lev * price_shock  # 음수 = 손실
                elif direction == "SHORT":
                    pos_loss = alloc * lev * (-price_shock)  # 하락이면 숏은 이익
                else:
                    pos_loss = 0.0

                total_loss_pct += pos_loss

            scenario_results[scenario_name] = round(float(total_loss_pct), 2)

            if total_loss_pct < worst_loss:  # 더 큰 손실 (더 음수)
                worst_loss = total_loss_pct
                worst_scenario = scenario_name

        # 통과 기준
        passed = abs(worst_loss) <= _MAX_TOLERABLE_DRAWDOWN_PCT

        # 권장 사항
        if abs(worst_loss) > _MAX_SINGLE_STRESS_LOSS_PCT:
            recommendation = (
                f"VETO: 최악 시나리오({worst_scenario}) 손실 "
                f"{worst_loss:.1f}%가 한도 {_MAX_SINGLE_STRESS_LOSS_PCT}% 초과"
            )
        elif abs(worst_loss) > _MAX_TOLERABLE_DRAWDOWN_PCT:
            scale = _MAX_TOLERABLE_DRAWDOWN_PCT / abs(worst_loss)
            recommendation = (
                f"SIZE DOWN: 포지션 {scale*100:.0f}%로 축소. "
                f"({worst_scenario} 시나리오 {worst_loss:.1f}% 손실 우려)"
            )
        else:
            recommendation = (
                f"PASS: 최악 시나리오({worst_scenario}) 손실 "
                f"{worst_loss:.1f}% — 허용 범위 내"
            )

        logger.info(
            f"[RiskBudget/Stress] worst={worst_scenario} "
            f"loss={worst_loss:.1f}% passed={passed}"
        )

        return {
            "passed": passed,
            "worst_scenario": worst_scenario,
            "worst_scenario_desc": _STRESS_SCENARIOS.get(worst_scenario, {}).get(
                "description", worst_scenario
            ),
            "worst_loss_pct": round(float(worst_loss), 2),
            "scenario_results": scenario_results,
            "max_tolerable_pct": _MAX_TOLERABLE_DRAWDOWN_PCT,
            "recommendation": recommendation,
        }

    # ── 파라메트릭 VaR ───────────────────────────────────────────────────────

    def compute_var(
        self,
        portfolio_vol_pct: float,   # 일별 변동성 (%)
        allocation_pct: float,       # 0~100
        confidence_level: float = 0.95,
        horizon_days: int = 1,
    ) -> float:
        """파라메트릭 VaR 계산 (정규분포 가정).

        VaR = allocation × σ_portfolio × Z_score × √horizon
        """
        # Z-score for confidence level
        z_scores = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}
        z = z_scores.get(confidence_level, 1.645)
        alloc = allocation_pct / 100.0
        var = alloc * portfolio_vol_pct * z * math.sqrt(horizon_days)
        return round(float(var), 4)

    # ── 진입 승인 결정 ────────────────────────────────────────────────────────

    def approve_or_scale(
        self,
        draft_decision: Dict,
        regime: str,
        symbol: str,
        portfolio_risk: Optional[Dict] = None,
        other_open_positions: Optional[List[Dict]] = None,
        account_balance_usd: float = 2000.0,
    ) -> Dict:
        """스트레스 테스트 + 예산 체크로 포지션 승인/축소/VETO.

        Args:
            draft_decision: Judge/Risk Manager 결정 dict
            regime: 현재 시장 레짐
            symbol: 거래 대상 심볼
            portfolio_risk: portfolio_optimizer.analyze() 결과
            other_open_positions: 현재 오픈 포지션 목록
            account_balance_usd: 계좌 잔액

        Returns:
            updated draft_decision with risk_budget_check 필드 추가
        """
        decision_dir = str(draft_decision.get("decision", "HOLD")).upper()
        allocation = float(draft_decision.get("allocation_pct", 0) or 0)
        leverage = float(draft_decision.get("leverage", 1) or 1)

        result = dict(draft_decision)

        # HOLD/CANCEL은 스트레스 테스트 불필요
        if decision_dir not in ["LONG", "SHORT"]:
            result["risk_budget_check"] = {
                "passed": True,
                "reason": f"direction={decision_dir} — no position risk",
                "stress_test": None,
            }
            return result

        # 1. 예산 확인
        available_budget = self.get_available_budget(regime)

        # 기존 오픈 포지션 상관관계 비용
        other_positions = other_open_positions or []
        btc_eth_corr = 0.87
        if portfolio_risk:
            btc_eth_corr = portfolio_risk.get("btc_eth_correlation", 0.87)

        other_alloc = sum(
            float(p.get("allocation_pct", 0))
            for p in other_positions
        )
        position_cost = self.calculate_position_cost(
            symbol=symbol,
            allocation_pct=allocation,
            leverage=leverage,
            correlation_with_other=btc_eth_corr if other_alloc > 0 else 0.0,
            other_allocation_pct=other_alloc,
        )

        budget_remaining = available_budget - position_cost
        budget_ok = budget_remaining >= 0

        # 2. 스트레스 테스트
        proposed_positions = [
            {
                "symbol": symbol,
                "allocation_pct": allocation,
                "leverage": leverage,
                "direction": decision_dir,
            }
        ]
        # 기존 포지션도 포함
        for op in other_positions:
            proposed_positions.append(op)

        stress = self.run_stress_test(
            positions=proposed_positions,
            account_balance_usd=account_balance_usd,
        )

        # 3. 포트폴리오 변동성 경고
        port_vol = 0.0
        scale_down = 1.0
        if portfolio_risk:
            pr = portfolio_risk.get("portfolio_risk") or {}
            port_vol = pr.get("portfolio_vol_pct", 0.0)
            scale_down = pr.get("scale_down_factor", 1.0)

        # 4. 최종 결정
        final_scale = 1.0
        veto_reasons = []

        if not budget_ok:
            # 예산 초과 → 예산에 맞게 축소
            if available_budget > 0 and position_cost > 0:
                final_scale = min(final_scale, available_budget / position_cost)
            veto_reasons.append(
                f"예산 초과: 필요={position_cost:.0f}pts / 가용={available_budget:.0f}pts"
            )

        if not stress["passed"]:
            stress_scale = (
                _MAX_TOLERABLE_DRAWDOWN_PCT / abs(stress["worst_loss_pct"])
                if abs(stress["worst_loss_pct"]) > 0
                else 1.0
            )
            final_scale = min(final_scale, stress_scale)
            veto_reasons.append(stress["recommendation"])

        if scale_down < 1.0:
            final_scale = min(final_scale, scale_down)
            veto_reasons.append(
                f"포트폴리오 변동성 {port_vol:.2f}%/day → {scale_down*100:.0f}% 축소"
            )

        # 최종 사이즈 적용
        if final_scale < 0.5:
            # 50% 미만으로 축소해야 하면 VETO
            result["decision"] = "HOLD"
            result["allocation_pct"] = 0
            veto_applied = True
            reason_str = "VETO: " + " | ".join(veto_reasons)
        elif final_scale < 1.0:
            new_alloc = round(allocation * final_scale, 2)
            result["allocation_pct"] = new_alloc
            veto_applied = False
            reason_str = f"SIZE DOWN to {new_alloc}% ({final_scale*100:.0f}%): " + " | ".join(veto_reasons)
        else:
            veto_applied = False
            reason_str = "PASS: 모든 리스크 체크 통과"

        result["risk_budget_check"] = {
            "passed": not veto_applied,
            "available_budget": available_budget,
            "position_cost": position_cost,
            "budget_remaining": budget_remaining,
            "stress_test": stress,
            "portfolio_vol_pct": port_vol,
            "final_scale_factor": round(final_scale, 4),
            "veto_reasons": veto_reasons,
            "reason": reason_str,
        }

        logger.info(
            f"[RiskBudget] {symbol} {decision_dir} alloc={allocation}% lev={leverage}x "
            f"→ scale={final_scale:.2f} passed={not veto_applied} "
            f"regime={regime}"
        )

        return result

    # ── Judge용 요약 ──────────────────────────────────────────────────────────

    def format_for_judge(
        self,
        regime: str,
        stress_result: Optional[Dict] = None,
    ) -> str:
        """리스크 예산 상황을 Judge에게 전달할 문자열로 포맷."""
        budget = self.get_available_budget(regime)
        lines = [
            f"[RISK BUDGET] Regime={regime}  Available Budget={budget:.0f}/100pts",
            "",
            "Stress Test Scenarios (worst-case drawdowns):",
        ]
        for name, data in _STRESS_SCENARIOS.items():
            desc = data.get("description", name)
            btc_drop = data.get("BTCUSDT", 0)
            eth_drop = data.get("ETHUSDT", 0)
            lines.append(f"  {name:<25} BTC:{btc_drop:.0f}%  ETH:{eth_drop:.0f}%  | {desc}")

        if stress_result:
            lines.append(
                f"\nResult: {stress_result.get('recommendation', 'N/A')}"
            )
        lines.append(
            f"\nMax Tolerable Drawdown: {_MAX_TOLERABLE_DRAWDOWN_PCT}% of account"
        )
        return "\n".join(lines)


# ── Singleton ──────────────────────────────────────────────────────────────────
risk_budget_controller = RiskBudgetController()
