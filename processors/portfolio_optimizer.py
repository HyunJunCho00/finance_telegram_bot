# -*- coding: utf-8 -*-
"""Layer 2: Portfolio Optimizer.

Institutional quant 방식:
  BTC와 ETH를 독립 자산으로 취급하지 않는다.
  공분산 행렬 기반으로 실제 포트폴리오 리스크를 계산하고
  상관관계를 반영한 최적 배분을 제안한다.

핵심 문제:
  현재 봇이 BTC 40% + ETH 40% = 80% 배분이라고 생각하지만,
  BTC-ETH 상관계수 ≈ 0.9 이면 실제 리스크는 약 72% 단방향 베팅과 같다.

수식:
  σ_p² = w_BTC² × σ_BTC² + w_ETH² × σ_ETH² + 2 × w_BTC × w_ETH × σ_BTC × σ_ETH × ρ
  effective_risk_multiplier = σ_p / σ_independent
  (독립 가정 대비 실제 리스크 배율)
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple
from loguru import logger


# 기본 변동성 (일별 %, 역사적 평균)
_DEFAULT_DAILY_VOL: Dict[str, float] = {
    "BTCUSDT": 3.5,   # BTC 일별 변동성 약 3.5%
    "ETHUSDT": 4.5,   # ETH 일별 변동성 약 4.5%
}

# 기본 BTC-ETH 상관계수 (역사적 평균, 2020-2025)
_DEFAULT_BTC_ETH_CORRELATION = 0.87

# 최대 허용 포트폴리오 변동성 (일별 %)
_MAX_PORTFOLIO_DAILY_VOL_PCT = 5.0

# 최소 독립성 임계치 (이 이하면 ETH를 독립 자산으로 취급)
_MIN_CORRELATION_FOR_ADJUSTMENT = 0.5


class PortfolioOptimizer:
    """BTC+ETH 포트폴리오의 실제 리스크 계산 및 배분 최적화."""

    # ── 상관계수 계산 ─────────────────────────────────────────────────────────

    def compute_correlation(
        self,
        symbol_a: str = "BTCUSDT",
        symbol_b: str = "ETHUSDT",
        lookback_days: int = 30,
    ) -> float:
        """DB에서 최근 N일 일별 수익률로 BTC-ETH 상관계수 계산."""
        try:
            from config.database import db
            limit_candles = lookback_days * 1440  # 1분봉 기준

            df_a = db.get_latest_market_data(
                symbol_a, limit=limit_candles, columns="timestamp,close"
            )
            df_b = db.get_latest_market_data(
                symbol_b, limit=limit_candles, columns="timestamp,close"
            )

            if df_a.empty or df_b.empty or len(df_a) < 10 or len(df_b) < 10:
                return _DEFAULT_BTC_ETH_CORRELATION

            # 일별 리샘플링
            import pandas as pd
            df_a = df_a.set_index("timestamp").resample("1D")["close"].last().dropna()
            df_b = df_b.set_index("timestamp").resample("1D")["close"].last().dropna()

            # 일별 수익률
            ret_a = df_a.pct_change().dropna()
            ret_b = df_b.pct_change().dropna()

            # 인덱스 맞추기
            common_idx = ret_a.index.intersection(ret_b.index)
            if len(common_idx) < 5:
                return _DEFAULT_BTC_ETH_CORRELATION

            ret_a = ret_a.loc[common_idx]
            ret_b = ret_b.loc[common_idx]

            corr = float(ret_a.corr(ret_b))
            if math.isnan(corr):
                return _DEFAULT_BTC_ETH_CORRELATION

            logger.info(
                f"[PortfolioOpt] Computed {symbol_a}/{symbol_b} "
                f"correlation={corr:.3f} (n={len(common_idx)}days)"
            )
            return round(corr, 4)

        except Exception as e:
            logger.warning(f"[PortfolioOpt] correlation compute failed: {e}")
            return _DEFAULT_BTC_ETH_CORRELATION

    # ── 변동성 계산 ───────────────────────────────────────────────────────────

    def compute_volatility(
        self,
        symbol: str,
        lookback_days: int = 30,
    ) -> float:
        """최근 N일 일별 수익률의 표준편차 (%)."""
        try:
            from config.database import db
            limit_candles = lookback_days * 1440

            df = db.get_latest_market_data(
                symbol, limit=limit_candles, columns="timestamp,close"
            )
            if df.empty or len(df) < 10:
                return _DEFAULT_DAILY_VOL.get(symbol, 4.0)

            import pandas as pd
            df = df.set_index("timestamp").resample("1D")["close"].last().dropna()
            returns = df.pct_change().dropna() * 100  # %로 변환

            if len(returns) < 5:
                return _DEFAULT_DAILY_VOL.get(symbol, 4.0)

            vol = float(returns.std())
            return round(vol, 4) if not math.isnan(vol) else _DEFAULT_DAILY_VOL.get(symbol, 4.0)

        except Exception as e:
            logger.warning(f"[PortfolioOpt] vol compute failed for {symbol}: {e}")
            return _DEFAULT_DAILY_VOL.get(symbol, 4.0)

    # ── 포트폴리오 리스크 계산 ────────────────────────────────────────────────

    def calculate_portfolio_risk(
        self,
        btc_allocation: float,     # 0~1 (예: 0.4 = 40%)
        eth_allocation: float,     # 0~1
        btc_vol: Optional[float] = None,
        eth_vol: Optional[float] = None,
        correlation: Optional[float] = None,
    ) -> Dict:
        """포트폴리오 실제 변동성 및 리스크 배율 계산.

        Returns:
            {
                "portfolio_vol_pct": float,        실제 포트폴리오 일별 변동성 (%)
                "independent_vol_pct": float,      독립 가정 시 변동성 (%)
                "effective_risk_multiplier": float, 독립 대비 실제 리스크 배율
                "correlation": float,
                "btc_vol": float,
                "eth_vol": float,
                "risk_level": str,                 LOW/MEDIUM/HIGH/CRITICAL
                "scale_down_factor": float,        권장 사이즈 조정 계수 (0~1)
                "warning": str or None
            }
        """
        w_btc = float(btc_allocation)
        w_eth = float(eth_allocation)

        if btc_vol is None:
            btc_vol = _DEFAULT_DAILY_VOL["BTCUSDT"]
        if eth_vol is None:
            eth_vol = _DEFAULT_DAILY_VOL["ETHUSDT"]
        if correlation is None:
            correlation = _DEFAULT_BTC_ETH_CORRELATION

        # 포트폴리오 분산
        # σ_p² = w_BTC² × σ_BTC² + w_ETH² × σ_ETH² + 2 × w_BTC × w_ETH × σ_BTC × σ_ETH × ρ
        var_p = (
            w_btc ** 2 * btc_vol ** 2
            + w_eth ** 2 * eth_vol ** 2
            + 2 * w_btc * w_eth * btc_vol * eth_vol * correlation
        )
        portfolio_vol = math.sqrt(max(var_p, 0.0))

        # 독립 가정 시 분산 (ρ = 0)
        var_independent = w_btc ** 2 * btc_vol ** 2 + w_eth ** 2 * eth_vol ** 2
        independent_vol = math.sqrt(max(var_independent, 0.0))

        # 리스크 배율 (독립 대비 실제 리스크)
        effective_multiplier = (
            portfolio_vol / independent_vol if independent_vol > 1e-9 else 1.0
        )

        # 리스크 레벨 판단
        if portfolio_vol > _MAX_PORTFOLIO_DAILY_VOL_PCT * 1.5:
            risk_level = "CRITICAL"
        elif portfolio_vol > _MAX_PORTFOLIO_DAILY_VOL_PCT:
            risk_level = "HIGH"
        elif portfolio_vol > _MAX_PORTFOLIO_DAILY_VOL_PCT * 0.7:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # 권장 사이즈 조정 계수
        if portfolio_vol > 1e-9:
            scale_down = min(
                _MAX_PORTFOLIO_DAILY_VOL_PCT / portfolio_vol, 1.0
            )
        else:
            scale_down = 1.0

        warning = None
        if correlation > 0.8 and w_btc > 0.1 and w_eth > 0.1:
            warning = (
                f"BTC-ETH 상관계수 {correlation:.2f}로 매우 높음. "
                f"실제 리스크는 독립 가정의 {effective_multiplier:.2f}배. "
                f"권장 사이즈 축소: {scale_down*100:.0f}%로 조정."
            )

        return {
            "portfolio_vol_pct": round(portfolio_vol, 4),
            "independent_vol_pct": round(independent_vol, 4),
            "effective_risk_multiplier": round(effective_multiplier, 4),
            "correlation": round(correlation, 4),
            "btc_vol": round(btc_vol, 4),
            "eth_vol": round(eth_vol, 4),
            "risk_level": risk_level,
            "scale_down_factor": round(scale_down, 4),
            "warning": warning,
        }

    # ── 배분 최적화 ───────────────────────────────────────────────────────────

    def optimize_allocation(
        self,
        btc_alpha: float,         # BTC IC-가중 알파 점수 (-1 ~ +1)
        eth_alpha: float,         # ETH IC-가중 알파 점수 (-1 ~ +1)
        total_risk_budget: float, # Layer 3에서 허용한 전체 리스크 예산 (0~100)
        current_btc_alloc: float = 0.0,  # 현재 BTC 배분 (0~1)
        current_eth_alloc: float = 0.0,  # 현재 ETH 배분 (0~1)
        correlation: Optional[float] = None,
    ) -> Dict:
        """리스크 예산 내에서 알파 가중 최적 배분 계산.

        Returns:
            {
                "btc_recommended_pct": float,
                "eth_recommended_pct": float,
                "total_recommended_pct": float,
                "portfolio_risk": dict,
                "rationale": str
            }
        """
        if correlation is None:
            correlation = _DEFAULT_BTC_ETH_CORRELATION

        # 알파 점수로 상대 배분 비율 결정
        # 알파 점수가 높은 자산에 더 많이 배분
        btc_raw = max(0.0, btc_alpha)
        eth_raw = max(0.0, eth_alpha)
        total_raw = btc_raw + eth_raw

        # 예산에서 최대 배분 가능 비율 계산 (0~1)
        max_total = min(total_risk_budget / 100.0, 0.8)

        if total_raw < 1e-9:
            # 둘 다 신호 없으면 균등 분할
            btc_alloc = max_total * 0.5
            eth_alloc = max_total * 0.5
        else:
            btc_alloc = max_total * (btc_raw / total_raw)
            eth_alloc = max_total * (eth_raw / total_raw)

        # 상관관계 보정: 높은 상관계수면 두 배분 합계 축소
        if correlation > _MIN_CORRELATION_FOR_ADJUSTMENT:
            corr_penalty = (correlation - _MIN_CORRELATION_FOR_ADJUSTMENT) / 0.5
            reduction = 1.0 - (corr_penalty * 0.3)  # 최대 30% 축소
            btc_alloc *= reduction
            eth_alloc *= reduction

        # 포트폴리오 리스크 계산
        portfolio_risk = self.calculate_portfolio_risk(
            btc_allocation=btc_alloc,
            eth_allocation=eth_alloc,
            correlation=correlation,
        )

        # 최대 변동성 초과 시 추가 축소
        if portfolio_risk["scale_down_factor"] < 1.0:
            btc_alloc *= portfolio_risk["scale_down_factor"]
            eth_alloc *= portfolio_risk["scale_down_factor"]
            portfolio_risk = self.calculate_portfolio_risk(
                btc_allocation=btc_alloc,
                eth_allocation=eth_alloc,
                correlation=correlation,
            )

        rationale = (
            f"BTC alpha={btc_alpha:+.3f}, ETH alpha={eth_alpha:+.3f}. "
            f"Correlation={correlation:.2f}. "
            f"Portfolio vol={portfolio_risk['portfolio_vol_pct']:.2f}% "
            f"(risk_level={portfolio_risk['risk_level']}). "
        )
        if portfolio_risk.get("warning"):
            rationale += portfolio_risk["warning"]

        logger.info(
            f"[PortfolioOpt] btc={btc_alloc*100:.1f}% eth={eth_alloc*100:.1f}% "
            f"port_vol={portfolio_risk['portfolio_vol_pct']:.2f}% "
            f"corr={correlation:.2f}"
        )

        return {
            "btc_recommended_pct": round(btc_alloc * 100, 2),
            "eth_recommended_pct": round(eth_alloc * 100, 2),
            "total_recommended_pct": round((btc_alloc + eth_alloc) * 100, 2),
            "portfolio_risk": portfolio_risk,
            "rationale": rationale,
        }

    # ── 통합 분석 (오케스트레이터에서 호출) ──────────────────────────────────

    def analyze(
        self,
        current_symbol: str,
        current_allocation_pct: float,
        other_symbol_allocation_pct: float = 0.0,
        lookback_days: int = 30,
    ) -> Dict:
        """현재 포지션의 포트폴리오 레벨 리스크 분석.

        Args:
            current_symbol: 분석 대상 심볼 (예: "BTCUSDT")
            current_allocation_pct: 현재 심볼 배분 (0~100)
            other_symbol_allocation_pct: 다른 심볼의 현재 배분 (0~100)
        """
        symbols = ["BTCUSDT", "ETHUSDT"]
        other_symbol = [s for s in symbols if s != current_symbol]
        other = other_symbol[0] if other_symbol else "ETHUSDT"

        # 변동성 및 상관계수 계산 (캐시 없이 매번 계산 — 4h 사이클에서 부담 없음)
        btc_vol = self.compute_volatility("BTCUSDT", lookback_days=lookback_days)
        eth_vol = self.compute_volatility("ETHUSDT", lookback_days=lookback_days)
        correlation = self.compute_correlation(
            "BTCUSDT", "ETHUSDT", lookback_days=lookback_days
        )

        # 현재 배분을 비율로 변환
        if current_symbol == "BTCUSDT":
            btc_alloc = current_allocation_pct / 100.0
            eth_alloc = other_symbol_allocation_pct / 100.0
        else:
            eth_alloc = current_allocation_pct / 100.0
            btc_alloc = other_symbol_allocation_pct / 100.0

        portfolio_risk = self.calculate_portfolio_risk(
            btc_allocation=btc_alloc,
            eth_allocation=eth_alloc,
            btc_vol=btc_vol,
            eth_vol=eth_vol,
            correlation=correlation,
        )

        return {
            "symbol": current_symbol,
            "btc_vol_pct": btc_vol,
            "eth_vol_pct": eth_vol,
            "btc_eth_correlation": correlation,
            "portfolio_risk": portfolio_risk,
            "recommendation": (
                "현재 배분 유지 가능" if portfolio_risk["risk_level"] in ["LOW", "MEDIUM"]
                else f"배분 {portfolio_risk['scale_down_factor']*100:.0f}%로 축소 권장"
            ),
        }

    # ── Judge용 요약 ──────────────────────────────────────────────────────────

    def format_for_judge(self, analysis: Dict) -> str:
        """포트폴리오 최적화 결과를 Judge에게 전달할 문자열로 포맷."""
        pr = analysis.get("portfolio_risk") or {}
        lines = [
            "[PORTFOLIO RISK ANALYSIS]",
            f"BTC vol={analysis.get('btc_vol_pct', 0):.2f}%/day  "
            f"ETH vol={analysis.get('eth_vol_pct', 0):.2f}%/day",
            f"BTC-ETH Correlation: {analysis.get('btc_eth_correlation', 0):.3f}",
            f"Portfolio Vol: {pr.get('portfolio_vol_pct', 0):.2f}%/day "
            f"(vs Independent: {pr.get('independent_vol_pct', 0):.2f}%)",
            f"Effective Risk Multiplier: {pr.get('effective_risk_multiplier', 1):.2f}x",
            f"Risk Level: {pr.get('risk_level', 'UNKNOWN')}",
            f"Recommended Scale: {pr.get('scale_down_factor', 1)*100:.0f}%",
        ]
        if pr.get("warning"):
            lines.append(f"WARNING: {pr['warning']}")
        lines.append(f"Action: {analysis.get('recommendation', 'N/A')}")
        return "\n".join(lines)


# ── Singleton ──────────────────────────────────────────────────────────────────
portfolio_optimizer = PortfolioOptimizer()
