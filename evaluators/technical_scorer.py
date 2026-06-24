import pandas as pd
import numpy as np
import pandas_ta_classic as ta
from loguru import logger

class TechnicalScorer:
    """
    Stage 2: Alpha Confluence Engine
    단순 Z-Score를 넘어, 기관 수준의 4가지 독립 팩터를 계산하여 0~100점의 Alpha Score를 산출합니다.
    """

    @staticmethod
    def calculate_scores(price_df: pd.DataFrame) -> dict:
        """
        OHLCV 데이터프레임을 받아 다중 팩터 Alpha Score를 계산합니다.
        """
        if price_df is None or price_df.empty or len(price_df) < 200: # 200MA 계산을 위해 최소 200일 필요
            return {"error": "Not enough data (minimum 200 candles required)"}
            
        try:
            # 원본 데이터 보호
            df = price_df.copy()
            close = df['close']
            volume = df['volume']
            current_price = close.iloc[-1]
            
            # --- FACTOR 1: Price Disparity Z-Score (과매도 지표) ---
            ma20 = close.rolling(window=20).mean()
            disparity = (close - ma20) / ma20 * 100
            hist_mean = disparity.mean()
            hist_std = disparity.std()
            z_score = 0 if pd.isna(hist_std) or hist_std == 0 else (disparity.iloc[-1] - hist_mean) / hist_std
            
            # --- FACTOR 2: Volume Anomaly (개미 털기 / 항복 시그널) ---
            # 오늘 거래량이 최근 20일 평균의 몇 배인가?
            vol_ma20 = volume.rolling(window=20).mean()
            current_vol_ratio = volume.iloc[-1] / vol_ma20.iloc[-1] if vol_ma20.iloc[-1] > 0 else 1
            
            # --- FACTOR 3: Trend Alignment (장기 추세) ---
            # 하락장이 아닌, 200일선 위에서 노는 '우상향 종목'의 단기 눌림목인가?
            ma200 = close.rolling(window=200).mean()
            is_macro_uptrend = current_price > ma200.iloc[-1]
            
            # --- FACTOR 4: RSI (상대강도지수) ---
            df.ta.rsi(length=14, append=True)
            current_rsi = df['RSI_14'].iloc[-1] if 'RSI_14' in df.columns else 50
            
            # ==========================================
            # SCORING ENGINE (Alpha Score: 0 ~ 100)
            # ==========================================
            alpha_score = 0
            reasons = []
            
            # 1. 과매도 점수 (최대 40점)
            # Z-Score가 -2.0 이하면 극단적 과매도 (가장 중요)
            if z_score <= -2.5:
                alpha_score += 40
                reasons.append("극단적 통계적 투매 (Z-Score <= -2.5)")
            elif z_score <= -1.5:
                alpha_score += 20
                reasons.append("단기 과매도 진입 (Z-Score <= -1.5)")
                
            # 2. RSI 바닥 확인 (최대 20점)
            if current_rsi < 30:
                alpha_score += 20
                reasons.append("RSI 30 미만 (기술적 과매도)")
            elif current_rsi < 40:
                alpha_score += 10
                
            # 3. 항복 거래량 (최대 20점)
            # 주가가 빠지는데 거래량이 3배 이상 터졌다면 누군가(기관)가 밑에서 다 받아먹었다는 뜻
            if z_score < 0 and current_vol_ratio >= 3.0:
                alpha_score += 20
                reasons.append(f"패닉 셀 물량 흡수 (거래량 {current_vol_ratio:.1f}배 폭발)")
            elif z_score < 0 and current_vol_ratio >= 1.5:
                alpha_score += 10
                
            # 4. 장기 추세 정렬 (최대 20점)
            # 200일선 위에 있는데 단기로 빠진 거면 아주 좋은 매수 찬스
            if is_macro_uptrend and z_score < 0:
                alpha_score += 20
                reasons.append("장기 상승 추세 속 단기 눌림목 (Price > 200MA)")
                
            # ==========================================
            
            return {
                "current_price": float(current_price),
                "disparity_z_score": float(z_score),
                "momentum_6m_pct": float((current_price / close.iloc[-130] - 1) * 100) if len(close) >= 130 else 0.0,
                "volume_ratio": float(current_vol_ratio),
                "rsi_14": float(current_rsi),
                "alpha_score": int(alpha_score),
                "alpha_reasons": reasons,
                "trend": "Strong Buy" if alpha_score >= 80 else ("Watch" if alpha_score >= 50 else "Neutral/Avoid")
            }
            
        except Exception as e:
            logger.error(f"[TechnicalScorer] Error during calculation: {e}")
            return {"error": str(e)}
