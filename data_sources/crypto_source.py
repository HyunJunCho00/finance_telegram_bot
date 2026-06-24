import yfinance as yf
import pandas as pd
from typing import Dict, Any, List
from loguru import logger
from .base import BaseMarketCollector

class CryptoCollector(BaseMarketCollector):
    """
    암호화폐(Crypto) 데이터 수집기
    - 가격 데이터: yfinance (일봉 단위 스크리닝용)
    - 종목 유니버스: BTC, ETH (시총 1, 2위)
    - 재무 데이터: 크립토는 전통적 재무제표가 없으므로 Pass용 Dummy 데이터 반환
    """

    def fetch_price_history(self, symbol: str, timeframe: str = "1d", limit: int = 1000) -> pd.DataFrame:
        """
        yfinance를 이용해 과거 일봉 데이터(OHLCV)를 수집합니다.
        (단타용 분봉/틱 데이터가 아니라, Z-Score 매크로 스크리닝을 위한 일봉 전용)
        """
        # yfinance에서는 크립토 심볼 뒤에 '-USD'를 붙여야 합니다.
        yf_symbol = f"{symbol}-USD" if not symbol.endswith("-USD") else symbol
        
        logger.info(f"[Crypto] Fetching price history for {yf_symbol}")
        try:
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period="5y", interval=timeframe)
            
            if df.empty:
                logger.warning(f"[Crypto] No price data found for {yf_symbol}")
                return pd.DataFrame()
                
            df.columns = [col.lower() for col in df.columns]
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            expected_columns = ['open', 'high', 'low', 'close', 'volume']
            df = df[[col for col in expected_columns if col in df.columns]]
            
            return df.tail(limit)
            
        except Exception as e:
            logger.error(f"[Crypto] Failed to fetch price for {yf_symbol}: {e}")
            return pd.DataFrame()

    def fetch_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """
        암호화폐는 대차대조표나 영업현금흐름이 존재하지 않으므로, 
        Fundamental Gate를 무조건 통과할 수 있도록 강제 Pass용 데이터를 반환합니다.
        """
        logger.info(f"[Crypto] Bypassing fundamentals for {symbol}")
        return {
            "market_cap": 1000000000,      # 통과 조건 충족을 위한 임의의 큰 값
            "total_assets": 100000,        # 자본잠식 아님(자산 > 부채)을 증명하기 위한 값
            "total_liabilities": 0,
            "net_income": 10000,           # 흑자
            "operating_cash_flow": 10000   # 양수 현금흐름
        }

    def get_screener_universe(self) -> List[str]:
        """
        크립토 스크리닝 대상: 시총 1, 2위 대장주
        """
        logger.info("[Crypto] Fetching target universe (BTC, ETH)")
        return ["BTC", "ETH"]
