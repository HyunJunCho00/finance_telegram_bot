import yfinance as yf
import pandas as pd
from typing import Dict, Any, List
from loguru import logger
from .base import BaseMarketCollector

class USStockCollector(BaseMarketCollector):
    """
    미국 주식 시장(US Stock) 데이터 수집기
    - 가격 데이터: yfinance
    - 재무 데이터: yfinance
    - 종목 유니버스: S&P 500 (위키피디아 파싱)
    """

    def fetch_price_history(self, symbol: str, timeframe: str = "1d", limit: int = 1000) -> pd.DataFrame:
        """
        yfinance를 이용해 과거 주가 데이터(OHLCV)를 수집합니다.
        """
        logger.info(f"[USStock] Fetching price history for {symbol}")
        try:
            # yfinance는 interval 파라미터로 timeframe을 받습니다 (예: '1d', '1wk', '1mo')
            # 1000일 정도의 데이터를 원한다면 period="5y" 정도가 넉넉합니다 (거래일 기준).
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="5y", interval=timeframe)
            
            if df.empty:
                logger.warning(f"[USStock] No price data found for {symbol}")
                return pd.DataFrame()
                
            # 컬럼 이름 소문자화 (파이프라인 표준 규격)
            df.columns = [col.lower() for col in df.columns]
            
            # 멀티인덱스 제거 (yfinance 최신 버전 이슈 방지)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            # 필요한 컬럼만 추출
            expected_columns = ['open', 'high', 'low', 'close', 'volume']
            df = df[[col for col in expected_columns if col in df.columns]]
            
            # 최근 데이터만 limit 개수만큼 잘라서 반환
            return df.tail(limit)
            
        except Exception as e:
            logger.error(f"[USStock] Failed to fetch price for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """
        yfinance를 이용해 기초 재무 데이터를 수집합니다. (Stage 1 Fundamental Gate 용)
        """
        logger.info(f"[USStock] Fetching fundamentals for {symbol}")
        result = {
            "market_cap": 0,
            "total_assets": 0,
            "total_liabilities": 0,
            "net_income": 0,
            "operating_cash_flow": 0
        }
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # 1. 시가총액 (Market Cap)
            result["market_cap"] = info.get("marketCap", 0)
            
            # 2. 대차대조표 (Balance Sheet) - 연간 데이터
            try:
                bs = ticker.balance_sheet
                if not bs.empty:
                    latest_bs = bs.iloc[:, 0] # 가장 최근 연도
                    # yfinance 필드명은 계속 변동될 수 있어 안전하게 get
                    result["total_assets"] = latest_bs.get("Total Assets", 0)
                    # 부채는 Total Liab 또는 Total Liabilities Net Minority Interest 등으로 나옴
                    result["total_liabilities"] = latest_bs.get("Total Liabilities Net Minority Interest", latest_bs.get("Total Liab", 0))
            except Exception as e:
                logger.debug(f"[USStock] Balance sheet parsing error for {symbol}: {e}")

            # 3. 손익계산서 (Income Statement)
            try:
                financials = ticker.financials
                if not financials.empty:
                    latest_fin = financials.iloc[:, 0]
                    result["net_income"] = latest_fin.get("Net Income", 0)
            except Exception as e:
                logger.debug(f"[USStock] Income statement parsing error for {symbol}: {e}")

            # 4. 현금흐름표 (Cash Flow)
            try:
                cf = ticker.cashflow
                if not cf.empty:
                    latest_cf = cf.iloc[:, 0]
                    result["operating_cash_flow"] = latest_cf.get("Operating Cash Flow", 0)
            except Exception as e:
                logger.debug(f"[USStock] Cash flow parsing error for {symbol}: {e}")
                
            return result
            
        except Exception as e:
            logger.error(f"[USStock] Failed to fetch fundamentals for {symbol}: {e}")
            return result

    def get_screener_universe(self) -> List[str]:
        """
        미국 주식 스크리닝 대상: 핵심 5개 종목으로 압축 (Magnificent 5)
        """
        logger.info("[USStock] Fetching target universe (Top 5 Tech)")
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
