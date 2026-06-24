from abc import ABC, abstractmethod
from typing import Dict, Any, List
import pandas as pd

class BaseMarketCollector(ABC):
    """
    모든 시장(Crypto, US Stock, KR Stock)의 데이터 수집기가 공통으로 지켜야 하는 인터페이스.
    이 규격을 강제함으로써 파이프라인(Stage 1 & 2)이 출처와 무관하게 동일한 로직으로 작동하도록 합니다.
    """

    @abstractmethod
    def fetch_price_history(self, symbol: str, timeframe: str = "1d", limit: int = 1000) -> pd.DataFrame:
        """
        주어진 심볼의 역사적 가격 데이터(OHLCV)를 가져옵니다.
        
        Returns:
            pd.DataFrame: 반드시 다음의 컬럼을 포함해야 합니다.
            ['open', 'high', 'low', 'close', 'volume']
            인덱스는 datetime 형식이어야 합니다.
        """
        pass

    @abstractmethod
    def fetch_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """
        Stage 1 (Fundamental Gate)에서 사용할 거친 재무 데이터를 가져옵니다.
        
        Returns:
            Dict[str, Any]: 최소한 아래의 키값을 포함해야 합니다.
            - "market_cap" (시가총액)
            - "total_assets" (총자산)
            - "total_liabilities" (총부채)
            - "net_income" (당기순이익)
            - "operating_cash_flow" (영업현금흐름)
            (Crypto의 경우 해당 없는 필드는 None 또는 0으로 처리)
        """
        pass

    @abstractmethod
    def get_screener_universe(self) -> List[str]:
        """
        이 수집기가 담당하는 시장에서 스크리닝을 진행할 타겟 유니버스(종목 리스트)를 반환합니다.
        예: S&P 500 리스트, KOSPI 200 리스트, 바이낸스 선물 상장 코인 리스트 등.
        """
        pass
