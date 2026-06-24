from typing import Dict, Any
from loguru import logger

class FundamentalGate:
    """
    Stage 1: Fundamental Gate (Pass/Fail)
    재무 데이터를 기반으로 상장폐지 위험군이나 심각한 밸류트랩을 걸러내는 문지기 역할을 합니다.
    """

    @staticmethod
    def evaluate(fundamentals: Dict[str, Any]) -> bool:
        """
        주어진 재무 데이터가 최소한의 안전 기준을 통과하는지 평가합니다.
        
        Args:
            fundamentals: fetch_fundamentals()에서 반환된 딕셔너리
            
        Returns:
            bool: 통과(True), 탈락(False)
        """
        try:
            total_assets = fundamentals.get("total_assets", 0) or 0
            total_liabilities = fundamentals.get("total_liabilities", 0) or 0
            net_income = fundamentals.get("net_income", 0) or 0
            operating_cash_flow = fundamentals.get("operating_cash_flow", 0) or 0
            
            # 기준 1: 자본잠식(부채가 자산보다 많음) 제외
            if total_assets <= total_liabilities:
                return False
                
            # 기준 2: 당기순이익이 0보다 작고(적자), 영업현금흐름도 0보다 작으면(현금 유출) 위험군으로 간주
            if net_income < 0 and operating_cash_flow < 0:
                return False
                
            # 기준 3: 데이터 누락(자산이 0)인 종목 제외
            if total_assets <= 0:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"[FundamentalGate] Error during evaluation: {e}")
            return False
