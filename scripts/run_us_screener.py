import sys
import os
from loguru import logger
import pandas as pd

# Add root directory to python path so we can import from data_sources and evaluators
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_sources.us_stock_source import USStockCollector
from evaluators.fundamental_gate import FundamentalGate
from evaluators.technical_scorer import TechnicalScorer

def run_screener():
    logger.info("Starting US Stock Screening Pipeline (2-Stage Hybrid Architecture)")
    
    collector = USStockCollector()
    
    # 1. 대상 종목 가져오기 (S&P 500 중 빠른 테스트를 위해 상위 10개만 샘플링)
    # 실제 운영 시에는 universe = collector.get_screener_universe() 전체를 사용
    universe = collector.get_screener_universe()
    if not universe:
        logger.error("Failed to load universe. Exiting.")
        return
        
    test_symbols = universe[:10]  # 빠른 실행을 위해 10개만 테스트
    logger.info(f"Target Universe Size: {len(universe)} (Testing first {len(test_symbols)} symbols: {test_symbols})")
    
    results = []
    
    for symbol in test_symbols:
        logger.info(f"--- Processing {symbol} ---")
        
        # ----------------------------------------------------
        # STAGE 1: Fundamental Gate (무료 재무 데이터로 거르기)
        # ----------------------------------------------------
        fundamentals = collector.fetch_fundamentals(symbol)
        
        is_safe = FundamentalGate.evaluate(fundamentals)
        if not is_safe:
            logger.warning(f"[{symbol}] FAILED Fundamental Gate. Discarding.")
            continue
            
        logger.info(f"[{symbol}] PASSED Fundamental Gate.")
        
        # ----------------------------------------------------
        # STAGE 2: Technical & Z-Score Engine (정교한 가격 데이터로 랭킹)
        # ----------------------------------------------------
        price_df = collector.fetch_price_history(symbol, timeframe="1d", limit=1000)
        
        scores = TechnicalScorer.calculate_scores(price_df)
        if "error" in scores:
            logger.warning(f"[{symbol}] Failed to calculate technical scores: {scores['error']}")
            continue
            
        # 결과 저장
        results.append({
            "Symbol": symbol,
            "Price": round(scores["current_price"], 2),
            "Disparity_Z_Score": round(scores["disparity_z_score"], 2),
            "Momentum_6M(%)": round(scores["momentum_6m_pct"], 2),
            "Trend": scores["trend"]
        })
        
    # ----------------------------------------------------
    # 결과 출력 및 정렬
    # ----------------------------------------------------
    if not results:
        logger.info("No symbols passed the screening.")
        return
        
    df_results = pd.DataFrame(results)
    # Z-Score 기준으로 오름차순 정렬 (음수일수록 과거 자기자신 대비 과매도/저평가 상태)
    df_results = df_results.sort_values(by="Disparity_Z_Score", ascending=True).reset_index(drop=True)
    
    logger.info("\n=== Final Screening Results (Sorted by Oversold Z-Score) ===")
    print("\n" + df_results.to_string())
    logger.info("=========================================================")

if __name__ == "__main__":
    run_screener()
