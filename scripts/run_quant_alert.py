import sys
import os
import time
from loguru import logger
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_sources.us_stock_source import USStockCollector
from data_sources.kr_stock_source import KRStockCollector
from data_sources.crypto_source import CryptoCollector
from evaluators.fundamental_gate import FundamentalGate
from evaluators.technical_scorer import TechnicalScorer

# Telegram Outbox 연동
from executors.execution_repository import execution_repository
from executors.outbox_dispatcher import outbox_dispatcher

# 알람 기준 (Alpha Score: 0~100)
ALPHA_SCORE_THRESHOLD = 80

# ----------------- 추가된 알라딘 아키텍처 연동 -----------------
from agents.risk_manager_agent import risk_manager_agent
from config.settings import TradingMode
import json
# ----------------------------------------------------------------

def run_multi_market_screener():
    logger.info("Starting Multi-Market Quant Screener (Alpha Confluence Engine)")
    
    collectors = {
        "US_STOCK": USStockCollector(),
        "KR_STOCK": KRStockCollector(),
        "CRYPTO": CryptoCollector()
    }
    
    anomalies = []
    
    for market_name, collector in collectors.items():
        logger.info(f"========== Screening {market_name} ==========")
        universe = collector.get_screener_universe()
        
        for symbol in universe:
            # 1. Fundamental Gate (펀더멘털 통과 여부 확인)
            fundamentals = collector.fetch_fundamentals(symbol)
            if not FundamentalGate.evaluate(fundamentals):
                logger.info(f"[{market_name}] {symbol} - FAILED Fundamental Gate. Skipping.")
                continue
                
            # 2. Alpha Confluence Engine (다중 팩터 점수 계산)
            price_df = collector.fetch_price_history(symbol, timeframe="1d", limit=1000)
            scores = TechnicalScorer.calculate_scores(price_df)
            
            if "error" in scores:
                logger.warning(f"[{market_name}] {symbol} - {scores['error']}")
                continue
                
            alpha_score = scores.get("alpha_score", 0)
            logger.info(f"[{market_name}] {symbol} - Alpha Score: {alpha_score}/100")
            
            # 다중 팩터 알파 모델의 임계점(80점) 돌파 시 리스트에 추가
            if alpha_score >= ALPHA_SCORE_THRESHOLD:
                anomalies.append({
                    "market": market_name,
                    "symbol": symbol,
                    "price": scores["current_price"],
                    "alpha_score": alpha_score,
                    "z_score": scores["disparity_z_score"],
                    "rsi": scores["rsi_14"],
                    "vol_ratio": scores["volume_ratio"],
                    "reasons": scores["alpha_reasons"]
                })
                
    return anomalies

def send_telegram_alert_and_execute(anomalies: list):
    """
    발견된 퀀트 픽을 PM(Judge) 가초안으로 변환하여
    CRO(RiskManager)의 결재를 받은 후 텔레그램 및 매매 큐에 전송합니다.
    """
    if not anomalies:
        logger.info("No Alpha targets found today. Silence is golden.")
        return
        
    logger.info(f"Found {len(anomalies)} Alpha targets. Sending to Risk Manager (CRO).")
    
    lines = ["🚨 **[Quant Alpha Alert] 강력 매수 후보 포착** 🚨", ""]
    lines.append(f"다중 팩터 교집합을 통해 Alpha Score {ALPHA_SCORE_THRESHOLD}점 이상 종목 포착.\n")
    
    for item in anomalies:
        symbol = item['symbol']
        market = item['market']
        
        # 1. PM 가초안(Draft) 생성
        draft_decision = {
            "decision": "LONG",
            "allocation_pct": 10,
            "leverage": 1,
            "target_exchange": "BINANCE" if market == "CRYPTO" else "UPBIT",
            "recommended_execution_style": "SMART_DCA",
            "symbol": symbol
        }
        
        # 2. CRO(Risk Manager) 심사 호출
        # 독립 스크립트 특성상, 거시경제 컨텍스트는 현재 DB에서 가져오거나 기본값 전달
        # (향후 scheduler.py 연동 시 db.get_market_context() 등을 연결 가능)
        logger.info(f"[{symbol}] Submitting draft to Risk Manager...")
        final_decision = risk_manager_agent.evaluate_trade(
            draft_decision=draft_decision,
            funding_context="Funding data unavailable for Quant fallback",
            deribit_context="Volatility data unavailable for Quant fallback",
            mode=TradingMode.SWING
        )
        
        cro_status = final_decision.get('decision', 'HOLD')
        cro_note = final_decision.get('risk_manager_note', '')
        
        lines.append(f"🎯 **{symbol}** ({market})")
        lines.append(f"- **현재가:** {item['price']:.2f}")
        lines.append(f"- **Alpha Score:** 🔥 {item['alpha_score']}/100 점")
        lines.append(f"- **선정 사유 (Confluence):**")
        for reason in item['reasons']:
            lines.append(f"  ✅ {reason}")
            
        lines.append(f"\n👔 **[CRO Risk Review]**")
        if cro_status == 'LONG':
            lines.append(f"✅ **승인 (APPROVED)**")
            lines.append(f"- 승인 비중: {final_decision.get('allocation_pct', 0)}%")
            lines.append(f"- 레버리지: {final_decision.get('leverage', 1)}x")
            # 3. Execution (추후 연동)
            # LocalStateManager.add_intent(...) 호출 로직이 이곳에 들어갑니다.
        else:
            lines.append(f"❌ **거절 (VETOED - HOLD)**")
            
        if cro_note:
            lines.append(f"- CRO 코멘트: {cro_note}")
        lines.append("")
        
    message_text = "\n".join(lines)
    
    try:
        logger.info("Enqueueing finalized message to execution_outbox...")
        execution_repository.enqueue_outbox_event(
            event_type="telegram_message",
            payload={"text": message_text},
            idempotency_key=f"quant_alert_{int(time.time())}"
        )
        outbox_dispatcher.publish_pending(limit=10)
        logger.info("Telegram alert sent successfully!")
        
    except Exception as e:
        logger.error(f"Failed to send telegram alert: {e}")

if __name__ == "__main__":
    anomalies = run_multi_market_screener()
    send_telegram_alert_and_execute(anomalies)
