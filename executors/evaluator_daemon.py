import time
import ccxt
from loguru import logger
from config.settings import settings
from config.database import db
from executors.post_mortem import write_post_mortem

class EvaluatorDaemon:
    """
    V6 Self-Healing Loop:
    Monitors recent trade reports. If the current price hits the Stop Loss (SL)
    or Take Profit (TP) defined in the original AI Report, it "closes" the trade
    conceptually and triggers the LLM Post-Mortem pipeline.
    """
    def __init__(self):
        self.exchange = ccxt.binance()
        
    def get_current_price(self, symbol: str) -> float:
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return float(ticker['last'])
        except Exception as e:
            logger.error(f"Daemon price fetch error for {symbol}: {e}")
            return 0.0

    def evaluate_recent_trades(self):
        logger.info("Evaluator Daemon: Checking for completed trades to analyze...")
        
        # In a real environment, we'd query Supabase for all OPEN trades.
        # For this architecture MVP, we check the latest AI report.
        symbols = settings.get_active_symbols()
        
        for symbol in symbols:
            report = db.get_latest_report(symbol)
            if not report:
                continue
                
            decision = report.get('final_decision', {})
            if not isinstance(decision, dict):
                continue
                
            direction = decision.get('decision', 'HOLD')
            if direction in ['HOLD', 'CANCEL_AND_CLOSE']:
                continue
                
            # Check if this report has already been post-mortemed to avoid looping
            # (We would typically use a 'resolved' boolean column in Postgres).
            
            entry = decision.get('entry_price', 0)
            tp = decision.get('take_profit', 0)
            sl = decision.get('stop_loss', 0)
            
            if not tp or not sl or not entry:
                continue

            current_price = self.get_current_price(symbol)
            if current_price == 0:
                continue

            outcome = None
            if direction == 'LONG':
                if current_price >= tp:
                    outcome = "SUCCESS (TP Hit)"
                elif current_price <= sl:
                    outcome = "FAILURE (SL Hit)"
            elif direction == 'SHORT':
                if current_price <= tp:
                    outcome = "SUCCESS (TP Hit)"
                elif current_price >= sl:
                    outcome = "FAILURE (SL Hit)"

            if outcome:
                logger.info(f"[{symbol}] Trade Closed: {outcome}. Triggering LLM Post-Mortem.")
                
                # Mock a slightly modified report dict so post_mortem doesn't fail
                try:
                    write_post_mortem(report, current_price)
                except Exception as e:
                    logger.error(f"Failed to generate post-mortem: {e}")
                    
                # In a real scenario, we would mark this report as 'resolved' in DB.

if __name__ == "__main__":
    daemon = EvaluatorDaemon()
    daemon.evaluate_recent_trades()
