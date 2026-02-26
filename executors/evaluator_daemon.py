import json
from loguru import logger
from config.settings import settings
from config.database import db
from executors.post_mortem import write_post_mortem
from executors.metrics_logger import metrics_logger
class EvaluatorDaemon:
    """
    V6 Self-Healing Loop:
    Monitors recent trade reports. If the current price hits the Stop Loss (SL)
    or Take Profit (TP) defined in the original AI Report, it "closes" the trade
    conceptually and triggers the LLM Post-Mortem pipeline.
    """
    def __init__(self):
        # [FIX] Reuse authenticated CCXT instance from trade_executor
        from executors.trade_executor import trade_executor
        self.exchange = trade_executor.binance
        self._evaluated_report_ids = set()  # Avoid re-evaluating same report

    def get_current_price(self, symbol: str) -> float:
        try:
            # [FIX] CCXT requires slash format: BTCUSDT → BTC/USDT
            ccxt_symbol = symbol
            if 'USDT' in symbol and '/' not in symbol:
                ccxt_symbol = symbol.replace('USDT', '/USDT')
            ticker = self.exchange.fetch_ticker(ccxt_symbol)
            return float(ticker['last'])
        except Exception as e:
            logger.error(f"Daemon price fetch error for {symbol}: {e}")
            return 0.0

    def evaluate_recent_trades(self):
        logger.info("Evaluator Daemon: Checking for completed trades to analyze...")

        # [FIX] settings.get_active_symbols() doesn't exist → use trading_symbols
        symbols = settings.trading_symbols

        for symbol in symbols:
            report = db.get_latest_report(symbol)
            if not report:
                continue

            # Guard: skip if already evaluated this exact report
            report_id = report.get('id') or report.get('created_at', '')
            if report_id in self._evaluated_report_ids:
                continue

            decision = report.get('final_decision', {})
            if isinstance(decision, str):
                try:
                    decision = json.loads(decision)
                except Exception:
                    continue
            if not isinstance(decision, dict):
                continue

            direction = decision.get('decision', 'HOLD')
            if direction in ['HOLD', 'CANCEL_AND_CLOSE']:
                continue

            entry = float(decision.get('entry_price', 0) or 0)
            tp = float(decision.get('take_profit', 0) or 0)
            sl = float(decision.get('stop_loss', 0) or 0)

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
                self._evaluated_report_ids.add(report_id)

                pnl_pct = 0.0
                if direction == 'LONG':
                    pnl_pct = ((current_price - entry) / entry) * 100
                elif direction == 'SHORT':
                    pnl_pct = ((entry - current_price) / entry) * 100

                mistake_summary = ""
                try:
                    if "FAILURE" in outcome:
                        post_mortem_result = write_post_mortem(report, current_price)
                        if post_mortem_result and isinstance(post_mortem_result, dict):
                            mistake_summary = post_mortem_result.get("mistake_summary", "")
                    else:
                        # Even if success, write post mortem if configured, else just skip
                        pass
                except Exception as e:
                    logger.error(f"Failed to generate post-mortem: {e}")

                # Log resolution for academic/quantitative evaluation
                try:
                    metrics_logger.log_resolution(
                        symbol=symbol,
                        direction=direction,
                        outcome=outcome,
                        pnl_pct=round(pnl_pct, 2),
                        mistake_summary=mistake_summary
                    )
                except Exception as e:
                    logger.error(f"Failed to log resolution metric: {e}")


if __name__ == "__main__":
    daemon = EvaluatorDaemon()
    daemon.evaluate_recent_trades()
