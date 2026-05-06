import ccxt
import time
from loguru import logger
from config.settings import settings
from config.database import db
import json

def check_and_update_stop_to_be():
    logger.info("Running Stop to Break-Even (BE) Monitor...")
    from executors.trade_executor import trade_executor
    binance = trade_executor.binance

    try:
        positions = binance.fetch_positions()
        open_positions = [p for p in positions if float(p.get('contracts', p.get('info', {}).get('positionAmt', 0))) != 0]
        
        if not open_positions:
            logger.info("No open positions. Exiting.")
            return

        for pos in open_positions:
            symbol = pos['symbol']
            pos_amt = float(pos.get('contracts', pos.get('info', {}).get('positionAmt', 0)))
            side = 'LONG' if pos_amt > 0 else 'SHORT'
            entry_price = float(pos.get('entryPrice', pos.get('info', {}).get('entryPrice', 0)))
            
            report = db.get_latest_report(symbol.replace("/", ""))
            if not report:
                continue
                
            decision = report.get('final_decision', {})
            if isinstance(decision, str):
                try:
                    decision = json.loads(decision)
                except Exception:
                    pass
            
            if not isinstance(decision, dict):
                continue
                
            tp1_price = float(decision.get('tp1_price', decision.get('take_profit', 0)))
            
            tp1_hit = False
            current_price = float(pos.get('markPrice', pos.get('info', {}).get('markPrice', 0)))
            if current_price == 0:
                ticker = binance.fetch_ticker(symbol)
                current_price = float(ticker['last'])
                
            if side == 'LONG' and current_price >= tp1_price and tp1_price > 0:
                tp1_hit = True
            elif side == 'SHORT' and current_price <= tp1_price and tp1_price > 0:
                tp1_hit = True
                
            if tp1_hit:
                logger.info(f"[{symbol}] TP1 ({tp1_price}) seems hit or crossed. Checking SL to BE ({entry_price}).")
                
                open_orders = binance.fetch_open_orders(symbol)
                sl_orders = [o for o in open_orders if o['type'] == 'stop_market' or o['type'] == 'STOP_MARKET']
                
                for sl_order in sl_orders:
                    stop_price = float(sl_order.get('stopPrice', sl_order.get('info', {}).get('stopPrice', 0)))
                    
                    if abs(stop_price - entry_price) > (entry_price * 0.002):
                        logger.info(f"[{symbol}] Cancelling old SL at {stop_price} and moving to BE {entry_price}")
                        binance.cancel_order(sl_order['id'], symbol)
                        
                        close_side = 'sell' if side == 'LONG' else 'buy'
                        try:
                            binance.create_order(
                                symbol=symbol,
                                type='STOP_MARKET',
                                side=close_side,
                                amount=abs(pos_amt),
                                params={'stopPrice': entry_price, 'reduceOnly': True}
                            )
                            logger.info(f"[{symbol}] Successfully moved SL to BE for {symbol} at {entry_price}")
                        except Exception as e:
                            logger.error(f"[{symbol}] Failed to create new SL order: {e}")

    except Exception as e:
        logger.error(f"Error in Stop to BE Monitor: {e}")

if __name__ == "__main__":
    check_and_update_stop_to_be()
