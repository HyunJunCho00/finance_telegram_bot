from loguru import logger
from config.local_state import state_manager
from executors.trade_executor import trade_executor
from config.settings import settings
import time

class ExecutionDesk:
    """The Stateful Order Manager.
    Processes PENDING and ACTIVE intents from local_state.db based on Execution Styles.
    """
    
    def __init__(self):
        self.dca_steps = 3
        
    def process_intents(self):
        """Called periodically by the scheduler/orchestrator to advance active orders."""
        active_orders = state_manager.get_active_orders()
        
        for order in active_orders:
            intent_id = order['intent_id']
            symbol = order['symbol']
            direction = order['direction']
            style = order['execution_style']
            remaining = order['remaining_amount']
            exchange = order['exchange']
            status = order['status']
            
            if status == 'PENDING':
                state_manager.update_status(intent_id, 'ACTIVE')
                status = 'ACTIVE'
                
            if status != 'ACTIVE':
                continue
                
            logger.info(f"ExecutionDesk processing {style} for {symbol} on {exchange}. Remaining: {remaining}")
            
            try:
                if style == "MOMENTUM_SNIPER":
                    # Sweep 100% immediately.
                    self._execute_chunk(intent_id, symbol, direction, remaining, exchange, style)
                    
                elif style == "SMART_DCA":
                    # Retail DCA: 33% chunk. (In reality, we'd check CVD here).
                    chunk_size = order['total_target_amount'] / self.dca_steps
                    actual_chunk = min(chunk_size, remaining)
                    self._execute_chunk(intent_id, symbol, direction, actual_chunk, exchange, style)
                    
                elif style == "PASSIVE_MAKER":
                    # For paper trading, wait for price to dip then execute (we simulate execution here simply for now)
                    # In V7 simulation realism, this gets harder.
                    self._execute_chunk(intent_id, symbol, direction, remaining, exchange, style)
                    
                elif style == "CASINO_EXIT":
                    # Panic exit everything currently open.
                    self._execute_chunk(intent_id, symbol, direction, remaining, exchange, style)
                    
            except Exception as e:
                logger.error(f"Error processing intent {intent_id}: {e}")
                
        # Run the TTL flusher to prevent memory bloat
        state_manager.flush_expired()
        
    def _execute_chunk(self, intent_id: str, symbol: str, direction: str, amount: float, exchange: str, style: str):
        if amount <= 0: return
        
        # Fire the actual execution directly to exchange or paper engine
        res = trade_executor.execute(
            symbol=symbol,
            side=direction,
            amount=amount,
            leverage=1, # Leverage constraints were applied by CRO sizing
            exchange=exchange,
            style=style
        )
        
        if res.get('success'):
            state_manager.update_order_fill(intent_id, amount)
            logger.info(f"ExecutionDesk filled {amount} of {intent_id} via {exchange}")
        else:
            logger.error(f"ExecutionDesk failed to fill: {res.get('error')}")

# Global instance
execution_desk = ExecutionDesk()
