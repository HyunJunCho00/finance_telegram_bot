from typing import Dict, Optional
import pandas as pd
from config.settings import settings
from config.database import db
from loguru import logger
from datetime import datetime, timezone


class VolatilityMonitor:
    def __init__(self):
        self.threshold = settings.VOLATILITY_THRESHOLD
        # BTC and ETH only
        self.symbols = ['BTCUSDT', 'ETHUSDT']

    def calculate_price_change(self, symbol: str) -> Optional[float]:
        try:
            df = db.get_latest_market_data(symbol, limit=2)
            if len(df) < 2:
                return None

            current_price = float(df.iloc[-1]['close'])
            previous_price = float(df.iloc[-2]['close'])
            return ((current_price - previous_price) / previous_price) * 100
        except Exception as e:
            logger.error(f"Price change calculation error for {symbol}: {e}")
            return None

    def check_volatility_spike(self) -> Dict:
        spike_detected = False
        spike_details = []

        for symbol in self.symbols:
            change_pct = self.calculate_price_change(symbol)
            if change_pct is not None and abs(change_pct) >= self.threshold:
                spike_detected = True
                spike_details.append({
                    'symbol': symbol,
                    'change_pct': change_pct,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
                logger.warning(
                    f"VOLATILITY SPIKE: {symbol} changed {change_pct:.2f}% "
                    f"(threshold: {self.threshold}%)"
                )

        return {'spike_detected': spike_detected, 'details': spike_details}

    def trigger_emergency_analysis(self, spike_details: list) -> None:
        from collectors.telegram_collector import telegram_collector
        from executors.orchestrator import orchestrator

        logger.critical(f"EMERGENCY ANALYSIS TRIGGERED: {len(spike_details)} spikes")
        telegram_collector.run(hours=1)
        for detail in spike_details:
            orchestrator.run_emergency_analysis(detail['symbol'])

    def run(self) -> None:
        result = self.check_volatility_spike()
        if result['spike_detected']:
            self.trigger_emergency_analysis(result['details'])


volatility_monitor = VolatilityMonitor()
