#!/bin/bash

set -euo pipefail

INSTALL_DIR="/opt/crypto_trading_system"

echo "=== Cold Start Data Seeding ==="
echo "Supabase schema.sql 먼저 실행했는지 확인하세요!"
echo ""

cd "$INSTALL_DIR"
source venv/bin/activate

echo "[1/6] Fear & Greed Index..."
python3 -c "
from collectors.fear_greed_collector import fear_greed_collector
fear_greed_collector.run()
print('  → OK')
"

echo "[2/6] Price & CVD (Binance + Upbit)..."
python3 -c "
from collectors.price_collector import collector
collector.run()
print('  → OK')
"

echo "[3/6] Funding Rate & Global OI..."
python3 -c "
from collectors.funding_collector import funding_collector
funding_collector.run()
print('  → OK')
"

echo "[4/6] Microstructure (Orderbook)..."
python3 -c "
from collectors.microstructure_collector import microstructure_collector
microstructure_collector.run()
print('  → OK')
"

echo "[5/6] Deribit Options (DVOL, PCR, IV, Skew)..."
python3 -c "
from collectors.deribit_collector import deribit_collector
deribit_collector.run()
print('  → OK')
"

echo "[6/6] Macro (FRED + yfinance)..."
python3 -c "
from collectors.macro_collector import macro_collector
macro_collector.run()
print('  → OK')
"

echo ""
echo "=== Cold Start Complete ==="
echo "이제 스케줄러를 시작하세요:"
echo "  sudo systemctl start scheduler.service"
echo "  sudo systemctl start mcp_server.service"
