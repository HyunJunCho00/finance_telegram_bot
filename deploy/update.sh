#!/bin/bash

set -euo pipefail

INSTALL_DIR="/opt/crypto_trading_system"

echo "=== Bot Update ==="

cd "$INSTALL_DIR"

echo "[1/3] Git pull..."
git pull origin main

echo "[2/3] Install dependencies..."
venv/bin/pip install -r requirements.txt --quiet

echo "[3/3] Restart services..."
systemctl restart scheduler.service
systemctl restart mcp_server.service

echo ""
echo "=== Update Complete ==="
systemctl status scheduler.service --no-pager | tail -5
