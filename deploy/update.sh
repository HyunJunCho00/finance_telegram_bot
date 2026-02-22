#!/bin/bash

set -euo pipefail

INSTALL_DIR="/opt/crypto_trading_system"

echo "=== Bot Update ==="

cd "$INSTALL_DIR"

# 소유자 불일치로 인한 git 권한 에러 방지
git config --global --add safe.directory "$INSTALL_DIR" 2>/dev/null || true

# GCP Secret Manager 환경변수 (서비스 재시작 시 필요)
export USE_SECRET_MANAGER=true
export PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
export VERTEX_REGION="${VERTEX_REGION:-us-central1}"

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
