#!/bin/bash
# Prod 자동 롤백 스크립트 — deploy-prod 실패 시 CI/CD가 자동 호출

set -e

echo "Starting production rollback..."

# 실패한 신규 컨테이너 종료
echo "Stopping failed deployment containers..."
docker stop --time 30 tgbot-blue-executor tgbot-blue-scheduler 2>/dev/null || true
docker compose rm -fs blue-scheduler blue-executor 2>/dev/null || true

# finance-bot:previous 이미지가 있으면 복구, 없으면 실패
if docker image inspect finance-bot:previous &>/dev/null; then
    echo "Restoring previous image..."
    docker tag finance-bot:previous finance-bot:latest
    docker compose up -d blue-scheduler blue-executor
    echo "Rollback successful. Previous version is now live."
else
    echo "ERROR: No previous image found (finance-bot:previous missing)."
    echo "Manual intervention required: check 'docker images' and restore manually."
    exit 1
fi
