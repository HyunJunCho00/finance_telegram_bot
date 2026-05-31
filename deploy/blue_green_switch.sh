#!/bin/bash
# ==============================================================================
# 배포 스크립트 (Institutional Grade Deployment Switch)
# ==============================================================================
# 
# 일반 웹 서버(NGINX)는 포트만 스위칭하지만, 
# 트레이딩 봇은 "중복 주문(Double Execution)"을 방지하는 것이 최우선입니다.
# 


echo "🚀 Starting Graceful Deployment Process..."

# 0. Python SDK를 사용해 Secret Manager에서 DATABASE_URL을 읽어와 패스워드를 추출해 환경 변수로 주입합니다.
echo "🔑 Fetching database password from GCP Secret Manager via Python SDK..."
export POSTGRES_DB_PASS=$(python3 -c "
import sys
try:
    import google.cloud.secretmanager as sm
    from urllib.parse import urlparse
    client = sm.SecretManagerServiceClient()
    name = f'projects/${PROJECT_ID:-}/secrets/DATABASE_URL/versions/latest'
    dsn = client.access_secret_version(request={'name': name}).payload.data.decode('UTF-8').strip()
    print(urlparse(dsn).password or '')
except Exception as e:
    pass
" 2>/dev/null)

if [ -z "${POSTGRES_DB_PASS:-}" ]; then
    echo "⚠️ Warning: Could not fetch DATABASE_URL from Secret Manager via Python SDK. Using default DB password."
fi

# 1. 현재 이미지를 previous로 백업 (롤백용)
if docker image inspect finance-bot:latest &>/dev/null; then
    echo "📦 Saving current image as finance-bot:previous (rollback snapshot)..."
    docker tag finance-bot:latest finance-bot:previous
fi

# 2. 이미지 준비: CI가 DEPLOY_IMAGE를 넘기면 pull, 아니면 로컬 빌드 (수동 배포 폴백)
if [ -n "${DEPLOY_IMAGE:-}" ]; then
    echo "📥 Pulling image from Artifact Registry: $DEPLOY_IMAGE"
    gcloud auth configure-docker asia-southeast1-docker.pkg.dev --quiet
    docker pull "$DEPLOY_IMAGE"
    docker tag "$DEPLOY_IMAGE" finance-bot:latest
    # [최적화] 원본 SHA 태그를 삭제해야 나중에 찌꺼기 이미지로 분류되어 자동 청소(Prune)의 대상이 됩니다.
    docker rmi "$DEPLOY_IMAGE" || true
else
    echo "🔨 No DEPLOY_IMAGE set — building locally (manual deploy fallback)..."
    docker build -t finance-bot:latest .
fi

# 2. 🟢 먼저 그린 환경(Shadow)을 띄워 로직 무결성을 확인합니다 (Paper Trading 모드)
echo "🟢 Spinning up Green Environment in Shadow(Paper) mode..."
docker compose --profile shadow up -d green-brain green-executor

echo "⏳ Waiting for 30 seconds to observe container stability..."
sleep 30

# 정상적으로 켜져 있는지 확인 (죽었으면 롤백)
if [ "$(docker inspect -f '{{.State.Running}}' tgbot-green-executor)" != "true" ]; then
    echo "❌ FATAL: Green executor crashed on startup. Aborting deployment."
    docker compose --profile shadow stop green-brain green-executor
    exit 1
fi
echo "✅ Green environment is stable in Shadow mode."

if [ "$AUTO_PROMOTE" = "1" ]; then
    echo "🤖 CI/CD Auto-Promotion Triggered!"
    answer="y"
else
    echo "⚠️ Are you ready to promote Green to Live(Blue)? (y/n)"
    read answer
fi

if [ "$answer" != "${answer#[Yy]}" ] ;then
    echo "🔄 Promoting Process Commencing..."
    
    # 3. 🔵 (CRITICAL) Blue 실행기를 먼저 가장 안전하게 종료 (단순 kill 아님)
    echo "🛑 Sending SIGTERM to Blue Executor to finish pending orders..."
    # execution_main.py는 SIGTERM 수신 시, 진행중인 의도(Intent)를 마무리하고 종료해야 함
    docker stop --timeout 60 tgbot-blue-executor
    
    echo "🛑 Stopping Blue Brain..."
    docker stop --timeout 30 tgbot-blue-brain
    
    # 4. 컨테이너 롤체인지 
    # Green 컨테이너를 끄고, Live 세팅을 주입하여 Blue 컨테이너로 리로드
    echo "🔃 Tearing down shadow green containers..."
    docker compose --profile shadow rm -fs green-brain green-executor
    
    echo "🟦 Spinning up NEW Blue Environment..."
    docker compose up -d blue-brain blue-executor
    
    echo "🌐 Automatically updating Shared Services (Collectors & Listener)..."
    docker compose up -d shared-data shared-listener shared-bot
    
    echo "🎉 Deployment Successful! New code is now Live on ALL environments."
    
    echo "🧹 Automatically cleaning up unused Docker cache to save disk space..."
    docker builder prune -f > /dev/null 2>&1
    docker image prune -f > /dev/null 2>&1
else
    echo "Deployment aborted. Green shadow node will remain running for observation."
    echo "To stop it manually: docker compose --profile shadow down"
fi
