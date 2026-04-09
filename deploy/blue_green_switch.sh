#!/bin/bash
# ==============================================================================
# 기관급 배포 스크립트 (Institutional Grade Deployment Switch)
# ==============================================================================
# 
# 일반 웹 서버(NGINX)는 포트만 스위칭하지만, 
# 트레이딩 봇은 "중복 주문(Double Execution)"을 방지하는 것이 최우선입니다.
# 
# 이 스크립트의 철학:
# "두 개의 실행기(Executor)가 한 순간이라도 라이브 모드로 동시 실행되어서는 안 된다."

echo "🚀 Starting Graceful Deployment Process..."

# 1. 새 버전 이미지 빌드
echo "🔨 Building new Docker image..."
docker-compose build

# 2. 🟢 먼저 그린 환경(Shadow)을 띄워 로직 무결성을 확인합니다 (Paper Trading 모드)
echo "🟢 Spinning up Green Environment in Shadow(Paper) mode..."
docker-compose --profile shadow up -d green-scheduler green-executor

echo "⏳ Waiting for 30 seconds to observe container stability..."
sleep 30

# 정상적으로 켜져 있는지 확인 (죽었으면 롤백)
if [ "$(docker inspect -f '{{.State.Running}}' tgbot-green-executor)" != "true" ]; then
    echo "❌ FATAL: Green executor crashed on startup. Aborting deployment."
    docker-compose --profile shadow stop green-scheduler green-executor
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
    docker stop --time 60 tgbot-blue-executor
    
    echo "🛑 Stopping Blue Scheduler..."
    docker stop tgbot-blue-scheduler
    
    # 4. 컨테이너 롤체인지 
    # Green 컨테이너를 끄고, Live 세팅을 주입하여 Blue 컨테이너로 리로드
    echo "🔃 Tearing down shadow green containers..."
    docker-compose --profile shadow rm -fs green-scheduler green-executor
    
    echo "🟦 Spinning up NEW Blue Environment..."
    docker-compose up -d blue-scheduler blue-executor
    
    echo "🎉 Deployment Successful! New code is now Live."
else
    echo "Deployment aborted. Green shadow node will remain running for observation."
    echo "To stop it manually: docker-compose --profile shadow down"
fi
