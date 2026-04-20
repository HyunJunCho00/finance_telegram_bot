#!/bin/bash
# Staging 환경 컨테이너 헬스체크 — 배포 후 30초 대기 후 실행됨

set -e

echo "Waiting 30s for containers to stabilize..."
sleep 30

echo "Running smoke tests on staging..."

# 스케줄러 / 실행기 컨테이너 실행 여부 확인
SCHEDULER_RUNNING=$(docker inspect -f '{{.State.Running}}' tgbot-blue-scheduler 2>/dev/null || echo "false")
EXECUTOR_RUNNING=$(docker inspect -f '{{.State.Running}}' tgbot-blue-executor 2>/dev/null || echo "false")

if [ "$SCHEDULER_RUNNING" != "true" ]; then
    echo "FAILED: Scheduler container is not running"
    docker logs --tail 30 tgbot-blue-scheduler 2>/dev/null || true
    exit 1
fi

if [ "$EXECUTOR_RUNNING" != "true" ]; then
    echo "FAILED: Executor container is not running"
    docker logs --tail 30 tgbot-blue-executor 2>/dev/null || true
    exit 1
fi

# 크래시 루프 감지 (재시작 횟수 > 2 이면 실패)
SCHEDULER_RESTARTS=$(docker inspect -f '{{.RestartCount}}' tgbot-blue-scheduler 2>/dev/null || echo "999")
EXECUTOR_RESTARTS=$(docker inspect -f '{{.RestartCount}}' tgbot-blue-executor 2>/dev/null || echo "999")

if [ "$SCHEDULER_RESTARTS" -gt 2 ]; then
    echo "FAILED: Scheduler is crash-looping (restarts: $SCHEDULER_RESTARTS)"
    exit 1
fi

if [ "$EXECUTOR_RESTARTS" -gt 2 ]; then
    echo "FAILED: Executor is crash-looping (restarts: $EXECUTOR_RESTARTS)"
    exit 1
fi

# 최근 로그에서 치명적 에러 감지
SCHEDULER_LOGS=$(docker logs --tail 50 tgbot-blue-scheduler 2>&1)
if echo "$SCHEDULER_LOGS" | grep -qE "FATAL|Traceback \(most recent|SystemExit"; then
    echo "FAILED: Fatal error detected in scheduler logs"
    echo "$SCHEDULER_LOGS" | tail -20
    exit 1
fi

echo "All smoke tests passed. Staging is healthy."
