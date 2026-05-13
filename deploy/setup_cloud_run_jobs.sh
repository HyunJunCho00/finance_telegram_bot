#!/bin/bash
# Cloud Run Jobs + Cloud Scheduler 설정 스크립트
#
# 사용법:
#   PROJECT_ID=your-project bash deploy/setup_cloud_run_jobs.sh
#
# 전제조건:
#   - Docker 이미지가 이미 빌드/푸시되어 있을 것
#   - Cloud Scheduler API 활성화: gcloud services enable cloudscheduler.googleapis.com
#   - Cloud Run API 활성화:       gcloud services enable run.googleapis.com

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-asia-southeast1}"
IMAGE="${IMAGE:-gcr.io/${PROJECT_ID}/crypto-bot:latest}"
SA_EMAIL="crypto-trading-sa@${PROJECT_ID}.iam.gserviceaccount.com"

if [[ -z "$PROJECT_ID" || "$PROJECT_ID" == "(unset)" ]]; then
  echo "ERROR: PROJECT_ID가 설정되지 않았습니다"
  exit 1
fi

echo "Project : $PROJECT_ID"
echo "Region  : $REGION"
echo "Image   : $IMAGE"
echo ""

# Cloud Run Jobs에 필요한 공통 env vars
COMMON_ENV="USE_SECRET_MANAGER=true,PROJECT_ID=${PROJECT_ID}"

# ── Cloud Run Job 생성/업데이트 함수 ──────────────────────────────────────────
upsert_job() {
  local name="$1"
  local job_env="JOB_NAME=${name},${COMMON_ENV}"
  local extra_args="${2:-}"

  if gcloud run jobs describe "$name" --project="$PROJECT_ID" --region="$REGION" \
       >/dev/null 2>&1; then
    echo "Updating job: $name"
    gcloud run jobs update "$name" \
      --project="$PROJECT_ID" \
      --region="$REGION" \
      --image="$IMAGE" \
      --set-env-vars="$job_env" \
      --service-account="$SA_EMAIL" \
      --max-retries=2 \
      --task-timeout=600 \
      $extra_args
  else
    echo "Creating job: $name"
    gcloud run jobs create "$name" \
      --project="$PROJECT_ID" \
      --region="$REGION" \
      --image="$IMAGE" \
      --command="python" \
      --args="cloud_jobs/entrypoint.py" \
      --set-env-vars="$job_env" \
      --service-account="$SA_EMAIL" \
      --max-retries=2 \
      --task-timeout=600 \
      $extra_args
  fi
}

# ── Cloud Scheduler 트리거 생성/업데이트 함수 ────────────────────────────────
upsert_scheduler() {
  local scheduler_name="$1"
  local job_name="$2"
  local schedule="$3"

  local job_uri="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${job_name}:run"

  if gcloud scheduler jobs describe "$scheduler_name" \
       --project="$PROJECT_ID" --location="$REGION" >/dev/null 2>&1; then
    echo "Updating scheduler: $scheduler_name ($schedule)"
    gcloud scheduler jobs update http "$scheduler_name" \
      --project="$PROJECT_ID" \
      --location="$REGION" \
      --schedule="$schedule" \
      --uri="$job_uri" \
      --oauth-service-account-email="$SA_EMAIL" \
      --time-zone="UTC"
  else
    echo "Creating scheduler: $scheduler_name ($schedule)"
    gcloud scheduler jobs create http "$scheduler_name" \
      --project="$PROJECT_ID" \
      --location="$REGION" \
      --schedule="$schedule" \
      --uri="$job_uri" \
      --oauth-service-account-email="$SA_EMAIL" \
      --time-zone="UTC"
  fi
}

echo "=== Cloud Run Jobs 생성/업데이트 ==="

# 데이터 수집 잡
upsert_job "etf-flow"
upsert_job "stablecoin"
upsert_job "coinglass"
upsert_job "dune"

# 분석/RAG 잡
upsert_job "telegram-batch"
upsert_job "crypto-news"
upsert_job "market-status"
upsert_job "snapshot-narrative"
upsert_job "hourly-monitor"       "--memory=2Gi --cpu=2 --task-timeout=3600"  # 분석 잡: 더 많은 리소스

# daily_precision: 심볼별 분리 (JOB_SYMBOL 환경변수 포함)
if gcloud run jobs describe "daily-precision-btcusdt" --project="$PROJECT_ID" --region="$REGION" >/dev/null 2>&1; then
  gcloud run jobs update "daily-precision-btcusdt" \
    --project="$PROJECT_ID" --region="$REGION" --image="$IMAGE" \
    --set-env-vars="JOB_NAME=daily_precision,JOB_SYMBOL=BTCUSDT,${COMMON_ENV}" \
    --service-account="$SA_EMAIL" --max-retries=1 --task-timeout=5400 \
    --memory=2Gi --cpu=2
else
  gcloud run jobs create "daily-precision-btcusdt" \
    --project="$PROJECT_ID" --region="$REGION" --image="$IMAGE" \
    --command="python" --args="cloud_jobs/entrypoint.py" \
    --set-env-vars="JOB_NAME=daily_precision,JOB_SYMBOL=BTCUSDT,${COMMON_ENV}" \
    --service-account="$SA_EMAIL" --max-retries=1 --task-timeout=5400 \
    --memory=2Gi --cpu=2
fi
if gcloud run jobs describe "daily-precision-ethusdt" --project="$PROJECT_ID" --region="$REGION" >/dev/null 2>&1; then
  gcloud run jobs update "daily-precision-ethusdt" \
    --project="$PROJECT_ID" --region="$REGION" --image="$IMAGE" \
    --set-env-vars="JOB_NAME=daily_precision,JOB_SYMBOL=ETHUSDT,${COMMON_ENV}" \
    --service-account="$SA_EMAIL" --max-retries=1 --task-timeout=5400 \
    --memory=2Gi --cpu=2
else
  gcloud run jobs create "daily-precision-ethusdt" \
    --project="$PROJECT_ID" --region="$REGION" --image="$IMAGE" \
    --command="python" --args="cloud_jobs/entrypoint.py" \
    --set-env-vars="JOB_NAME=daily_precision,JOB_SYMBOL=ETHUSDT,${COMMON_ENV}" \
    --service-account="$SA_EMAIL" --max-retries=1 --task-timeout=5400 \
    --memory=2Gi --cpu=2
fi

# 평가 잡
upsert_job "evaluation"
upsert_job "evaluation-rollup"
upsert_job "evaluation-24h"

echo ""
echo "=== Cloud Scheduler 트리거 설정 ==="

# NY 일봉 캔들 확정 후 분석 (1x/일): BTC 22:05 UTC / ETH 22:10 UTC
upsert_scheduler "sched-etf-flow"                  "etf-flow"                  "0 6 * * *"
upsert_scheduler "sched-stablecoin"                "stablecoin"                "15 6 * * *"
upsert_scheduler "sched-coinglass"                 "coinglass"                 "30 0,4,8,12,16,20 * * *"
upsert_scheduler "sched-dune"                      "dune"                      "*/15 * * * *"
upsert_scheduler "sched-telegram-batch"            "telegram-batch"            "5 * * * *"
upsert_scheduler "sched-crypto-news"               "crypto-news"               "10 * * * *"
upsert_scheduler "sched-market-status"             "market-status"             "20 * * * *"
upsert_scheduler "sched-snapshot-narrative"        "snapshot-narrative"        "2,32 * * * *"
upsert_scheduler "sched-hourly-monitor"            "hourly-monitor"            "15 * * * *"
upsert_scheduler "sched-daily-precision-btcusdt"   "daily-precision-btcusdt"   "5 22 * * *"
upsert_scheduler "sched-daily-precision-ethusdt"   "daily-precision-ethusdt"   "10 22 * * *"
upsert_scheduler "sched-evaluation"                "evaluation"                "45 * * * *"
upsert_scheduler "sched-evaluation-rollup"         "evaluation-rollup"         "40 0 * * *"
upsert_scheduler "sched-evaluation-24h"            "evaluation-24h"            "30 0 * * *"

echo ""
echo "=== IAM: Cloud Scheduler → Cloud Run Jobs 실행 권한 부여 ==="
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/run.invoker" \
  --quiet

echo ""
echo "완료!"
echo ""
echo "잡 목록 확인:         gcloud run jobs list --region=$REGION"
echo "스케줄러 목록 확인:   gcloud scheduler jobs list --location=$REGION"
echo "수동 실행 테스트:     gcloud run jobs execute etf-flow --region=$REGION --wait"
