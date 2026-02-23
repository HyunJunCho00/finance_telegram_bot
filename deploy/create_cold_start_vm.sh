#!/bin/bash
# Cold Start 전용 임시 VM 생성 → 실행 → 삭제
# Usage:
#   bash deploy/create_cold_start_vm.sh                        # 기본: 2020-01-01부터 오늘까지 자동 계산
#   bash deploy/create_cold_start_vm.sh --from 2020-01-01      # 시작일 지정
#   bash deploy/create_cold_start_vm.sh --mode ohlcv           # 모드 지정

set -euo pipefail

# ─── 설정 ───────────────────────────────────────────────────────────
INSTANCE_NAME="crypto-cold-start"
MACHINE_TYPE="e2-highmem-2"          # 16GB RAM (cold_start 6년치 여유롭게)
REGION="${REGION:-asia-northeast3}"
ZONE="${ZONE:-${REGION}-a}"
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"
BOOT_DISK_SIZE_GB="50"

COLD_START_FROM="${COLD_START_FROM:-2020-01-01}"   # 기본 시작일
COLD_START_MODE="${COLD_START_MODE:-all}"

# CLI 인자 파싱
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from)   COLD_START_FROM="$2"; shift 2 ;;
    --mode)   COLD_START_MODE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# 시작일 → 오늘까지 days 자동 계산
COLD_START_DAYS=$(( ( $(date -u +%s) - $(date -u -d "${COLD_START_FROM}" +%s) ) / 86400 ))
echo "기간: ${COLD_START_FROM} ~ $(date -u +%Y-%m-%d) = ${COLD_START_DAYS}일"

# ─── PROJECT_ID 확인 ─────────────────────────────────────────────────
PROJECT_ID="${PROJECT_ID:-}"
if [[ -z "$PROJECT_ID" ]]; then
  PROJECT_ID="$(gcloud config get-value project 2>/dev/null || true)"
fi
if [[ -z "$PROJECT_ID" || "$PROJECT_ID" == "(unset)" ]]; then
  echo "ERROR: PROJECT_ID가 설정되지 않았습니다."
  echo "  gcloud config set project YOUR_PROJECT_ID 후 재실행하세요."
  exit 1
fi

REPO_URL="${REPO_URL:-https://github.com/HyunJunCho00/finance_telegram_bot.git}"

SA_EMAIL="crypto-trading-sa@${PROJECT_ID}.iam.gserviceaccount.com"

echo "================================================"
echo "  Cold Start VM 생성"
echo "  Project  : $PROJECT_ID"
echo "  Machine  : $MACHINE_TYPE (16GB RAM)"
echo "  Zone     : $ZONE"
echo "  Mode     : $COLD_START_MODE"
echo "  Days     : $COLD_START_DAYS"
echo "  VM 이름  : $INSTANCE_NAME (완료 후 자동 삭제)"
echo "================================================"

# ─── VM 생성 ─────────────────────────────────────────────────────────
if gcloud compute instances describe "$INSTANCE_NAME" \
    --project="$PROJECT_ID" --zone="$ZONE" >/dev/null 2>&1; then
  echo "ERROR: $INSTANCE_NAME 이미 존재합니다."
  echo "  삭제 후 재실행: gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE"
  exit 1
fi

echo "[1/4] VM 생성 중..."
gcloud compute instances create "$INSTANCE_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --image-family="$IMAGE_FAMILY" \
  --image-project="$IMAGE_PROJECT" \
  --boot-disk-size="${BOOT_DISK_SIZE_GB}GB" \
  --boot-disk-type="pd-balanced" \
  --service-account="$SA_EMAIL" \
  --scopes="https://www.googleapis.com/auth/cloud-platform" \
  --network-interface="network-tier=STANDARD,subnet=default"

echo "[2/4] SSH 준비 대기 중 (30초)..."
sleep 30

# ─── VM 내부 셋업 + cold_start 실행 ──────────────────────────────────
echo "[3/4] Cold Start 실행 중... (시간이 걸립니다)"

gcloud compute ssh "$INSTANCE_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --command="
    set -e
    echo '=== 패키지 설치 ==='
    sudo apt-get update -q
    sudo apt-get install -y software-properties-common git
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -q
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip

    echo '=== 코드 클론 ==='
    git clone ${REPO_URL} /opt/app
    cd /opt/app

    echo '=== 패키지 설치 (pip) ==='
    python3.11 -m venv venv
    source venv/bin/activate
    pip install -q -r requirements.txt

    echo '=== Cold Start 시작 (mode=${COLD_START_MODE}, days=${COLD_START_DAYS}) ==='
    export USE_SECRET_MANAGER=true
    export PROJECT_ID=${PROJECT_ID}
    python -m tools.cold_start --mode ${COLD_START_MODE} --days ${COLD_START_DAYS}

    echo '=== Cold Start 완료 ==='
  " \
  --ssh-flag="-o ConnectTimeout=60" \
  --ssh-flag="-o ServerAliveInterval=60"

# ─── VM 삭제 ─────────────────────────────────────────────────────────
echo "[4/4] VM 삭제 중..."
gcloud compute instances delete "$INSTANCE_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --quiet

echo ""
echo "================================================"
echo "  Cold Start 완료!"
echo "  데이터는 Supabase + GCS에 저장되었습니다."
echo "  이제 실전 VM(e2-medium)을 시작하세요."
echo "================================================"
