#!/bin/bash
# One-click GCP deploy + optional GCS migration
#
# Usage (fresh deploy):
#   bash deploy/one_click_gcloud_deploy.sh .env
#
# Usage (account migration — copies GCS data from old bucket):
#   OLD_GCS_BUCKET=old-bucket-name bash deploy/one_click_gcloud_deploy.sh .env
#
# Required env vars (in .env or shell):
#   PROJECT_ID        — new GCP project ID
#   GCS_ARCHIVE_BUCKET — new bucket name to create
#
# Optional env vars for migration:
#   OLD_GCS_BUCKET    — source bucket to migrate FROM (old account)
#   OLD_PROJECT_ID    — old project ID (used only in log messages)

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-}"
REGION="${REGION:-}"
VERTEX_REGION="${VERTEX_REGION:-}"
BOOT_DISK_SIZE_GB="${BOOT_DISK_SIZE_GB:-}"
BOOT_DISK_TYPE="${BOOT_DISK_TYPE:-}"
OLD_GCS_BUCKET="${OLD_GCS_BUCKET:-}"
OLD_PROJECT_ID="${OLD_PROJECT_ID:-}"
ENV_FILE="${1:-.env}"

extract_env_value() {
  local key="$1"
  local file="$2"
  local raw
  raw="$(grep -E "^${key}=" "$file" | tail -n 1 | cut -d'=' -f2- || true)"
  raw="${raw%\"}"
  raw="${raw#\"}"
  raw="${raw%\'}"
  raw="${raw#\'}"
  printf "%s" "$raw"
}

# ─── 사전 검사 ──────────────────────────────────────────────────────────────
if ! command -v gcloud >/dev/null 2>&1; then
  echo "ERROR: gcloud CLI is not installed or not in PATH."
  exit 1
fi

if ! command -v gsutil >/dev/null 2>&1; then
  echo "ERROR: gsutil is not installed. Install the Google Cloud SDK."
  exit 1
fi

if [[ ! -r "$ENV_FILE" ]]; then
  echo "ERROR: env file not found or not readable: $ENV_FILE"
  echo "Create it from template: cp .env.example .env"
  exit 1
fi

if [[ "$ENV_FILE" == ".env.example" ]]; then
  echo "ERROR: Refusing to deploy with .env.example."
  echo "Use a local secrets file: cp .env.example .env"
  exit 1
fi

# ─── 변수 추출 ──────────────────────────────────────────────────────────────
if [[ -z "$PROJECT_ID" ]]; then
  PROJECT_ID="$(gcloud config get-value project 2>/dev/null || true)"
fi
if [[ -z "$PROJECT_ID" || "$PROJECT_ID" == "(unset)" || "$PROJECT_ID" == "your-gcp-project-id" ]]; then
  echo "ERROR: PROJECT_ID is not set."
  echo "Set it with: gcloud config set project <PROJECT_ID>"
  echo "Or run with: PROJECT_ID=<PROJECT_ID> bash deploy/one_click_gcloud_deploy.sh"
  exit 1
fi

ACTIVE_ACCOUNT="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null || true)"
if [[ -z "$ACTIVE_ACCOUNT" ]]; then
  echo "ERROR: No active gcloud account found."
  echo "Run: gcloud auth login"
  exit 1
fi

if [[ -z "${REPO_URL:-}" ]]; then
  REPO_URL="$(extract_env_value "REPO_URL" "$ENV_FILE")"
fi
if [[ -z "${REPO_URL:-}" ]]; then
  REPO_URL="https://github.com/<your-org>/finance_telegram_bot.git"
fi

if [[ -z "$REGION" ]]; then
  REGION="$(extract_env_value "REGION" "$ENV_FILE")"
fi
if [[ -z "$REGION" ]]; then
  REGION="asia-northeast3"
fi

if [[ -z "$VERTEX_REGION" ]]; then
  VERTEX_REGION="$(extract_env_value "VERTEX_REGION" "$ENV_FILE")"
fi
if [[ -z "$VERTEX_REGION" ]]; then
  VERTEX_REGION="us-central1"
fi

if [[ -z "$BOOT_DISK_SIZE_GB" ]]; then
  BOOT_DISK_SIZE_GB="$(extract_env_value "BOOT_DISK_SIZE_GB" "$ENV_FILE")"
fi
if [[ -z "$BOOT_DISK_SIZE_GB" ]]; then
  BOOT_DISK_SIZE_GB="50"
fi

if [[ -z "$BOOT_DISK_TYPE" ]]; then
  BOOT_DISK_TYPE="$(extract_env_value "BOOT_DISK_TYPE" "$ENV_FILE")"
fi
if [[ -z "$BOOT_DISK_TYPE" ]]; then
  BOOT_DISK_TYPE="pd-balanced"
fi

NEW_GCS_BUCKET="$(extract_env_value "GCS_ARCHIVE_BUCKET" "$ENV_FILE")"
ENABLE_GCS_ARCHIVE="$(extract_env_value "ENABLE_GCS_ARCHIVE" "$ENV_FILE")"

# ─── 설정 요약 출력 ─────────────────────────────────────────────────────────
echo "============================================"
echo "  One-click Deploy"
echo "============================================"
echo "  Project      : $PROJECT_ID"
echo "  Account      : $ACTIVE_ACCOUNT"
echo "  VM Region    : $REGION"
echo "  Vertex Region: $VERTEX_REGION"
echo "  Boot Disk    : ${BOOT_DISK_SIZE_GB}GB ${BOOT_DISK_TYPE}"
echo "  Env file     : $ENV_FILE"
if [[ -n "$NEW_GCS_BUCKET" ]]; then
  echo "  New bucket   : gs://$NEW_GCS_BUCKET"
fi
if [[ -n "$OLD_GCS_BUCKET" ]]; then
  echo "  Migrate FROM : gs://$OLD_GCS_BUCKET"
  [[ -n "$OLD_PROJECT_ID" ]] && echo "  Old project  : $OLD_PROJECT_ID"
fi
echo "============================================"

export CLOUDSDK_CORE_DISABLE_PROMPTS=1
gcloud config set project "$PROJECT_ID" >/dev/null

# ─── Step 1: API 활성화 ─────────────────────────────────────────────────────
echo ""
echo "Step 1/4: Enabling required APIs..."
gcloud services enable \
  compute.googleapis.com \
  iam.googleapis.com \
  secretmanager.googleapis.com \
  aiplatform.googleapis.com \
  storage.googleapis.com \
  --project "$PROJECT_ID" \
  --quiet >/dev/null
echo "  Done."

# ─── Step 2: GCS 버킷 생성 + 데이터 이전 ───────────────────────────────────
if [[ -n "$NEW_GCS_BUCKET" && "$ENABLE_GCS_ARCHIVE" == "true" ]]; then
  echo ""
  echo "Step 2/4: GCS bucket setup..."

  # 버킷이 없으면 생성
  if gsutil ls "gs://$NEW_GCS_BUCKET" >/dev/null 2>&1; then
    echo "  Bucket gs://$NEW_GCS_BUCKET already exists — skipping create."
  else
    echo "  Creating gs://$NEW_GCS_BUCKET in $REGION..."
    gcloud storage buckets create "gs://$NEW_GCS_BUCKET" \
      --project="$PROJECT_ID" \
      --location="$REGION" \
      --uniform-bucket-level-access \
      --quiet
    echo "  Bucket created."
  fi

  # 구 버킷 → 신 버킷 데이터 이전
  if [[ -n "$OLD_GCS_BUCKET" ]]; then
    echo ""
    echo "  Migrating GCS data: gs://$OLD_GCS_BUCKET → gs://$NEW_GCS_BUCKET"
    echo "  (gsutil rsync: 중단해도 재실행 시 이어서 복사됩니다)"
    echo ""

    # 구 버킷 접근 가능 여부 확인
    if ! gsutil ls "gs://$OLD_GCS_BUCKET" >/dev/null 2>&1; then
      echo "  ERROR: gs://$OLD_GCS_BUCKET 에 접근할 수 없습니다."
      echo "  현재 계정($ACTIVE_ACCOUNT)이 해당 버킷의 읽기 권한을 갖고 있는지 확인하세요."
      echo ""
      echo "  해결 방법:"
      echo "    1) 구 계정으로 로그인한 뒤 새 버킷에 쓰기 권한 부여:"
      echo "       gcloud storage buckets add-iam-policy-binding gs://$OLD_GCS_BUCKET \\"
      echo "         --member=user:$ACTIVE_ACCOUNT --role=roles/storage.objectViewer"
      echo "    2) 또는 구 프로젝트 오너 계정으로 gcloud auth login 후 재실행"
      echo ""
      echo "  GCS 이전을 건너뜁니다. 나머지 배포는 계속됩니다."
    else
      # 데이터 크기 미리 확인
      echo "  Source bucket size:"
      gsutil du -sh "gs://$OLD_GCS_BUCKET" 2>/dev/null || echo "  (size check failed — continuing anyway)"
      echo ""

      # rsync: -m 병렬, -r 재귀, -c checksum 검증 (이어받기 안전)
      gsutil -m rsync -r -c \
        "gs://$OLD_GCS_BUCKET" \
        "gs://$NEW_GCS_BUCKET"

      echo ""
      echo "  GCS migration complete."
      echo "  Transferred:"
      gsutil du -sh "gs://$NEW_GCS_BUCKET" 2>/dev/null || true
    fi
  fi
else
  echo ""
  echo "Step 2/4: GCS skipped (ENABLE_GCS_ARCHIVE != true or GCS_ARCHIVE_BUCKET not set)."
fi

# ─── Step 3: Secrets + IAM ──────────────────────────────────────────────────
echo ""
echo "Step 3/4: Secrets + IAM setup..."
PROJECT_ID="$PROJECT_ID" bash deploy/setup_secrets.sh "$ENV_FILE"

# ─── Step 4: VM 생성 ────────────────────────────────────────────────────────
echo ""
echo "Step 4/4: VM creation..."
PROJECT_ID="$PROJECT_ID" \
  REGION="$REGION" \
  VERTEX_REGION="$VERTEX_REGION" \
  BOOT_DISK_SIZE_GB="$BOOT_DISK_SIZE_GB" \
  BOOT_DISK_TYPE="$BOOT_DISK_TYPE" \
  REPO_URL="$REPO_URL" \
  bash deploy/create_vm.sh

# ─── 완료 ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Deploy complete!"
echo "============================================"
if [[ -n "$OLD_GCS_BUCKET" && -n "$NEW_GCS_BUCKET" ]]; then
  echo "  GCS: gs://$OLD_GCS_BUCKET → gs://$NEW_GCS_BUCKET  ✓"
fi
echo ""
echo "  VM 시작 로그 확인:"
echo "    gcloud compute ssh crypto-trading-vm --tunnel-through-iap \\"
echo "      --command='tail -f /var/log/startup.log'"
echo "============================================"
