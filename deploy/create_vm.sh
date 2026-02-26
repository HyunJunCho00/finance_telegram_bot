#!/bin/bash

set -euo pipefail

# ─── Config ────────────────────────────────────────────────────────────────
PROJECT_ID="${PROJECT_ID:-}"
if [[ -z "$PROJECT_ID" ]]; then
  PROJECT_ID="$(gcloud config get-value project 2>/dev/null || true)"
fi
if [[ -z "$PROJECT_ID" || "$PROJECT_ID" == "(unset)" ]]; then
  echo "ERROR: PROJECT_ID is not set."
  echo "Run: gcloud config set project <PROJECT_ID>"
  exit 1
fi

INSTANCE_NAME="${INSTANCE_NAME:-crypto-trading-vm}"
VERTEX_REGION="${VERTEX_REGION:-global}"
BOOT_DISK_SIZE_GB="${BOOT_DISK_SIZE_GB:-50}"
BOOT_DISK_TYPE="${BOOT_DISK_TYPE:-pd-standard}"
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"
SA_NAME="crypto-trading-sa"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
REPO_URL="${REPO_URL:-https://github.com/HyunJunCho00/finance_telegram_bot.git}"

# ─── Zone + Machine-type fallback list ────────────────────────────────────
# Seoul (asia-northeast3) confirmed full — excluded.
# Format: "zone:machine_type"
CANDIDATES=(
  "asia-southeast1-a:e2-medium"   # Singapore
  "asia-southeast1-b:e2-medium"
  "asia-southeast1-c:e2-medium"
  "asia-east1-b:e2-medium"        # Taiwan
  "asia-east1-a:e2-medium"
  "asia-northeast1-b:e2-medium"   # Tokyo
  "asia-northeast1-a:e2-medium"
  "asia-northeast2-a:e2-medium"   # Osaka
  "asia-southeast1-a:n1-standard-2"
  "asia-east1-b:n1-standard-2"
  "asia-northeast1-b:n1-standard-2"
)

# ─── Repo validation ───────────────────────────────────────────────────────
if [[ "$REPO_URL" == *"<your-org>"* ]]; then
  echo "ERROR: REPO_URL not set. Pass REPO_URL=https://github.com/org/repo.git"
  exit 1
fi

# ─── Check if instance already exists in any candidate zone ───────────────
for pair in "${CANDIDATES[@]}"; do
  zone="${pair%%:*}"
  if gcloud compute instances describe "$INSTANCE_NAME" \
      --project="$PROJECT_ID" --zone="$zone" &>/dev/null; then
    echo "ERROR: $INSTANCE_NAME already exists in $zone. Delete it first:"
    echo "  gcloud compute instances delete $INSTANCE_NAME --zone=$zone --project=$PROJECT_ID"
    exit 1
  fi
done

# ─── Service account setup ────────────────────────────────────────────────
if ! gcloud iam service-accounts describe "$SA_EMAIL" \
    --project="$PROJECT_ID" &>/dev/null; then
  echo "Creating service account: $SA_EMAIL"
  gcloud iam service-accounts create "$SA_NAME" \
    --project="$PROJECT_ID" \
    --display-name="Crypto Trading System Service Account"
fi

echo "Granting IAM roles to service account..."
for role in roles/aiplatform.user roles/secretmanager.secretAccessor \
            roles/storage.objectAdmin roles/logging.logWriter; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="$role" >/dev/null
done

# ─── Startup script ───────────────────────────────────────────────────────
STARTUP_SCRIPT='#!/bin/bash
exec >> /var/log/startup.log 2>&1
set -euo pipefail
echo "=== Startup: $(date) ==="

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git python3.11 python3.11-venv python3.11-dev python3-pip

APP_DIR="/opt/app"
if [[ -d "$APP_DIR/.git" ]]; then
  echo "Repo exists — pulling latest..."
  git -C "$APP_DIR" pull --ff-only
else
  echo "Cloning repo..."
  git clone "PLACEHOLDER_REPO_URL" "$APP_DIR"
fi

cd "$APP_DIR"
mkdir -p data

if [[ ! -d venv ]]; then
  python3.11 -m venv venv
fi
source venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

# ── Create app user (required by scheduler.service User=crypto_trader) ──
useradd -m -s /bin/bash crypto_trader 2>/dev/null || true
chown -R crypto_trader:crypto_trader "$APP_DIR"

sed -i \
  -e "s|Environment=\"PROJECT_ID=.*\"|Environment=\"PROJECT_ID=PLACEHOLDER_PROJECT_ID\"|g" \
  -e "s|Environment=\"VERTEX_REGION=.*\"|Environment=\"VERTEX_REGION=PLACEHOLDER_VERTEX_REGION\"|g" \
  deploy/scheduler.service deploy/mcp_server.service

cp deploy/scheduler.service /etc/systemd/system/
cp deploy/mcp_server.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable scheduler.service mcp_server.service
systemctl start scheduler.service mcp_server.service

echo "=== Startup complete: $(date) ==="'

STARTUP_SCRIPT="${STARTUP_SCRIPT//PLACEHOLDER_REPO_URL/$REPO_URL}"
STARTUP_SCRIPT="${STARTUP_SCRIPT//PLACEHOLDER_PROJECT_ID/$PROJECT_ID}"
STARTUP_SCRIPT="${STARTUP_SCRIPT//PLACEHOLDER_VERTEX_REGION/$VERTEX_REGION}"

# ─── Try each zone/machine combination until one succeeds ─────────────────
ZONE_USED=""
MACHINE_USED=""

for pair in "${CANDIDATES[@]}"; do
  zone="${pair%%:*}"
  machine="${pair##*:}"
  echo "Trying zone=$zone machine=$machine ..."
  if gcloud compute instances create "$INSTANCE_NAME" \
      --project="$PROJECT_ID" \
      --zone="$zone" \
      --machine-type="$machine" \
      --network-interface="network-tier=STANDARD,subnet=default" \
      --maintenance-policy="MIGRATE" \
      --service-account="$SA_EMAIL" \
      --scopes="https://www.googleapis.com/auth/cloud-platform" \
      --image-family="$IMAGE_FAMILY" \
      --image-project="$IMAGE_PROJECT" \
      --boot-disk-size="${BOOT_DISK_SIZE_GB}GB" \
      --boot-disk-type="$BOOT_DISK_TYPE" \
      --boot-disk-device-name="$INSTANCE_NAME" \
      --tags="crypto-trading" \
      --metadata="USE_SECRET_MANAGER=true,ENABLE_GCS_ARCHIVE=true,startup-script=${STARTUP_SCRIPT}" \
      2>/dev/null; then
    ZONE_USED="$zone"
    MACHINE_USED="$machine"
    break
  else
    echo "  Unavailable, trying next..."
  fi
done

if [[ -z "$ZONE_USED" ]]; then
  echo ""
  echo "ERROR: All zone/machine combinations exhausted. Try again later or add more candidates."
  exit 1
fi

echo ""
echo "Instance created!"
echo "  Name   : $INSTANCE_NAME"
echo "  Zone   : $ZONE_USED"
echo "  Machine: $MACHINE_USED"
echo ""
echo "SSH:"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE_USED --project=$PROJECT_ID"
echo ""
echo "Watch startup log:"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE_USED --project=$PROJECT_ID --command='tail -f /var/log/startup.log'"
echo ""