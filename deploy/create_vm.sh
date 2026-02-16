#!/bin/bash

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-}"
if [[ -z "$PROJECT_ID" ]]; then
  PROJECT_ID="$(gcloud config get-value project 2>/dev/null || true)"
fi

if [[ -z "$PROJECT_ID" || "$PROJECT_ID" == "(unset)" || "$PROJECT_ID" == "your-gcp-project-id" ]]; then
  echo "ERROR: PROJECT_ID is not set."
  echo "Set it with: gcloud config set project <PROJECT_ID>"
  echo "Or run with: PROJECT_ID=<PROJECT_ID> bash deploy/create_vm.sh"
  exit 1
fi

INSTANCE_NAME="crypto-trading-vm"
REGION="${REGION:-asia-northeast3}"
ZONE="${ZONE:-${REGION}-a}"
VERTEX_REGION="${VERTEX_REGION:-us-central1}"
MACHINE_TYPE="e2-small"
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"
SA_NAME="crypto-trading-sa"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
REPO_URL="${REPO_URL:-https://github.com/<your-org>/finance_telegram_bot.git}"

if [[ -z "$REPO_URL" || "$REPO_URL" == *"<your-org>"* ]]; then
  echo "ERROR: REPO_URL is not set to a real repository URL."
  echo "Run with: REPO_URL=https://github.com/<org>/finance_telegram_bot.git bash deploy/create_vm.sh"
  exit 1
fi

if gcloud compute instances describe "$INSTANCE_NAME" --project="$PROJECT_ID" --zone="$ZONE" >/dev/null 2>&1; then
  echo "ERROR: Instance already exists: $INSTANCE_NAME ($ZONE)"
  echo "Delete it first or set a different INSTANCE_NAME in the script."
  exit 1
fi

if ! gcloud iam service-accounts describe "$SA_EMAIL" --project="$PROJECT_ID" >/dev/null 2>&1; then
  echo "Creating service account: $SA_EMAIL"
  gcloud iam service-accounts create "$SA_NAME" \
    --project="$PROJECT_ID" \
    --display-name="Crypto Trading System Service Account"
fi

echo "Granting required IAM roles to service account..."
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/aiplatform.user" >/dev/null

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/secretmanager.secretAccessor" >/dev/null

read -r -d '' STARTUP_SCRIPT <<'SCRIPT' || true
#!/bin/bash
apt-get update
apt-get install -y python3-pip python3-venv git

cd /opt
git clone "__REPO_URL__" /opt/crypto_trading_system
cd /opt/crypto_trading_system
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

useradd -m -s /bin/bash crypto_trader || true
chown -R crypto_trader:crypto_trader /opt/crypto_trading_system

sed -i 's|Environment="PROJECT_ID=.*"|Environment="PROJECT_ID=__PROJECT_ID__"|' deploy/scheduler.service deploy/mcp_server.service
sed -i 's|Environment="VERTEX_REGION=.*"|Environment="VERTEX_REGION=__VERTEX_REGION__"|' deploy/scheduler.service deploy/mcp_server.service
cp deploy/scheduler.service /etc/systemd/system/
cp deploy/mcp_server.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable scheduler.service
systemctl enable mcp_server.service
systemctl start scheduler.service
systemctl start mcp_server.service
SCRIPT

STARTUP_SCRIPT="${STARTUP_SCRIPT//__REPO_URL__/$REPO_URL}"
STARTUP_SCRIPT="${STARTUP_SCRIPT//__PROJECT_ID__/$PROJECT_ID}"
STARTUP_SCRIPT="${STARTUP_SCRIPT//__VERTEX_REGION__/$VERTEX_REGION}"

echo "Creating GCP Compute Engine instance..."
echo "  Project: $PROJECT_ID"
echo "  Machine: $MACHINE_TYPE (~\$12/month)"
echo "  Disk: 20GB pd-standard"

gcloud compute instances create "$INSTANCE_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --network-interface="network-tier=STANDARD,subnet=default" \
  --maintenance-policy="MIGRATE" \
  --service-account="$SA_EMAIL" \
  --scopes="https://www.googleapis.com/auth/cloud-platform" \
  --image-family="$IMAGE_FAMILY" \
  --image-project="$IMAGE_PROJECT" \
  --boot-disk-size="20GB" \
  --boot-disk-type="pd-standard" \
  --boot-disk-device-name="$INSTANCE_NAME" \
  --tags="crypto-trading" \
  --metadata="startup-script=${STARTUP_SCRIPT}"

echo ""
echo "Instance created successfully!"
echo "SSH: gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID"
echo ""