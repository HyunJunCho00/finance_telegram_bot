#!/bin/bash

set -e

PROJECT_ID="your-gcp-project-id"
INSTANCE_NAME="crypto-trading-vm"
ZONE="us-central1-a"
# e2-small: 0.5 vCPU, 2GB RAM = ~$12/month (sufficient for this workload)
MACHINE_TYPE="e2-small"
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"

echo "Creating GCP Compute Engine instance..."
echo "  Machine: $MACHINE_TYPE (~\$12/month)"
echo "  Disk: 20GB pd-standard"

gcloud compute instances create $INSTANCE_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --network-interface=network-tier=STANDARD,subnet=default \
    --maintenance-policy=MIGRATE \
    --service-account=crypto-trading-sa@${PROJECT_ID}.iam.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --image-family=$IMAGE_FAMILY \
    --image-project=$IMAGE_PROJECT \
    --boot-disk-size=20GB \
    --boot-disk-type=pd-standard \
    --boot-disk-device-name=$INSTANCE_NAME \
    --tags=crypto-trading \
    --metadata=startup-script='#!/bin/bash
apt-get update
apt-get install -y python3-pip python3-venv git

cd /opt
git clone https://github.com/<your-org>/finance_telegram_bot.git /opt/crypto_trading_system
cd /opt/crypto_trading_system
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

useradd -m -s /bin/bash crypto_trader || true
chown -R crypto_trader:crypto_trader /opt/crypto_trading_system

cp deploy/scheduler.service /etc/systemd/system/
cp deploy/mcp_server.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable scheduler.service
systemctl enable mcp_server.service
systemctl start scheduler.service
systemctl start mcp_server.service'

echo ""
echo "Instance created successfully!"
echo "SSH: gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID"
echo ""
echo "Cost estimate:"
echo "  VM (e2-small): ~\$12/month"
echo "  AI (Gemini):   ~\$4/month"
echo "  Perplexity:    ~\$5/month"
echo "  Supabase:      \$0 (free tier)"
echo "  Total:         ~\$21/month"
