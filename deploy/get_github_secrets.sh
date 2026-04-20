#!/bin/bash
# GitHub Secrets 정보 자동 수집 및 가이드 스크립트

# 색상 정의
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}==========================================================${NC}"
echo -e "${YELLOW}   GitHub Actions Secrets 설정 가이드 (New GCP Project)${NC}"
echo -e "${BLUE}==========================================================${NC}"
echo "이 정보를 복사하여 GitHub 저장소의 Settings > Secrets > Actions에 입력하세요."
echo ""

# 1. SERVER_HOST
INSTANCE_NAME="crypto-trading-vm"
ZONE="asia-southeast1-a"
IP=$(gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --format='get(networkInterfaces[0].accessConfigs[0].natIP)' 2>/dev/null)
echo -e "${GREEN}[1] SERVER_HOST${NC}"
echo "    Value: $IP"
echo ""

# 2. SERVER_USER
USER=$(gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="whoami" --quiet 2>/dev/null)
if [[ -z "$USER" ]]; then USER="ubuntu"; fi
echo -e "${GREEN}[2] SERVER_USER${NC}"
echo "    Value: $USER"
echo ""

# 3. SERVER_DEPLOY_PATH
echo -e "${GREEN}[3] SERVER_DEPLOY_PATH${NC}"
echo "    Value: /opt/app"
echo ""

# 4. GCP_PROJECT_ID
PROJECT=$(gcloud config get-value project 2>/dev/null)
echo -e "${GREEN}[4] GCP_PROJECT_ID${NC}"
echo "    Value: $PROJECT"
echo ""

# 5. SERVER_SSH_KEY
echo -e "${GREEN}[5] SERVER_SSH_KEY (Private Key)${NC}"
echo "    다음 명령어를 입력하여 출력되는 내용을 통째로(-----BEGIN...부터 끝까지) 복사하세요:"
echo -e "${YELLOW}    cat ~/.ssh/id_rsa${NC}"
echo ""
echo -e "${BLUE}==========================================================${NC}"
