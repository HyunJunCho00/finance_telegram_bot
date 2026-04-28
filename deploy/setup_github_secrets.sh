#!/bin/bash
set -e

PROJECT_ID="mcpknu-493907"
SA_EMAIL="github-actions-deploy@${PROJECT_ID}.iam.gserviceaccount.com"
VM_NAME="crypto-trading-vm-new"
ZONE="asia-southeast1-b"
REPO="HyunJunCho00/finance_telegram_bot"

echo "=== STEP 1: gh CLI 설치 ==="
sudo apt install gh -y

echo "=== STEP 2: GitHub 로그인 ==="
gh auth login

echo "=== STEP 3: IAM 권한 부여 ==="
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/compute.osAdminLogin"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/iap.tunnelResourceAccessor"

echo "=== STEP 4: SA 키 생성 ==="
gcloud iam service-accounts keys create gcp-key.json \
    --iam-account="$SA_EMAIL"

echo "=== STEP 5: SSH 키 생성 ==="
ssh-keygen -t rsa -f ./github_actions_key -C "github-actions" -N ""

echo "=== STEP 6: SSH 공개키 VM에 등록 ==="
gcloud compute instances add-metadata "$VM_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT_ID" \
    --metadata-from-file ssh-keys=./github_actions_key.pub

echo "=== STEP 7: GitHub Secrets 업로드 ==="
gh secret set GCP_PROJECT_ID --body "$PROJECT_ID" --repo "$REPO"
gh secret set GCP_SA_KEY < gcp-key.json --repo "$REPO"
gh secret set SERVER_SSH_KEY < github_actions_key --repo "$REPO"

echo "=== STEP 8: 키 파일 삭제 ==="
rm gcp-key.json github_actions_key github_actions_key.pub

echo "=== 완료: Secret 목록 확인 ==="
gh secret list --repo "$REPO"
