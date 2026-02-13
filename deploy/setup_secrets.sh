#!/bin/bash

set -e

PROJECT_ID="your-gcp-project-id"

echo "Setting up GCP Secret Manager..."

gcloud config set project $PROJECT_ID

echo "Creating secrets..."

echo -n "your-supabase-url" | gcloud secrets create SUPABASE_URL --data-file=- --replication-policy="automatic"
echo -n "your-supabase-key" | gcloud secrets create SUPABASE_KEY --data-file=- --replication-policy="automatic"

echo -n "your-binance-api-key" | gcloud secrets create BINANCE_API_KEY --data-file=- --replication-policy="automatic"
echo -n "your-binance-api-secret" | gcloud secrets create BINANCE_API_SECRET --data-file=- --replication-policy="automatic"

echo -n "your-upbit-access-key" | gcloud secrets create UPBIT_ACCESS_KEY --data-file=- --replication-policy="automatic"
echo -n "your-upbit-secret-key" | gcloud secrets create UPBIT_SECRET_KEY --data-file=- --replication-policy="automatic"

echo -n "your-telegram-api-id" | gcloud secrets create TELEGRAM_API_ID --data-file=- --replication-policy="automatic"
echo -n "your-telegram-api-hash" | gcloud secrets create TELEGRAM_API_HASH --data-file=- --replication-policy="automatic"
echo -n "+1234567890" | gcloud secrets create TELEGRAM_PHONE --data-file=- --replication-policy="automatic"
echo -n "your-bot-token" | gcloud secrets create TELEGRAM_BOT_TOKEN --data-file=- --replication-policy="automatic"
echo -n "your-chat-id" | gcloud secrets create TELEGRAM_CHAT_ID --data-file=- --replication-policy="automatic"

echo "Creating service account..."

gcloud iam service-accounts create crypto-trading-sa \
    --display-name="Crypto Trading System Service Account"

SA_EMAIL="crypto-trading-sa@${PROJECT_ID}.iam.gserviceaccount.com"

echo "Granting permissions..."

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/secretmanager.secretAccessor"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/aiplatform.user"

echo "Secret Manager setup complete"
echo "Service Account: $SA_EMAIL"
