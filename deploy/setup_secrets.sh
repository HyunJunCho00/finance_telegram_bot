#!/bin/bash

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-}"
if [[ -z "$PROJECT_ID" ]]; then
  PROJECT_ID="$(gcloud config get-value project 2>/dev/null || true)"
fi

if [[ -z "$PROJECT_ID" || "$PROJECT_ID" == "(unset)" || "$PROJECT_ID" == "your-gcp-project-id" ]]; then
  echo "ERROR: PROJECT_ID is not set."
  echo "Set it with: gcloud config set project <PROJECT_ID>"
  echo "Or run with: PROJECT_ID=<PROJECT_ID> bash deploy/setup_secrets.sh"
  exit 1
fi

ENV_FILE="${1:-.env}"

if [[ "$ENV_FILE" == ".env.example" ]]; then
  echo "ERROR: Refusing to upload secrets from .env.example."
  echo "Use: cp .env.example .env and edit .env"
  exit 1
fi

if [[ ! -r "$ENV_FILE" ]]; then
  echo "ERROR: env file not found or not readable: $ENV_FILE"
  echo "Create it from template: cp .env.example .env"
  exit 1
fi

SECRET_KEYS=(
  SUPABASE_URL
  SUPABASE_KEY
  ANTHROPIC_API_KEY
  OPENAI_API_KEY
  BINANCE_API_KEY
  BINANCE_API_SECRET
  UPBIT_ACCESS_KEY
  UPBIT_SECRET_KEY
  TELEGRAM_API_ID
  TELEGRAM_API_HASH
  TELEGRAM_PHONE
  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID
  PERPLEXITY_API_KEY
  FRED_API_KEY
  NEO4J_URI
  NEO4J_PASSWORD
  MILVUS_URI
  MILVUS_TOKEN
  GCS_ARCHIVE_BUCKET
  ENABLE_GCS_ARCHIVE
  DUNE_API_KEY
)

SA_NAME="crypto-trading-sa"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

extract_env_value() {
  local key="$1"
  local file="$2"

  local raw
  raw="$(grep -E "^${key}=" "$file" | tail -n 1 | cut -d'=' -f2- || true)"

  # Trim surrounding single/double quotes if present.
  raw="${raw%\"}"
  raw="${raw#\"}"
  raw="${raw%\'}"
  raw="${raw#\'}"

  printf "%s" "$raw"
}

upsert_secret() {
  local key="$1"
  local value="$2"

  if gcloud secrets describe "$key" --project "$PROJECT_ID" >/dev/null 2>&1; then
    printf "%s" "$value" | gcloud secrets versions add "$key" \
      --project "$PROJECT_ID" \
      --data-file=- >/dev/null
    echo "Updated secret version: $key"
  else
    printf "%s" "$value" | gcloud secrets create "$key" \
      --project "$PROJECT_ID" \
      --replication-policy="automatic" \
      --data-file=- >/dev/null
    echo "Created secret: $key"
  fi
}

echo "Using project: $PROJECT_ID"
gcloud config set project "$PROJECT_ID" >/dev/null

echo "Creating/updating secrets from $ENV_FILE"
for key in "${SECRET_KEYS[@]}"; do
  value="$(extract_env_value "$key" "$ENV_FILE")"
  if [[ -n "$value" ]]; then
    upsert_secret "$key" "$value"
  else
    echo "Skipped empty value: $key"
  fi
done

if ! gcloud iam service-accounts describe "$SA_EMAIL" --project "$PROJECT_ID" >/dev/null 2>&1; then
  echo "Creating service account: $SA_EMAIL"
  gcloud iam service-accounts create "$SA_NAME" \
    --project "$PROJECT_ID" \
    --display-name="Crypto Trading System Service Account" >/dev/null
fi

echo "Granting required IAM roles..."
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/secretmanager.secretAccessor" >/dev/null

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/aiplatform.user" >/dev/null

echo "Done. Service Account: $SA_EMAIL"