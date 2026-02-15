#!/bin/bash

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-}"
REGION="${REGION:-us-central1}"
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

if ! command -v gcloud >/dev/null 2>&1; then
  echo "ERROR: gcloud CLI is not installed or not in PATH."
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

echo "Project: $PROJECT_ID"
echo "Region : $REGION"
echo "Env    : $ENV_FILE"
echo "Account: $ACTIVE_ACCOUNT"

export CLOUDSDK_CORE_DISABLE_PROMPTS=1

gcloud config set project "$PROJECT_ID" >/dev/null

echo "Enabling required APIs..."
gcloud services enable \
  compute.googleapis.com \
  iam.googleapis.com \
  secretmanager.googleapis.com \
  aiplatform.googleapis.com \
  --project "$PROJECT_ID" \
  --quiet >/dev/null

echo "Step 1/2: secrets + IAM setup"
PROJECT_ID="$PROJECT_ID" bash deploy/setup_secrets.sh "$ENV_FILE"

echo "Step 2/2: VM creation"
PROJECT_ID="$PROJECT_ID" REGION="$REGION" REPO_URL="$REPO_URL" bash deploy/create_vm.sh

echo "One-click deploy complete."
