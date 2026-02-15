#!/bin/bash

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-}"
REGION="${REGION:-us-central1}"
ENV_FILE="${1:-.env}"

if [[ -z "$PROJECT_ID" ]]; then
  PROJECT_ID="$(gcloud config get-value project 2>/dev/null || true)"
fi

if [[ -z "$PROJECT_ID" || "$PROJECT_ID" == "(unset)" || "$PROJECT_ID" == "your-gcp-project-id" ]]; then
  echo "ERROR: PROJECT_ID is not set."
  echo "Set it with: gcloud config set project <PROJECT_ID>"
  echo "Or run with: PROJECT_ID=<PROJECT_ID> bash deploy/one_click_gcloud_deploy.sh"
  exit 1
fi


if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: env file not found: $ENV_FILE"
  echo "Create it from template: cp .env.example .env"
  exit 1
fi

echo "Project: $PROJECT_ID"
echo "Region : $REGION"
echo "Env    : $ENV_FILE"

gcloud config set project "$PROJECT_ID" >/dev/null

echo "Enabling required APIs..."
gcloud services enable \
  compute.googleapis.com \
  iam.googleapis.com \
  secretmanager.googleapis.com \
  aiplatform.googleapis.com \
  --project "$PROJECT_ID" >/dev/null

echo "Step 1/2: secrets + IAM setup"
PROJECT_ID="$PROJECT_ID" bash deploy/setup_secrets.sh "$ENV_FILE"

echo "Step 2/2: VM creation"
PROJECT_ID="$PROJECT_ID" REGION="$REGION" bash deploy/create_vm.sh

echo "One-click deploy complete."
