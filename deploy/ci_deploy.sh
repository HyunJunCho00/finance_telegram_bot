#!/bin/bash
set -euo pipefail

export AUTO_PROMOTE=1
export PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"

cd /opt/app
bash deploy/blue_green_switch.sh
