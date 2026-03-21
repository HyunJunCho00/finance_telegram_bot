#!/bin/bash
# Grafana Agent 설치 + Grafana Cloud 연동 스크립트
# GCP Secret Manager에서 자동으로 값을 가져옵니다.
# 사용법: PROJECT_ID=mcpknu bash grafana_setup.sh

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-mcpknu}"

echo "=== GCP Secret Manager에서 Grafana 설정 로드 ==="
_secret() { gcloud secrets versions access latest --secret="$1" --project="$PROJECT_ID"; }

GRAFANA_CLOUD_METRICS_URL=$(_secret GRAFANA_CLOUD_METRICS_URL)
GRAFANA_CLOUD_USER=$(_secret GRAFANA_CLOUD_USER)
GRAFANA_CLOUD_API_KEY=$(_secret GRAFANA_CLOUD_API_KEY)

AGENT_VERSION="v0.43.3"
ARCH="amd64"

echo "=== Grafana Agent 설치 ==="

# 이미 설치된 경우 스킵
if command -v grafana-agent &>/dev/null; then
  echo "grafana-agent already installed: $(grafana-agent --version 2>&1 | head -1)"
else
  cd /tmp
  wget -q "https://github.com/grafana/agent/releases/download/${AGENT_VERSION}/grafana-agent-linux-${ARCH}.zip"
  unzip -q "grafana-agent-linux-${ARCH}.zip"
  sudo mv "grafana-agent-linux-${ARCH}" /usr/local/bin/grafana-agent
  sudo chmod +x /usr/local/bin/grafana-agent
  echo "grafana-agent installed."
fi

echo "=== Config 복사 및 환경변수 주입 ==="
APP_DIR="/opt/app"
sudo cp "$APP_DIR/deploy/grafana_agent.yml" /etc/grafana-agent.yaml
sudo sed -i \
  -e "s|PLACEHOLDER_METRICS_URL|${GRAFANA_CLOUD_METRICS_URL}|g" \
  -e "s|PLACEHOLDER_USER|${GRAFANA_CLOUD_USER}|g" \
  -e "s|PLACEHOLDER_API_KEY|${GRAFANA_CLOUD_API_KEY}|g" \
  /etc/grafana-agent.yaml

echo "=== systemd 서비스 등록 ==="
sudo tee /etc/systemd/system/grafana-agent.service > /dev/null << 'EOF'
[Unit]
Description=Grafana Agent
After=network.target

[Service]
Type=simple
User=root
ExecStart=/usr/local/bin/grafana-agent -config.file=/etc/grafana-agent.yaml
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable grafana-agent
sudo systemctl restart grafana-agent

echo ""
echo "=== 완료 ==="
echo "상태 확인: sudo systemctl status grafana-agent"
echo "로그 확인: sudo journalctl -u grafana-agent -f"
echo ""
echo "Grafana Cloud에서 메트릭 확인:"
echo "  Explore → Metrics → job=\"execution\""
