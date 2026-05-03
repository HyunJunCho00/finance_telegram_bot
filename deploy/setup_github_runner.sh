#!/bin/bash
# VM에 GitHub Actions self-hosted runner 설치
#
# 사용법 (VM에서):
#   GITHUB_REPO=your-org/your-repo \
#   GITHUB_TOKEN=your-runner-token \
#   sudo -u crypto_trader bash deploy/setup_github_runner.sh
#
# GitHub runner token 발급:
#   https://github.com/{org}/{repo}/settings/actions/runners/new
#
# 설치 후 ci.yml의 deploy-vm job에서:
#   runs-on: ubuntu-latest  →  runs-on: [self-hosted, linux]
# 로 변경하면 SSH/IAP 완전히 제거됨

set -euo pipefail

REPO="${GITHUB_REPO:-}"
TOKEN="${GITHUB_TOKEN:-}"
RUNNER_VERSION="2.317.0"
RUNNER_DIR="/opt/actions-runner"
RUNNER_USER="${SUDO_USER:-crypto_trader}"

if [[ -z "$REPO" || -z "$TOKEN" ]]; then
  echo "ERROR: GITHUB_REPO와 GITHUB_TOKEN을 설정하세요"
  echo "  예: GITHUB_REPO=your-org/repo GITHUB_TOKEN=xxx bash $0"
  exit 1
fi

echo "Runner 설치 경로: $RUNNER_DIR"
echo "실행 유저: $RUNNER_USER"
echo "대상 레포: $REPO"

# Runner 디렉토리 생성
mkdir -p "$RUNNER_DIR"
cd "$RUNNER_DIR"

# 다운로드 (이미 있으면 스킵)
if [[ ! -f "run.sh" ]]; then
  echo "GitHub Actions runner $RUNNER_VERSION 다운로드..."
  curl -sL "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" \
    | tar xz
fi

# 권한 설정
chown -R "$RUNNER_USER:$RUNNER_USER" "$RUNNER_DIR"

# Runner 등록
echo "Runner 등록 중..."
sudo -u "$RUNNER_USER" ./config.sh \
  --url "https://github.com/$REPO" \
  --token "$TOKEN" \
  --name "$(hostname)-runner" \
  --labels "self-hosted,linux,crypto-vm" \
  --work "$RUNNER_DIR/_work" \
  --unattended \
  --replace

# systemd 서비스로 등록 (VM 재시작 시 자동 시작)
./svc.sh install "$RUNNER_USER"
./svc.sh start

echo ""
echo "완료! Runner가 서비스로 등록되었습니다."
echo ""
echo "다음 단계: .github/workflows/ci.yml의 deploy-vm job에서"
echo "  runs-on: ubuntu-latest"
echo "  →"
echo "  runs-on: [self-hosted, linux, crypto-vm]"
echo "으로 변경하면 SSH/IAP 없이 바로 VM에서 배포됩니다."
echo ""
echo "서비스 상태 확인: sudo systemctl status actions.runner.*"
