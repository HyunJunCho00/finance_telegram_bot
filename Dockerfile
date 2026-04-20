# Finance Telegram Bot - Base Image
FROM python:3.10-slim

# 보안 및 성능을 위한 파이썬 환경 변수 설정
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 운영 체제 레벨 필수 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 파이썬 종속성 설치 (캐싱 최적화를 위해 먼저 복사)
COPY requirements.txt .
RUN pip install -r requirements.txt

# 소스 코드 복사
COPY . .

# 실행 시 기본적으로 scheduler.py를 바라보도록 설정
CMD ["python", "scheduler.py"]
