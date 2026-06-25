#!/bin/bash
# PostgreSQL 설치 및 초기 설정 스크립트 (Ubuntu 22.04 / GCP VM)
# 사용법: sudo bash tools/setup_postgres.sh
set -e

DB_NAME="financebot"
DB_USER="botuser"
DB_PASS="${POSTGRES_PASSWORD:-changeme_strong_password}"

echo "=== [1/5] PostgreSQL 설치 ==="
apt-get update -qq
apt-get install -y postgresql postgresql-contrib

echo "=== [2/5] PostgreSQL 서비스 시작 ==="
systemctl enable postgresql
systemctl start postgresql

echo "=== [3/5] DB 유저 및 데이터베이스 생성 ==="
sudo -u postgres psql -tc "SELECT 1 FROM pg_roles WHERE rolname='${DB_USER}'" | grep -q 1 || \
    sudo -u postgres psql -c "CREATE USER ${DB_USER} WITH PASSWORD '${DB_PASS}';"

sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" | grep -q 1 || \
    sudo -u postgres psql -c "CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};"

sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE ${DB_NAME} TO ${DB_USER};"
sudo -u postgres psql -d ${DB_NAME} -c "GRANT ALL ON SCHEMA public TO ${DB_USER};"

echo "=== [4/5] PostgreSQL 메모리 튜닝 (4GB VM 기준) ==="
PG_CONF=$(sudo -u postgres psql -t -c "SHOW config_file;" | xargs)
# shared_buffers=256MB, work_mem=8MB, max_connections=50 (봇 워크로드에 적합)
sudo -u postgres psql -c "ALTER SYSTEM SET shared_buffers = '256MB';"
sudo -u postgres psql -c "ALTER SYSTEM SET work_mem = '8MB';"
sudo -u postgres psql -c "ALTER SYSTEM SET maintenance_work_mem = '64MB';"
sudo -u postgres psql -c "ALTER SYSTEM SET max_connections = '50';"
sudo -u postgres psql -c "ALTER SYSTEM SET effective_cache_size = '1GB';"
sudo -u postgres psql -c "ALTER SYSTEM SET wal_buffers = '16MB';"
sudo -u postgres psql -c "ALTER SYSTEM SET checkpoint_completion_target = '0.9';"
sudo -u postgres psql -c "ALTER SYSTEM SET log_min_duration_statement = '1000';"  # 1초 이상 쿼리 로깅

systemctl restart postgresql

echo "=== [5/5] 스키마 적용 ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SQL_DIR="${SCRIPT_DIR}/../sql"

sudo -u postgres psql -d ${DB_NAME} -c "GRANT CREATE ON SCHEMA public TO ${DB_USER};"

# 메인 스키마
sudo -u postgres psql -d ${DB_NAME} -f "${SQL_DIR}/schema_quant.sql"
sudo -u postgres psql -d ${DB_NAME} -f "${SQL_DIR}/schema_text.sql"
sudo -u postgres psql -d ${DB_NAME} -f "${SQL_DIR}/20260407_blackrock_layers.sql"
sudo -u postgres psql -d ${DB_NAME} -f "${SQL_DIR}/20260408_paper_orders.sql"
sudo -u postgres psql -d ${DB_NAME} -f "${SQL_DIR}/20260505_news_impact_log.sql"

# OMS 스키마 (EXECUTION_DB_URL을 같은 DB로 통합 — Aiven 불필요)
sudo -u postgres psql -d ${DB_NAME} -f "${SQL_DIR}/schema_oms_postgres.sql"

# 소유권 정리
sudo -u postgres psql -d ${DB_NAME} -c "
    DO \$\$
    DECLARE r RECORD;
    BEGIN
      FOR r IN SELECT tablename FROM pg_tables WHERE schemaname='public' LOOP
        EXECUTE 'ALTER TABLE public.' || quote_ident(r.tablename) || ' OWNER TO ${DB_USER}';
      END LOOP;
    END\$\$;
"

echo ""
echo "=========================================="
echo "  PostgreSQL 설치 완료!"
echo "  DB:   ${DB_NAME}"
echo "  User: ${DB_USER}"
echo "  Pass: ${DB_PASS}"
echo ""
echo "  .env / Secret Manager에 추가 (VM 외부 IP 사용):"
echo "  DATABASE_URL=postgresql://${DB_USER}:${DB_PASS}@<VM_EXTERNAL_IP>:5432/${DB_NAME}"
echo "  EXECUTION_DB_URL=postgresql://${DB_USER}:${DB_PASS}@<VM_EXTERNAL_IP>:5432/${DB_NAME}"
echo "=========================================="
