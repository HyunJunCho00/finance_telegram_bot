"""CI test conftest — 외부 연결을 모두 차단한 뒤 테스트 모듈 임포트를 허용.

config/database.py는 모듈 레벨에서 DatabaseClient()를 생성하면서
Supabase create_client()를 호출한다.
pytest가 conftest.py를 테스트 파일보다 먼저 실행하므로
여기서 supabase.create_client를 MagicMock으로 교체하면
이후 모든 임포트에서 실제 연결 시도가 차단된다.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

# ── 환경 변수 (settings.py가 읽기 전에 설정) ───────────────────────────────────
os.environ.setdefault("USE_SECRET_MANAGER", "false")
os.environ.setdefault("SUPABASE_URL", "https://dummy.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "dummy-key-for-ci")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "0:dummy")
os.environ.setdefault("TELEGRAM_CHAT_ID", "0")
os.environ.setdefault("BINANCE_API_KEY", "dummy")
os.environ.setdefault("BINANCE_API_SECRET", "dummy")
os.environ.setdefault("GEMINI_API_KEY", "dummy")
os.environ.setdefault("ANTHROPIC_API_KEY", "dummy")
os.environ.setdefault("DATABASE_URL", "postgresql://dummy:dummy@localhost/dummy")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "dummy")
os.environ.setdefault("ZILLIZ_URI", "https://dummy.zillizcloud.com")
os.environ.setdefault("ZILLIZ_TOKEN", "dummy")

# ── supabase.create_client 차단 (config/database.py 모듈 레벨 실행 전) ─────────
_mock_client = MagicMock()

import supabase as _supabase_module
import supabase._sync.client as _supabase_sync

_supabase_module.create_client = MagicMock(return_value=_mock_client)
_supabase_sync.create_client = MagicMock(return_value=_mock_client)

# ── psycopg2 connect 차단 (execution_repository 모듈 레벨 실행 전) ────────────
import psycopg2 as _psycopg2
_psycopg2.connect = MagicMock(return_value=MagicMock())
