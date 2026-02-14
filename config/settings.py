from pydantic_settings import BaseSettings
from google.cloud import secretmanager
from functools import lru_cache
from enum import Enum
import os
import json


class TradingMode(str, Enum):
    """SWING = long-term (days~weeks), SCALP = short-term (minutes~hours)"""
    SWING = "swing"
    SCALP = "scalp"


class Settings(BaseSettings):
    PROJECT_ID: str = ""
    REGION: str = "us-central1"

    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""

    BINANCE_API_KEY: str = ""
    BINANCE_API_SECRET: str = ""

    UPBIT_ACCESS_KEY: str = ""
    UPBIT_SECRET_KEY: str = ""

    TELEGRAM_API_ID: str = ""
    TELEGRAM_API_HASH: str = ""
    TELEGRAM_PHONE: str = ""
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""

    # Perplexity API for market narrative search
    PERPLEXITY_API_KEY: str = ""

    # Neo4j Aura Free (Knowledge Graph)
    NEO4J_URI: str = ""  # e.g. neo4j+s://xxxxxx.databases.neo4j.io
    NEO4J_PASSWORD: str = ""

    # Zilliz Cloud Free (Milvus vector DB)
    MILVUS_URI: str = ""  # e.g. https://in03-xxxxxx.api.gcp-us-west1.zillizcloud.com
    MILVUS_TOKEN: str = ""

    VOLATILITY_THRESHOLD: float = 3.0
    ANALYSIS_INTERVAL_HOURS: int = 4

    # ===== AI Models =====
    # Agents (bull/bear/risk): Gemini 2.0 Flash via Vertex AI
    # Judge: Claude Opus 4.6 via Vertex AI Model Garden
    MODEL_ENDPOINT: str = "gemini-2.0-flash-001"

    # ===== Trading Mode =====
    # "swing" = long-term position trading (default, your preference)
    # "scalp" = short-term day trading
    TRADING_MODE: str = "swing"

    # ===== Chart Image / VLM Cost Control =====
    # Smart image strategy:
    #   SWING mode: chart sent to Judge only (512x512, ~1024 tokens)
    #   SCALP mode: no images (text-only, pure speed)
    USE_CHART_IMAGES: bool = True
    CHART_LOW_RES: bool = True  # default True for cost savings
    CHART_IMAGE_WIDTH: int = 1200
    CHART_IMAGE_HEIGHT: int = 800
    CHART_IMAGE_DPI: int = 100

    # ===== Candle Limits per Mode =====
    # SWING: 4320 candles (3 days of 1m) for proper weekly/daily resampling
    # SCALP: 720 candles (12h of 1m) for fast 5m/15m/1h analysis
    SWING_CANDLE_LIMIT: int = 4320
    SCALP_CANDLE_LIMIT: int = 720

    # ===== Data Retention (days) =====
    # PostgreSQL(Supabase): 시계열 데이터만 보존 (30일)
    # Neo4j/Milvus: 뉴스/그래프 데이터 영구 보존 (cleanup 없음)
    RETENTION_MARKET_DATA_DAYS: int = 30
    RETENTION_TELEGRAM_DAYS: int = 90  # 원본 텍스트 (Neo4j/Milvus에도 저장됨)
    RETENTION_REPORTS_DAYS: int = 365  # AI 리포트 영구에 가깝게
    RETENTION_CVD_DAYS: int = 30
    RETENTION_GRAPH_DAYS: int = 0  # 0 = 영구 보존 (Neo4j Aura free: 200K nodes)

    class Config:
        env_file = ".env"
        case_sensitive = True

    @property
    def trading_mode(self) -> TradingMode:
        return TradingMode(self.TRADING_MODE.lower())

    @property
    def candle_limit(self) -> int:
        if self.trading_mode == TradingMode.SCALP:
            return self.SCALP_CANDLE_LIMIT
        return self.SWING_CANDLE_LIMIT

    @property
    def should_use_chart(self) -> bool:
        """SWING: yes (Judge only). SCALP: no."""
        if self.trading_mode == TradingMode.SCALP:
            return False
        return self.USE_CHART_IMAGES


class SecretManager:
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.client = secretmanager.SecretManagerServiceClient()

    def get_secret(self, secret_id: str, version: str = "latest") -> str:
        name = f"projects/{self.project_id}/secrets/{secret_id}/versions/{version}"
        response = self.client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")

    def load_all_secrets(self) -> dict:
        secrets = {}
        secret_names = [
            "SUPABASE_URL",
            "SUPABASE_KEY",
            "BINANCE_API_KEY",
            "BINANCE_API_SECRET",
            "UPBIT_ACCESS_KEY",
            "UPBIT_SECRET_KEY",
            "TELEGRAM_API_ID",
            "TELEGRAM_API_HASH",
            "TELEGRAM_PHONE",
            "TELEGRAM_BOT_TOKEN",
            "TELEGRAM_CHAT_ID",
            "PERPLEXITY_API_KEY",
            "NEO4J_URI",
            "NEO4J_PASSWORD",
            "MILVUS_URI",
            "MILVUS_TOKEN",
        ]

        for name in secret_names:
            try:
                secrets[name] = self.get_secret(name)
            except Exception:
                secrets[name] = ""

        return secrets


@lru_cache()
def get_settings() -> Settings:
    if os.getenv("USE_SECRET_MANAGER", "false").lower() == "true":
        project_id = os.getenv("PROJECT_ID", "")
        sm = SecretManager(project_id)
        secrets = sm.load_all_secrets()

        settings = Settings(
            PROJECT_ID=project_id,
            **secrets
        )
    else:
        settings = Settings()

    return settings


settings = get_settings()
