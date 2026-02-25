from pydantic_settings import BaseSettings
from google.cloud import secretmanager
from functools import lru_cache
from typing import List
from enum import Enum
import os
import json


class TradingMode(str, Enum):
    """Two trading modes with distinct timeframes and analysis intervals.
    SWING: days~2weeks, 4h analysis cycle
    POSITION: weeks~months, 1d analysis cycle
    """
    SWING = "swing"
    POSITION = "position"


class Settings(BaseSettings):
    PROJECT_ID: str = ""
    REGION: str = "asia-northeast3"          # VM 인프라 리전 (서울)
    VERTEX_REGION: str = "global"            # 모델 호출 리전 (Gemini + Claude)
    VERTEX_REGION_GEMINI: str = "global"     # Gemini 전용 (하위호환)

    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""

    # ===== Direct AI API Keys (Multi-LLM) =====
    ANTHROPIC_API_KEY: str = ""
    OPENAI_API_KEY: str = ""

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

    # FRED API for macro regime data
    FRED_API_KEY: str = ""

    # Neo4j Aura Free (Knowledge Graph)
    NEO4J_URI: str = ""  # e.g. neo4j+s://xxxxxx.databases.neo4j.io
    NEO4J_PASSWORD: str = ""

    # Zilliz Cloud Free (Milvus vector DB)
    MILVUS_URI: str = ""  # e.g. https://in03-xxxxxx.api.gcp-us-west1.zillizcloud.com
    MILVUS_TOKEN: str = ""


    # ===== Trading Symbols (single source of truth) =====
    # Comma-separated, Binance Futures format (no slashes).
    # Adding a symbol here automatically propagates to ALL collectors.
    # Upbit KRW pairs and Deribit currencies are auto-derived.
    TRADING_SYMBOLS: str = "BTCUSDT,ETHUSDT"

    # Deribit public options data (DVOL, PCR, IV Term Structure, 25d Skew)
    # No API key required — free public REST API
    DERIBIT_ENABLED: bool = True

    # Dune API (on-chain/DEX macro signals)
    DUNE_API_KEY: str = ""
    DUNE_ENABLED: bool = False
    # Cost guardrails for free-tier Dune credits (monthly 2500 credits)
    # Keep scheduler cadence unchanged, but throttle collector execution internally.
    DUNE_BUDGET_GUARD: bool = True
    DUNE_GLOBAL_MIN_INTERVAL_MINUTES: int = 360  # at most once per 6h globally
    DUNE_MAX_QUERY_RUNS_PER_DAY: int = 8
    DUNE_MAX_QUERIES_PER_RUN: int = 1
    DUNE_PRIORITY_QUERY_IDS: str = "21689,4319,6638261"

    # Optional low-cost long-term archive on GCS
    ENABLE_GCS_ARCHIVE: bool = False
    GCS_ARCHIVE_BUCKET: str = ""


    # ===== Paper trading / sandbox safety =====
    PAPER_TRADING_MODE: bool = True  # default safe mode (no real orders)
    PAPER_TRADING_PRICE_SOURCE: str = "ticker"  # ticker | last_report
    BINANCE_USE_TESTNET: bool = False
    UPBIT_PAPER_ONLY: bool = True
    
    # ===== V8: Retail Multi-Exchange Execution Limits =====
    # Hardcoded by User Request for Retail Scale limits
    BINANCE_PAPER_BALANCE_USD: float = 2000.0
    UPBIT_PAPER_BALANCE_KRW: float = 2000000.0
    MAX_LEVERAGE: int = 3
    
    COINBASE_API_KEY: str = ""
    COINBASE_API_SECRET: str = ""

    VOLATILITY_THRESHOLD: float = 3.0
    ANALYSIS_INTERVAL_HOURS: int = 4

    # ===== AI Models =====
    # Supports Gemini, Claude, and GPT models via routing.
    # Model status (2026 SOTA):
    #   gemini-3.1-pro-preview: Google's latest multimodal, extreme context window, unbeatable for big data crunching.
    #   claude-opus-4.6 / claude-sonnet-4.6: Anthropic's flagship models, absolute SOTA in pure mathematical/logical reasoning.
    #   gpt-5.2: OpenAI's latest general intelligence, SOTA in world knowledge, macro analysis and news.
    MODEL_ENDPOINT: str = "gemini-3.1-pro-preview"


    # ===== Role-based model routing (2026 SOTA tuning) =====
    # 1. High-frequency / Big Data (Lowest cost, largest context API)
    # Using gemini-3-flash-preview (Thinking Level: LOW)
    MODEL_LIQUIDITY: str = "gemini-3-flash-preview"
    MODEL_MICROSTRUCTURE: str = "gemini-3-flash-preview"
    MODEL_RAG_EXTRACTION: str = "gemini-3-flash-preview"
    
    # 2. Vision / Multimodal Geometry
    # Using gemini-3.1-pro-preview (Thinking Level: HIGH)
    MODEL_VLM_GEOMETRIC: str = "gemini-3.1-pro-preview"
    
    # 3. World Knowledge / Macro Economy
    # gpt-5.2: OpenAI's latest flagship (Dec 2025), unparalleled for macro text analysis and global intelligence
    MODEL_MACRO: str = "gpt-5.2"
    
    # 4. Supreme Logical Reasoning / Trade Execution
    # claude-sonnet-4-6: The actual SOTA as of Feb 2026, featuring hybrid reasoning (extended thinking)
    # and sweeping coding/math benchmarks (e.g. 70.3% SWE-bench, 96.2% MATH 500).
    MODEL_JUDGE: str = "claude-sonnet-4-6"
    MODEL_SELF_CORRECTION: str = "claude-sonnet-4-6"

    # Soft input caps (character-based) to improve token efficiency
    MAX_INPUT_CHARS_LIQUIDITY: int = 15000
    MAX_INPUT_CHARS_MICROSTRUCTURE: int = 15000
    MAX_INPUT_CHARS_MACRO: int = 15000
    MAX_INPUT_CHARS_JUDGE: int = 25000
    MAX_INPUT_CHARS_SELF_CORRECTION: int = 10000
    MAX_INPUT_CHARS_RAG_EXTRACTION: int = 5000

    # ===== Trading Mode =====
    # "swing"       = multi-day (days~2weeks), 4h analysis cycle
    # "position"    = long-term (weeks~months), 1d analysis cycle
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

    # ===== Candle Limits per Mode (1m candles needed from DB) =====
    # SWING: 4320 (3 days) → for 1h/4h + needs 1d from GCS
    # POSITION: 10080 (7 days) → for 4h + needs 1d/1w from GCS
    SWING_CANDLE_LIMIT: int = 4320
    POSITION_CANDLE_LIMIT: int = 10080

    # ===== Analysis Intervals per Mode =====
    # SWING: every 4 hours
    # POSITION: every 24 hours (once daily at 09:00 KST)
    SWING_INTERVAL_HOURS: int = 4
    POSITION_INTERVAL_HOURS: int = 24

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
        extra = "allow"

    @property
    def vertex_region(self) -> str:
        """Vertex AI region. Falls back to REGION for backward compatibility."""
        return self.VERTEX_REGION or self.REGION

    @property
    def trading_mode(self) -> TradingMode:
        return TradingMode(self.TRADING_MODE.lower())

    @property
    def candle_limit(self) -> int:
        mode = self.trading_mode
        if mode == TradingMode.POSITION:
            return self.POSITION_CANDLE_LIMIT
        return self.SWING_CANDLE_LIMIT

    @property
    def analysis_interval_hours(self) -> int:
        mode = self.trading_mode
        if mode == TradingMode.POSITION:
            return self.POSITION_INTERVAL_HOURS
        return self.SWING_INTERVAL_HOURS

    @property
    def should_use_chart(self) -> bool:
        """All 3 modes generate structure chart for Judge VLM."""
        return self.USE_CHART_IMAGES

    @property
    def chart_timeframe(self) -> str:
        """Primary chart timeframe per mode."""
        mode = self.trading_mode
        if mode == TradingMode.POSITION:
            return "1d"
        return "4h"

    @property
    def analysis_timeframes(self) -> list:
        """Timeframes to analyze per mode."""
        mode = self.trading_mode
        if mode == TradingMode.POSITION:
            return ["4h", "1d", "1w"]
        return ["1h", "4h", "1d"]

    @property
    def trading_symbols(self) -> List[str]:
        """['BTCUSDT', 'ETHUSDT', ...] — canonical format used throughout."""
        return [s.strip().upper() for s in self.TRADING_SYMBOLS.split(',') if s.strip()]

    @property
    def trading_symbols_slash(self) -> List[str]:
        """['BTC/USDT', 'ETH/USDT', ...] — ccxt / Binance API format."""
        return [f"{s[:-4]}/USDT" for s in self.trading_symbols if s.endswith('USDT')]

    @property
    def trading_symbols_base(self) -> List[str]:
        """['BTC', 'ETH', ...] — base currency only."""
        return [s[:-4] for s in self.trading_symbols if s.endswith('USDT')]

    @property
    def trading_symbols_krw(self) -> List[str]:
        """['BTC/KRW', 'ETH/KRW', ...] — Upbit format."""
        return [f"{s[:-4]}/KRW" for s in self.trading_symbols if s.endswith('USDT')]

    @property
    def deribit_currencies(self) -> List[str]:
        """Deribit options exist only for BTC and ETH (exchange constraint)."""
        _available = {'BTC', 'ETH'}
        return [b for b in self.trading_symbols_base if b in _available]

    @property
    def data_lookback_hours(self) -> int:
        """How far back to look for supplementary data (funding, CVD, liquidations)."""
        mode = self.trading_mode
        if mode == TradingMode.POSITION:
            return 168  # 7 days
        return 24


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
            "COINBASE_API_KEY",
            "COINBASE_API_SECRET",
            "TELEGRAM_API_ID",
            "TELEGRAM_API_HASH",
            "TELEGRAM_PHONE",
            "TELEGRAM_BOT_TOKEN",
            "TELEGRAM_CHAT_ID",
            "PERPLEXITY_API_KEY",
            "FRED_API_KEY",
            "NEO4J_URI",
            "NEO4J_PASSWORD",
            "MILVUS_URI",
            "MILVUS_TOKEN",
            "GCS_ARCHIVE_BUCKET",
            "DUNE_API_KEY",
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
