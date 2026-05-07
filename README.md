# Finance Telegram Bot — Architecture Reference

**Last updated: 2026-05-07 (V15.x)**
**Scope**: `scheduler.py`, `executors/`, `agents/`, `collectors/`, `evaluators/`, `processors/`, `config/`, `bot/`, `liquidation_cascade/`

> 코드 위치를 빠르게 찾기 위한 레퍼런스 문서.
> 섹션 번호로 먼저 탐색하고 해당 파일로 이동하세요.

---

## 목차

1. [시스템 개요](#1-시스템-개요)
2. [런타임 토폴로지](#2-런타임-토폴로지)
3. [AI 모델 라우팅](#3-ai-모델-라우팅)
4. [데이터 레이어](#4-데이터-레이어)
5. [스케줄러 잡 맵](#5-스케줄러-잡-맵)
6. [오케스트레이터 (LangGraph)](#6-오케스트레이터)
7. [에이전트 레이어](#7-에이전트-레이어)
8. [컬렉터 레이어](#8-컬렉터-레이어)
9. [프로세서 레이어](#9-프로세서-레이어)
10. [익스큐터 레이어](#10-익스큐터-레이어)
11. [이벨류에이터 레이어](#11-이벨류에이터-레이어)
12. [리퀴데이션 캐스케이드 (ML)](#12-리퀴데이션-캐스케이드)
13. [텔레그램 봇](#13-텔레그램-봇)
14. [DB 테이블 전체 목록](#14-db-테이블-전체-목록)
15. [핵심 설정 파라미터](#15-핵심-설정-파라미터)

---

## 1. 시스템 개요

BTC/ETH 자동 분석·실행 봇. 14개 텔레그램 채널 + 파생상품/온체인 데이터를 LightRAG 지식 그래프로 통합하고, LangGraph 멀티에이전트 파이프라인으로 트레이딩 결정을 내린다.

| 항목 | 값 |
|------|-----|
| 심볼 | `BTCUSDT`, `ETHUSDT` (`settings.TRADING_SYMBOLS`) |
| 트레이딩 모드 | `SWING` (4h 분석 사이클, days~2주) |
| 스팟 모드 | `spot_swing` / `spot_position` (`SpotMode`) |
| AI 파이프라인 | Meta → Judge → Risk (LangGraph `StateGraph`) |
| 실행 | Paper-first 기본; Binance Futures / Upbit Spot |
| 평가 루프 | PerformanceEvaluator → FeedbackGenerator |
| 모니터링 | Prometheus metrics + Telegram 알림 |

---

## 2. 런타임 토폴로지

프로세스는 **2개**로 분리된다.

```
scheduler.py           메인 프로세스 (APScheduler)
├── Telegram Bot         bot/telegram_bot.py        (daemon thread)
├── Telegram Listener    collectors/telegram_listener.py  (daemon thread)
├── WS Price Feed        collectors/ws_price_feed.py      (daemon thread)
├── WS User Stream       collectors/ws_user_stream.py     (daemon thread)
└── WebSocket Collector  collectors/websocket_collector.py (daemon thread)

execution_main.py      실행 전용 프로세스 (EXECUTION_PROCESS_SEPARATE=True)
└── 1min execution job, 8h funding fee job
```

```mermaid
flowchart LR
    subgraph Collectors
        TL[Telegram Listener]
        PC[Price / Funding / Micro]
        WS[WebSocket: Liq + Whale]
    end

    subgraph Storage
        SBQ[(Supabase QUANT\nmarket/funding/liq/micro)]
        SBT[(Supabase TEXT\ntelegram/narrative/reports)]
        LR[LightRAG\nNeo4j + Milvus]
        GCS[(GCS Parquet\n장기 히스토리)]
        REDIS[(Redis\n실행 캐시)]
        SQLITE[(SQLite\nactive_orders / agent_state)]
    end

    TL --> SBT & LR
    PC --> SBQ
    WS --> SBQ

    subgraph Orchestrator [LangGraph Orchestrator]
        MA[meta_agent] --> JA[judge_agent]
        JA --> RM[risk_manager]
        VLM[vlm_geometric] -.조건부.-> JA
    end

    SBQ & SBT & LR --> Orchestrator
    GCS --> Orchestrator

    RM --> TE[TradeExecutor]
    TE --> SQLITE
    SQLITE --> ED[ExecutionDesk / PaperExchange]
    ED --> REDIS

    Orchestrator --> RG[ReportGenerator]
    RG --> TGM[Telegram]
    RG --> SBT

    SBQ & SBT --> EV[PerformanceEvaluator]
    EV --> FB[FeedbackGenerator]
    FB --> SBT
```

---

## 3. AI 모델 라우팅

`config/settings.py` — `MODEL_*` 필드가 단일 정책 테이블.

| 역할 | 모델 | 제공자 |
|------|------|--------|
| Judge / Self-correction | `gemini-2.5-pro` | Google AI Studio (Project A) |
| VLM / 차트 기하 분석 | `gemini-2.5-flash` | Google AI Studio (Project B) |
| Meta Regime 분류 | `qwen-3-235b-a22b-instruct-2507` | Cerebras |
| Risk Eval | `qwen-3-235b-a22b-instruct-2507` (fallback: Groq `qwen/qwen3-32b`) | Cerebras |
| RAG 추출 | `meta-llama/llama-4-scout-17b-16e-instruct` | Groq |
| 뉴스 요약 | `llama-3.1-8b-instant` | Groq |
| 뉴스 클러스터링 | `llama-3.3-70b-versatile` | Groq |
| 뉴스 최종 합성 | `openai/gpt-oss-120b` (fallback: `qwen/qwen3-32b`) | OpenRouter |
| 시장 모니터 (hourly) | `llama3.1-8b` | Cerebras |
| Perplexity 내러티브 | `sonar-pro` (targeted: `sonar`) | Perplexity |

---

## 4. 데이터 레이어

### 4.1 Supabase — 2-project 분리

**파일**: `config/database.py` — `DatabaseClient` (싱글톤: `db`)

| 프로젝트 | 환경변수 | 저장 데이터 |
|---------|---------|------------|
| QUANT | `SUPABASE_URL_QUANT` / `SUPABASE_KEY_QUANT` | market, funding, cvd, liquidations, micro, deribit, fear_greed, macro |
| TEXT | `SUPABASE_URL_TEXT` / `SUPABASE_KEY_TEXT` | telegram, narrative, ai_reports, evaluations, trade_executions, feedback |

주요 조회 메서드:

| 메서드 | 테이블 | 용도 |
|--------|--------|------|
| `get_latest_market_data(symbol, limit)` | `market_data` | 1분봉 최근 N개 |
| `get_market_data_gap(symbol, since, limit)` | `market_data` | GCS 캐시 이후 갭만 fetch |
| `get_cvd_data(symbol, limit)` | `cvd_data` | CVD |
| `get_liquidation_data(symbol, limit)` | `liquidations` | 청산 데이터 |
| `get_latest_fear_greed()` | `fear_greed_index` | `{value, label}` |
| `get_latest_macro_data()` | `macro_data` | `{dgs10, dxy, nasdaq, ust_2s10s_spread}` |
| `get_funding_history(symbol, limit)` | `funding_rates` | funding rate DataFrame |
| `get_latest_narrative_data(symbol, source)` | `narrative_data` | 최신 퍼플렉시티 내러티브 |
| `cleanup_old_data()` | 다수 | 보존기간 초과 삭제 |

### 4.2 GCS Parquet — 장기 히스토리

**파일**: `processors/gcs_parquet.py` — `GCSParquetStore` (싱글톤: `gcs_parquet_store`)
- 활성화: `ENABLE_GCS_ARCHIVE=True` + `GCS_ARCHIVE_BUCKET` 설정 시
- 비활성화 시에도 `_read_local_cache()` 경로(로컬 `cache/gcs_parquet/`)로 폴백

**데이터 로딩 원칙** (SWING 6개월 룩백):

```
1. gcs_parquet_store.load_ohlcv("1m", symbol, months_back)
   └── 로컬 캐시(cache/gcs_parquet/)에서 역사 데이터 로드

2. db.get_market_data_gap(symbol, since=last_cached_ts)
   └── 캐시 마지막 ts ~ 현재 갭만 Supabase에서 페이지네이션 fetch

3. pd.concat([df_cached, df_recent]).drop_duplicates()
```

### 4.3 LightRAG (지식 그래프)

**파일**: `processors/light_rag.py`
- Neo4j Aura: 핫 그래프 (엔티티 관계 / 코러보레이션 토폴로지)
- Zilliz/Milvus: 벡터 인덱스
- Cloudflare Workers AI `bge-reranker-base`: 경계 구간 크로스인코더 (dedup)
- 채널 가중치 없음 — 그래프 토폴로지(corroboration)가 중요도를 결정

### 4.4 Redis / SQLite

| 저장소 | 파일 | 용도 |
|--------|------|------|
| Redis | `utils/redis_client.py` | 실행 캐시, 속도 제한 |
| SQLite | `executors/agent_state_store.py` | AgentStateStore (스냅샷) |
| SQLite | `executors/execution_repository.py` | active_orders OMS |

---

## 5. 스케줄러 잡 맵

**파일**: `scheduler.py`

| 주기 | 잡 ID | 주요 동작 |
|------|-------|----------|
| 매 1분 | `job_1min_tick` | price, funding, microstructure, volatility 수집 |
| 매 1분 | `job_1min_execution` | intent 처리, paper TP/SL 체크, 청산 모니터 (`execution_main.py`) |
| 매 5분 | `job_5m_ws_health` | WebSocket 스레드 생존 확인 + 재시작 |
| 매시 :15 | `job_hourly_monitor` | 시장 상태 소프트 트리거 모니터 |
| 매시 :20 | `job_routine_market_status` | market_status pressure 평가 |
| 매시 :45 | `job_1hour_evaluation` | PerformanceEvaluator 평가 사이클 |
| 1h | `job_1hour_deribit` | Deribit DVOL, PCR, IV Term, 25d Skew |
| 일 01:00 UTC | `job_daily_precision_symbol` | LangGraph 전체 분석 (BTC/ETH 순차 실행) |
| 일 13:00 UTC | `job_daily_precision_symbol` | LangGraph 2차 분석 |
| 일 | `job_daily_fear_greed` | Fear & Greed Index 수집 |
| 일 | `job_daily_coinmetrics` | CoinMetrics 온체인 지표 수집 |
| 일 | `job_daily_archive_to_gcs` | GCS Parquet 아카이빙 |
| 일 | `job_daily_safe_cleanup` | 보존기간 초과 데이터 삭제 |
| 일 | `job_daily_refresh_higher_tf_cache` | 4h/1d TF 캐시 갱신 |
| 트리거 | `job_snapshot_refresh_fast` | AgentState 스냅샷 빠른 갱신 |
| 트리거 | `job_snapshot_refresh_narrative` | AgentState 내러티브 갱신 |
| 트리거 | `job_liquidation_cascade_monitor` | ML 청산 캐스케이드 확률 계산 |

---

## 6. 오케스트레이터

**파일**: `executors/orchestrator.py` — `LangGraph StateGraph`

```
collect_data
  → context_gathering
  → meta_agent          (Cerebras: 시장 레짐 분류)
  → triage              (VLM 투입 여부 결정)
  → generate_chart
  → rule_based_chart
  → vlm_expert          (조건부: 불확실/스트레스 구간만)
  → judge_agent         (Gemini: LONG/SHORT/HOLD + 승률/EV 계산)
  → risk_manager        (Cerebras: 포지션 사이징 / 리스크 예산)
  → execute_trade
  → generate_report
```

**게이트 조건** (deterministic, LLM 추가 호출 없음):

| 파라미터 | 기본값 | 의미 |
|---------|--------|------|
| `JUDGE_MIN_WIN_PROB_PCT` | 55.0% | 코인플립 초과 + 수수료 마진 |
| `JUDGE_MIN_RR_FOR_ENTRY` | 1.9 | `POLICY_MIN_RR=2.0` 허용 오차 |
| `JUDGE_MIN_EV_FOR_ENTRY_PCT` | 0.20% | 최소 기대 수익률 |

---

## 7. 에이전트 레이어

**디렉토리**: `agents/`

| 파일 | 역할 | 모델 |
|------|------|------|
| `meta_agent.py` | 시장 레짐 분류, 컨텍스트 통합 | Cerebras `qwen-3-235b` |
| `judge_agent.py` | 최종 LONG/SHORT/HOLD 결정, 승률·EV 계산 | Gemini `gemini-2.5-pro` |
| `risk_manager_agent.py` | 포지션 사이징, 리스크 예산 배분 | Cerebras `qwen-3-235b` |
| `vlm_geometric_agent.py` | 차트 기하 패턴 분석 (조건부) | Gemini `gemini-2.5-flash` |
| `liquidity_agent.py` | 유동성 구조 분석 | Gemini |
| `macro_options_agent.py` | 매크로 + 옵션 시장 분석 | Gemini |
| `microstructure_agent.py` | 주문서 미시 구조 분석 | Gemini |
| `market_monitor_agent.py` | 시간별 소프트 트리거 모니터 | OpenRouter (무료 티어) |
| `emergency_replan_agent.py` | 포지션 보유 중 긴급 재계획 | Gemini |
| `ai_router.py` | 에이전트 라우팅 유틸 | — |

---

## 8. 컬렉터 레이어

**디렉토리**: `collectors/`

### 8.1 텔레그램 채널 — 4-tier 분류

**파일**: `processors/telegram_batcher.py`

| 티어 | 채널 | 처리 방식 |
|------|------|----------|
| `BTC_ETH_ONCHAIN` | CryptoQuant, Glassnode, Lookonchain | 전량 LightRAG 인제스트 |
| `SMART_MONEY_FLOW` | Whale_Alert, Arkham_Alerter, DeFi_Million | 전량 LightRAG 인제스트 |
| `MARKET_INTELLIGENCE` | Wu_Blockchain, Unfolded, PeckShield | 전량 LightRAG 인제스트 |
| `BREAKING_FILTER` | WalterBloomberg, Tree_News, Watcher_Guru, Cointelegraph, Binance_Announcements | LLM 필터링 → `NO_BTC_ETH_SIGNAL` 시 인제스트 스킵 |

### 8.2 시장 데이터 컬렉터

| 파일 | 데이터 | 주기 |
|------|--------|------|
| `price_collector.py` | 1분봉 OHLCV, CVD | 1분 |
| `funding_collector.py` | Funding Rate | 1분 |
| `microstructure_collector.py` | 주문서 미시 구조 | 1분 |
| `volatility_monitor.py` | 변동성 지표 | 1분 |
| `websocket_collector.py` | 실시간 청산 + 웨일 거래 | 상시 WS |
| `ws_price_feed.py` | 실시간 가격 (주문 실행용) | 상시 WS |
| `ws_user_stream.py` | 체결 확인 User Data Stream | 상시 WS |
| `deribit_collector.py` | DVOL, PCR, IV Term, 25d Skew | 1시간 |
| `dune_collector.py` | 온체인/DEX 매크로 (월 2,500 크레딧 가드) | 1시간 |
| `coinmetrics_collector.py` | 일별 온체인 레짐 오버레이 | 1일 |
| `fear_greed_collector.py` | Fear & Greed Index | 1일 |
| `macro_collector.py` | DGS10, DXY, NASDAQ, 2s10s spread (FRED) | 1일 |
| `perplexity_collector.py` | 시장 내러티브 검색 (200콜/일 쿼타, ~6콜/일 사용) | 분석 트리거 시 |
| `etf_flow_collector.py` | BTC/ETH ETF 자금 흐름 | 분석 트리거 시 |
| `coinglass_collector.py` | OI, OI Divergence, MFI Proxy | 분석 트리거 시 |
| `stablecoin_collector.py` | 스테이블코인 유통량 | 분석 트리거 시 |

---

## 9. 프로세서 레이어

**디렉토리**: `processors/`

| 파일 | 싱글톤 | 역할 |
|------|--------|------|
| `light_rag.py` | `light_rag` | LightRAG 인제스트·쿼리 (Neo4j + Milvus) |
| `gcs_parquet.py` | `gcs_parquet_store` | GCS Parquet 읽기/쓰기 |
| `gcs_archive.py` | `gcs_archive_exporter` | 일별 GCS 아카이빙 |
| `market_snapshot_builder.py` | — | 오케스트레이터용 종합 스냅샷 빌드 |
| `telegram_batcher.py` | — | 4-tier 채널 분류 + 배치 처리 |
| `chart_generator.py` | — | VLM용 캔들 차트 생성 |
| `onchain_signal_engine.py` | — | 온체인 시그널 정규화 |
| `flow_confirm_engine.py` | — | 자금 흐름 확인 |
| `math_engine.py` | `math_engine` | 기술 지표 계산 |
| `cvd_normalizer.py` | — | CVD 정규화 |
| `factor_ic_tracker.py` | — | 팩터 IC 추적 |
| `playbook_service.py` | — | 플레이북 룰 서비스 |
| `portfolio_optimizer.py` | — | 포트폴리오 최적화 |

---

## 10. 익스큐터 레이어

**디렉토리**: `executors/`

| 파일 | 싱글톤 | 역할 |
|------|--------|------|
| `orchestrator.py` | `orchestrator` | LangGraph 파이프라인 진입점 |
| `trade_executor.py` | `trade_executor` | Binance/Upbit 주문 실행 |
| `paper_exchange.py` | `paper_engine` | Paper Trading 엔진 |
| `order_manager.py` | `execution_desk` | OMS 주문 관리 |
| `execution_planner.py` | — | SMART_DCA 지정가 플래닝 |
| `execution_repository.py` | — | SQLite active_orders CRUD |
| `outbox_dispatcher.py` | — | 체결 확인 Outbox 패턴 |
| `risk_policy_engine.py` | — | 정책 헌법 (하드코딩 리스크 한도) |
| `risk_budget_controller.py` | — | 동적 리스크 예산 조율 |
| `agent_state_store.py` | `agent_state_store` | SQLite 에이전트 스냅샷 캐시 |
| `agent_snapshot_refresher.py` | — | 스냅샷 백그라운드 갱신 |
| `data_synthesizer.py` | — | 다중 소스 데이터 합성 |
| `report_generator.py` | `report_generator` | Telegram 분석 리포트 생성 |
| `report_hot_path.py` | — | 핫패스 리포트 (즉시 전송) |
| `cascade_warning_engine.py` | `cascade_warning_engine` | 청산 캐스케이드 경보 |
| `exchange_circuit_breaker.py` | — | 거래소 서킷브레이커 |
| `playbook_guard.py` | — | 플레이북 규칙 검증 |
| `gate_tuner.py` | — | 게이트 임계값 자동 조정 |
| `metrics_logger.py` | — | 실행 메트릭 로깅 |
| `performance_telemetry.py` | — | 성과 텔레메트리 |
| `post_mortem.py` | — | 트레이드 사후 분석 |
| `policy_engine.py` | — | 정책 엔진 (보조) |
| `evaluator_daemon.py` | — | 평가 데몬 |
| `spot_orchestrator.py` | — | 스팟 전용 오케스트레이터 |
| `stats_thresholds.py` | — | 통계 임계값 계산 |

---

## 11. 이벨류에이터 레이어

**디렉토리**: `evaluators/`

| 파일 | 싱글톤 | 역할 |
|------|--------|------|
| `performance_evaluator.py` | `performance_evaluator` | 고정 호라이즌 결과 계산 → `evaluation_outcomes` |
| `feedback_generator.py` | `feedback_generator` | 오답 LLM 피드백 → `feedback_logs` |
| `evaluation_rollup.py` | `evaluation_rollup_service` | 일별 KPI 집계 → `evaluation_rollups_daily` |
| `trade_attribution_engine.py` | — | 트레이드 귀인 분석 |

---

## 12. 리퀴데이션 캐스케이드

**디렉토리**: `liquidation_cascade/`

청산 연쇄 반응 확률을 실시간으로 추정하는 ML 모듈.

| 파일 | 역할 |
|------|------|
| `model.py` | LightGBM 모델 정의 |
| `features.py` | 피처 엔지니어링 |
| `labels.py` | 레이블 생성 |
| `dataset.py` | 학습 데이터셋 빌더 |
| `inference.py` | 실시간 추론 |
| `schema.py` | 데이터 스키마 |

**확률 임계값** (`settings.py`):

| 단계 | 파라미터 | 기본값 |
|------|---------|--------|
| 주시 (WATCH) | `LIQUIDATION_CASCADE_WATCH_PROB` | 0.45 |
| 경고 (WARN) | `LIQUIDATION_CASCADE_WARN_PROB` | 0.60 |
| 확정 (CONFIRM) | `LIQUIDATION_CASCADE_CONFIRM_PROB` | 0.75 |

---

## 13. 텔레그램 봇

**파일**: `bot/telegram_bot.py`

`scheduler.py`에서 daemon 스레드로 기동. 주요 커맨드:

| 커맨드 | 동작 |
|--------|------|
| `/status` | 현재 포지션·잔고 조회 |
| `/analyze` | 즉시 분석 트리거 |
| `/report` | 최신 AI 리포트 조회 |
| `/mode` | SWING / POSITION 모드 확인 |

**MCP 서버**: `mcp_server/server.py` + `mcp_server/tools.py`

| 툴 | 설명 |
|----|------|
| `analyze_market(symbol)` | 핫패스 분석 실행 |
| `get_news_summary(hours)` | 최근 뉴스 요약 |

---

## 14. DB 테이블 전체 목록

### Supabase QUANT (수치 데이터)

| 테이블 | 주요 Writer | 보존 |
|--------|-----------|------|
| `market_data` | `price_collector` | 30일 |
| `cvd_data` | `price_collector`, `websocket_collector` | 30일 |
| `funding_rates` | `funding_collector` | 90일 |
| `liquidations` | `websocket_collector` | 30일 |
| `microstructure` | `microstructure_collector` | 30일 |
| `deribit_data` | `deribit_collector` | 90일 |
| `fear_greed_index` | `fear_greed_collector` | 365일 |
| `macro_data` | `macro_collector` | 365일 |

### Supabase TEXT (텍스트/AI 데이터)

| 테이블 | 주요 Writer | 보존 |
|--------|-----------|------|
| `telegram_messages` | `telegram_listener` | 30일 |
| `narrative_data` | `perplexity_collector` | 90일 |
| `ai_reports` | `report_generator` | 365일 |
| `trade_executions` | `trade_executor`, `paper_engine` | 365일 |
| `evaluation_outcomes` | `performance_evaluator` | 365일 |
| `evaluation_rollups_daily` | `evaluation_rollup_service` | 영구 |
| `feedback_logs` | `feedback_generator` | 365일 |

---

## 15. 핵심 설정 파라미터

**파일**: `config/settings.py` — `Settings` (Pydantic, `.env` 로드)

### 실행 안전

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `PAPER_TRADING_MODE` | `True` | 실주문 차단 (기본 안전 모드) |
| `MAX_LEVERAGE` | `3` | 최대 레버리지 |
| `MAX_ENTRY_DEVIATION_PCT` | `2.0%` | 분석가 대비 현재가 편차 한도 |
| `SMART_DCA_LIMIT_TTL_MINUTES` | `240` | 지정가 미체결 자동 취소 (4h) |
| `BINANCE_PAPER_BALANCE_USD` | `2,000` | 페이퍼 잔고 |

### 분석 캐던스

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `ANALYSIS_INTERVAL_HOURS` | `4` | SWING 분석 사이클 |
| `DAILY_PRECISION_HOURS_UTC` | `"1,13"` | 전체 LangGraph 실행 시각 (UTC) |
| `MONITOR_SOFT_TRIGGER_THRESHOLD` | `0.7` | 시장 모니터 소프트 트리거 임계값 |

### Dune API 비용 가드

| 파라미터 | 기본값 |
|---------|--------|
| `DUNE_GLOBAL_MIN_INTERVAL_MINUTES` | `60` |
| `DUNE_MAX_QUERY_RUNS_PER_DAY` | `24` |
| `DUNE_MAX_QUERIES_PER_RUN` | `5` |
