# Finance Telegram Bot — 에이전트 아키텍처 문서

> BTC/ETH 자동매매 봇 | LightRAG(Graph RAG) + 14개 텔레그램 채널 + Perplexity API
> Last updated: 2026-03-03

---

## 목차

1. [시스템 전체 구조](#1-시스템-전체-구조)
2. [데이터 수집 계층](#2-데이터-수집-계층-collectors)
3. [데이터 처리 계층](#3-데이터-처리-계층-processors)
4. [에이전트 계층](#4-에이전트-계층-agents)
5. [오케스트레이터 파이프라인](#5-오케스트레이터-파이프라인-langgraph)
6. [데이터베이스 스키마](#6-데이터베이스-스키마)
7. [스케줄러 Job 목록](#7-스케줄러-job-목록)
8. [트레이딩 모드 비교](#8-트레이딩-모드-비교)
9. [LightRAG 지식 그래프](#9-lightrag-지식-그래프)
10. [보고서 & 알림 출력](#10-보고서--알림-출력)
11. [핵심 설계 원칙](#11-핵심-설계-원칙)
12. [비용 & 인프라](#12-비용--인프라)

---

## 1. 시스템 전체 구조

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                         FINANCE TELEGRAM BOT — SYSTEM MAP                      ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  ┌─────────────────────────────────────────────────────────────────────────┐    ║
║  │                      EXTERNAL DATA SOURCES                              │    ║
║  │                                                                         │    ║
║  │  📡 Binance Futures   📡 Upbit Spot    📡 Bybit/OKX (OI only)          │    ║
║  │  📊 FRED / yfinance   📊 Deribit       📊 Alternative.me (Fear&Greed)  │    ║
║  │  🔍 Perplexity API    🔗 Dune Analytics 📱 14 Telegram Channels        │    ║
║  └──────────────────────────────────┬──────────────────────────────────────┘    ║
║                                     │                                            ║
║                                     ▼                                            ║
║  ┌─────────────────────────────────────────────────────────────────────────┐    ║
║  │                       COLLECTORS  (데이터 수집)                          │    ║
║  │                                                                         │    ║
║  │  price_collector   funding_collector   telegram_listener   dune_coll.  │    ║
║  │  macro_collector   deribit_collector   perplexity_coll.   fear_greed   │    ║
║  │  microstructure_collector   volatility_monitor   websocket_collector   │    ║
║  └──────────────────────────────────┬──────────────────────────────────────┘    ║
║                                     │                                            ║
║                                     ▼                                            ║
║  ┌─────────────────────────────────────────────────────────────────────────┐    ║
║  │                    SUPABASE PostgreSQL  (Hot Storage)                   │    ║
║  │                                                                         │    ║
║  │  market_data  cvd_data  funding_data  liquidations  microstructure     │    ║
║  │  macro_data   deribit_data  fear_greed  telegram_messages  narrative   │    ║
║  │  dune_query_results  ai_reports  trade_executions  feedback_logs       │    ║
║  └───────┬────────────────────────────────────────────────┬───────────────┘    ║
║          │                                                │                     ║
║          ▼                                                ▼                     ║
║  ┌───────────────────┐                      ┌────────────────────────────┐     ║
║  │ PROCESSORS        │                      │  KNOWLEDGE GRAPH (RAG)     │     ║
║  │                   │                      │                            │     ║
║  │ telegram_batcher  │──────────────────▶  │  Neo4j Aura (그래프 DB)   │     ║
║  │ chart_generator   │                      │  Zilliz Cloud (벡터 DB)   │     ║
║  │ math_engine       │                      │  GCS Parquet (Cold Archive)│     ║
║  │ gcs_parquet       │                      │                            │     ║
║  └───────────────────┘                      └────────────────┬───────────┘     ║
║                                                              │                  ║
║                                     ┌────────────────────────┘                  ║
║                                     ▼                                            ║
║  ┌─────────────────────────────────────────────────────────────────────────┐    ║
║  │                   ORCHESTRATOR  (LangGraph StateGraph)                   │    ║
║  │                                                                         │    ║
║  │  collect_data → perplexity → rag_ingest → [parallel retrieval]        │    ║
║  │  → triage → [expert agents] → generate_chart → judge → risk → trade   │    ║
║  └──────────────────────────────────┬──────────────────────────────────────┘    ║
║                                     │                                            ║
║                                     ▼                                            ║
║  ┌─────────────────────────────────────────────────────────────────────────┐    ║
║  │                          OUTPUT LAYER                                   │    ║
║  │                                                                         │    ║
║  │  📱 Telegram Notification    📊 DB 저장 (ai_reports)                   │    ║
║  │  💹 Trade Executor (Binance Futures / Upbit Spot)                      │    ║
║  │  📝 Report Generator                                                    │    ║
║  └─────────────────────────────────────────────────────────────────────────┘    ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

---

## 2. 데이터 수집 계층 (Collectors)

### 2-1. 수집 주기별 분류

```
┌────────────────────────────────────────────────────────────────────────────┐
│  REAL-TIME  (1분 주기)                                                       │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────┐   ┌──────────────────────────────────┐   │
│  │  price_collector.py          │   │  funding_collector.py            │   │
│  │                              │   │                                  │   │
│  │  Source: Binance Futures     │   │  Source: Binance + Bybit + OKX  │   │
│  │          Upbit Spot          │   │                                  │   │
│  │                              │   │  - Funding Rate (8h interval)   │   │
│  │  - OHLCV 1m 캔들             │   │  - Global OI (3-exchange 합산)  │   │
│  │  - CVD 계산                  │   │  - Long/Short Ratio             │   │
│  │    (Taker Buy - Taker Sell)  │   │                                  │   │
│  │  - Whale CVD 분리 추적       │   │  CVD 공식:                      │   │
│  │                              │   │  delta = TakerBuy - TakerSell   │   │
│  │  Output: market_data         │   │  CVD = cumsum(delta)            │   │
│  │          cvd_data            │   │                                  │   │
│  └──────────────────────────────┘   │  Output: funding_data           │   │
│                                     └──────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────┐   ┌──────────────────────────────────┐   │
│  │  microstructure_collector    │   │  websocket_collector.py          │   │
│  │                              │   │                                  │   │
│  │  Source: Binance Orderbook   │   │  Source: Binance WS Stream       │   │
│  │                              │   │                                  │   │
│  │  - Bid/Ask Spread (%)        │   │  - 실시간 청산 데이터            │   │
│  │  - Orderbook Imbalance       │   │  - Whale Trade ($1M+)           │   │
│  │  - Slippage (100k qty)       │   │                                  │   │
│  │                              │   │  Health Check: 5분마다          │   │
│  │  Output: microstructure_data │   │  Output: liquidations            │   │
│  └──────────────────────────────┘   └──────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│  SCHEDULED  (주기별)                                                         │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────┐   ┌──────────────────────────────────┐   │
│  │  telegram_listener.py        │   │  dune_collector.py               │   │
│  │  (Real-Time + VLM)          │   │  (15분 / 1h / 6h)               │   │
│  │                              │   │                                  │   │
│  │  14개 채널 실시간 감청        │   │  5개 Core Dune 쿼리:            │   │
│  │  ─────────────────           │   │  ┌──────────────────────────┐   │   │
│  │  Tier 1 (On-Chain):          │   │  │ #6638261 ETH 거래소 넷플로│   │   │
│  │  CryptoQuant, Glassnode,     │   │  │ #3378085 BTC ETF Tracking │   │   │
│  │  Lookonchain                 │   │  │ #21689   DEX 거래량(집합) │   │   │
│  │                              │   │  │ #4319    DEX 거래량       │   │   │
│  │  Tier 2 (Smart Money):       │   │  │ #3383110 Lido 스테이킹   │   │   │
│  │  Whale_Alert, Arkham,        │   │  └──────────────────────────┘   │   │
│  │  DeFi_Million                │   │                                  │   │
│  │                              │   │  Budget Guard:                  │   │
│  │  Tier 3 (Intelligence):      │   │  - 최대 24쿼리/일               │   │
│  │  Wu_Blockchain, Unfolded,    │   │  - 글로벌 최소 60분 간격        │   │
│  │  PeckShield                  │   │                                  │   │
│  │                              │   │  Output: dune_query_results     │   │
│  │  Tier 4 (Breaking News):     │   └──────────────────────────────────┘   │
│  │  WalterBloomberg, Tree_News, │                                           │
│  │  Watcher_Guru, Cointelegraph,│   ┌──────────────────────────────────┐   │
│  │  Binance_Announcements       │   │  deribit_collector.py  (1h)     │   │
│  │                              │   │                                  │   │
│  │  VLM 처리 (Gemini Flash):    │   │  - DVOL (변동성 지수)           │   │
│  │  - 캡션 앵커링 추출           │   │  - PCR (Put/Call Ratio)        │   │
│  │  - TREND / MISMATCH 감지     │   │  - IV Term Structure            │   │
│  │                              │   │  - 25d Skew                     │   │
│  │  Output: telegram_messages   │   │  Output: deribit_data           │   │
│  └──────────────────────────────┘   └──────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────┐   ┌──────────────────────────────────┐   │
│  │  perplexity_collector.py     │   │  macro_collector.py  (4h)       │   │
│  │  (분석 사이클마다)            │   │                                  │   │
│  │                              │   │  Source: FRED + yfinance         │   │
│  │  Mode-aware 검색:            │   │                                  │   │
│  │  SWING   → 1-7일 촉매        │   │  - DGS2, DGS10 (국채 금리)     │   │
│  │  POSITION → 2-8주 사이클     │   │  - CPI, FEDFUNDS                │   │
│  │                              │   │  - DXY (달러 인덱스)            │   │
│  │  Live State 주입:            │   │  - Nasdaq, Gold, Oil            │   │
│  │  Fear&Greed + DGS10 + DXY   │   │                                  │   │
│  │  + Funding Rate + Price      │   │  Output: macro_data             │   │
│  │  + 30d Return                │   └──────────────────────────────────┘   │
│  │                              │                                           │
│  │  Quota: 200 calls/day        │   ┌──────────────────────────────────┐   │
│  │  실사용: ~6 calls/day         │   │  fear_greed_collector.py (일 1회)│   │
│  │  Output: narrative_data      │   │  Source: alternative.me API      │   │
│  └──────────────────────────────┘   │  Output: fear_greed_data        │   │
│                                     └──────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘
```

### 2-2. 텔레그램 채널 신뢰도 가중치

```
┌─────────────────────────────────────────────────────────┐
│              Channel Credibility Weights                │
├──────────────────────────────┬──────────┬───────────────┤
│  Channel                     │  Weight  │  Tier         │
├──────────────────────────────┼──────────┼───────────────┤
│  Whale_Alert                 │  1.00    │  Smart Money  │
│  Arkham_Alerter              │  1.00    │  Smart Money  │
│  CryptoQuant                 │  0.95    │  On-Chain     │
│  Glassnode                   │  0.95    │  On-Chain     │
│  Lookonchain                 │  0.90    │  On-Chain     │
│  Wu_Blockchain               │  0.85    │  Intelligence │
│  Tree_News                   │  0.85    │  Breaking     │
│  WalterBloomberg             │  0.85    │  Breaking     │
│  Unfolded                    │  0.80    │  Intelligence │
│  DeFi_Million                │  0.75    │  Smart Money  │
│  Watcher_Guru                │  0.70    │  Breaking     │
│  PeckShield                  │  0.70    │  Intelligence │
│  Cointelegraph               │  0.65    │  Breaking     │
│  Binance_Announcements       │  0.60    │  Breaking     │
├──────────────────────────────┴──────────┴───────────────┤
│  Note: 가중치는 rule-based 적용 없음                     │
│        Neo4j 코로버레이션이 중요도를 자동 결정            │
└─────────────────────────────────────────────────────────┘
```

---

## 3. 데이터 처리 계층 (Processors)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         PROCESSORS OVERVIEW                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ┌─────────────────────────────────┐                                          │
│  │  telegram_batcher.py            │                                          │
│  │                                 │                                          │
│  │  Input: DB의 텔레그램 메시지     │                                          │
│  │  (최근 4시간 lookback)          │                                          │
│  │                                 │  ┌───────────────────────────────────┐  │
│  │  4-Tier 분류 후 LLM 합성:       │  │  light_rag.py (LightRAG Core)    │  │
│  │                                 │  │                                   │  │
│  │  BTC_ETH_ONCHAIN ──────────────▶│  │  ┌─────────────────────────────┐ │  │
│  │  (CryptoQuant, Glassnode,       │  │  │  Extraction (Gemini Flash)  │ │  │
│  │   Lookonchain)                  │  │  │  ─────────────────────────  │ │  │
│  │                                 │  │  │  Entity: institution /      │ │  │
│  │  SMART_MONEY_FLOW ─────────────▶│  │  │   regulator / exchange /   │ │  │
│  │  (Whale_Alert, Arkham,          │  │  │   macro_event / narrative  │ │  │
│  │   DeFi_Million)                 │  │  │                             │ │  │
│  │                                 │  │  │  Relationship: weight 누적 │ │  │
│  │  MARKET_INTELLIGENCE ──────────▶│  │  └──────────┬──────────────────┘ │  │
│  │  (Wu_Blockchain, Unfolded,      │  │             │                     │  │
│  │   PeckShield)                   │  │             ▼                     │  │
│  │                                 │  │  ┌─────────────────────────────┐ │  │
│  │  BREAKING_FILTER ──────────────▶│  │  │  Dual-Level Indexing        │ │  │
│  │  (5개 뉴스 채널)                │  │  │  ─────────────────────────  │ │  │
│  │  → "NO_BTC_ETH_SIGNAL"         │  │  │  Low:  Entity + Direct Rel │ │  │
│  │    Sentinel = ingest 스킵       │  │  │  High: Topic Communities   │ │  │
│  └─────────────────────────────────┘  │  └──────────┬──────────────────┘ │  │
│                                       │             │                     │  │
│                                       │    ┌────────┴───────┐            │  │
│                                       │    ▼                ▼            │  │
│                                       │  ┌──────┐      ┌────────┐        │  │
│                                       │  │Neo4j │      │Zilliz  │        │  │
│                                       │  │Graph │      │Vectors │        │  │
│                                       │  └──────┘      └────────┘        │  │
│                                       └───────────────────────────────────┘  │
│                                                                                │
│  ┌─────────────────────────────────┐   ┌───────────────────────────────────┐  │
│  │  chart_generator.py             │   │  math_engine.py                   │  │
│  │                                 │   │                                   │  │
│  │  Structure-only 차트 생성:      │   │  Technical Indicators 계산:       │  │
│  │  ✅ 캔들스틱 (가격)             │   │  - RSI, MACD, ADX                │  │
│  │  ✅ Pivot point markers         │   │  - Fibonacci Retracement         │  │
│  │  ✅ 대각선 Trendline            │   │  - Swing High/Low                │  │
│  │  ✅ Fibonacci Lines             │   │  - ATR (Volatility)              │  │
│  │  ✅ Swing High/Low             │   │                                   │  │
│  │  ✅ Liquidation markers         │   │  Output: 숫자값만 Judge에 전달   │  │
│  │                                 │   │  (차트에는 오버레이 없음)         │  │
│  │  ❌ RSI/MACD 오버레이 없음      │   └───────────────────────────────────┘  │
│  │  ❌ OI/Funding 오버레이 없음    │                                           │
│  │                                 │   ┌───────────────────────────────────┐  │
│  │  SWING:    4h 캔들 ~10일        │   │  gcs_parquet.py / gcs_archive.py  │  │
│  │  POSITION: 1d 캔들 ~90일        │   │                                   │  │
│  │  Size: 1280 x 800 px            │   │  Cold Archive to GCS:             │  │
│  └─────────────────────────────────┘   │  - DB 만료 전 Parquet 변환        │  │
│                                         │  - 장기 보존 (무제한)             │  │
│                                         └───────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 에이전트 계층 (Agents)

### 4-1. 에이전트 역할 & 모델 분배

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         AGENT ROSTER & MODEL ROUTING                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│   ┌─────────────────────────────────────────────────────────────────────┐    │
│   │  ⭐ JUDGE AGENT  (claude-sonnet-4-6)              agents/judge_agent │    │
│   │                                                                     │    │
│   │  역할: 모든 에이전트 분석을 종합 → 최종 매매 결정                    │    │
│   │                                                                     │    │
│   │  입력:                                                               │    │
│   │  ├── 📊 Chart Image (1280×800px) ← 유일하게 이미지 받는 에이전트    │    │
│   │  ├── 📈 Market Data (OHLCV + Indicators as text)                   │    │
│   │  ├── 💬 Blackboard (Expert 3종 요약)                                │    │
│   │  ├── 💰 Funding Context (OI + Funding Rate)                        │    │
│   │  ├── 🌐 Narrative Context (Perplexity + RAG)                       │    │
│   │  ├── 🏛️ Macro Context (FRED + Options)                             │    │
│   │  ├── 📋 Previous Decision (일관성 유지)                             │    │
│   │  └── 🎯 Trading Mode Rules (SWING vs POSITION)                     │    │
│   │                                                                     │    │
│   │  출력: { decision, allocation_pct, leverage, entry, SL, TP,        │    │
│   │          win_prob, ev_ratio, counter_scenario, reasoning }          │    │
│   └─────────────────────────────────────────────────────────────────────┘    │
│                                                                                │
│   ┌──────────────────────┐  ┌──────────────────────┐  ┌─────────────────┐   │
│   │  LIQUIDITY AGENT     │  │  MICROSTRUCTURE AGENT│  │  MACRO/OPTIONS  │   │
│   │  (gemini-flash)      │  │  (gemini-flash)      │  │  AGENT (flash)  │   │
│   │                      │  │                      │  │                 │   │
│   │  CVD anomaly 분석    │  │  Orderbook 분석       │  │  Deribit + FRED │   │
│   │  Whale activity      │  │  Spread/Imbalance    │  │  DVOL, PCR     │   │
│   │  Liquidation density │  │  Slippage estimation │  │  IV Term, Skew │   │
│   │                      │  │                      │  │                 │   │
│   │  anomaly:            │  │  anomaly:            │  │  anomaly:       │   │
│   │  whale_cvd_divergence│  │  microstr_imbalance  │  │  options_panic  │   │
│   │  liquidation_hunting │  │  fake_wall           │  │  macro_diverge  │   │
│   │  none                │  │  none                │  │  none           │   │
│   │                      │  │                      │  │                 │   │
│   │  confidence: 0.0-1.0 │  │  directional_bias:   │  │  options_bias:  │   │
│   │  target_entry: float │  │  UP|DOWN|NEUTRAL     │  │  BULL|BEAR|NEUT │   │
│   └──────────────────────┘  └──────────────────────┘  └─────────────────┘   │
│                                                                                │
│   ┌──────────────────────┐  ┌──────────────────────┐  ┌─────────────────┐   │
│   │  META AGENT          │  │  RISK MANAGER AGENT  │  │  MARKET MONITOR │   │
│   │  (gemini-flash)      │  │  (gemini-flash)      │  │  (gemini-flash) │   │
│   │                      │  │                      │  │                 │   │
│   │  시장 레짐 컨텍스트   │  │  포지션 사이징 검증   │  │  1시간 주기      │   │
│   │  High-risk 여부      │  │  드로우다운 한도      │  │  Free-First     │   │
│   │  Market phase        │  │  레버리지 검증        │  │  시장 요약      │   │
│   └──────────────────────┘  └──────────────────────┘  └─────────────────┘   │
│                                                                                │
│   ┌──────────────────────────────────────────────────────────────────────┐   │
│   │  AI ROUTER  (agents/ai_router.py)                                    │   │
│   │                                                                      │   │
│   │  Role               → Model                      Cost Tier          │   │
│   │  ─────────────────────────────────────────────────────────────────  │   │
│   │  liquidity          → gemini-3-flash-preview      💰 Low            │   │
│   │  microstructure     → gemini-3-flash-preview      💰 Low            │   │
│   │  macro              → gemini-3-flash-preview      💰 Low            │   │
│   │  rag_extraction     → gemini-3-flash-preview      💰 Low            │   │
│   │  vlm_telegram_chart → gemini-3-flash-preview      💰 Low            │   │
│   │  vlm_geometric      → gemini-3.1-pro-preview      💎 Premium        │   │
│   │  judge              → claude-sonnet-4-6            🧠 Best Logic     │   │
│   │  chat (Tg Bot)      → gemini-3-flash-preview      💰 Low            │   │
│   └──────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 4-2. Judge Agent 결정 로직

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                       JUDGE AGENT DECISION FRAMEWORK                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│   SWING Mode                          POSITION Mode                           │
│   ─────────────────────────────────   ─────────────────────────────────────  │
│   Top-down MTF 분석:                   Top-down MTF 분석:                      │
│   1d bias (전체 방향성)                 1w macro bias                           │
│     └→ 4h setup (진입존)               1d structural shift                    │
│          └→ 1h entry (정밀 진입)                                               │
│                                                                                │
│   Key Rules:                           Key Rules:                             │
│   · Fib 38.2/50/61.8% = 핵심 존        · 매크로 사이클 + ATH = 절대 경계선      │
│   · Funding >0.05% = 역발상 상단        · 극단 음수 Funding (수개월) = 바닥      │
│   · OI-Price diverge = 취약 움직임     · DVOL 급등 = 항복/바닥                  │
│   · CVD diverge 추적                   · CME Basis 백워데이션 = 기관 헤지       │
│                                         · Deribit Skew = OTC 포지션 추적       │
│   포지션 사이징:                         포지션 사이징:                           │
│   · Kelly의 10~25%                     · Kelly의 5~15% (초보수적)              │
│   · 레버리지: 1~3x                      · 레버리지: 1x or 최대 1.5x             │
│   · SL: 가격의 5~10%                   · SL: 10~25% 넓은 폭                    │
│   · RR: 최소 1.5:1 또는 2:1            · RR: 3:1+                              │
│                                                                                │
│   Conviction Hierarchy (충돌 해소):                                            │
│   SWING:     Microstructure > Liquidity > Macro > Visual                     │
│   POSITION:  Macro/Options > Liquidity > Visual > Microstructure             │
│                                                                                │
│   Falsifiability Rule (V9 필수):                                               │
│   "최종 결정 전, 내 주된 편향에 반하는 가장 강한 증거를 명시적으로 서술하라"      │
│                                                                                │
│   Output JSON:                                                                 │
│   {                                                                            │
│     "decision": "LONG | SHORT | HOLD | CANCEL_AND_CLOSE",                    │
│     "allocation_pct": 0~100,                                                  │
│     "leverage": 1~3,                                                          │
│     "entry_price": float,                                                     │
│     "stop_loss": float,                                                       │
│     "take_profit": float,                                                     │
│     "hold_duration": "2-5 days | 2-4 weeks",                                 │
│     "win_probability_pct": 0~100,                                             │
│     "expected_profit_pct": float,                                             │
│     "expected_loss_pct": float,                                               │
│     "reasoning": {                                                            │
│       "counter_scenario": "반대 시나리오",                                      │
│       "technical": "MTF 분석",                                                │
│       "derivatives": "Funding/OI/CVD",                                       │
│       "experts": "Blackboard 요약",                                           │
│       "narrative": "뉴스+RAG 컨텍스트",                                        │
│       "final_logic": "최종 합성"                                               │
│     }                                                                         │
│   }                                                                            │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. 오케스트레이터 파이프라인 (LangGraph)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│             ORCHESTRATOR LANGGRAPH PIPELINE  (executors/orchestrator.py)      │
└──────────────────────────────────────────────────────────────────────────────┘

   INPUT: Symbol (BTCUSDT / ETHUSDT) + Mode (SWING / POSITION)
      │
      ▼
  ┌─────────────────────────────────────────────────────┐
  │  [1] COLLECT_DATA                                    │
  │  - DB에서 최신 OHLCV 로드                            │
  │    SWING: 10,000 캔들  /  POSITION: 60,000 캔들     │
  │  - CVD, Liquidation, Funding, Microstructure 로드   │
  └────────────────────────┬────────────────────────────┘
                           │
                           ▼
  ┌─────────────────────────────────────────────────────┐
  │  [2] PERPLEXITY_SEARCH                               │
  │  - search_market_narrative(symbol) 호출              │
  │  - Mode-aware 프롬프트 + Live market state 주입      │
  │  - Fear&Greed, DGS10, DXY, Funding, Price, 30d Ret  │
  └────────────────────────┬────────────────────────────┘
                           │
                           ▼
  ┌─────────────────────────────────────────────────────┐
  │  [3] RAG_INGEST                                      │
  │  - Perplexity 내러티브 → Neo4j/Milvus 인제스트       │
  │  - telegram_batcher 실행 (4-tier 분류 + 배치 합성)   │
  │  - 엔티티/관계 추출 → 지식 그래프 업데이트            │
  └────────────────────────┬────────────────────────────┘
                           │
                           ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  [4] PARALLEL RETRIEVAL  (동시 실행)                                     │
  │                                                                         │
  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
  │  │ funding_ctx  │  │  cvd_ctx     │  │  liq_ctx     │                  │
  │  │ OI-Price div │  │  Whale 누적  │  │  청산 밀집도  │                  │
  │  └──────────────┘  └──────────────┘  └──────────────┘                  │
  │                                                                         │
  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
  │  │  rag_query   │  │  macro_ctx   │  │  deribit_ctx │                  │
  │  │  Neo4j+Vec   │  │  FRED+yf     │  │  DVOL,PCR,Sk │                  │
  │  └──────────────┘  └──────────────┘  └──────────────┘                  │
  └────────────────────────────────────────┬────────────────────────────────┘
                                           │
                                           ▼
  ┌─────────────────────────────────────────────────────┐
  │  [5] TRIAGE                                          │
  │  - 모든 데이터를 에이전트 입력 형식으로 포맷         │
  │  - 인디케이터: 숫자값만 (시각화 아님)                │
  └────────────────────────┬────────────────────────────┘
                           │
                           ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  [6] PARALLEL EXPERTS  (동시 실행)                                       │
  │                                                                         │
  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐      │
  │  │  liquidity_agent │  │ microstr._agent  │  │  macro_agent     │      │
  │  │  (Gemini Flash)  │  │  (Gemini Flash)  │  │  (Gemini Flash)  │      │
  │  │                  │  │                  │  │                  │      │
  │  │  CVD + 청산      │  │  Orderbook       │  │  Deribit + FRED  │      │
  │  └──────────────────┘  └──────────────────┘  └──────────────────┘      │
  │              └──────────────────┬──────────────────┘                   │
  │                           Blackboard 합성                               │
  └────────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
  ┌─────────────────────────────────────────────────────┐
  │  [7] GENERATE_CHART                                  │
  │  - structure-only 차트 생성 (1280×800px)             │
  │  - 캔들 + Pivot + Trendline + Fibonacci + Liq.Mark  │
  │  - RSI/MACD/OI 오버레이 없음 (텍스트로 Judge에 전달)│
  └────────────────────────┬────────────────────────────┘
                           │
                           ▼
  ┌─────────────────────────────────────────────────────┐
  │  [8] JUDGE_AGENT  ⭐                                 │
  │  - Claude Sonnet 4.6                                │
  │  - 유일하게 Chart Image 수신                        │
  │  - 모든 컨텍스트 + 차트 → 최종 결정                 │
  │  - Falsifiability 분석 필수                         │
  └────────────────────────┬────────────────────────────┘
                           │
                           ▼
  ┌─────────────────────────────────────────────────────┐
  │  [9] RISK_MANAGER                                    │
  │  - 포지션 사이징 검증                                │
  │  - 드로우다운 한도 체크                              │
  │  - 레버리지 범위 확인                                │
  └────────────────────────┬────────────────────────────┘
                           │
                           ▼
  ┌─────────────────────────────────────────────────────┐
  │  [10] EXECUTE_TRADE                                  │
  │  - LONG/SHORT → Binance Futures / Upbit Spot       │
  │  - CANCEL_AND_CLOSE → 긴급 청산                     │
  │  - HOLD → 스킵                                      │
  │  - Paper Trading → paper_engine 위임                │
  └────────────────────────┬────────────────────────────┘
                           │
                           ▼
  ┌─────────────────────────────────────────────────────┐
  │  [11] GENERATE_REPORT                                │
  │  - DB 저장 (ai_reports)                             │
  │  - 텔레그램 요약 메시지 전송                          │
  │  - 텔레그램 상세 분석 버튼 포함                       │
  └─────────────────────────────────────────────────────┘

   OUTPUT: Trade Receipt + Telegram Notification
```

---

## 6. 데이터베이스 스키마

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Supabase PostgreSQL — 테이블 목록 & 보존 기간                                 │
├─────────────────────────┬──────────────────────────────────────┬─────────────┤
│  Table                  │  Key Columns                         │  Retention  │
├─────────────────────────┼──────────────────────────────────────┼─────────────┤
│  market_data            │  timestamp, symbol, exchange,         │  30일       │
│                         │  open/high/low/close/volume,          │             │
│                         │  taker_buy_volume, quote_volume       │             │
├─────────────────────────┼──────────────────────────────────────┼─────────────┤
│  cvd_data               │  timestamp, symbol,                   │  30일       │
│                         │  volume_delta, cvd,                   │             │
│                         │  whale_buy/sell_vol, whale_cvd        │             │
├─────────────────────────┼──────────────────────────────────────┼─────────────┤
│  liquidations           │  timestamp, symbol,                   │  30일       │
│                         │  long_liq_qty, short_liq_qty,         │             │
│                         │  liq_price_levels                     │             │
├─────────────────────────┼──────────────────────────────────────┼─────────────┤
│  funding_data           │  timestamp, symbol,                   │  30일       │
│                         │  funding_rate, next_funding_time,     │             │
│                         │  mark_price, global_oi (3 exch. sum)  │             │
├─────────────────────────┼──────────────────────────────────────┼─────────────┤
│  microstructure_data    │  timestamp, symbol, exchange,         │  30일       │
│                         │  bid_ask_spread_pct,                  │             │
│                         │  orderbook_imbalance,                 │             │
│                         │  slippage_100k_qty                    │             │
├─────────────────────────┼──────────────────────────────────────┼─────────────┤
│  deribit_data           │  timestamp, symbol,                   │  30일       │
│                         │  dvol, pcr,                           │             │
│                         │  iv_term_structure_json, skew_25d     │             │
├─────────────────────────┼──────────────────────────────────────┼─────────────┤
│  macro_data             │  timestamp, source,                   │  30일       │
│                         │  dgs2, dgs10, cpiaucsl, fedfunds,     │             │
│                         │  dxy, nasdaq, gold, oil               │             │
├─────────────────────────┼──────────────────────────────────────┼─────────────┤
│  fear_greed_data        │  timestamp, value, label              │  30일       │
├─────────────────────────┼──────────────────────────────────────┼─────────────┤
│  telegram_messages      │  channel, message_id, text,           │  90일       │
│                         │  media_urls, media_descriptions       │             │
├─────────────────────────┼──────────────────────────────────────┼─────────────┤
│  narrative_data         │  timestamp, symbol, source,           │  90일       │
│                         │  summary, sentiment,                  │             │
│                         │  bullish/bearish_factors, reasoning   │             │
├─────────────────────────┼──────────────────────────────────────┼─────────────┤
│  dune_query_results     │  query_id, collected_at,              │  30일       │
│                         │  result_json                          │             │
├─────────────────────────┼──────────────────────────────────────┼─────────────┤
│  ai_reports             │  id, symbol, timestamp,               │  365일      │
│                         │  market_data (JSON), bull/bear,       │             │
│                         │  risk_assessment, final_decision (JSON)│            │
├─────────────────────────┼──────────────────────────────────────┼─────────────┤
│  trade_executions       │  id, symbol, direction, leverage,     │  영구       │
│                         │  entry_price, exit_price, pnl_pct     │             │
├─────────────────────────┼──────────────────────────────────────┼─────────────┤
│  feedback_logs          │  id, trade_id, accuracy_assessment,   │  영구       │
│                         │  lessons_learned, sentiment           │             │
└─────────────────────────┴──────────────────────────────────────┴─────────────┘
```

---

## 7. 스케줄러 Job 목록

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     SCHEDULER JOBS  (scheduler.py — APScheduler)              │
├────────────────────────────┬────────────────────────────┬─────────────────────┤
│  Job                       │  Trigger                   │  Purpose            │
├────────────────────────────┼────────────────────────────┼─────────────────────┤
│  job_1min_tick             │  every 1 min               │  Price, Funding,    │
│                            │                            │  Microstructure,    │
│                            │                            │  Volatility 수집    │
├────────────────────────────┼────────────────────────────┼─────────────────────┤
│  job_1min_execution        │  every 1 min               │  Order 처리,        │
│                            │                            │  청산 체크, TP/SL   │
├────────────────────────────┼────────────────────────────┼─────────────────────┤
│  job_15min_dune            │  every 15 min              │  Dune 온체인 쿼리   │
├────────────────────────────┼────────────────────────────┼─────────────────────┤
│  job_1hour_deribit         │  every 1 hour              │  DVOL, PCR, IV,    │
│                            │                            │  Skew 옵션 데이터   │
├────────────────────────────┼────────────────────────────┼─────────────────────┤
│  job_8hour_funding_fee     │  0/8/16 UTC                │  Paper Trading      │
│                            │                            │  펀딩비 차감 시뮬   │
├────────────────────────────┼────────────────────────────┼─────────────────────┤
│  job_daily_fear_greed      │  daily 00:15 UTC           │  Fear&Greed 지수    │
├────────────────────────────┼────────────────────────────┼─────────────────────┤
│  job_daily_precision       │  daily 00:00 UTC           │  POSITION 플레이북  │
│  job_hourly_monitor        │  every hour :15            │  SWING/POS 모니터링 │
├────────────────────────────┼────────────────────────────┼─────────────────────┤
│  job_routine_market_status │  every 1 hour              │  Free-First 시장   │
│                            │                            │  요약 (텔레그램)    │
├────────────────────────────┼────────────────────────────┼─────────────────────┤
│  job_24hour_evaluation     │  daily 00:30 UTC           │  매매 피드백 생성   │
│                            │                            │  (에피소딕 메모리)  │
├────────────────────────────┼────────────────────────────┼─────────────────────┤
│  job_1hour_evaluation      │  every 1 hour              │  RAG 에피소딕 평가  │
├────────────────────────────┼────────────────────────────┼─────────────────────┤
│  job_1hour_telegram        │  every 1 hour              │  텔레그램 배칭 +    │
│                            │                            │  Perplexity 삼각검증│
├────────────────────────────┼────────────────────────────┼─────────────────────┤
│  job_daily_cleanup         │  daily 01:00 UTC           │  DB 정리 + GCS      │
│                            │                            │  아카이브 + RAG 정리 │
└────────────────────────────┴────────────────────────────┴─────────────────────┘

   Cold Start 시 즉시 실행:
   price, funding, microstructure, volatility, deribit, fear_greed

   백그라운드 데몬 스레드:
   ├── Telegram Bot (bot/telegram_bot.py)
   ├── Real-time Telegram Listener (V13.1 Alpha)
   └── WebSocket Health Check (5분마다 자동 재시작)
```

---

## 8. 트레이딩 모드 비교

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        TRADING MODE COMPARISON                                │
├────────────────────────────────────┬─────────────────────────────────────────┤
│           SWING MODE               │           POSITION MODE                  │
├────────────────────────────────────┼─────────────────────────────────────────┤
│  보유 기간: 일 ~ 2주               │  보유 기간: 주 ~ 수개월                  │
│  분석 주기: 8시간 (0/8/16 UTC)     │  분석 주기: 일 1회 (00:00 UTC)          │
│  차트 TF:  4h 캔들                 │  차트 TF:  1d 캔들                      │
│  분석 TF:  1h / 4h / 1d            │  분석 TF:  4h / 1d / 1w                │
│  캔들 한도: 10,000개               │  캔들 한도: 60,000개                    │
│  레버리지: 1x ~ 3x                 │  레버리지: 1x (최대 1.5x)              │
│  포지션%:  Kelly 10~25%            │  포지션%:  Kelly 5~15%                 │
│  SL 폭:    5~10%                   │  SL 폭:    10~25%                      │
│  RR 비율:  최소 1.5:1 또는 2:1     │  RR 비율:  3:1+                        │
│  Perplexity: 1~7일 촉매 탐색       │  Perplexity: 2~8주 사이클 탐색         │
│  핵심 지표: Fibonacci + CVD + Fnd  │  핵심 지표: DVOL + CME Basis + ETF     │
└────────────────────────────────────┴─────────────────────────────────────────┘
```

---

## 9. LightRAG 지식 그래프

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         LIGHTRAG KNOWLEDGE GRAPH                              │
└──────────────────────────────────────────────────────────────────────────────┘

  INGEST SOURCES                   EXTRACTION                    STORAGE
  ──────────────                   ──────────                    ───────
                                                          ┌──────────────────┐
  Telegram Messages ──────────────▶  Gemini Flash         │   NEO4J AURA     │
  (14채널 → 4-tier 분류 후 배치)       (BTC/ETH 전용)      │                  │
                                      │                   │  Entity Node     │
  Perplexity Narrative ─────────────▶ │ Entity 추출:      │  ─────────────   │
  (Symbol별 내러티브)                  │ - institution      │  name            │
                                      │ - regulator        │  entity_type     │
                                      │ - exchange         │  description     │
                                      │ - macro_event      │  status:         │
                                      │ - narrative        │   PENDING (1 src)│
                                      │                   │   CORROBORATED   │
                                      │ Relationship 추출: │   (2+ src)       │
                                      │ weight = 누적      │                  │
                                      │                   │  Relationship    │
                                      ▼                   │  ─────────────   │
                               ┌────────────────┐        │  source → target │
                               │  Dual-Level    │        │  rel_type        │
                               │  Indexing      │        │  weight (누적)   │
                               │                │        │  description     │
                               │  Low-Level:    │        └──────────────────┘
                               │  Entity +      │
                               │  Direct Rel    │        ┌──────────────────┐
                               │                │        │   ZILLIZ CLOUD   │
                               │  High-Level:   │        │                  │
                               │  Topic         │────────▶ Collections:     │
                               │  Communities   │        │  telegram_entities│
                               └────────────────┘        │  events          │
                                                         │                  │
                                                         │  Embedding:      │
                                                         │  Voyage AI       │
                                                         │                  │
                                                         │  Reranker:       │
                                                         │  Cloudflare      │
                                                         │  Workers AI      │
                                                         └──────────────────┘

  TRIANGULATION WORKFLOW:
  ──────────────────────
  1. Neo4j에서 PENDING 관계 조회 (get_triangulation_candidates)
  2. Perplexity search_targeted() 로 웹 검증
  3. 검증 성공 시 → CORROBORATED 상태로 업데이트
  4. weight 누적 (같은 엣지 반복 등장 시)

  QUERY AT ANALYSIS TIME:
  ──────────────────────
  query_context(query_text, top_k) 호출
  → Zilliz 벡터 검색 + Neo4j 그래프 탐색
  → Judge Agent에 관련 컨텍스트 전달
```

---

## 10. 보고서 & 알림 출력

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         REPORT & NOTIFICATION FORMAT                          │
└──────────────────────────────────────────────────────────────────────────────┘

  TELEGRAM 요약 메시지:
  ─────────────────────
  ┌────────────────��─────────────────────────────────────────────────────┐
  │  🟢 [ SWING (스윙) ] AI 분석 리포트 | BTCUSDT                        │
  │  🕒 2026-03-03 08:00 UTC                                             │
  │                                                                      │
  │  🎯 최종 결정: 📈 강세 (LONG)   (확신도: 85%)                        │
  │                                                                      │
  │  🏁 진입가:  $42,500                                                 │
  │  🛑 손절가:  $40,800   (-4.0%)                                       │
  │  🏆 목표가:  $45,200   (+6.4%)   RR = 1.6:1                         │
  │  ⚡ 레버리지: 2x  |  비중: 20%                                       │
  │  ⏱️ 예상 보유: 3~5일                                                  │
  │                                                                      │
  │  💡 요약:                                                             │
  │  Fib 61.8% 지지 컨펌 + CVD whale 누적 + 긍정적 매크로 다이버전스.    │
  │  펀딩레이트 중립권, OI 소폭 감소 (포지션 청소 완료).                  │
  │                                                                      │
  │  [📊 상세 분석 보기]  [📋 현재 포지션]                               │
  └──────────────────────────────────────────────────────────────────────┘

  TELEGRAM 상세 분석 메시지:
  ───────────────────────────
  ┌──────────────────────────────────────────────────────────────────────┐
  │  🔍 DEEP ANALYSIS | BTCUSDT                                          │
  │                                                                      │
  │  📏 기술적 분석                                                       │
  │  1d 편향: 상승 구조 유지 (HH/HL 패턴)                                │
  │  4h 설업: Fib 61.8% ($42,400) 리테스트 진행 중                       │
  │  1h 진입: RSI 42 과매도 반등 + 상승 다이버전스                        │
  │                                                                      │
  │  ⛓️ 파생상품 심리                                                     │
  │  Funding: +0.01% (중립) | OI: $28.4B (-2.1% 24h)                   │
  │  CVD: 최근 12h +$180M (기관 누적)                                    │
  │  청산: $41,200~$43,800 구간 청산 클러스터 부재                        │
  │                                                                      │
  │  🧠 전문가 스웜                                                       │
  │  Liquidity: whale_cvd_divergence 감지, 신뢰도 0.82                   │
  │  Microstructure: orderbook BULLISH, ask 벽 얕음                      │
  │  Macro/Options: DVOL 하락 중, PCR 0.68 (콜 우세)                    │
  │                                                                      │
  │  🌐 내러티브 & RAG                                                   │
  │  Perplexity: ETF 넷플로우 플러스 전환, FOMC 비둘기파 신호             │
  │  RAG: BlackRock 매집 패턴 CORROBORATED (3 독립 소스)                  │
  │                                                                      │
  │  ⚠️ 반대 시나리오 (Falsifiability)                                    │
  │  "DXY 급등 재개 시 지지선 이탈 가능. $41,800 이탈 = 손절 확정"       │
  └──────────────────────────────────────────────────────────────────────┘
```

---

## 11. 핵심 설계 원칙

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         CORE DESIGN PRINCIPLES                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  1. NO PRE-ASSIGNED CHANNEL WEIGHTS                                           │
│     채널별 가중치 수동 할당 없음.                                              │
│     Neo4j 그래프 토폴로지(코로버레이션)가 중요도를 자동 결정.                  │
│     - 단일 소스  → PENDING      (미검증)                                      │
│     - 2+ 독립 소스 → CORROBORATED  (신뢰)                                    │
│     - 반복 엣지  → weight 누적  (자연스러운 중요도 상승)                       │
│                                                                                │
│  2. NO RULE-BASED MARKET PHASE CLASSIFIER                                     │
│     "상승장/하락장/횡보" 규칙 기반 분류 없음.                                   │
│     Fear&Greed, DGS10, DXY, Funding Rate를 프롬프트에 직접 주입.              │
│     LLM이 현재 상태를 읽고 스스로 컨텍스트 적응.                               │
│                                                                                │
│  3. BTC/ETH FOCUS ONLY                                                        │
│     알트코인 시그널 → 수집 단계에서 폐기.                                       │
│     telegram_batcher: NO_BTC_ETH_SIGNAL 센티넬로 스킵.                        │
│     Perplexity: BTC/ETH 전용 엔티티 타입만 검색.                               │
│                                                                                │
│  4. TEXT-ONLY EXPERTS, IMAGE-ONLY JUDGE                                       │
│     전문가 에이전트 (Gemini Flash): 텍스트만 (비용 최소화)                      │
│     Judge (Claude Sonnet 4.6): 유일하게 차트 이미지 수신                       │
│     차트에는 인디케이터 오버레이 없음 → 숫자값으로 Judge에 별도 전달.           │
│                                                                                │
│  5. FALSIFIABILITY RULE (V9)                                                  │
│     Judge는 최종 결정 전 반드시 반대 증거를 명시해야 함.                        │
│     "내 결정에 반하는 가장 강한 증거는 무엇인가?"                               │
│     Meta-Agent가 High-Risk 레짐 신호 시 → 더 높은 확신도 요구.                 │
│                                                                                │
│  6. HOT / COLD STORAGE SEPARATION                                             │
│     Hot (PostgreSQL): 30~365일 보존                                           │
│     Cold (GCS Parquet): 만료 전 아카이브 → 무제한 보존                         │
│     Graph (Neo4j): 영구 (200K 노드 Aura Free 한도 내)                         │
│                                                                                │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 12. 비용 & 인프라

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          INFRASTRUCTURE & COST                                │
├────────────────────────────────────┬────────────────────────┬─────────────────┤
│  Service                           │  Plan / Limit          │  Est. Monthly   │
├────────────────────────────────────┼────────────────────────┼─────────────────┤
│  Supabase PostgreSQL               │  Free Tier / Pro       │  $0 ~ $25       │
│  Neo4j Aura Free                   │  200K nodes, 400K rels │  $0             │
│  Zilliz Cloud (Milvus)             │  1M vectors, 2 coll.   │  $0             │
│  Google Cloud Storage              │  Standard              │  ~$0.50         │
│  Perplexity API                    │  200 calls/day incl.   │  ~$5            │
│  Gemini Flash (Extraction+Experts) │  LightRAG + 3 Experts  │  ~$0.55         │
│  Gemini 3.1 Pro (VLM)             │  Chart geometric only  │  ~$0.10         │
│  Claude Sonnet 4.6 (Judge)        │  ~6 calls/day          │  ~$2            │
│  Voyage AI (Embeddings)            │  Free tier             │  ~$0.05         │
│  Cloudflare Workers AI (Reranker)  │  10K neurons/day free  │  $0             │
│  Dune Analytics                    │  Free tier             │  $0             │
│  Binance / Upbit APIs              │  Free                  │  $0             │
│  alternative.me Fear&Greed         │  Free                  │  $0             │
├────────────────────────────────────┼────────────────────────┼─────────────────┤
│  TOTAL ESTIMATED                   │                        │  ~$8 ~ $33/월   │
└────────────────────────────────────┴────────────────────────┴─────────────────┘

  Key File Index:
  ──────────────────────────────────────────────────────────────────
  config/settings.py           모든 설정 + 모델 라우팅 (403 lines)
  config/database.py           Supabase PostgreSQL 클라이언트 (503 lines)
  scheduler.py                 APScheduler 잡 오케스트레이션 (502 lines)
  executors/orchestrator.py    LangGraph StateGraph 파이프라인 (56KB+)
  executors/report_generator   리포트 생성 + 텔레그램 발송
  executors/trade_executor.py  멀티 거래소 주문 실행
  executors/paper_exchange.py  Paper Trading 시뮬레이터
  agents/judge_agent.py        최종 결정자 (Claude Sonnet 4.6)
  agents/liquidity_agent.py    CVD + 청산 전문가 (Gemini Flash)
  agents/microstructure_agent  Orderbook 전문가 (Gemini Flash)
  agents/macro_options_agent   Options + 매크로 전문가 (Gemini Flash)
  agents/ai_router.py          역할별 모델 라우팅
  collectors/price_collector   OHLCV + CVD (Binance + Upbit)
  collectors/funding_collector Funding + Global OI (3 거래소 합산)
  collectors/perplexity_coll.  내러티브 수집 (Mode-aware)
  collectors/telegram_listener 14개 채널 실시간 + VLM 차트 분석
  collectors/dune_collector.py 온체인 스냅샷 (5 쿼리)
  collectors/macro_collector   FRED + yfinance (4h 주기)
  processors/light_rag.py      Graph RAG 핵심 (Neo4j + Milvus)
  processors/chart_generator   Structure-only 차트 생성
  processors/telegram_batcher  4-tier BTC/ETH 시그널 분류
  bot/telegram_bot.py          인터랙티브 텔레그램 봇
  ──────────────────────────────────────────────────────────────────
```

---

*이 문서는 코드베이스 전체 탐색 기반으로 작성된 아키텍처 레퍼런스입니다.*
*업데이트 필요 시 `AGENTS_ARCHITECTURE.md`를 직접 수정하거나 Claude에게 요청하세요.*
