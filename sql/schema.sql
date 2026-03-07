CREATE TABLE IF NOT EXISTS market_data (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(timestamp, symbol, exchange)
);

CREATE INDEX idx_market_data_symbol_timestamp ON market_data(symbol, timestamp DESC);
CREATE INDEX idx_market_data_timestamp ON market_data(timestamp DESC);
CREATE INDEX idx_market_data_exchange ON market_data(exchange, symbol, timestamp DESC);

-- CVD (Cumulative Volume Delta) data from Binance Futures
-- Taker Buy/Sell volume breakdown for whale activity detection
CREATE TABLE IF NOT EXISTS cvd_data (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    taker_buy_volume DECIMAL(20, 8) NOT NULL DEFAULT 0,
    taker_sell_volume DECIMAL(20, 8) NOT NULL DEFAULT 0,
    volume_delta DECIMAL(20, 8) NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(timestamp, symbol)
);

CREATE INDEX idx_cvd_data_symbol_timestamp ON cvd_data(symbol, timestamp DESC);

-- Funding data with Global OI (Binance + Bybit + OKX)
CREATE TABLE IF NOT EXISTS funding_data (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    funding_rate DECIMAL(10, 8),
    next_funding_time BIGINT,
    open_interest DECIMAL(20, 8),
    open_interest_value DECIMAL(20, 2),
    oi_binance DECIMAL(20, 2) DEFAULT 0,
    oi_bybit DECIMAL(20, 2) DEFAULT 0,
    oi_okx DECIMAL(20, 2) DEFAULT 0,
    long_short_ratio DECIMAL(10, 4),
    long_account DECIMAL(10, 4),
    short_account DECIMAL(10, 4),
    basis_pct DECIMAL(10, 6),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(timestamp, symbol)
);

CREATE INDEX idx_funding_data_symbol_timestamp ON funding_data(symbol, timestamp DESC);

CREATE TABLE IF NOT EXISTS telegram_messages (
    id BIGSERIAL PRIMARY KEY,
    channel VARCHAR(100) NOT NULL,
    message_id BIGINT NOT NULL,
    text TEXT NOT NULL,
    views INTEGER DEFAULT 0,
    forwards INTEGER DEFAULT 0,
    timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(channel, message_id)
);

CREATE INDEX idx_telegram_messages_created_at ON telegram_messages(created_at DESC);
CREATE INDEX idx_telegram_messages_channel ON telegram_messages(channel);

CREATE TABLE IF NOT EXISTS narrative_data (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    source VARCHAR(30) NOT NULL DEFAULT 'perplexity',
    status VARCHAR(20) DEFAULT 'unknown',
    sentiment VARCHAR(20) DEFAULT 'neutral',
    summary TEXT,
    reasoning TEXT,
    key_events JSONB,
    bullish_factors JSONB,
    bearish_factors JSONB,
    macro_context TEXT,
    sources JSONB,
    raw_payload JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(timestamp, symbol, source)
);

CREATE INDEX idx_narrative_data_symbol_timestamp ON narrative_data(symbol, timestamp DESC);
CREATE INDEX idx_narrative_data_created_at ON narrative_data(created_at DESC);

CREATE TABLE IF NOT EXISTS ai_reports (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    market_data JSONB,
    bull_opinion TEXT,
    bear_opinion TEXT,
    risk_assessment TEXT,
    final_decision JSONB,
    onchain_context TEXT,
    onchain_snapshot JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_ai_reports_symbol ON ai_reports(symbol);
CREATE INDEX idx_ai_reports_created_at ON ai_reports(created_at DESC);

CREATE TABLE IF NOT EXISTS feedback_logs (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    prediction_time TIMESTAMPTZ NOT NULL,
    predicted_direction VARCHAR(10),
    actual_direction VARCHAR(10),
    actual_change_pct DECIMAL(10, 4),
    mistake_summary TEXT,
    lesson_learned TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_feedback_logs_created_at ON feedback_logs(created_at DESC);

CREATE TABLE IF NOT EXISTS trade_executions (
    id BIGSERIAL PRIMARY KEY,
    exchange VARCHAR(20) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    amount DECIMAL(20, 8) NOT NULL,
    leverage INTEGER DEFAULT 1,
    order_id VARCHAR(100),
    filled_price DECIMAL(20, 8),
    success BOOLEAN DEFAULT FALSE,
    error TEXT,
    timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_trade_executions_symbol ON trade_executions(symbol);
CREATE INDEX idx_trade_executions_created_at ON trade_executions(created_at DESC);

-- Migration for existing funding_data tables (add global OI columns)
-- Run these only if upgrading from older schema:
-- ALTER TABLE funding_data ADD COLUMN IF NOT EXISTS oi_binance DECIMAL(20, 2) DEFAULT 0;
-- ALTER TABLE funding_data ADD COLUMN IF NOT EXISTS oi_bybit DECIMAL(20, 2) DEFAULT 0;
-- ALTER TABLE funding_data ADD COLUMN IF NOT EXISTS oi_okx DECIMAL(20, 2) DEFAULT 0;
-- ALTER TABLE funding_data ADD COLUMN IF NOT EXISTS basis_pct DECIMAL(10, 6);

-- Dune query snapshots (raw rows JSON for macro/on-chain signals)
CREATE TABLE IF NOT EXISTS dune_query_results (
    id BIGSERIAL PRIMARY KEY,
    query_id BIGINT NOT NULL,
    query_name VARCHAR(120),
    category VARCHAR(50),
    cadence_minutes INTEGER,
    execution_ended_at TIMESTAMPTZ,
    row_count INTEGER DEFAULT 0,
    rows JSONB NOT NULL DEFAULT '[]'::jsonb,
    collected_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(query_id, collected_at)
);

CREATE INDEX idx_dune_query_results_query_time ON dune_query_results(query_id, collected_at DESC);
CREATE INDEX idx_dune_query_results_collected_at ON dune_query_results(collected_at DESC);


-- Microstructure snapshots (lightweight orderbook proxies)
CREATE TABLE IF NOT EXISTS microstructure_data (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(20) NOT NULL DEFAULT 'binance',
    bid_price DECIMAL(20, 8),
    ask_price DECIMAL(20, 8),
    spread_bps DECIMAL(12, 6),
    bid_qty_top5 DECIMAL(20, 8),
    ask_qty_top5 DECIMAL(20, 8),
    orderbook_imbalance DECIMAL(14, 8),
    slippage_buy_100k_bps DECIMAL(12, 6),
    depth_levels INTEGER DEFAULT 5,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(timestamp, symbol, exchange)
);

CREATE INDEX idx_microstructure_symbol_timestamp ON microstructure_data(symbol, timestamp DESC);

-- Macro regime snapshots (FRED + yfinance proxies)
CREATE TABLE IF NOT EXISTS macro_data (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    source VARCHAR(40) NOT NULL DEFAULT 'macro_collector',
    dgs2 DECIMAL(10, 4),
    dgs10 DECIMAL(10, 4),
    ust_2s10s_spread DECIMAL(10, 4),
    cpiaucsl DECIMAL(20, 4),
    fedfunds DECIMAL(10, 4),
    dxy DECIMAL(12, 4),
    nasdaq DECIMAL(20, 4),
    gold DECIMAL(20, 4),
    oil DECIMAL(20, 4),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(timestamp, source)
);

CREATE INDEX idx_macro_data_timestamp ON macro_data(timestamp DESC);


-- ─────────────────────────────────────────────────────────────────────────────
-- Deribit Options Data  (DVOL, PCR, IV Term Structure, 25-delta Skew)
-- Collected hourly via public Deribit REST API (no auth required)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS deribit_data (
    id           BIGSERIAL PRIMARY KEY,
    symbol       VARCHAR(10)     NOT NULL,          -- BTC | ETH
    timestamp    TIMESTAMPTZ     NOT NULL,
    dvol         DECIMAL(10, 4),                    -- 30-day IV index (DVOL)
    spot_price   DECIMAL(20, 4),
    -- Put/Call Ratio
    pcr_oi       DECIMAL(10, 6),                    -- by Open Interest
    pcr_vol      DECIMAL(10, 6),                    -- by Volume
    put_oi       DECIMAL(20, 4),
    call_oi      DECIMAL(20, 4),
    put_vol      DECIMAL(20, 8),
    call_vol     DECIMAL(20, 8),
    -- IV Term Structure (average ATM IV per expiry bucket)
    iv_1w        DECIMAL(10, 4),
    iv_2w        DECIMAL(10, 4),
    iv_1m        DECIMAL(10, 4),
    iv_3m        DECIMAL(10, 4),
    iv_6m        DECIMAL(10, 4),
    term_inverted BOOLEAN        DEFAULT FALSE,     -- front IV > back IV = panic
    -- 25-delta Skew (put_IV - call_IV; positive = fear, negative = greed)
    skew_1w      DECIMAL(10, 4),
    skew_2w      DECIMAL(10, 4),
    skew_1m      DECIMAL(10, 4),
    skew_3m      DECIMAL(10, 4),
    created_at   TIMESTAMPTZ     DEFAULT NOW(),
    UNIQUE(symbol, timestamp)
);

CREATE INDEX idx_deribit_data_symbol_timestamp ON deribit_data(symbol, timestamp DESC);


-- ─────────────────────────────────────────────────────────────────────────────
-- Crypto Fear & Greed Index  (alternative.me, daily, no API key)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS fear_greed_data (
    id                  BIGSERIAL PRIMARY KEY,
    timestamp           TIMESTAMPTZ  NOT NULL,      -- daily midnight UTC
    value               INTEGER      NOT NULL,       -- 0-100
    classification      VARCHAR(30)  NOT NULL,       -- Extreme Fear … Extreme Greed
    value_prev          INTEGER,                     -- yesterday's value
    classification_prev VARCHAR(30),
    change              INTEGER,                     -- today - yesterday
    created_at          TIMESTAMPTZ  DEFAULT NOW(),
    UNIQUE(timestamp)
);

CREATE INDEX idx_fear_greed_timestamp ON fear_greed_data(timestamp DESC);

-- Liquidation aggregates from Binance force-order websocket
CREATE TABLE IF NOT EXISTS liquidations (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    long_liq_usd DECIMAL(20, 2) NOT NULL DEFAULT 0,
    short_liq_usd DECIMAL(20, 2) NOT NULL DEFAULT 0,
    long_liq_count INTEGER NOT NULL DEFAULT 0,
    short_liq_count INTEGER NOT NULL DEFAULT 0,
    largest_single_usd DECIMAL(20, 2) NOT NULL DEFAULT 0,
    largest_single_side VARCHAR(10) NOT NULL DEFAULT '',
    largest_single_price DECIMAL(20, 8) NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(timestamp, symbol)
);

CREATE INDEX idx_liquidations_symbol_timestamp ON liquidations(symbol, timestamp DESC);

-- Daily playbooks used by the hourly monitor
CREATE TABLE IF NOT EXISTS daily_playbooks (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    mode TEXT NOT NULL,
    playbook JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ttl_hours INT NOT NULL DEFAULT 24,
    source_decision TEXT NOT NULL DEFAULT 'HOLD',
    UNIQUE (symbol, mode)
);

CREATE INDEX idx_daily_playbooks_symbol_mode ON daily_playbooks(symbol, mode);
CREATE INDEX idx_daily_playbooks_created_at ON daily_playbooks(created_at DESC);

-- Hourly monitor debug logs
CREATE TABLE IF NOT EXISTS monitor_logs (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    symbol TEXT NOT NULL,
    mode TEXT NOT NULL,
    status TEXT NOT NULL,
    matched_conditions JSONB,
    unmatched_conditions JSONB,
    live_indicators JSONB,
    playbook_id BIGINT REFERENCES daily_playbooks(id),
    reasoning TEXT
);

CREATE INDEX idx_monitor_logs_symbol_time ON monitor_logs(symbol, created_at DESC);
CREATE INDEX idx_monitor_logs_playbook ON monitor_logs(playbook_id);

-- Hourly market-status technical snapshots for threshold tuning
CREATE TABLE IF NOT EXISTS market_status_events (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    symbol TEXT NOT NULL,
    regime TEXT,
    price DOUBLE PRECISION,
    funding_rate DOUBLE PRECISION,
    volatility DOUBLE PRECISION,
    technical_snapshot JSONB NOT NULL
);

CREATE INDEX idx_market_status_events_symbol_time ON market_status_events(symbol, created_at DESC);
CREATE INDEX idx_market_status_events_regime_time ON market_status_events(regime, created_at DESC);

CREATE TABLE IF NOT EXISTS evaluation_predictions (
    id BIGSERIAL PRIMARY KEY,
    source_type TEXT NOT NULL,
    source_id BIGINT,
    ai_report_id BIGINT REFERENCES ai_reports(id) ON DELETE SET NULL,
    prediction_time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    mode TEXT,
    decision TEXT NOT NULL,
    prediction_label TEXT,
    confidence DOUBLE PRECISION,
    entry_price DOUBLE PRECISION,
    take_profit DOUBLE PRECISION,
    stop_loss DOUBLE PRECISION,
    regime TEXT,
    model_version TEXT,
    prompt_version TEXT,
    rag_version TEXT,
    strategy_version TEXT,
    consensus_rate DOUBLE PRECISION,
    anomalies_detected JSONB NOT NULL DEFAULT '[]'::jsonb,
    input_context JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(source_type, source_id, mode)
);

CREATE INDEX idx_evaluation_predictions_time ON evaluation_predictions(prediction_time DESC);
CREATE INDEX idx_evaluation_predictions_symbol_mode_time ON evaluation_predictions(symbol, mode, prediction_time DESC);
CREATE INDEX idx_evaluation_predictions_source ON evaluation_predictions(source_type, source_id);

CREATE TABLE IF NOT EXISTS evaluation_outcomes (
    id BIGSERIAL PRIMARY KEY,
    prediction_id BIGINT NOT NULL REFERENCES evaluation_predictions(id) ON DELETE CASCADE,
    horizon_minutes INTEGER NOT NULL,
    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    target_time TIMESTAMPTZ,
    entry_price DOUBLE PRECISION,
    exit_price DOUBLE PRECISION,
    actual_direction TEXT,
    realized_return_pct DOUBLE PRECISION,
    fee_adjusted_pnl_pct DOUBLE PRECISION,
    benchmark_return_pct DOUBLE PRECISION,
    excess_return_pct DOUBLE PRECISION,
    correct BOOLEAN,
    tp_hit BOOLEAN,
    sl_hit BOOLEAN,
    mfe_pct DOUBLE PRECISION,
    mae_pct DOUBLE PRECISION,
    sample_count INTEGER,
    data_delay_minutes DOUBLE PRECISION,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(prediction_id, horizon_minutes)
);

CREATE INDEX idx_evaluation_outcomes_horizon ON evaluation_outcomes(horizon_minutes, evaluated_at DESC);
CREATE INDEX idx_evaluation_outcomes_prediction ON evaluation_outcomes(prediction_id, horizon_minutes);

CREATE TABLE IF NOT EXISTS evaluation_component_scores (
    id BIGSERIAL PRIMARY KEY,
    prediction_id BIGINT NOT NULL REFERENCES evaluation_predictions(id) ON DELETE CASCADE,
    component_type TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value DOUBLE PRECISION,
    metric_label TEXT,
    scope_key TEXT NOT NULL DEFAULT '',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(prediction_id, component_type, metric_name, scope_key)
);

CREATE INDEX idx_evaluation_component_scores_prediction ON evaluation_component_scores(prediction_id, component_type);
CREATE INDEX idx_evaluation_component_scores_component ON evaluation_component_scores(component_type, metric_name, created_at DESC);

CREATE TABLE IF NOT EXISTS evaluation_rollups_daily (
    id BIGSERIAL PRIMARY KEY,
    rollup_date DATE NOT NULL,
    symbol TEXT NOT NULL,
    mode TEXT NOT NULL,
    scope TEXT NOT NULL DEFAULT 'system',
    horizon_minutes INTEGER NOT NULL DEFAULT 0,
    metric_name TEXT NOT NULL,
    metric_value DOUBLE PRECISION,
    sample_size INTEGER NOT NULL DEFAULT 0,
    bucket_key TEXT NOT NULL DEFAULT '',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(rollup_date, symbol, mode, scope, horizon_minutes, metric_name, bucket_key)
);

CREATE INDEX idx_evaluation_rollups_daily_lookup
ON evaluation_rollups_daily(rollup_date DESC, symbol, mode, scope, horizon_minutes);

CREATE TABLE IF NOT EXISTS archive_manifests (
    id BIGSERIAL PRIMARY KEY,
    table_name TEXT NOT NULL,
    partition_key TEXT NOT NULL,
    gcs_path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    schema_version TEXT NOT NULL DEFAULT 'v1',
    time_column TEXT NOT NULL,
    pk_column TEXT NOT NULL DEFAULT 'id',
    symbol_column TEXT,
    partition_start TIMESTAMPTZ,
    partition_end TIMESTAMPTZ,
    partition_start_date DATE,
    partition_end_date DATE,
    partition_symbol TEXT,
    archive_started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    archive_completed_at TIMESTAMPTZ,
    verified_at TIMESTAMPTZ,
    cleanup_completed_at TIMESTAMPTZ,
    retention_cutoff TIMESTAMPTZ,
    retention_cutoff_date DATE,
    row_count_db INTEGER NOT NULL DEFAULT 0,
    row_count_parquet INTEGER NOT NULL DEFAULT 0,
    min_pk BIGINT,
    max_pk BIGINT,
    min_time TIMESTAMPTZ,
    max_time TIMESTAMPTZ,
    min_date DATE,
    max_date DATE,
    file_size_bytes BIGINT,
    md5_hash TEXT,
    error_message TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(table_name, partition_key)
);

CREATE INDEX idx_archive_manifests_status
ON archive_manifests(status, cleanup_completed_at, table_name);

CREATE INDEX idx_archive_manifests_table_partition
ON archive_manifests(table_name, partition_key);

CREATE INDEX idx_archive_manifests_verified
ON archive_manifests(verified_at DESC, cleanup_completed_at);
