-- ============================================================
-- QUANT Project SQL Schema
-- 수치 데이터: market_data, cvd_data, funding_data,
--              liquidations, microstructure_data, macro_data,
--              deribit_data, fear_greed_data, onchain_daily_snapshots,
--              liquidation_cascade_features/predictions, archive_manifests
-- Supabase SQL Editor에 전체 붙여넣기 후 실행
-- ============================================================

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
    taker_buy_volume DECIMAL(20, 8),
    taker_sell_volume DECIMAL(20, 8),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(timestamp, symbol, exchange)
);
CREATE INDEX idx_market_data_symbol_timestamp ON market_data(symbol, timestamp DESC);
CREATE INDEX idx_market_data_timestamp ON market_data(timestamp DESC);
CREATE INDEX idx_market_data_exchange ON market_data(exchange, symbol, timestamp DESC);

CREATE TABLE IF NOT EXISTS cvd_data (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    taker_buy_volume DECIMAL(20, 8) NOT NULL DEFAULT 0,
    taker_sell_volume DECIMAL(20, 8) NOT NULL DEFAULT 0,
    volume_delta DECIMAL(20, 8) NOT NULL DEFAULT 0,
    whale_buy_vol DECIMAL(20, 2),
    whale_sell_vol DECIMAL(20, 2),
    whale_buy_count INTEGER,
    whale_sell_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(timestamp, symbol)
);
CREATE INDEX idx_cvd_data_symbol_timestamp ON cvd_data(symbol, timestamp DESC);

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
    btc_cme_basis DECIMAL(20, 4),
    btc_cme_basis_pct DECIMAL(10, 6),
    eth_cme_basis DECIMAL(20, 4),
    eth_cme_basis_pct DECIMAL(10, 6),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(timestamp, source)
);
CREATE INDEX idx_macro_data_timestamp ON macro_data(timestamp DESC);

CREATE TABLE IF NOT EXISTS deribit_data (
    id           BIGSERIAL PRIMARY KEY,
    symbol       VARCHAR(10)     NOT NULL,
    timestamp    TIMESTAMPTZ     NOT NULL,
    dvol         DECIMAL(10, 4),
    spot_price   DECIMAL(20, 4),
    pcr_oi       DECIMAL(10, 6),
    pcr_vol      DECIMAL(10, 6),
    put_oi       DECIMAL(20, 4),
    call_oi      DECIMAL(20, 4),
    put_vol      DECIMAL(20, 8),
    call_vol     DECIMAL(20, 8),
    iv_1w        DECIMAL(10, 4),
    iv_2w        DECIMAL(10, 4),
    iv_1m        DECIMAL(10, 4),
    iv_3m        DECIMAL(10, 4),
    iv_6m        DECIMAL(10, 4),
    term_inverted BOOLEAN        DEFAULT FALSE,
    skew_1w      DECIMAL(10, 4),
    skew_2w      DECIMAL(10, 4),
    skew_1m      DECIMAL(10, 4),
    skew_3m      DECIMAL(10, 4),
    created_at   TIMESTAMPTZ     DEFAULT NOW(),
    UNIQUE(symbol, timestamp)
);
CREATE INDEX idx_deribit_data_symbol_timestamp ON deribit_data(symbol, timestamp DESC);

CREATE TABLE IF NOT EXISTS fear_greed_data (
    id                  BIGSERIAL PRIMARY KEY,
    timestamp           TIMESTAMPTZ  NOT NULL,
    value               INTEGER      NOT NULL,
    classification      VARCHAR(30)  NOT NULL,
    value_prev          INTEGER,
    classification_prev VARCHAR(30),
    change              INTEGER,
    created_at          TIMESTAMPTZ  DEFAULT NOW(),
    UNIQUE(timestamp)
);
CREATE INDEX idx_fear_greed_timestamp ON fear_greed_data(timestamp DESC);

CREATE TABLE IF NOT EXISTS public.onchain_daily_snapshots (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    asset TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'coinmetrics',
    as_of_date DATE NOT NULL,
    raw_metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    derived_features JSONB NOT NULL DEFAULT '{}'::jsonb,
    regime_flags JSONB NOT NULL DEFAULT '{}'::jsonb,
    bias_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    risk_bias TEXT NOT NULL DEFAULT 'NEUTRAL',
    data_quality TEXT NOT NULL DEFAULT 'unknown',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (symbol, as_of_date, source)
);
CREATE INDEX IF NOT EXISTS idx_onchain_daily_snapshots_symbol_date
ON public.onchain_daily_snapshots (symbol, as_of_date DESC);
CREATE INDEX IF NOT EXISTS idx_onchain_daily_snapshots_risk_bias_date
ON public.onchain_daily_snapshots (risk_bias, as_of_date DESC);

CREATE TABLE IF NOT EXISTS public.liquidation_cascade_features (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('down', 'up')),
    event_candidate BOOLEAN NOT NULL DEFAULT FALSE,
    vulnerability_score DOUBLE PRECISION,
    ignition_score DOUBLE PRECISION,
    ignition_ema DOUBLE PRECISION,
    ignition_slope DOUBLE PRECISION,
    ignition_r2 DOUBLE PRECISION,
    vulnerability_pct_rank DOUBLE PRECISION,
    ignition_pct_rank DOUBLE PRECISION,
    slope_pct_rank DOUBLE PRECISION,
    r2_pct_rank DOUBLE PRECISION,
    feature_version TEXT NOT NULL,
    features JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (timestamp, symbol, side, feature_version)
);
CREATE INDEX IF NOT EXISTS idx_liq_cascade_features_symbol_time
ON public.liquidation_cascade_features (symbol, side, timestamp DESC);

CREATE TABLE IF NOT EXISTS public.liquidation_cascade_predictions (
    id BIGSERIAL PRIMARY KEY,
    feature_id BIGINT REFERENCES public.liquidation_cascade_features(id) ON DELETE SET NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('down', 'up')),
    horizon_minutes INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    feature_version TEXT NOT NULL,
    probability DOUBLE PRECISION NOT NULL,
    predicted_label BOOLEAN,
    diagnostics JSONB NOT NULL DEFAULT '{}'::jsonb,
    realized_label BOOLEAN,
    realized_at TIMESTAMPTZ,
    evaluation JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_liq_cascade_predictions_symbol_time
ON public.liquidation_cascade_predictions (symbol, side, timestamp DESC);

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
CREATE INDEX idx_archive_manifests_status ON archive_manifests(status, cleanup_completed_at, table_name);
CREATE INDEX idx_archive_manifests_table_partition ON archive_manifests(table_name, partition_key);
CREATE INDEX idx_archive_manifests_verified ON archive_manifests(verified_at DESC, cleanup_completed_at);
