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
