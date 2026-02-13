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

CREATE TABLE IF NOT EXISTS funding_data (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    funding_rate DECIMAL(10, 8),
    next_funding_time BIGINT,
    open_interest DECIMAL(20, 8),
    open_interest_value DECIMAL(20, 2),
    long_short_ratio DECIMAL(10, 4),
    long_account DECIMAL(10, 4),
    short_account DECIMAL(10, 4),
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
