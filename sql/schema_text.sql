-- ============================================================
-- TEXT Project SQL Schema
-- 텍스트/AI 데이터: telegram_messages, narrative_data, ai_reports,
--                   feedback_logs, trade_executions, dune_query_results,
--                   daily_playbooks, monitor_logs, market_status_events,
--                   evaluation_predictions/outcomes/component_scores/rollups_daily
-- Supabase SQL Editor에 전체 붙여넣기 후 실행
-- ============================================================

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
    feedback_type VARCHAR(10) DEFAULT 'negative', -- 'positive' | 'negative'
    mistake_summary TEXT,
    lesson_learned TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_feedback_logs_created_at ON feedback_logs(created_at DESC);
CREATE INDEX idx_feedback_logs_type ON feedback_logs(feedback_type, created_at DESC);
-- Migration for existing deployments:
-- ALTER TABLE feedback_logs ADD COLUMN IF NOT EXISTS feedback_type VARCHAR(10) DEFAULT 'negative';

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
