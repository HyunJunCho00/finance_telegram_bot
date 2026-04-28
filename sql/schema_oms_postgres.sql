-- ========================================================
-- [Execution DB / OMS] PostgreSQL Migration Schema
-- Target: Aiven / Neon (PostgreSQL)
-- ========================================================

-- 1. System Config (시스템 상태 및 전역 변수 관리)
CREATE TABLE IF NOT EXISTS system_config (
    key VARCHAR(255) PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. Paper Wallets (가상 지갑 잔고)
CREATE TABLE IF NOT EXISTS paper_wallets (
    exchange VARCHAR(50) PRIMARY KEY,
    balance NUMERIC(20, 8) NOT NULL DEFAULT 0.0,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 초기 시드머니 설정 (필요에 따라 변경)
INSERT INTO paper_wallets (exchange, balance) VALUES ('binance', 2000.0) ON CONFLICT DO NOTHING;
INSERT INTO paper_wallets (exchange, balance) VALUES ('upbit', 2000000.0) ON CONFLICT DO NOTHING;

-- 3. Active Orders (의도 및 대기 중인 주문)
CREATE TABLE IF NOT EXISTS active_orders (
    intent_id VARCHAR(36) PRIMARY KEY,
    symbol VARCHAR(50) NOT NULL,
    direction VARCHAR(20) NOT NULL,
    execution_style VARCHAR(50) NOT NULL,
    total_target_amount NUMERIC(20, 8) NOT NULL,
    remaining_amount NUMERIC(20, 8) NOT NULL,
    filled_amount NUMERIC(20, 8) DEFAULT 0.0,
    exchange VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    leverage NUMERIC(10, 2) DEFAULT 1.0,
    tp_price NUMERIC(20, 8) DEFAULT 0.0,
    sl_price NUMERIC(20, 8) DEFAULT 0.0,
    tp2_price NUMERIC(20, 8) DEFAULT 0.0,
    tp1_exit_pct NUMERIC(5, 2) DEFAULT 50.0,
    playbook_id VARCHAR(255),
    source_decision TEXT,
    strategy_version VARCHAR(50),
    trigger_reason TEXT,
    thesis_id VARCHAR(255),
    status VARCHAR(50) DEFAULT 'PENDING'
);

-- 4. Paper Positions (현재 오픈된 모의투자 포지션)
CREATE TABLE IF NOT EXISTS paper_positions (
    position_id VARCHAR(36) PRIMARY KEY,
    exchange VARCHAR(50) NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    side VARCHAR(20) NOT NULL,
    size NUMERIC(20, 8) NOT NULL,
    entry_price NUMERIC(20, 8) NOT NULL,
    leverage NUMERIC(10, 2) DEFAULT 1.0,
    is_open SMALLINT DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
    tp_price NUMERIC(20, 8) DEFAULT 0.0,
    sl_price NUMERIC(20, 8) DEFAULT 0.0,
    tp2_price NUMERIC(20, 8) DEFAULT 0.0,
    tp1_done SMALLINT DEFAULT 0,
    tp1_exit_pct NUMERIC(5, 2) DEFAULT 50.0
);

-- 5. Execution Outbox (외부 이벤트/메시지 발행 보장용 큐)
CREATE TABLE IF NOT EXISTS execution_outbox (
    event_id VARCHAR(36) PRIMARY KEY,
    event_type VARCHAR(100) NOT NULL,
    idempotency_key VARCHAR(255) UNIQUE,
    payload_json JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'PENDING',
    attempts INTEGER DEFAULT 0,
    last_error TEXT,
    processing_started_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
    published_at TIMESTAMP WITH TIME ZONE
);

-- 성능 최적화를 위한 인덱스
CREATE INDEX IF NOT EXISTS idx_active_orders_symbol_exchange ON active_orders(symbol, exchange);
CREATE INDEX IF NOT EXISTS idx_active_orders_status ON active_orders(status);
CREATE INDEX IF NOT EXISTS idx_paper_positions_is_open ON paper_positions(is_open, position_id);
CREATE INDEX IF NOT EXISTS idx_execution_outbox_status_created ON execution_outbox(status, created_at);
