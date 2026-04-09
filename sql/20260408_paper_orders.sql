-- ============================================================
-- Paper Order Tracking: lifecycle management for virtual positions
-- Separate from trade_executions — tracks entry → monitoring → exit → P&L
-- Used by: node_paper_order_record (orchestrator) + ws_price_feed (monitoring)
-- ============================================================

CREATE TABLE IF NOT EXISTS paper_orders (
    id                      BIGSERIAL PRIMARY KEY,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol                  VARCHAR(20)  NOT NULL,
    mode                    VARCHAR(20)  NOT NULL,   -- swing / position
    decision_id             TEXT,                    -- ai_reports.id 링크
    factor_signals_id       BIGINT,                  -- factor_signals.id 링크

    -- 진입
    side                    VARCHAR(10)  NOT NULL,   -- LONG / SHORT
    size_usdt               FLOAT,
    entry_price             FLOAT,
    sl_price                FLOAT,
    tp1_price               FLOAT,
    tp2_price               FLOAT,
    allocation_pct          FLOAT,
    leverage                FLOAT        DEFAULT 1,

    -- 상태
    status                  VARCHAR(20)  NOT NULL DEFAULT 'OPEN',
    -- OPEN / CLOSED_TP1 / CLOSED_TP2 / CLOSED_SL / CLOSED_MANUAL / EXPIRED

    -- 청산 (ws_price_feed가 채움)
    exit_price              FLOAT,
    exit_at                 TIMESTAMPTZ,
    realized_pnl_pct        FLOAT,
    holding_hours           FLOAT,

    -- Self-correction 결과 스냅샷 (why this trade passed)
    self_correction_passed  BOOLEAN,
    self_correction_result  JSONB,       -- {challenges, loop_back_reason, confidence_delta}
    correction_loop_count   INT DEFAULT 0,

    -- Attribution용 팩터 스냅샷
    alpha_score             FLOAT,
    factor_signals_snapshot JSONB,       -- factor_ic_result 전체
    regime                  TEXT
);

CREATE INDEX IF NOT EXISTS idx_paper_orders_symbol_status
    ON paper_orders (symbol, status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_paper_orders_decision_id
    ON paper_orders (decision_id);

CREATE INDEX IF NOT EXISTS idx_paper_orders_open
    ON paper_orders (status)
    WHERE status = 'OPEN';
