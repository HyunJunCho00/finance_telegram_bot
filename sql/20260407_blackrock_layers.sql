-- ============================================================
-- Institutional Quant Framework: DB Migration
-- Layer 1: Factor Signal Storage
-- Layer 6: Post-Trade Attribution & IC History
-- Target: TEXT project (Supabase)
-- Created: 2026-04-07
-- ============================================================

-- ── Factor Signals: snapshot of all factor values at decision time ──────────
CREATE TABLE IF NOT EXISTS factor_signals (
    id              BIGSERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol          VARCHAR(20)  NOT NULL,
    mode            VARCHAR(20)  NOT NULL,   -- swing / position
    decision_id     TEXT,                    -- links to ai_reports.id
    regime          TEXT,                    -- regime at decision time

    -- Normalized signal values (-1.0 ~ +1.0)
    -- Positive = bullish direction, Negative = bearish direction
    funding_rate_signal      FLOAT,   -- high positive funding = contrarian bearish → negative
    oi_change_signal         FLOAT,   -- rapid OI growth = crowded → slightly negative
    liquidation_trap_signal  FLOAT,   -- trap confirmed toward direction → positive
    onchain_flow_signal      FLOAT,   -- exchange net outflow = accumulation → positive
    narrative_sentiment_signal FLOAT, -- perplexity/RAG bullish vs bearish factors
    rsi_divergence_signal    FLOAT,   -- positive divergence = reversal bullish → positive
    macro_regime_signal      FLOAT,   -- bull_momentum = +1, bear = -1, range = 0
    microstructure_signal    FLOAT,   -- bid-heavy imbalance → positive

    -- IC weights used at decision time (snapshot of current IC knowledge)
    ic_weights      JSONB,            -- {factor_name: ic_value}

    -- Weighted composite alpha score
    alpha_score     FLOAT,            -- IC-weighted directional score (-1 ~ +1)

    -- Decision made
    final_decision  TEXT,             -- LONG / SHORT / HOLD
    allocation_pct  FLOAT
);

CREATE INDEX IF NOT EXISTS idx_factor_signals_symbol_created
    ON factor_signals (symbol, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_factor_signals_decision_id
    ON factor_signals (decision_id);

-- ── Factor IC History: rolling IC per factor per regime ─────────────────────
CREATE TABLE IF NOT EXISTS factor_ic_history (
    id              BIGSERIAL PRIMARY KEY,
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    factor_name     TEXT        NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    regime          TEXT,                    -- NULL = all regimes aggregated

    ic_7t           FLOAT,                   -- IC over last 7 trades
    ic_30t          FLOAT,                   -- IC over last 30 trades
    ic_all          FLOAT,                   -- IC over all available trades
    sample_count    INT,                     -- number of completed trades used

    -- Decay-weighted IC (recent trades count more)
    ic_decay_weighted FLOAT,                 -- exponential decay weighted IC

    UNIQUE (factor_name, symbol, regime, computed_at)
);

CREATE INDEX IF NOT EXISTS idx_factor_ic_name_symbol
    ON factor_ic_history (factor_name, symbol, computed_at DESC);

-- ── Trade Attribution: post-trade factor contribution decomposition ──────────
CREATE TABLE IF NOT EXISTS trade_attribution (
    id                      BIGSERIAL PRIMARY KEY,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol                  VARCHAR(20)  NOT NULL,
    mode                    VARCHAR(20)  NOT NULL,

    -- Trade linkage
    decision_id             TEXT,            -- links to ai_reports.id
    factor_signals_id       BIGINT,          -- links to factor_signals.id

    -- Trade outcome
    entry_price             FLOAT,
    exit_price              FLOAT,
    realized_pnl_pct        FLOAT,           -- fee-adjusted realized return %
    holding_hours           FLOAT,           -- how long held
    regime_at_entry         TEXT,

    -- Factor contributions: {factor_name: contribution_pct}
    factor_contributions    JSONB,

    -- Signals at entry (snapshot for IC recomputation)
    factor_signals_snapshot JSONB,

    -- Attribution summary
    dominant_factor         TEXT,            -- factor with highest absolute contribution
    dominant_factor_correct BOOL,            -- was the dominant factor's direction correct?
    execution_cost_pct      FLOAT,           -- estimated slippage cost
    residual_pct            FLOAT            -- unexplained return
);

CREATE INDEX IF NOT EXISTS idx_trade_attribution_symbol
    ON trade_attribution (symbol, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_trade_attribution_decision_id
    ON trade_attribution (decision_id);
