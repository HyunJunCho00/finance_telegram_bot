-- hh:20 market-status technical snapshots + detected event triggers
-- Used for threshold calibration and post-hoc analysis.

CREATE TABLE IF NOT EXISTS public.market_status_events (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    symbol TEXT NOT NULL,
    regime TEXT,
    price DOUBLE PRECISION,
    funding_rate DOUBLE PRECISION,
    volatility DOUBLE PRECISION,
    technical_snapshot JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_market_status_events_symbol_time
ON public.market_status_events (symbol, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_market_status_events_regime_time
ON public.market_status_events (regime, created_at DESC);

COMMENT ON TABLE public.market_status_events IS
'Hourly market-status event snapshots for ATR/regime threshold calibration.';
