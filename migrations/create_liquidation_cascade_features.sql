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
