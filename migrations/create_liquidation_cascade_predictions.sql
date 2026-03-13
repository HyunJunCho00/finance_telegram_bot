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
