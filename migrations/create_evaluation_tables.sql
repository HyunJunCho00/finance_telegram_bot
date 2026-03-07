-- Structured evaluation storage for predictions, fixed-horizon outcomes,
-- component diagnostics, and daily KPI rollups.

CREATE TABLE IF NOT EXISTS public.evaluation_predictions (
    id BIGSERIAL PRIMARY KEY,
    source_type TEXT NOT NULL,
    source_id BIGINT,
    ai_report_id BIGINT REFERENCES public.ai_reports(id) ON DELETE SET NULL,
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
    UNIQUE (source_type, source_id, mode)
);

CREATE INDEX IF NOT EXISTS idx_evaluation_predictions_time
ON public.evaluation_predictions (prediction_time DESC);

CREATE INDEX IF NOT EXISTS idx_evaluation_predictions_symbol_mode_time
ON public.evaluation_predictions (symbol, mode, prediction_time DESC);

CREATE INDEX IF NOT EXISTS idx_evaluation_predictions_source
ON public.evaluation_predictions (source_type, source_id);

CREATE TABLE IF NOT EXISTS public.evaluation_outcomes (
    id BIGSERIAL PRIMARY KEY,
    prediction_id BIGINT NOT NULL REFERENCES public.evaluation_predictions(id) ON DELETE CASCADE,
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
    UNIQUE (prediction_id, horizon_minutes)
);

CREATE INDEX IF NOT EXISTS idx_evaluation_outcomes_horizon
ON public.evaluation_outcomes (horizon_minutes, evaluated_at DESC);

CREATE INDEX IF NOT EXISTS idx_evaluation_outcomes_prediction
ON public.evaluation_outcomes (prediction_id, horizon_minutes);

CREATE TABLE IF NOT EXISTS public.evaluation_component_scores (
    id BIGSERIAL PRIMARY KEY,
    prediction_id BIGINT NOT NULL REFERENCES public.evaluation_predictions(id) ON DELETE CASCADE,
    component_type TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value DOUBLE PRECISION,
    metric_label TEXT,
    scope_key TEXT NOT NULL DEFAULT '',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (prediction_id, component_type, metric_name, scope_key)
);

CREATE INDEX IF NOT EXISTS idx_evaluation_component_scores_prediction
ON public.evaluation_component_scores (prediction_id, component_type);

CREATE INDEX IF NOT EXISTS idx_evaluation_component_scores_component
ON public.evaluation_component_scores (component_type, metric_name, created_at DESC);

CREATE TABLE IF NOT EXISTS public.evaluation_rollups_daily (
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
    UNIQUE (rollup_date, symbol, mode, scope, horizon_minutes, metric_name, bucket_key)
);

CREATE INDEX IF NOT EXISTS idx_evaluation_rollups_daily_lookup
ON public.evaluation_rollups_daily (rollup_date DESC, symbol, mode, scope, horizon_minutes);

COMMENT ON TABLE public.evaluation_predictions IS
'Immutable prediction events for ai_reports and realtime signals.';

COMMENT ON TABLE public.evaluation_outcomes IS
'Fixed-horizon realized outcomes keyed by prediction and horizon.';

COMMENT ON TABLE public.evaluation_component_scores IS
'Per-prediction LLM, RAG, and system diagnostic metrics.';

COMMENT ON TABLE public.evaluation_rollups_daily IS
'Daily KPI rollups for dashboards and regression monitoring.';
