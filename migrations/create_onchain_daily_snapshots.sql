-- Daily Coin Metrics snapshot cache for BTC/ETH on-chain regime filtering.
-- Stores both raw metrics and deterministic derived regime/gate features.

CREATE TABLE IF NOT EXISTS public.onchain_daily_snapshots (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    asset TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'coinmetrics',
    as_of_date DATE NOT NULL,
    raw_metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    derived_features JSONB NOT NULL DEFAULT '{}'::jsonb,
    regime_flags JSONB NOT NULL DEFAULT '{}'::jsonb,
    bias_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    risk_bias TEXT NOT NULL DEFAULT 'NEUTRAL',
    data_quality TEXT NOT NULL DEFAULT 'unknown',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (symbol, as_of_date, source)
);

CREATE INDEX IF NOT EXISTS idx_onchain_daily_snapshots_symbol_date
ON public.onchain_daily_snapshots (symbol, as_of_date DESC);

CREATE INDEX IF NOT EXISTS idx_onchain_daily_snapshots_risk_bias_date
ON public.onchain_daily_snapshots (risk_bias, as_of_date DESC);

COMMENT ON TABLE public.onchain_daily_snapshots IS
'Daily on-chain snapshot cache used as a regime/risk overlay for swing and position trading.';
