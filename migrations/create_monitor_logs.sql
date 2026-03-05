-- Hourly monitor debug logs (Supabase / PostgreSQL)
-- playbook_id type must match daily_playbooks.id (BIGSERIAL -> BIGINT)

CREATE TABLE IF NOT EXISTS public.monitor_logs (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    symbol TEXT NOT NULL,
    mode TEXT NOT NULL,
    status TEXT NOT NULL, -- TRIGGER | SOFT_TRIGGER | WATCH | NO_ACTION
    matched_conditions JSONB,
    unmatched_conditions JSONB,
    live_indicators JSONB,
    playbook_id BIGINT REFERENCES public.daily_playbooks(id),
    reasoning TEXT
);

CREATE INDEX IF NOT EXISTS idx_monitor_logs_symbol_time
ON public.monitor_logs (symbol, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_monitor_logs_playbook
ON public.monitor_logs (playbook_id);

COMMENT ON TABLE public.monitor_logs IS
'Hourly monitor results for debugging and strategy tuning.';
