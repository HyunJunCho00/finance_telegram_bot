-- Daily Playbook table (Supabase / PostgreSQL)
-- One row per (symbol, mode) — upserted daily at 00:00 UTC

CREATE TABLE IF NOT EXISTS daily_playbooks (
    id              BIGSERIAL PRIMARY KEY,
    symbol          TEXT        NOT NULL,
    mode            TEXT        NOT NULL,         -- 'swing' | 'position'
    playbook        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    ttl_hours       INT         NOT NULL DEFAULT 24,
    source_decision TEXT        NOT NULL DEFAULT 'HOLD',
    UNIQUE (symbol, mode)   -- upsert key
);

CREATE INDEX IF NOT EXISTS idx_daily_playbooks_symbol_mode ON daily_playbooks (symbol, mode);
CREATE INDEX IF NOT EXISTS idx_daily_playbooks_created_at  ON daily_playbooks (created_at DESC);

-- Enable Row Level Security (optional — safe default for Supabase)
ALTER TABLE daily_playbooks ENABLE ROW LEVEL SECURITY;
