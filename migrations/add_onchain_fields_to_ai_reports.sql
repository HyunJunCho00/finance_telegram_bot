ALTER TABLE public.ai_reports
ADD COLUMN IF NOT EXISTS onchain_context TEXT,
ADD COLUMN IF NOT EXISTS onchain_snapshot JSONB DEFAULT '{}'::jsonb;
