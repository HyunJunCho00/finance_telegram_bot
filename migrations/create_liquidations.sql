-- Liquidation aggregates from Binance force-order websocket
-- Required by websocket_collector, orchestrator chart overlays, and DB cleanup.

CREATE TABLE IF NOT EXISTS public.liquidations (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    long_liq_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
    short_liq_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
    long_liq_count INTEGER NOT NULL DEFAULT 0,
    short_liq_count INTEGER NOT NULL DEFAULT 0,
    largest_single_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
    largest_single_side TEXT NOT NULL DEFAULT '',
    largest_single_price DOUBLE PRECISION NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (timestamp, symbol)
);

ALTER TABLE public.liquidations
    ADD COLUMN IF NOT EXISTS long_liq_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS short_liq_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS long_liq_count INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS short_liq_count INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS largest_single_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS largest_single_side TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS largest_single_price DOUBLE PRECISION NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ;

UPDATE public.liquidations
SET created_at = now()
WHERE created_at IS NULL;

ALTER TABLE public.liquidations
    ALTER COLUMN created_at SET DEFAULT now(),
    ALTER COLUMN created_at SET NOT NULL;

CREATE INDEX IF NOT EXISTS idx_liquidations_symbol_timestamp
ON public.liquidations (symbol, timestamp DESC);
