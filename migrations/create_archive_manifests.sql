-- Verified GCS Parquet archive manifests.
-- Cleanup is only allowed for rows covered by a verified manifest.

CREATE TABLE IF NOT EXISTS public.archive_manifests (
    id BIGSERIAL PRIMARY KEY,
    table_name TEXT NOT NULL,
    partition_key TEXT NOT NULL,
    gcs_path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    schema_version TEXT NOT NULL DEFAULT 'v1',
    time_column TEXT NOT NULL,
    pk_column TEXT NOT NULL DEFAULT 'id',
    symbol_column TEXT,
    partition_start TIMESTAMPTZ,
    partition_end TIMESTAMPTZ,
    partition_start_date DATE,
    partition_end_date DATE,
    partition_symbol TEXT,
    archive_started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    archive_completed_at TIMESTAMPTZ,
    verified_at TIMESTAMPTZ,
    cleanup_completed_at TIMESTAMPTZ,
    retention_cutoff TIMESTAMPTZ,
    retention_cutoff_date DATE,
    row_count_db INTEGER NOT NULL DEFAULT 0,
    row_count_parquet INTEGER NOT NULL DEFAULT 0,
    min_pk BIGINT,
    max_pk BIGINT,
    min_time TIMESTAMPTZ,
    max_time TIMESTAMPTZ,
    min_date DATE,
    max_date DATE,
    file_size_bytes BIGINT,
    md5_hash TEXT,
    error_message TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (table_name, partition_key)
);

CREATE INDEX IF NOT EXISTS idx_archive_manifests_status
ON public.archive_manifests (status, cleanup_completed_at, table_name);

CREATE INDEX IF NOT EXISTS idx_archive_manifests_table_partition
ON public.archive_manifests (table_name, partition_key);

CREATE INDEX IF NOT EXISTS idx_archive_manifests_verified
ON public.archive_manifests (verified_at DESC, cleanup_completed_at);

COMMENT ON TABLE public.archive_manifests IS
'Manifest records for verified GCS Parquet archives. Cleanup is allowed only after verification.';
