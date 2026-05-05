-- 뉴스 임팩트 예측 로그 테이블
-- predicted_at 이후 가격 변화를 채워서 calibration에 활용
CREATE TABLE IF NOT EXISTS news_impact_log (
    id                BIGSERIAL PRIMARY KEY,
    predicted_at      TIMESTAMPTZ NOT NULL,
    headline          TEXT NOT NULL,
    claim             TEXT,
    impact_score      SMALLINT NOT NULL CHECK (impact_score BETWEEN 1 AND 5),
    already_priced_in BOOLEAN NOT NULL DEFAULT FALSE,
    why               TEXT,
    btc_price_at_time NUMERIC(18, 2),
    eth_price_at_time NUMERIC(18, 2),
    -- 사후 가격 변화 (calibration용, 별도 잡에서 채움)
    btc_price_1h      NUMERIC(18, 2),
    btc_price_4h      NUMERIC(18, 2),
    btc_price_24h     NUMERIC(18, 2),
    eth_price_1h      NUMERIC(18, 2),
    eth_price_4h      NUMERIC(18, 2),
    eth_price_24h     NUMERIC(18, 2),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_news_impact_log_predicted_at
    ON news_impact_log (predicted_at DESC);

CREATE INDEX IF NOT EXISTS idx_news_impact_log_impact_score
    ON news_impact_log (impact_score, already_priced_in);
