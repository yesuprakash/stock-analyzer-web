-- Stock research snapshots. Run as the same PostgreSQL role configured in .env.
-- The runner stores the latest five trading dates with successful refreshes.

BEGIN;

CREATE TABLE IF NOT EXISTS public.stock_research_runs (
    id BIGSERIAL PRIMARY KEY,
    trading_date DATE NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    error TEXT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS stock_research_runs_date_status_idx
    ON public.stock_research_runs (trading_date DESC, status);

CREATE TABLE IF NOT EXISTS public.stock_research_recommendations (
    id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL
        REFERENCES public.stock_research_runs(id) ON DELETE CASCADE,
    symbol VARCHAR(32) NOT NULL
        REFERENCES public.stock_master(symbol) ON DELETE CASCADE,
    source_name VARCHAR(120) NOT NULL,
    analyst_name VARCHAR(120) NOT NULL DEFAULT '',
    recommendation TEXT NOT NULL CHECK (recommendation IN (
        'STRONG_BUY', 'BUY', 'ACCUMULATE', 'OUTPERFORM',
        'HOLD', 'NEUTRAL', 'UNDERPERFORM', 'REDUCE', 'SELL', 'STRONG_SELL'
    )),
    target_price NUMERIC(18, 4),
    currency CHAR(3) NOT NULL DEFAULT 'INR',
    horizon VARCHAR(120) NOT NULL DEFAULT '',
    published_at TIMESTAMPTZ,
    source_url TEXT,
    CHECK (target_price IS NULL OR target_price > 0),
    CHECK (currency ~ '^[A-Z]{3}$')
);

CREATE INDEX IF NOT EXISTS stock_research_recommendations_symbol_run_idx
    ON public.stock_research_recommendations (symbol, run_id);

CREATE TABLE IF NOT EXISTS public.stock_research_articles (
    id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL
        REFERENCES public.stock_research_runs(id) ON DELETE CASCADE,
    symbol VARCHAR(32) NOT NULL
        REFERENCES public.stock_master(symbol) ON DELETE CASCADE,
    source_name VARCHAR(160) NOT NULL DEFAULT 'NewsAPI',
    author VARCHAR(250) NOT NULL DEFAULT '',
    title VARCHAR(1000) NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    source_url TEXT NOT NULL,
    published_at TIMESTAMPTZ,
    UNIQUE (run_id, symbol, source_url)
);

CREATE INDEX IF NOT EXISTS stock_research_articles_symbol_run_idx
    ON public.stock_research_articles (symbol, run_id);

COMMIT;
