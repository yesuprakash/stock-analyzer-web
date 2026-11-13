-- Stock Analyzer: new stock-master and email-alert tables.
-- Safe to run more than once. Does not alter predictions or prediction_summary.
-- Run against the same PostgreSQL database used by the Stock Analyzer.

BEGIN;

CREATE TABLE IF NOT EXISTS public.stock_master (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(32) NOT NULL UNIQUE,
    company_name VARCHAR(120) NOT NULL DEFAULT '',
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (symbol = UPPER(symbol)),
    CHECK (symbol ~ '^[A-Z0-9][A-Z0-9.&-]{0,28}\.(NS|BO)$')
);

INSERT INTO public.stock_master (symbol) VALUES
    ('ABB.NS'),
    ('ABBOTINDIA.NS'),
    ('ACC.NS'),
    ('AEGISLOG.NS'),
    ('AJANTPHARM.NS'),
    ('ALKEM.NS'),
    ('AMBER.NS'),
    ('AMBUJACEM.NS'),
    ('ANANTRAJ.NS'),
    ('APLLTD.NS'),
    ('APOLLOHOSP.NS'),
    ('APOLLOTYRE.NS'),
    ('ASHOKLEY.NS'),
    ('ASIANPAINT.NS'),
    ('AUROPHARMA.NS'),
    ('AXISBANK.NS'),
    ('BAJAJ-AUTO.NS'),
    ('BAJAJFINSV.NS'),
    ('BAJFINANCE.NS'),
    ('BATAINDIA.NS'),
    ('BEL.NS'),
    ('BEML.NS'),
    ('BHARATFORG.NS'),
    ('BHARTIARTL.NS'),
    ('BIOCON.NS'),
    ('BLUESTARCO.NS'),
    ('BPCL.NS'),
    ('BRIGADE.NS'),
    ('CENTURYPLY.NS'),
    ('CERA.NS'),
    ('CIPLA.NS'),
    ('COALINDIA.NS'),
    ('COFORGE.NS'),
    ('CROMPTON.NS'),
    ('DIVISLAB.NS'),
    ('DIXON.NS'),
    ('DLF.NS'),
    ('DRREDDY.NS'),
    ('EICHERMOT.NS'),
    ('EXIDEIND.NS'),
    ('FEDERALBNK.NS'),
    ('GLAND.NS'),
    ('GLENMARK.NS'),
    ('GODREJCP.NS'),
    ('GODREJPROP.NS'),
    ('GRASIM.NS'),
    ('GAIL.NS'),
    ('HAVELLS.NS'),
    ('HCLTECH.NS'),
    ('HDFCBANK.NS'),
    ('HDFCLIFE.NS'),
    ('HEROMOTOCO.NS'),
    ('HINDALCO.NS'),
    ('HINDPETRO.NS'),
    ('HINDUNILVR.NS'),
    ('ICICIBANK.NS'),
    ('IDFCFIRSTB.NS'),
    ('IGL.NS'),
    ('INFY.NS'),
    ('IOC.NS'),
    ('ITC.NS'),
    ('JKLAKSHMI.NS'),
    ('JINDALSTEL.NS'),
    ('JIOFIN.NS'),
    ('JSWSTEEL.NS'),
    ('JSL.NS'),
    ('JUBLFOOD.NS'),
    ('KAJARIACER.NS'),
    ('KALYANKJIL.NS'),
    ('KOTAKBANK.NS'),
    ('LAURUSLABS.NS'),
    ('LICHSGFIN.NS'),
    ('LODHA.NS'),
    ('LT.NS'),
    ('LTM.NS'),
    ('LTTS.NS'),
    ('LUPIN.NS'),
    ('M&M.NS'),
    ('MANKIND.NS'),
    ('MARUTI.NS'),
    ('MAXHEALTH.NS'),
    ('MGL.NS'),
    ('MOTHERSON.NS'),
    ('MPHASIS.NS'),
    ('NATIONALUM.NS'),
    ('NESTLEIND.NS'),
    ('NTPC.NS'),
    ('OBEROIRLTY.NS'),
    ('OFSS.NS'),
    ('ONGC.NS'),
    ('PETRONET.NS'),
    ('PHOENIXLTD.NS'),
    ('POWERGRID.NS'),
    ('PRESTIGE.NS'),
    ('PERSISTENT.NS'),
    ('RADICO.NS'),
    ('RAMCOIND.NS'),
    ('RELIANCE.NS'),
    ('SBILIFE.NS'),
    ('SBIN.NS'),
    ('SHRIRAMFIN.NS'),
    ('SIEMENS.NS'),
    ('SIGNATURE.NS'),
    ('SONACOMS.NS'),
    ('SUNPHARMA.NS'),
    ('TATACONSUM.NS'),
    ('TATASTEEL.NS'),
    ('TMPV.NS'),
    ('TMCV.NS'),
    ('TATAPOWER.NS'),
    ('TCS.NS'),
    ('TECHM.NS'),
    ('TIINDIA.NS'),
    ('TITAN.NS'),
    ('TORNTPHARM.NS'),
    ('TRENT.NS'),
    ('TVSMOTOR.NS'),
    ('ULTRACEMCO.NS'),
    ('UNOMINDA.NS'),
    ('UPL.NS'),
    ('VGUARD.NS'),
    ('VENKEYS.NS'),
    ('VOLTAS.NS'),
    ('WELCORP.NS'),
    ('WHIRLPOOL.NS'),
    ('WIPRO.NS'),
    ('ZYDUSLIFE.NS')
ON CONFLICT (symbol) DO NOTHING;

CREATE TABLE IF NOT EXISTS public.stock_alert_recipients (
    id BIGSERIAL PRIMARY KEY,
    email TEXT NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    alert_rises BOOLEAN NOT NULL DEFAULT TRUE,
    alert_falls BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS stock_alert_recipients_email_lower_uq
    ON public.stock_alert_recipients (LOWER(email));

CREATE TABLE IF NOT EXISTS public.stock_alert_settings (
    id SMALLINT PRIMARY KEY CHECK (id = 1),
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    thresholds JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO public.stock_alert_settings (id, enabled, thresholds)
VALUES (
    1,
    TRUE,
    '{"1":{"fall":5.0,"rise":5.0},"2":{"fall":6.5,"rise":6.5},"3":{"fall":8.0,"rise":8.0},"4":{"fall":10.0,"rise":10.0},"5":{"fall":12.0,"rise":12.0}}'::jsonb
)
ON CONFLICT (id) DO NOTHING;

CREATE TABLE IF NOT EXISTS public.stock_alert_events (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('rise', 'fall')),
    period_sessions SMALLINT NOT NULL CHECK (period_sessions IN (1, 2, 3, 4, 5)),
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    change_pct NUMERIC(10,4) NOT NULL,
    start_close NUMERIC(18,4) NOT NULL,
    end_close NUMERIC(18,4) NOT NULL,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (symbol, direction, period_sessions, end_date)
);

CREATE TABLE IF NOT EXISTS public.stock_alert_deliveries (
    id BIGSERIAL PRIMARY KEY,
    event_id BIGINT NOT NULL
        REFERENCES public.stock_alert_events(id) ON DELETE CASCADE,
    recipient_id BIGINT NOT NULL
        REFERENCES public.stock_alert_recipients(id) ON DELETE CASCADE,
    status TEXT NOT NULL CHECK (status IN ('sent', 'failed')),
    error TEXT,
    sent_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (event_id, recipient_id)
);

CREATE TABLE IF NOT EXISTS public.stock_alert_runs (
    run_date DATE PRIMARY KEY,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    event_count INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

COMMIT;
