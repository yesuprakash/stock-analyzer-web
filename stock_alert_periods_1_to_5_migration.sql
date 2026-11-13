BEGIN;

ALTER TABLE public.stock_alert_events
    DROP CONSTRAINT IF EXISTS stock_alert_events_period_sessions_check;

ALTER TABLE public.stock_alert_events
    ADD CONSTRAINT stock_alert_events_period_sessions_check
    CHECK (period_sessions IN (1, 2, 3, 4, 5));

UPDATE public.stock_alert_settings
SET thresholds = COALESCE(thresholds, '{}'::jsonb)
        || CASE
            WHEN COALESCE(thresholds, '{}'::jsonb) ? '2' THEN '{}'::jsonb
            ELSE '{"2":{"fall":6.5,"rise":6.5}}'::jsonb
        END
        || CASE
            WHEN COALESCE(thresholds, '{}'::jsonb) ? '4' THEN '{}'::jsonb
            ELSE '{"4":{"fall":10.0,"rise":10.0}}'::jsonb
        END,
    updated_at = NOW()
WHERE id = 1;

COMMIT;
