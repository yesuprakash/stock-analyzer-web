BEGIN;

UPDATE public.stock_master AS current_symbol
SET enabled = current_symbol.enabled OR old_symbol.enabled,
    company_name = CASE
        WHEN current_symbol.company_name = '' THEN 'LTM Limited'
        ELSE current_symbol.company_name
    END,
    updated_at = NOW()
FROM public.stock_master AS old_symbol
WHERE current_symbol.symbol = 'LTM.NS'
  AND old_symbol.symbol = 'LTIM.NS';

UPDATE public.stock_master AS old_symbol
SET symbol = 'LTM.NS',
    company_name = 'LTM Limited',
    updated_at = NOW()
WHERE old_symbol.symbol = 'LTIM.NS'
  AND NOT EXISTS (
      SELECT 1
      FROM public.stock_master AS current_symbol
      WHERE current_symbol.symbol = 'LTM.NS'
  );

UPDATE public.stock_master
SET enabled = FALSE,
    updated_at = NOW()
WHERE symbol IN ('GSPL.NS', 'GUJGASLTD.NS', 'JBCHEPHARM.NS', 'LTIM.NS');

INSERT INTO public.stock_master (symbol, company_name, enabled)
VALUES ('LTM.NS', 'LTM Limited', TRUE)
ON CONFLICT (symbol) DO NOTHING;

COMMIT;
