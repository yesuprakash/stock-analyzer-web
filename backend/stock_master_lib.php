<?php
declare(strict_types=1);

require_once __DIR__ . DIRECTORY_SEPARATOR . 'stock_alert_lib.php';

function stock_master_ensure_schema($connection): void
{
    stock_alert_query(
        $connection,
        "CREATE TABLE IF NOT EXISTS stock_master (
            id BIGSERIAL PRIMARY KEY,
            symbol VARCHAR(32) NOT NULL UNIQUE,
            company_name VARCHAR(120) NOT NULL DEFAULT '',
            enabled BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CHECK (symbol = UPPER(symbol)),
            CHECK (symbol ~ '^[A-Z0-9][A-Z0-9.&-]{0,28}\\.(NS|BO)$')
        )"
    );

    $seedPath = __DIR__ . DIRECTORY_SEPARATOR . 'stock_master_seed.json';
    $seed = json_decode((string)file_get_contents($seedPath), true, 512, JSON_THROW_ON_ERROR);
    if (!is_array($seed)) {
        throw new RuntimeException('Stock master seed list is invalid.');
    }

    foreach ($seed as $symbol) {
        if (!is_string($symbol)) {
            throw new RuntimeException('Stock master seed contains an invalid symbol.');
        }
    }
    stock_alert_query(
        $connection,
        'INSERT INTO stock_master (symbol)
         SELECT jsonb_array_elements_text($1::jsonb)
         ON CONFLICT (symbol) DO NOTHING',
        [json_encode($seed, JSON_THROW_ON_ERROR)]
    );
}

function stock_master_symbols($connection, bool $enabledOnly = true): array
{
    $query = $enabledOnly
        ? 'SELECT symbol FROM stock_master WHERE enabled = TRUE ORDER BY id'
        : 'SELECT symbol FROM stock_master ORDER BY id';
    $result = stock_alert_query($connection, $query);
    $rows = pg_fetch_all($result) ?: [];
    return array_map(static fn(array $row): string => $row['symbol'], $rows);
}
