import json
import os

from backend.db import get_connection


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEED_PATH = os.path.join(BASE_DIR, "stock_master_seed.json")


def get_active_stock_symbols():
    with open(SEED_PATH, encoding="utf-8") as seed_file:
        seed_symbols = json.load(seed_file)
    if not isinstance(seed_symbols, list) or not all(
        isinstance(symbol, str) for symbol in seed_symbols
    ):
        raise RuntimeError("Stock master seed list is invalid.")

    connection = get_connection()
    try:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS stock_master (
                        id BIGSERIAL PRIMARY KEY,
                        symbol VARCHAR(32) NOT NULL UNIQUE,
                        company_name VARCHAR(120) NOT NULL DEFAULT '',
                        enabled BOOLEAN NOT NULL DEFAULT TRUE,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        CHECK (symbol = UPPER(symbol)),
                        CHECK (symbol ~ '^[A-Z0-9][A-Z0-9.&-]{0,28}\\.(NS|BO)$')
                    )
                    """
                )
                cursor.execute(
                    """
                    INSERT INTO stock_master (symbol)
                    SELECT jsonb_array_elements_text(%s::jsonb)
                    ON CONFLICT (symbol) DO NOTHING
                    """,
                    (json.dumps(seed_symbols),),
                )
                cursor.execute(
                    "SELECT symbol FROM stock_master WHERE enabled = TRUE ORDER BY id"
                )
                symbols = [row[0] for row in cursor.fetchall()]
    finally:
        connection.close()

    return symbols
