#!/usr/bin/env python3
"""Compare NSE index constituents with the symbols stored in stock_master."""

import argparse
import csv
import re
import sys
import time
from collections import defaultdict
from urllib.parse import parse_qs, urlparse

import psycopg2
import requests

from backend.db import get_connection


INDEX_PAGE_URLS = (
    "https://www.nseindia.in/market-data/live-equity-market?symbol=NIFTY%2050",
    "https://www.nseindia.in/market-data/live-equity-market?symbol=NIFTY%20NEXT%2050",
    "https://www.nseindia.in/market-data/live-equity-market?symbol=NIFTY%20BANK",
    "https://www.nseindia.in/market-data/live-equity-market?symbol=NIFTY%20MID%20SELECT",
    "https://www.nseindia.in/market-data/live-equity-market?symbol=NIFTY%20MIDCAP%2050",
    "https://www.nseindia.in/market-data/live-equity-market?symbol=NIFTY%20SMLCAP%20100",
    "https://www.nseindia.in/market-data/live-equity-market?symbol=NIFTY%20SMLCAP%2050",
    "https://www.nseindia.in/market-data/live-equity-market?symbol=NIFTY%20AUTO",
    "https://www.nseindia.in/market-data/live-equity-market?symbol=NIFTY%20FMCG",
    "https://www.nseindia.in/market-data/live-equity-market?symbol=NIFTY%20IT",
    "https://www.nseindia.in/market-data/live-equity-market?symbol=NIFTY%20METAL",
    "https://www.nseindia.in/market-data/live-equity-market?symbol=NIFTY%20PVT%20BANK",
    "https://www.nseindia.in/market-data/live-equity-market?symbol=NIFTY%20REALTY",
    "https://www.nseindia.in/market-data/live-equity-market?symbol=NIFTY%20HEALTHCARE",
    "https://www.nseindia.in/market-data/live-equity-market?symbol=NIFTY%20CONSR%20DURBL",
    "https://www.nseindia.in/market-data/live-equity-market?symbol=NIFTY%20OIL%20AND%20GAS",
)

NSE_API_URL = "https://www.nseindia.in/api/NextApi/apiClient/marketWatchApi"
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.&-]{0,28}\.NS$")


def index_name_from_url(page_url):
    values = parse_qs(urlparse(page_url).query).get("symbol", [])
    if len(values) != 1 or not values[0].strip():
        raise ValueError(f"Index URL has no unique symbol parameter: {page_url}")
    return values[0].strip()


def extract_symbols(index_name, payload):
    data = payload.get("data")
    rows = data.get("data") if isinstance(data, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"NSE returned no constituent rows for {index_name}.")

    symbols = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"NSE returned an invalid row for {index_name}.")

        raw_symbol = row.get("symbol")
        if not isinstance(raw_symbol, str) or not raw_symbol.strip():
            raise ValueError(f"NSE returned a row without a symbol for {index_name}.")

        raw_symbol = raw_symbol.strip().upper()
        if str(row.get("priority", "")) == "1" or raw_symbol == index_name.upper():
            continue

        symbol = raw_symbol if raw_symbol.endswith(".NS") else f"{raw_symbol}.NS"
        if not SYMBOL_PATTERN.fullmatch(symbol):
            raise ValueError(
                f"Unexpected NSE symbol {raw_symbol!r} in index {index_name}."
            )
        symbols.add(symbol)

    if not symbols:
        raise ValueError(f"NSE returned no stock symbols for {index_name}.")
    return symbols


def fetch_index_symbols(session, page_url):
    index_name = index_name_from_url(page_url)
    page_response = session.get(page_url, timeout=30)
    page_response.raise_for_status()

    api_response = session.get(
        NSE_API_URL,
        params={
            "functionName": "getIndicesData",
            "symbol": index_name,
        },
        headers={
            "Accept": "application/json, text/plain, */*",
            "Referer": page_url,
        },
        timeout=30,
    )
    api_response.raise_for_status()
    try:
        payload = api_response.json()
    except ValueError as error:
        raise ValueError(f"NSE returned non-JSON data for {index_name}.") from error
    if not isinstance(payload, dict):
        raise ValueError(f"NSE returned an invalid response for {index_name}.")
    return index_name, extract_symbols(index_name, payload)


def fetch_all_index_symbols(delay_seconds):
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
            ),
            "Accept": "text/html,application/json, text/plain, */*",
        }
    )
    symbols_by_index = {}
    for index_number, page_url in enumerate(INDEX_PAGE_URLS):
        index_name, symbols = fetch_index_symbols(session, page_url)
        symbols_by_index[index_name] = symbols
        print(
            f"Fetched {index_name}: {len(symbols)} constituent symbols",
            file=sys.stderr,
        )
        if delay_seconds and index_number < len(INDEX_PAGE_URLS) - 1:
            time.sleep(delay_seconds)
    return symbols_by_index


def read_database_symbols():
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT symbol, enabled FROM stock_master ORDER BY symbol")
            return {
                str(symbol).strip().upper(): bool(enabled)
                for symbol, enabled in cursor.fetchall()
            }
    finally:
        connection.close()


def write_csv(path, symbols_by_index, database_symbols):
    indices_by_symbol = defaultdict(set)
    for index_name, symbols in symbols_by_index.items():
        for symbol in symbols:
            indices_by_symbol[symbol].add(index_name)

    all_symbols = sorted(set(indices_by_symbol) | set(database_symbols))
    with open(path, "w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            ["symbol", "nse_indices", "in_database", "enabled"]
        )
        for symbol in all_symbols:
            writer.writerow(
                [
                    symbol,
                    "; ".join(sorted(indices_by_symbol.get(symbol, set()))),
                    symbol in database_symbols,
                    database_symbols.get(symbol, ""),
                ]
            )


def print_comparison(symbols_by_index, database_symbols):
    nse_symbols = set().union(*symbols_by_index.values())
    missing = nse_symbols - set(database_symbols)
    database_only = set(database_symbols) - nse_symbols
    duplicate_memberships = {
        symbol: sorted(
            index_name
            for index_name, symbols in symbols_by_index.items()
            if symbol in symbols
        )
        for symbol in nse_symbols
    }

    print("\nPer-index comparison")
    for index_name, symbols in symbols_by_index.items():
        matched = symbols & set(database_symbols)
        print(
            f"{index_name}: {len(symbols)} NSE, {len(matched)} in database, "
            f"{len(symbols - set(database_symbols))} missing"
        )

    print(
        f"\nUnique NSE symbols: {len(nse_symbols)}"
        f"\nDatabase symbols: {len(database_symbols)}"
        f"\nNSE symbols missing from database: {len(missing)}"
        f"\nDatabase symbols outside these indices: {len(database_only)}"
    )

    if missing:
        print("\nMissing from database (symbol - NSE index membership)")
        for symbol in sorted(missing):
            print(f"{symbol} - {', '.join(duplicate_memberships[symbol])}")
    if database_only:
        print("\nIn database but not in these NSE indices")
        for symbol in sorted(database_only):
            state = "enabled" if database_symbols[symbol] else "disabled"
            print(f"{symbol} ({state})")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Read constituents from the supplied NSE indices and compare them "
            "with stock_master. This script does not change the database."
        )
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to wait between NSE index requests (default: 0.5).",
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        help="Optional path for a CSV containing the union of NSE and database symbols.",
    )
    args = parser.parse_args()
    if args.delay < 0:
        parser.error("--delay must be zero or greater.")
    return args


def main():
    args = parse_arguments()
    try:
        symbols_by_index = fetch_all_index_symbols(args.delay)
        database_symbols = read_database_symbols()
    except (requests.RequestException, ValueError, psycopg2.Error) as error:
        print(f"Comparison failed: {error}", file=sys.stderr)
        return 1

    print_comparison(symbols_by_index, database_symbols)
    if args.csv_path:
        write_csv(args.csv_path, symbols_by_index, database_symbols)
        print(f"\nCSV written to {args.csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
