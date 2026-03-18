#!/usr/bin/env python
# ↑ shebang MUST be first line

import os
import sys
import logging
from datetime import datetime
import math

# -------------------------------------------------
# PROJECT ROOT (DO NOT CHANGE)
# -------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# -------------------------------------------------
# LOG LOCATION
# -------------------------------------------------
PUBLIC_HTML = os.path.abspath(os.path.join(BASE_DIR, "public_html"))
LOG_DIR = os.path.join(PUBLIC_HTML, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "near_entry_api.log")

# CLEAR OLD LOG
open(LOG_FILE, "w").close()

# -------------------------------------------------
# LOGGER
# -------------------------------------------------
logger = logging.getLogger("near_entry_api")
logger.setLevel(logging.INFO)

if not logger.handlers:
    fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

logger.propagate = False

# -------------------------------------------------
# STARTUP LOGS
# -------------------------------------------------
logger.info("====================================")
logger.info("NEAR MARKET CORRECTION LOADED")
logger.info(f"Time: {datetime.now()}")
logger.info(f"BASE_DIR: {BASE_DIR}")
logger.info(f"CWD: {os.getcwd()}")
logger.info(f"sys.path: {sys.path}")
logger.info(f"sys.argv: {sys.argv}")
logger.info("====================================")

# -------------------------------------------------
# ARG SAFETY
# -------------------------------------------------
if len(sys.argv) < 2:
    logger.info("No args, exiting early")
    print("[]")
    sys.exit(0)

# -------------------------------------------------
# FORCE WORKING DIRECTORY
# -------------------------------------------------
os.chdir(BASE_DIR)
logger.info(f"Changed CWD to {BASE_DIR}")

# -------------------------------------------------
# IMPORTS
# -------------------------------------------------
import json
import pandas as pd
import numpy as np
import yfinance as yf
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from backend.db import get_connection
from backend.market_utils import (
    safe_float,
    fetch_price_df,
    calc_rsi,
    calc_macd_hist,
    calc_atr
)

# -------------------------------------------------
# ENV
# -------------------------------------------------
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

# -------------------------------------------------
# 🔥 NUMERIC CLEANER (CRITICAL FIX)
# -------------------------------------------------
def clean_num(v):
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        if math.isnan(v) or math.isinf(v):
            return None
    return float(v)

# -------------------------------------------------
# DB FETCH
# -------------------------------------------------
def fetch_latest_predictions():
    conn = get_connection()
    df = pd.read_sql("""
        SELECT stock_symbol, entry_price, target_price,
               stop_loss, trade_signal, probability_success
        FROM predictions
        ORDER BY stock_symbol, prediction_date DESC
    """, conn)
    conn.close()

    if df.empty:
        return df

    return df.groupby("stock_symbol", as_index=False).first()

# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    logger.info("Entered main()")

    skip_filters = False
    if len(sys.argv) >= 8:
        skip_filters = int(sys.argv[7]) == 1

    correction_window = int(sys.argv[1])
    min_corr = float(sys.argv[2])
    max_corr = float(sys.argv[3])
    rsi_min = float(sys.argv[4])
    max_atr_pct = float(sys.argv[5])
    min_rr = float(sys.argv[6])

    preds = fetch_latest_predictions()
    rows = []

    for r in preds.itertuples(index=False):
        logger.info(f"Processing {r.stock_symbol}")

        df = fetch_price_df(r.stock_symbol)
        if df.empty or len(df) < correction_window:
            continue

        close = df['Close']
        high = df['High']

        last_price = clean_num(safe_float(close.iloc[-1]))
        high_close = clean_num(safe_float(close.tail(correction_window).max()))
        high_intraday = clean_num(safe_float(high.tail(correction_window).max()))

        if None in (last_price, high_close, high_intraday):
            continue

        # -----------------------------
        # CORRECTIONS
        # -----------------------------
        corr_close = clean_num(((last_price - high_close) / high_close) * 100)
        corr_intraday = clean_num(((last_price - high_intraday) / high_intraday) * 100)

        if not skip_filters and not (max_corr <= corr_close <= min_corr):
            continue

        # -----------------------------
        # INDICATORS
        # -----------------------------
        rsi_val = clean_num(safe_float(calc_rsi(close).iloc[-1]))
        macd_val = clean_num(safe_float(calc_macd_hist(close).iloc[-1]))
        atr_val = clean_num(safe_float(calc_atr(df).iloc[-1]))

        if rsi_val is None or macd_val is None or atr_val is None:
            continue

        if not skip_filters and rsi_val < rsi_min:
            continue

        if not skip_filters and macd_val < 0:
            continue

        # -----------------------------
        # ATR %
        # -----------------------------
        if last_price == 0:
            continue

        atr_pct = clean_num((atr_val / last_price) * 100)

        if atr_pct is None:
            continue

        if not skip_filters and atr_pct > max_atr_pct:
            continue

        # -----------------------------
        # RR
        # -----------------------------
        if not (r.entry_price and r.target_price and r.stop_loss):
            continue

        den = (r.entry_price - r.stop_loss)
        if den == 0:
            continue

        rr = clean_num((r.target_price - r.entry_price) / den)

        if rr is None:
            continue

        if not skip_filters and rr < min_rr:
            continue

        # -----------------------------
        # FINAL ROW
        # -----------------------------
        row = {
            "stock": r.stock_symbol,
            "last_price": round(last_price, 2),
            "corr_close": round(corr_close, 2),
            "corr_intraday": round(corr_intraday, 2),
            "rsi": round(rsi_val, 2),
            "atr_pct": round(atr_pct, 2),
            "rr": round(rr, 2),
            "signal": r.trade_signal,
            "probability": clean_num(r.probability_success)
        }

        rows.append(row)

    rows.sort(key=lambda x: (x["corr_close"], -x["rr"]))

    # 🔥 STRICT JSON (CRITICAL)
    print(json.dumps(rows, allow_nan=False))

# -------------------------------------------------
# EXECUTION
# -------------------------------------------------
logger.info("About to call main()")

if __name__ == "__main__":
    main()

logger.info("Reached end of file (EOF)")