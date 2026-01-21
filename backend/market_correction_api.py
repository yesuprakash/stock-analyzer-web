#!/usr/bin/env python
# ↑ shebang MUST be first line

import os
import sys
import logging
from datetime import datetime



# -------------------------------------------------
# PROJECT ROOT
# -------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 🔥 THIS FIXES backend.* IMPORTS
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# -------------------------------------------------
# LOGGING
# -------------------------------------------------
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "market_correction_api.log")

# -------------------------------------------------
# CLEAR OLD LOG ON EVERY RUN
# -------------------------------------------------
if os.path.exists(LOG_FILE):
    open(LOG_FILE, "w").close()

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True
)

logger = logging.getLogger("market_correction_api")

logger.info("==== RAW sys.argv DUMP START ====")
logger.info(f"sys.argv (len={len(sys.argv)}): {sys.argv}")
for idx, val in enumerate(sys.argv):
    logger.info(f"argv[{idx}] = {repr(val)}")
logger.info("==== RAW sys.argv DUMP END ====")

logger.info("====================================")
logger.info("SCRIPT LOADED")
logger.info(f"Time: {datetime.now()}")
logger.info(f"BASE_DIR: {BASE_DIR}")
logger.info(f"CWD: {os.getcwd()}")
logger.info(f"sys.path: {sys.path}")
logger.info("====================================")


# -------------------------------------------------
# ARG SAFETY
# -------------------------------------------------
if len(sys.argv) < 2:
    logger.info("No args, exiting early")
    sys.exit(0)

# -------------------------------------------------
# FORCE WORKING DIRECTORY
# -------------------------------------------------
os.chdir(BASE_DIR)
logger.info(f"Changed CWD to {BASE_DIR}")

# -------------------------------------------------
# STANDARD IMPORTS
# -------------------------------------------------
import json
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

# -------------------------------------------------
# BACKEND IMPORTS (NOW 100% SAFE)
# -------------------------------------------------
from backend.db import get_connection
from backend.market_utils import (
    safe_float,
    fetch_price_df,
    calc_rsi,
    calc_macd_hist,
    calc_atr
)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# -------------------------------------------------
# ENV
# -------------------------------------------------
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)



# -------------------------------------------------
# SAFETY
# -------------------------------------------------
def safe_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, (pd.Series, pd.DataFrame)):
            x = x.iloc[-1]
        return float(x)
    except Exception:
        return None

# -------------------------------------------------
# PRICE FETCH
# -------------------------------------------------
def fetch_price_df(symbol, period="120d"):
    try:
        df = yf.download(symbol, period=period, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index).normalize()
        return df
    except Exception:
        return pd.DataFrame()

# -------------------------------------------------
# INDICATORS (unchanged)
# -------------------------------------------------
def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_macd_hist(close):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal

def calc_atr(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)

    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    return tr.ewm(alpha=1/period, adjust=False).mean()

# -------------------------------------------------
# DB FETCH (same query)
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
# MAIN (replaces Streamlit)
# -------------------------------------------------
def main():
    logger.info("Entered main()")
    skip_filters = False
    if len(sys.argv) >= 8:
        skip_filters = int(sys.argv[7]) == 1

    logger.info(f"Skip filters = {skip_filters}")
    # args from PHP
    correction_window = int(sys.argv[1])
    min_corr = float(sys.argv[2])
    max_corr = float(sys.argv[3])
    rsi_min = float(sys.argv[4])
    max_atr_pct = float(sys.argv[5])
    min_rr = float(sys.argv[6])
    logger.info(
    f"Params → window={correction_window}, min_corr={min_corr}, "
    f"max_corr={max_corr}, rsi_min={rsi_min}, max_atr={max_atr_pct}, min_rr={min_rr}, skip_filters={skip_filters}"
)
    preds = fetch_latest_predictions()
    rows = []

    for r in preds.itertuples(index=False):
        logger.info(f"Processing {r.stock_symbol}")
        df = fetch_price_df(r.stock_symbol)
        if df.empty or len(df) < correction_window:
            logger.info(f"{r.stock_symbol} skipped: insufficient price data")
            continue

        close = df['Close']
        high = df['High']

        last_price = safe_float(close.iloc[-1])
        high_close = safe_float(close.tail(correction_window).max())
        high_intraday = safe_float(high.tail(correction_window).max())

        if None in (last_price, high_close, high_intraday):
            continue

        corr_close = ((last_price - high_close) / high_close) * 100
        corr_intraday = ((last_price - high_intraday) / high_intraday) * 100
        if not skip_filters:
            if not (max_corr <= corr_close <= min_corr):
                continue

        rsi_val = safe_float(calc_rsi(close).iloc[-1])
        if not skip_filters:
            if rsi_val is None or rsi_val < rsi_min:
                logger.info(
            f"{r.stock_symbol} skipped: rsi={rsi_val}"
        )
                continue

        macd_val = safe_float(calc_macd_hist(close).iloc[-1])
        if not skip_filters:
            if macd_val is None or macd_val < 0:
                continue

        atr_val = safe_float(calc_atr(df).iloc[-1])
        if atr_val is None:
            continue

        atr_pct = (atr_val / last_price) * 100
        if not skip_filters:
            if atr_pct > max_atr_pct:
                logger.info(
            f"{r.stock_symbol} skipped: atr_pct={atr_pct:.2f}"
        )
                continue

        if not (r.entry_price and r.target_price and r.stop_loss):
            continue

        rr = (r.target_price - r.entry_price) / (r.entry_price - r.stop_loss)
        if not skip_filters:
            if rr < min_rr:
                logger.info(
            f"{r.stock_symbol} skipped: rr={rr:.2f}"
        )
                continue

        rows.append({
            "stock": r.stock_symbol,
            "last_price": round(last_price, 2),
            "corr_close": round(corr_close, 2),
            "corr_intraday": round(corr_intraday, 2),
            "rsi": round(rsi_val, 2),
            "atr_pct": round(atr_pct, 2),
            "rr": round(rr, 2),
            "signal": r.trade_signal,
            "probability": r.probability_success
        })

    rows.sort(key=lambda x: (x["corr_close"], -x["rr"]))
    print(json.dumps(rows))

logger.info("About to call main()")

if __name__ == "__main__":
    main()

logger.info("Reached end of file (EOF)")