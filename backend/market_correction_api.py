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
PUBLIC_HTML = os.path.abspath(
    os.path.join(BASE_DIR, "public_html")
    if os.name == "nt"
    else os.path.join(BASE_DIR, "..", "public_html")
)
LOG_DIR = os.path.join(PUBLIC_HTML, "logs")

LOG_FILE = os.path.join(LOG_DIR, "market_correction_api.log")

logger = logging.getLogger("market_correction_api")
logger.setLevel(logging.INFO)
logger.propagate = False

def configure_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    open(LOG_FILE, "w").close()
    handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)

# -------------------------------------------------
# IMPORTS
# -------------------------------------------------
import json
import pandas as pd
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from backend.db import get_connection
from backend.stock_master import get_active_stock_symbols
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
    try:
        value = float(v)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None

# -------------------------------------------------
# DB FETCH
# -------------------------------------------------
def fetch_latest_predictions():
    active_symbols = get_active_stock_symbols()
    if not active_symbols:
        return pd.DataFrame(columns=[
            "stock_symbol", "prediction_date", "entry_price", "target_price",
            "stop_loss", "trade_signal", "probability_success",
        ])

    conn = get_connection()
    df = pd.read_sql("""
        SELECT stock_symbol, prediction_date, entry_price, target_price,
               stop_loss, trade_signal, probability_success
        FROM predictions
        WHERE stock_symbol = ANY(%s)
        ORDER BY stock_symbol, prediction_date DESC
    """, conn, params=(active_symbols,))
    conn.close()

    if df.empty:
        return df

    df["prediction_date"] = pd.to_datetime(df["prediction_date"])
    return (
        df.sort_values(["stock_symbol", "prediction_date"], kind="mergesort")
          .drop_duplicates("stock_symbol", keep="last")
          .reset_index(drop=True)
    )

# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    if len(sys.argv) < 7:
        raise ValueError("Expected correction window and all strategy thresholds.")

    os.chdir(BASE_DIR)
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
    max_data_age_days = int(sys.argv[8]) if len(sys.argv) >= 9 else 5

    if correction_window not in (5, 10, 15, 30, 45, 60):
        raise ValueError("Window must be one of 5, 10, 15, 30, 45, or 60 sessions.")
    if min_corr > max_corr:
        raise ValueError("Minimum correction must be less than or equal to maximum correction.")
    if not 0 <= rsi_min <= 100:
        raise ValueError("RSI minimum must be between 0 and 100.")
    if max_atr_pct < 0 or min_rr < 0 or not 0 <= max_data_age_days <= 30:
        raise ValueError("ATR, reward/risk, or data-age threshold is outside its valid range.")

    preds = fetch_latest_predictions()
    rows = []
    market_today = pd.Timestamp.now(tz="Asia/Kolkata").date()

    for r in preds.itertuples(index=False):
        logger.info(f"Processing {r.stock_symbol}")

        df = fetch_price_df(r.stock_symbol, auto_adjust=False)
        if df.empty or len(df) < correction_window:
            continue
        if not {"Close", "High"}.issubset(df.columns):
            continue

        close = pd.to_numeric(df['Close'], errors='coerce').dropna().sort_index()
        high = pd.to_numeric(df['High'], errors='coerce').reindex(close.index)
        if len(close) < correction_window or high.dropna().empty:
            continue

        last_price = clean_num(close.iloc[-1])
        high_close = clean_num(safe_float(close.tail(correction_window).max()))
        high_intraday = clean_num(safe_float(high.tail(correction_window).max()))

        if None in (last_price, high_close, high_intraday):
            continue
        if last_price <= 0 or high_close <= 0 or high_intraday <= 0:
            continue

        price_data_date = pd.Timestamp(close.index[-1])
        if price_data_date.tzinfo is not None:
            price_data_date = price_data_date.tz_convert("Asia/Kolkata")
        price_age_days = (market_today - price_data_date.date()).days
        prediction_date = pd.Timestamp(r.prediction_date)
        if pd.isna(prediction_date):
            continue
        if prediction_date.tzinfo is not None:
            prediction_date = prediction_date.tz_convert("Asia/Kolkata")
        prediction_age_days = (market_today - prediction_date.date()).days
        if not (
            0 <= price_age_days <= max_data_age_days
            and 0 <= prediction_age_days <= max_data_age_days
        ):
            continue

        # -----------------------------
        # CORRECTIONS
        # -----------------------------
        corr_close = clean_num(((last_price - high_close) / high_close) * 100)
        corr_intraday = clean_num(((last_price - high_intraday) / high_intraday) * 100)

        if not skip_filters and not (min_corr <= corr_close <= max_corr):
            continue

        # -----------------------------
        # INDICATORS
        # -----------------------------
        rsi_val = clean_num(safe_float(calc_rsi(close).iloc[-1]))
        macd_val = clean_num(safe_float(calc_macd_hist(close).iloc[-1]))
        price_data = df.reindex(close.index)
        atr_val = None
        if {"High", "Low", "Close"}.issubset(price_data.columns):
            atr_val = clean_num(safe_float(calc_atr(price_data).iloc[-1]))

        if not skip_filters and (rsi_val is None or macd_val is None or atr_val is None):
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

        atr_pct = (
            clean_num((atr_val / last_price) * 100)
            if atr_val is not None
            else None
        )

        if not skip_filters and atr_pct is None:
            continue

        if not skip_filters and atr_pct > max_atr_pct:
            continue

        # Recalculate reward/risk using the price at which a trade could be entered now.
        target_price = clean_num(r.target_price)
        stop_loss = clean_num(r.stop_loss)
        rr = None
        if target_price is not None and stop_loss is not None:
            reward = target_price - last_price
            risk = last_price - stop_loss
            if reward > 0 and risk > 0:
                rr = clean_num(reward / risk)

        if not skip_filters and (rr is None or rr < min_rr):
            continue

        # -----------------------------
        # FINAL ROW
        # -----------------------------
        row = {
            "stock": r.stock_symbol,
            "prediction_date": prediction_date.strftime("%Y-%m-%d"),
            "price_data_date": price_data_date.strftime("%Y-%m-%d"),
            "price_age_days": price_age_days,
            "last_price": round(last_price, 2),
            "corr_close": round(corr_close, 2),
            "corr_from_high": round(corr_intraday, 2),
            "rsi": round(rsi_val, 2) if rsi_val is not None else None,
            "atr_pct": round(atr_pct, 2) if atr_pct is not None else None,
            "rr": round(rr, 2) if rr is not None else None,
            "signal": r.trade_signal,
            "probability": clean_num(r.probability_success)
        }

        rows.append(row)

    rows.sort(
        key=lambda x: (
            x["corr_close"],
            -x["rr"] if x["rr"] is not None else math.inf,
        )
    )

    # 🔥 STRICT JSON (CRITICAL)
    print(json.dumps(rows, allow_nan=False))

# -------------------------------------------------
# EXECUTION
# -------------------------------------------------
if __name__ == "__main__":
    configure_logging()
    logger.info("====================================")
    logger.info("NEAR MARKET CORRECTION LOADED")
    logger.info(f"Time: {datetime.now()}")
    logger.info(f"BASE_DIR: {BASE_DIR}")
    logger.info(f"CWD: {os.getcwd()}")
    logger.info(f"sys.path: {sys.path}")
    logger.info(f"sys.argv: {sys.argv}")
    logger.info("====================================")

    if len(sys.argv) < 2:
        logger.info("No args, exiting early")
        print("[]")
        sys.exit(0)

    logger.info("About to call main()")
    main()
    logger.info("Reached end of file (EOF)")