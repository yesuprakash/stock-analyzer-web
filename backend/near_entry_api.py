#!/usr/bin/env python
# ↑ shebang MUST be first line

import os
import sys
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------------------------------
# PROJECT ROOT (DO NOT CHANGE)
# -------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# REQUIRED for backend imports
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# -------------------------------------------------
# LOG LOCATION (same pattern as market_correction)
# -------------------------------------------------
PUBLIC_HTML = os.path.abspath(os.path.join(BASE_DIR, "public_html"))
LOG_DIR = os.path.join(PUBLIC_HTML, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "near_entry_api.log")

open(LOG_FILE, "w").close()

logger = logging.getLogger("near_entry_api")
logger.setLevel(logging.INFO)

if not logger.handlers:
    fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

logger.propagate = False

# -------------------------------------------------
# STARTUP LOGS
# -------------------------------------------------
logger.info("====================================")
logger.info("NEAR ENTRY API LOADED")
logger.info(f"Time: {datetime.now()}")
logger.info(f"BASE_DIR: {BASE_DIR}")
logger.info(f"CWD: {os.getcwd()}")
logger.info(f"sys.argv: {sys.argv}")
logger.info("====================================")

# -------------------------------------------------
# FORCE WORKING DIRECTORY
# -------------------------------------------------
os.chdir(BASE_DIR)

# -------------------------------------------------
# BACKEND IMPORTS (SAME AS STREAMLIT)
# -------------------------------------------------
from backend.db import get_connection

# -------------------------------------------------
# ENV
# -------------------------------------------------
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

# -------------------------------------------------
# HARD-CODED CONFIG (PHASE-1 — MATCH STREAMLIT)
# -------------------------------------------------
CONFIG = {
    "max_prediction_age_days": 3,
    "min_pct_tolerance": 2.0,
    "min_abs_tolerance": 0.0,

    "rsi_min": 45,
    "ma_period": 20,
    "min_reward_risk": 1.5,
    "max_atr_pct": 5.0,

    "use_rsi": True,
    "use_ma": True,
    "use_macd": True,
    "use_rr": True,
    "use_volatility": True,
}

# -------------------------------------------------
# HELPERS (SAME AS STREAMLIT)
# -------------------------------------------------
def safe_float(x):
    try:
        if x is None or pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None

def flatten_columns(df):
    df.columns = [
        "_".join(map(str, c)).strip("_") if isinstance(c, tuple) else str(c)
        for c in df.columns
    ]
    return df

def find_series(df, candidates):
    if df is None or df.empty:
        return None
    lowered = {str(c).lower(): c for c in df.columns}

    for cand in candidates:
        if cand.lower() in lowered:
            return df[lowered[cand.lower()]]

    for col in df.columns:
        for cand in candidates:
            if cand.lower() in str(col).lower():
                return df[col]

    nums = df.select_dtypes(include=[np.number]).columns
    return df[nums[0]] if len(nums) else None

# -------------------------------------------------
# PRICE FETCH (MATCH STREAMLIT)
# -------------------------------------------------
def fetch_price_df(symbol, period="60d"):
    try:
        df = yf.download(symbol, period=period, progress=False, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = flatten_columns(df)
        df.index = pd.to_datetime(df.index).normalize()
        return df
    except Exception:
        return pd.DataFrame()

# -------------------------------------------------
# INDICATORS (MATCH STREAMLIT)
# -------------------------------------------------
def calc_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))

def calc_macd_hist(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal

def calc_atr(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

# -------------------------------------------------
# DB
# -------------------------------------------------
def fetch_latest_predictions():
    conn = get_connection()
    df = pd.read_sql("""
        SELECT stock_symbol, prediction_date,
               entry_price, target_price, stop_loss,
               trade_signal, probability_success
        FROM predictions
        ORDER BY stock_symbol, prediction_date DESC
    """, conn)
    conn.close()

    if df.empty:
        return df

    df["prediction_date"] = pd.to_datetime(df["prediction_date"])
    return df.groupby("stock_symbol", as_index=False).first()

# -------------------------------------------------
# MAIN LOGIC (EXACT STREAMLIT BEHAVIOR)
# -------------------------------------------------
def main():
    preds = fetch_latest_predictions()
    rows = []

    today = datetime.now().date()

    for r in preds.itertuples(index=False):
        df = fetch_price_df(r.stock_symbol)
        if df.empty:
            continue

        close = find_series(df, ["close"])
        high = find_series(df, ["high"])
        low = find_series(df, ["low"])

        if close is None or high is None or low is None:
            continue

        last_price = safe_float(close.iloc[-1])
        entry = safe_float(r.entry_price)
        target = safe_float(r.target_price)
        stop = safe_float(r.stop_loss)

        if entry is None:
            continue

        pct_diff = ((last_price - entry) / entry) * 100

        # proximity (Streamlit logic)
        proximity_ok = False
        if abs(pct_diff) <= CONFIG["min_pct_tolerance"]:
            proximity_ok = True

        if CONFIG["min_abs_tolerance"] > 0:
            if abs(last_price - entry) <= CONFIG["min_abs_tolerance"]:
                proximity_ok = True

        if not proximity_ok:
            continue

        # -------- prediction age (🔥 MISSING EARLIER)
        pred_age_days = None
        if r.prediction_date is not None:
            pred_age_days = (today - r.prediction_date.date()).days

        allowed_signals = ["strong buy", "buy", "bullish"]

        orig_signal = str(r.trade_signal).strip().lower() if r.trade_signal else ""

        passes = {
            "age": pred_age_days is not None and pred_age_days <= CONFIG["max_prediction_age_days"],
            "signal_strength": orig_signal in allowed_signals,
            "rsi": True,
            "ma": True,
            "macd": True,
            "rr": True,
            "atr": True,
        }

        # RSI
        rsi_val = safe_float(calc_rsi(close).iloc[-1])
        if CONFIG["use_rsi"] and (rsi_val is None or rsi_val < CONFIG["rsi_min"]):
            passes["rsi"] = False

        # MA
        ma20 = safe_float(close.rolling(CONFIG["ma_period"]).mean().iloc[-1])
        if CONFIG["use_ma"] and (ma20 is None or last_price < ma20):
            passes["ma"] = False

        # MACD
        macd_hist = safe_float(calc_macd_hist(close).iloc[-1])
        if CONFIG["use_macd"] and (macd_hist is None or macd_hist < 0):
            passes["macd"] = False

        # RR
        rr = None
        if target and stop and entry > stop:
            rr = (target - entry) / (entry - stop)
        if CONFIG["use_rr"] and (rr is None or rr < CONFIG["min_reward_risk"]):
            passes["rr"] = False

        # ATR
        atr_pct = None
        if CONFIG["use_volatility"]:
            atr = calc_atr(pd.DataFrame({"High": high, "Low": low, "Close": close}))
            atr_last = safe_float(atr.iloc[-1])
            if atr_last:
                atr_pct = (atr_last / last_price) * 100
                if atr_pct > CONFIG["max_atr_pct"]:
                    passes["atr"] = False

        # -------- suggested_action (EXACT ORDER)
        if not passes["age"]:
            suggested_action = "SKIP (prediction too old)"
        elif not passes["signal_strength"]:
            suggested_action = "SKIP (weak original signal)"
        elif not (passes["rsi"] and passes["ma"] and passes["macd"]):
            suggested_action = "SKIP (trend not confirmed)"
        elif not passes["rr"]:
            suggested_action = "SKIP (reward-risk too low)"
        elif not passes["atr"]:
            suggested_action = "SKIP (high volatility)"
        else:
            suggested_action = "ACTIONABLE (passes all filters)"


        rows.append({
            "stock_symbol": r.stock_symbol,
            "entry_price": entry,
            "last_price": round(last_price, 2),
            "pct_diff": round(pct_diff, 2),
            "rsi": rsi_val,
            "ma20": ma20,
            "macd_hist": macd_hist,
            "atr_pct": atr_pct,
            "reward_risk": rr,
            "suggested_action": suggested_action,
            "prediction_age_days": pred_age_days,
            "probability": r.probability_success,
        })

    print(json.dumps(rows))

# -------------------------------------------------
# EXECUTE
# -------------------------------------------------
if __name__ == "__main__":
    main()
    logger.info("Execution completed")
