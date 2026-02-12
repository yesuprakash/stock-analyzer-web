#!/usr/bin/env python

import os
import sys
import json
from dotenv import load_dotenv

# -------------------------------------------------
# PROJECT ROOT
# -------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# -------------------------------------------------
# LOAD ENV
# -------------------------------------------------
load_dotenv(os.path.join(BASE_DIR, ".env"))

# -------------------------------------------------
# IMPORT CORE
# -------------------------------------------------
from backend.near_entry_core import run_near_entry_logic

# -------------------------------------------------
# HARD-CODE CONFIG (match Streamlit defaults)
# -------------------------------------------------
CONFIG = {
    "latest_only": True,
    "actionable_only": False,

    "max_prediction_age_days": 3,
    "min_pct_tolerance": 2.0,
    "min_abs_tolerance": 0.0,

    "rsi_period": 14,
    "rsi_min": 45,
    "use_rsi_filter": True,

    "ma_period": 20,
    "use_ma_filter": True,

    "use_macd_filter": True,
    "macd_hist_min": 0,

    "use_volume_filter": True,
    "min_volume_ratio": 1.2,

    "use_rr_filter": True,
    "min_reward_risk": 1.5,

    "use_gap_filter": True,
    "max_gap_pct": 1.5,

    "use_volatility_filter": True,
    "max_atr_pct": 5.0,
    "atr_period": 14,

    "allowed_signals": ["Strong Buy", "Buy", "Bullish"],

    "allow_short_on_reversal": False,
    "auto_adjust": False,
    "period_for_history": "60d"
}

# -------------------------------------------------
# EXECUTE
# -------------------------------------------------
if __name__ == "__main__":
    rows = run_near_entry_logic(CONFIG)
    print(json.dumps(rows, default=str))
