#!/usr/bin/env python

import datetime
import sys
import os
import traceback
from backend.logger import logger

def parse_args():
    args = {}
    for arg in sys.argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            args[k] = v
    return args

ARGS = parse_args()


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
config = {
    "max_prediction_age_days": int(ARGS.get("max_prediction_age_days", 3)),
    "min_pct_tolerance": float(ARGS.get("min_pct_tolerance", 2.0)),
    "min_abs_tolerance": 0.0,

    "rsi_min": int(ARGS.get("rsi_min", 45)),
    "ma_period": 20,
    "min_reward_risk": float(ARGS.get("min_reward_risk", 1.5)),
    "max_atr_pct": float(ARGS.get("max_atr_pct", 5.0)),

    "use_rsi_filter": ARGS.get("use_rsi", "1") == "1",
    "use_ma_filter": ARGS.get("use_ma", "1") == "1",
    "use_macd_filter": ARGS.get("use_macd", "1") == "1",
    "use_rr_filter": True,
    "use_gap_filter": True,
    "use_volume_filter": True,
    "use_volatility_filter": True,

    "allowed_signals": ["Strong Buy", "Buy", "Bullish"],

    "period_for_history": "60d",
    "auto_adjust": False,
    "latest_only": True,
    "actionable_only": ARGS.get("actionable_only", "0") == "1"
}


# -------------------------------------------------
# EXECUTE
# -------------------------------------------------
if __name__ == "__main__":
    try:
        logger.info("==== PYTHON START ==== " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        logger.info("ARGS: " + str(ARGS))

        rows = run_near_entry_logic(config)

        logger.info("Rows count: " + str(len(rows)))

        print(json.dumps(rows, default=str))

        logger.info("==== PYTHON END SUCCESS ====")

    except Exception as e:
        logger.error("ERROR OCCURRED")
        logger.error(traceback.format_exc())
        print(json.dumps({"error": str(e)}))
