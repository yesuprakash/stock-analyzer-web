import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from backend.near_entry_core import (
    evaluate_filters_for_symbol,
    fetch_active_predictions,
    fetch_price_df,
    run_near_entry_logic,
)


def near_entry_config():
    return {
        "max_prediction_age_days": 10,
        "min_pct_tolerance": 2.0,
        "min_abs_tolerance": 0.0,
        "rsi_min": 45,
        "rsi_period": 14,
        "ma_period": 20,
        "min_reward_risk": 1.5,
        "max_atr_pct": 5.0,
        "max_gap_pct": 1.5,
        "min_volume_ratio": 1.2,
        "use_rsi_filter": True,
        "use_ma_filter": True,
        "use_macd_filter": True,
        "use_rr_filter": True,
        "use_gap_filter": True,
        "use_volume_filter": True,
        "use_volatility_filter": True,
        "allowed_signals": ["Strong Buy", "Buy", "Bullish"],
        "period_for_history": "6mo",
        "auto_adjust": False,
        "latest_only": True,
        "actionable_only": False,
    }


def rising_prices(length=60, include_open=True):
    dates = pd.date_range(end=pd.Timestamp.now(tz="Asia/Kolkata").date(), periods=length, freq="B")
    close = np.arange(100.0, 100.0 + length)
    values = {
        "High": close + 1,
        "Low": close - 1,
        "Close": close,
        "Volume": np.full(length, 100.0),
    }
    values["Volume"][-1] = 200
    if include_open:
        values["Open"] = close - 0.25
    return pd.DataFrame(values, index=dates)


class NearEntryCoreTests(unittest.TestCase):
    def test_latest_prediction_keeps_fields_from_one_row(self):
        predictions = pd.DataFrame([
            {
                "stock_symbol": "TEST.NS",
                "prediction_date": "2026-11-23",
                "entry_price": 100.0,
                "target_price": 120.0,
                "stop_loss": 90.0,
                "trade_signal": "Buy",
                "probability_success": 65.0,
            },
            {
                "stock_symbol": "TEST.NS",
                "prediction_date": "2026-11-24",
                "entry_price": np.nan,
                "target_price": 125.0,
                "stop_loss": 95.0,
                "trade_signal": "Strong Buy",
                "probability_success": 75.0,
            },
        ])
        connection = Mock()

        with patch("backend.near_entry_core.get_active_stock_symbols", return_value=["TEST.NS"]), \
             patch("backend.near_entry_core.get_connection", return_value=connection), \
             patch("backend.near_entry_core.pd.read_sql", return_value=predictions):
            latest = fetch_active_predictions(latest_only=True)

        self.assertEqual(len(latest), 1)
        self.assertEqual(latest.iloc[0]["prediction_date"], pd.Timestamp("2026-11-24"))
        self.assertTrue(pd.isna(latest.iloc[0]["entry_price"]))
        self.assertEqual(latest.iloc[0]["target_price"], 125.0)

    def test_missing_gap_data_fails_closed(self):
        prices = rising_prices(include_open=False)
        current_price = float(prices["Close"].iloc[-1])
        prediction = {
            "prediction_date": pd.Timestamp.now(tz="Asia/Kolkata").date(),
            "entry_price": current_price / 1.01,
            "target_price": current_price / 1.01 + 23,
            "stop_loss": current_price / 1.01 - 7,
            "trade_signal": "Strong Buy",
        }

        result = evaluate_filters_for_symbol(prediction, prices, near_entry_config())

        self.assertFalse(result["passes"]["gap"])
        self.assertFalse(result["overall"])
        self.assertTrue(any("Gap" in reason for reason in result["failed_filters"]))

    def test_reward_risk_uses_current_price_not_original_entry(self):
        prices = rising_prices()
        prediction = {
            "prediction_date": pd.Timestamp.now(tz="Asia/Kolkata").date(),
            "entry_price": 158.0,
            "target_price": 170.0,
            "stop_loss": 150.0,
            "trade_signal": "Strong Buy",
        }

        result = evaluate_filters_for_symbol(prediction, prices, near_entry_config())

        self.assertAlmostEqual(result["reward_risk"], 11 / 9)
        self.assertFalse(result["passes"]["reward_risk"])

    def test_short_price_history_cannot_be_actionable(self):
        prices = rising_prices(length=30)
        current_price = float(prices["Close"].iloc[-1])
        prediction = {
            "prediction_date": pd.Timestamp.now(tz="Asia/Kolkata").date(),
            "entry_price": current_price / 1.01,
            "target_price": current_price / 1.01 + 23,
            "stop_loss": current_price / 1.01 - 7,
            "trade_signal": "Strong Buy",
        }

        result = evaluate_filters_for_symbol(prediction, prices, near_entry_config())

        self.assertFalse(result["passes"]["history"])
        self.assertFalse(result["overall"])

    def test_timezone_keeps_exchange_calendar_date_and_suffix_fallback(self):
        exchange_date = pd.Timestamp("2026-11-24", tz="Asia/Kolkata")
        price_data = pd.DataFrame(
            {"Close": [100.0]},
            index=pd.DatetimeIndex([exchange_date]),
        )

        with patch(
            "backend.near_entry_core._try_yf_history",
            side_effect=[pd.DataFrame(), price_data],
        ) as history:
            result = fetch_price_df("TEST.NS", try_suffixes=True)

        self.assertEqual(
            [call.args[0] for call in history.call_args_list],
            ["TEST.NS", "TEST.BO"],
        )
        self.assertEqual(result.index[0], pd.Timestamp("2026-11-24"))

    def test_entry_proximity_gate_remains_symmetric_and_two_percent(self):
        prices = rising_prices()
        current_price = float(prices["Close"].iloc[-1])
        predictions = pd.DataFrame([
            {
                "stock_symbol": symbol,
                "prediction_date": pd.Timestamp.now(tz="Asia/Kolkata").date(),
                "entry_price": entry,
                "target_price": entry + 23,
                "stop_loss": entry - 7,
                "trade_signal": "Strong Buy",
                "probability_success": 75.0,
            }
            for symbol, entry in [
                ("ABOVE.NS", current_price / 1.019),
                ("BELOW.NS", current_price / 0.981),
                ("OUTSIDE.NS", current_price / 1.021),
            ]
        ])

        with patch("backend.near_entry_core.fetch_active_predictions", return_value=predictions), \
             patch("backend.near_entry_core.fetch_price_df", return_value=prices):
            rows = run_near_entry_logic(near_entry_config())

        self.assertEqual({row["stock_symbol"] for row in rows}, {"ABOVE.NS", "BELOW.NS"})


if __name__ == "__main__":
    unittest.main()
