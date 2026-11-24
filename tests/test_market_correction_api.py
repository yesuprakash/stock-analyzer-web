import contextlib
import io
import json
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from backend import market_correction_api as api
from backend.market_utils import normalize_price_df


def prediction_row(prediction_date=None):
    return {
        "stock_symbol": "TEST.NS",
        "prediction_date": prediction_date or pd.Timestamp.now(tz="Asia/Kolkata"),
        "entry_price": 85.0,
        "target_price": 110.0,
        "stop_loss": 80.0,
        "trade_signal": "Buy",
        "probability_success": 75.0,
    }


def price_history(days_ago=0):
    end = (
        pd.Timestamp.now(tz="Asia/Kolkata").normalize().tz_localize(None)
        - pd.Timedelta(days=days_ago)
    )
    index = pd.date_range(end=end, periods=5, freq="D")
    return pd.DataFrame(
        {
            "High": [101.0, 102.0, 103.0, 104.0, 111.0],
            "Low": [99.0, 99.0, 99.0, 99.0, 89.0],
            "Close": [100.0, 100.0, 100.0, 100.0, 90.0],
        },
        index=index,
    )


class MarketCorrectionApiTests(unittest.TestCase):
    def test_normalizes_yfinance_multiindex_and_keeps_exchange_date(self):
        exchange_date = pd.Timestamp("2025-01-02", tz="Asia/Kolkata")
        columns = pd.MultiIndex.from_product(
            [["Close", "High"], ["TEST.NS"]]
        )
        raw = pd.DataFrame([[100.0, 105.0]], index=[exchange_date], columns=columns)

        normalized = normalize_price_df(raw)

        self.assertEqual(list(normalized.columns), ["Close", "High"])
        self.assertEqual(normalized.index[0], pd.Timestamp("2025-01-02"))

    def run_api(self, prices=None, skip_filters=0, min_rr=1.8, max_age=5):
        prediction = prediction_row()
        history = prices if prices is not None else price_history()
        index = history.index
        stdout = io.StringIO()
        with patch.object(
            api.sys,
            "argv",
            [
                api.__file__, "5", "-20", "-8", "40", "5.5",
                str(min_rr), str(skip_filters), str(max_age),
            ],
        ), patch.object(api, "fetch_latest_predictions", return_value=pd.DataFrame([prediction])), \
             patch.object(api, "fetch_price_df", return_value=history), \
             patch.object(api, "calc_rsi", return_value=pd.Series(60.0, index=index)), \
             patch.object(api, "calc_macd_hist", return_value=pd.Series(1.0, index=index)), \
             patch.object(api, "calc_atr", return_value=pd.Series(2.0, index=index)), \
             contextlib.redirect_stdout(stdout):
            api.main()
        return json.loads(stdout.getvalue())

    def test_latest_prediction_keeps_fields_from_one_complete_row(self):
        predictions = pd.DataFrame([
            prediction_row("2026-01-01"),
            {**prediction_row("2026-01-02"), "entry_price": np.nan},
        ])
        connection = Mock()

        with patch.object(api, "get_active_stock_symbols", return_value=["TEST.NS"]), \
             patch.object(api, "get_connection", return_value=connection), \
             patch.object(api.pd, "read_sql", return_value=predictions):
            latest = api.fetch_latest_predictions()

        self.assertEqual(len(latest), 1)
        self.assertEqual(latest.iloc[0]["prediction_date"], pd.Timestamp("2026-01-02"))
        self.assertTrue(pd.isna(latest.iloc[0]["entry_price"]))
        self.assertEqual(latest.iloc[0]["target_price"], 110.0)

    def test_reward_risk_filter_uses_current_price(self):
        rows = self.run_api(min_rr=3.0)

        self.assertEqual(rows, [])
        rows = self.run_api(min_rr=1.8)
        self.assertEqual(rows[0]["rr"], 2.0)
        self.assertEqual(rows[0]["corr_close"], -10.0)

    def test_skip_strategy_thresholds_still_requires_fresh_data(self):
        stale_rows = self.run_api(prices=price_history(days_ago=10), skip_filters=1)
        self.assertEqual(stale_rows, [])

    def test_skip_strategy_thresholds_allows_missing_indicators(self):
        prediction = prediction_row()
        history = price_history()
        index = history.index
        stdout = io.StringIO()
        with patch.object(
            api.sys,
            "argv",
            [api.__file__, "5", "-20", "-8", "40", "5.5", "1.8", "1", "5"],
        ), patch.object(api, "fetch_latest_predictions", return_value=pd.DataFrame([prediction])), \
             patch.object(api, "fetch_price_df", return_value=history), \
             patch.object(api, "calc_rsi", return_value=pd.Series(np.nan, index=index)), \
             patch.object(api, "calc_macd_hist", return_value=pd.Series(np.nan, index=index)), \
             patch.object(api, "calc_atr", return_value=pd.Series(np.nan, index=index)), \
             contextlib.redirect_stdout(stdout):
            api.main()

        rows = json.loads(stdout.getvalue())
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["rsi"])
        self.assertIsNone(rows[0]["atr_pct"])


if __name__ == "__main__":
    unittest.main()
