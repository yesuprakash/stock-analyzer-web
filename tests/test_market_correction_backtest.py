import unittest
from unittest.mock import patch

import pandas as pd

from backend.market_correction_backtest import BacktestConfig, backtest_symbol


def historical_prices(length=33):
    index = pd.bdate_range("2025-01-01", periods=length)
    prices = pd.DataFrame(
        {
            "Open": 100.0,
            "High": 110.0,
            "Low": 90.0,
            "Close": 100.0,
        },
        index=index,
    )
    signal_index = 30
    prices.iloc[signal_index + 1, prices.columns.get_loc("High")] = 105.0
    prices.iloc[signal_index + 1, prices.columns.get_loc("Low")] = 95.0
    prices.iloc[signal_index + 2, prices.columns.get_loc("High")] = 115.0
    prices.iloc[signal_index + 2, prices.columns.get_loc("Low")] = 85.0
    return prices, signal_index


def permissive_config(min_reward_risk=0.1):
    return BacktestConfig(
        correction_window=5,
        min_correction=-100,
        max_correction=0,
        rsi_min=40,
        max_atr_pct=5.5,
        min_reward_risk=min_reward_risk,
        holding_period=15,
    )


class MarketCorrectionBacktestTests(unittest.TestCase):
    def run_with_indicators(self, prices, signal_date, config):
        indexes = prices.index
        with patch(
            "backend.market_correction_backtest.calc_rsi",
            return_value=pd.Series(60.0, index=indexes),
        ), patch(
            "backend.market_correction_backtest.calc_macd_hist",
            return_value=pd.Series(1.0, index=indexes),
        ), patch(
            "backend.market_correction_backtest.calc_atr",
            return_value=pd.Series(2.0, index=indexes),
        ):
            return backtest_symbol(
                "TEST.NS", prices, signal_date, signal_date, config
            )

    def test_enters_next_open_and_stop_wins_same_bar_tie(self):
        prices, signal_index = historical_prices()
        signal_date = prices.index[signal_index]
        trades, censored = self.run_with_indicators(
            prices, signal_date, permissive_config()
        )

        self.assertEqual(censored, 0)
        self.assertEqual(len(trades), 1)
        trade = trades[0]
        self.assertEqual(trade["signal_date"], signal_date.strftime("%Y-%m-%d"))
        self.assertEqual(
            trade["entry_date"], prices.index[signal_index + 1].strftime("%Y-%m-%d")
        )
        self.assertEqual(trade["target_price"], 110.0)
        self.assertEqual(trade["exit_reason"], "stop_and_target_same_bar_stop_first")
        self.assertAlmostEqual(trade["exit_price"], 88.2)

    def test_current_close_reward_risk_threshold_filters_candidate(self):
        prices, signal_index = historical_prices()
        signal_date = prices.index[signal_index]
        trades, _ = self.run_with_indicators(
            prices, signal_date, permissive_config(min_reward_risk=0.9)
        )

        self.assertEqual(trades, [])


if __name__ == "__main__":
    unittest.main()
