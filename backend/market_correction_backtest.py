#!/usr/bin/env python

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.market_utils import calc_atr, calc_macd_hist, calc_rsi, normalize_price_df
from backend.stock_master import get_active_stock_symbols


@dataclass(frozen=True)
class BacktestConfig:
    correction_window: int = 30
    min_correction: float = -20.0
    max_correction: float = -8.0
    rsi_min: float = 40.0
    max_atr_pct: float = 5.5
    min_reward_risk: float = 1.8
    holding_period: int = 15
    skip_strategy_thresholds: bool = False


TRADE_FIELDS = [
    "symbol",
    "signal_date",
    "entry_date",
    "entry_price",
    "target_price",
    "stop_loss",
    "signal_close",
    "correction_pct",
    "rsi",
    "atr_pct",
    "reward_risk_at_signal",
    "exit_date",
    "exit_price",
    "exit_reason",
    "holding_sessions",
    "gross_return_pct",
]


def _finite(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _validate_config(config):
    if config.correction_window not in (5, 10, 15, 30, 45, 60):
        raise ValueError("Correction window must be 5, 10, 15, 30, 45, or 60 sessions.")
    if config.min_correction > config.max_correction:
        raise ValueError("Minimum correction must not exceed maximum correction.")
    if not 0 <= config.rsi_min <= 100:
        raise ValueError("Minimum RSI must be between 0 and 100.")
    if config.max_atr_pct < 0 or config.min_reward_risk < 0:
        raise ValueError("ATR and reward/risk thresholds must be non-negative.")
    if config.holding_period < 1:
        raise ValueError("Holding period must be at least one session.")


def _ohlc_data(frame):
    if frame is None or frame.empty or not {"Open", "High", "Low", "Close"}.issubset(frame.columns):
        return pd.DataFrame()
    data = frame[["Open", "High", "Low", "Close"]].apply(pd.to_numeric, errors="coerce")
    data.index = pd.to_datetime(data.index).normalize()
    return data.dropna().sort_index()


def _exit_trade(data, entry_index, target, stop, holding_period):
    final_index = entry_index + holding_period - 1
    available_final_index = min(final_index, len(data) - 1)

    for index in range(entry_index, available_final_index + 1):
        bar = data.iloc[index]
        if index > entry_index and bar["Open"] <= stop:
            return index, float(bar["Open"]), "stop_gap"
        if index > entry_index and bar["Open"] >= target:
            return index, float(bar["Open"]), "target_gap"

        stop_hit = bar["Low"] <= stop
        target_hit = bar["High"] >= target
        if stop_hit and target_hit:
            return index, stop, "stop_and_target_same_bar_stop_first"
        if stop_hit:
            return index, stop, "stop"
        if target_hit:
            return index, target, "target"

    if final_index >= len(data):
        return None
    return final_index, float(data.iloc[final_index]["Close"]), "time_exit"


def backtest_symbol(symbol, prices, start_date, end_date, config):
    _validate_config(config)
    data = _ohlc_data(prices)
    if len(data) < max(20, config.correction_window + 1):
        return [], 0

    close = data["Close"]
    rsi = calc_rsi(close)
    macd_hist = calc_macd_hist(close)
    atr = calc_atr(data)
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    trades = []
    censored = 0
    signal_index = max(19, config.correction_window - 1)

    while signal_index < len(data) - 1:
        signal_date = data.index[signal_index]
        if signal_date < start:
            signal_index += 1
            continue
        if signal_date > end:
            break

        current_close = _finite(close.iloc[signal_index])
        rsi_value = _finite(rsi.iloc[signal_index])
        macd_value = _finite(macd_hist.iloc[signal_index])
        atr_value = _finite(atr.iloc[signal_index])
        if current_close is None or current_close <= 0:
            signal_index += 1
            continue

        correction_high = _finite(
            close.iloc[signal_index - config.correction_window + 1:signal_index + 1].max()
        )
        correction = (
            (current_close - correction_high) / correction_high * 100
            if correction_high is not None and correction_high > 0 else None
        )
        level_start = signal_date - pd.Timedelta(days=90)
        level_data = data.loc[(data.index >= level_start) & (data.index <= signal_date)]
        recent_high = _finite(level_data["High"].max())
        recent_low = _finite(level_data["Low"].min())
        target = recent_high
        stop = recent_low * 0.98 if recent_low is not None else None

        reward_risk = None
        if target is not None and stop is not None:
            reward = target - current_close
            risk = current_close - stop
            if reward > 0 and risk > 0:
                reward_risk = reward / risk

        atr_pct = atr_value / current_close * 100 if atr_value is not None else None
        if not config.skip_strategy_thresholds:
            if (
                correction is None
                or not config.min_correction <= correction <= config.max_correction
                or rsi_value is None
                or rsi_value < config.rsi_min
                or macd_value is None
                or macd_value < 0
                or atr_pct is None
                or atr_pct > config.max_atr_pct
                or reward_risk is None
                or reward_risk < config.min_reward_risk
            ):
                signal_index += 1
                continue

        if target is None or stop is None or target <= stop:
            signal_index += 1
            continue

        entry_index = signal_index + 1
        entry_price = _finite(data.iloc[entry_index]["Open"])
        if (
            entry_price is None
            or entry_price <= stop
            or entry_price >= target
        ):
            signal_index += 1
            continue

        outcome = _exit_trade(
            data, entry_index, target, stop, config.holding_period
        )
        if outcome is None:
            censored += 1
            signal_index += 1
            continue

        exit_index, exit_price, exit_reason = outcome
        gross_return = (exit_price - entry_price) / entry_price * 100
        trades.append({
            "symbol": symbol,
            "signal_date": signal_date.strftime("%Y-%m-%d"),
            "entry_date": data.index[entry_index].strftime("%Y-%m-%d"),
            "entry_price": round(entry_price, 4),
            "target_price": round(target, 4),
            "stop_loss": round(stop, 4),
            "signal_close": round(current_close, 4),
            "correction_pct": round(correction, 4) if correction is not None else None,
            "rsi": round(rsi_value, 4) if rsi_value is not None else None,
            "atr_pct": round(atr_pct, 4) if atr_pct is not None else None,
            "reward_risk_at_signal": (
                round(reward_risk, 4) if reward_risk is not None else None
            ),
            "exit_date": data.index[exit_index].strftime("%Y-%m-%d"),
            "exit_price": round(exit_price, 4),
            "exit_reason": exit_reason,
            "holding_sessions": exit_index - entry_index + 1,
            "gross_return_pct": round(gross_return, 4),
        })
        signal_index = max(signal_index + 1, exit_index)

    return trades, censored


def run_backtest(prices_by_symbol, start_date, end_date, config):
    _validate_config(config)
    if start_date > end_date:
        raise ValueError("Start date must be on or before end date.")

    trades = []
    censored = 0
    for symbol, prices in prices_by_symbol.items():
        symbol_trades, symbol_censored = backtest_symbol(
            symbol, prices, start_date, end_date, config
        )
        trades.extend(symbol_trades)
        censored += symbol_censored
    return trades, censored


def _parse_cost_rates(value):
    try:
        rates = [float(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Cost rates must be comma-separated numbers.") from error
    if not rates or any(not math.isfinite(rate) or rate < 0 for rate in rates):
        raise argparse.ArgumentTypeError("Cost rates must be finite and non-negative.")
    return rates


def _report(trades, cost_rates, censored, start_date, end_date):
    print(f"Backtest dates: {start_date} through {end_date}")
    print("Selected current active symbols only; historical-universe survivorship bias remains.")
    print(f"Completed trades: {len(trades)} | Censored at data boundary: {censored}")
    print("Round-trip cost (%) | Trades | Win rate (%) | Average net return (%) | Profit factor")
    for cost in cost_rates:
        returns = [trade["gross_return_pct"] - cost for trade in trades]
        wins = [value for value in returns if value > 0]
        losses = [value for value in returns if value < 0]
        profit_factor = (
            f"{sum(wins) / abs(sum(losses)):.2f}"
            if losses else ("inf" if wins else "n/a")
        )
        win_rate = (len(wins) / len(returns) * 100) if returns else 0.0
        average = (sum(returns) / len(returns)) if returns else 0.0
        print(
            f"{cost:18.3f} | {len(returns):6d} | {win_rate:12.2f} | "
            f"{average:22.3f} | {profit_factor}"
        )


def _download_history(symbol, start, end):
    raw = yf.download(
        symbol,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False,
    )
    return normalize_price_df(raw)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Backtest Market Correction filters using historical daily OHLC data."
    )
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    parser.add_argument("--end", required=True, type=date.fromisoformat)
    parser.add_argument("--output", required=True, type=Path, help="Path for the trade CSV.")
    parser.add_argument("--symbols", help="Comma-separated symbols; defaults to active stock master.")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--min-correction", type=float, default=-20)
    parser.add_argument("--max-correction", type=float, default=-8)
    parser.add_argument("--rsi-min", type=float, default=40)
    parser.add_argument("--max-atr", type=float, default=5.5)
    parser.add_argument("--min-rr", type=float, default=1.8)
    parser.add_argument("--holding-period", type=int, default=15)
    parser.add_argument("--skip-strategy-thresholds", action="store_true")
    parser.add_argument(
        "--cost-rates",
        type=_parse_cost_rates,
        default=[0.0, 0.1, 0.2, 0.3],
        help="Round-trip cost percentages to compare (default: 0,0.1,0.2,0.3).",
    )
    args = parser.parse_args(argv)

    if args.start > args.end:
        parser.error("--start must be on or before --end.")
    config = BacktestConfig(
        correction_window=args.window,
        min_correction=args.min_correction,
        max_correction=args.max_correction,
        rsi_min=args.rsi_min,
        max_atr_pct=args.max_atr,
        min_reward_risk=args.min_rr,
        holding_period=args.holding_period,
        skip_strategy_thresholds=args.skip_strategy_thresholds,
    )
    try:
        _validate_config(config)
    except ValueError as error:
        parser.error(str(error))

    symbols = (
        [symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()]
        if args.symbols else get_active_stock_symbols()
    )
    if not symbols:
        parser.error("No symbols were provided or found in the active stock master.")

    fetch_start = args.start - timedelta(days=120)
    fetch_end = args.end + timedelta(days=config.holding_period * 2 + 11)
    histories = {}
    failures = []
    for symbol in symbols:
        try:
            history = _download_history(symbol, fetch_start, fetch_end)
            if history.empty:
                failures.append(f"{symbol}: no historical OHLC data returned")
            else:
                histories[symbol] = history
        except Exception as error:
            failures.append(f"{symbol}: {error}")
    for failure in failures:
        print(f"Warning: {failure}", file=sys.stderr)
    if not histories:
        raise RuntimeError("No symbols returned usable historical data.")

    trades, censored = run_backtest(
        histories, args.start, args.end, config
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=TRADE_FIELDS)
        writer.writeheader()
        writer.writerows(trades)
    _report(trades, args.cost_rates, censored, args.start, args.end)
    print(f"Trade details written to: {args.output}")


if __name__ == "__main__":
    main()
