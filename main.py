"""Momentum Trading Algorithm — Backtest CLI

Uses Rate of Change (ROC) + Bollinger Bands to generate buy/sell signals,
then backtests the strategy on historical stock data.

Usage:
    python main.py --ticker AAPL --start 2020-01-01 --end 2024-12-31
"""

import argparse

from momentum_trader.data import fetch
from momentum_trader.indicators import add_bollinger_bands, add_roc
from momentum_trader.signals import generate_signals
from momentum_trader.backtest import run_backtest
from momentum_trader.metrics import compute_metrics, print_metrics
from momentum_trader.visualize import plot_results


def main():
    parser = argparse.ArgumentParser(
        description="Momentum trading backtest using ROC + Bollinger Bands"
    )
    parser.add_argument("--ticker", default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=100_000, help="Initial capital")
    parser.add_argument("--roc-period", type=int, default=12, help="ROC lookback period")
    parser.add_argument("--roc-threshold", type=float, default=2.0, help="ROC crossover threshold")
    parser.add_argument("--bb-window", type=int, default=20, help="Bollinger Band window")
    parser.add_argument("--bb-std", type=float, default=2.0, help="Bollinger Band std multiplier")
    parser.add_argument("--save", type=str, default=None, help="Save chart to file path")
    args = parser.parse_args()

    print(f"Fetching data for {args.ticker} ({args.start} to {args.end})...")
    df = fetch(args.ticker, args.start, args.end)
    print(f"Got {len(df)} trading days of data.")

    # Compute indicators
    df = add_roc(df, period=args.roc_period)
    df = add_bollinger_bands(df, window=args.bb_window, num_std=args.bb_std)

    # Generate signals
    df = generate_signals(df, roc_threshold=args.roc_threshold)

    buy_count = (df["Signal"] == 1).sum()
    sell_count = (df["Signal"] == -1).sum()
    print(f"Signals generated: {buy_count} buys, {sell_count} sells")

    # Run backtest
    df, trades = run_backtest(df, initial_capital=args.capital)

    # Print metrics
    metrics = compute_metrics(df, trades)
    print_metrics(metrics)

    # Plot results
    plot_results(df, args.ticker, save_path=args.save)


if __name__ == "__main__":
    main()
