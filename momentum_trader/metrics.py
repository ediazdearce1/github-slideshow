import numpy as np
import pandas as pd


def compute_metrics(
    df: pd.DataFrame, trades: list[dict], risk_free_rate: float = 0.02
) -> dict:
    """Compute performance metrics from backtest results."""
    portfolio = df["Portfolio"]
    initial = portfolio.iloc[0]
    final = portfolio.iloc[-1]

    total_return = (final / initial) - 1
    trading_days = len(df)
    years = trading_days / 252
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # Sharpe ratio
    daily_returns = portfolio.pct_change().dropna()
    daily_rf = risk_free_rate / 252
    excess_returns = daily_returns - daily_rf
    sharpe = (
        (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        if excess_returns.std() > 0
        else 0
    )

    # Max drawdown
    cummax = portfolio.cummax()
    drawdown = (cummax - portfolio) / cummax
    max_drawdown = drawdown.max()

    # Trade statistics
    num_trades = len(trades)
    if num_trades > 0:
        pnls = [t["pnl"] for t in trades]
        returns = [t["return_pct"] for t in trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]
        win_rate = len(winners) / num_trades
        avg_return = np.mean(returns)
        gross_profit = sum(winners) if winners else 0
        gross_loss = abs(sum(losers)) if losers else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    else:
        win_rate = 0
        avg_return = 0
        profit_factor = 0

    return {
        "Total Return": f"{total_return:.2%}",
        "Annualized Return": f"{annualized_return:.2%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Max Drawdown": f"{max_drawdown:.2%}",
        "Number of Trades": num_trades,
        "Win Rate": f"{win_rate:.2%}",
        "Avg Trade Return": f"{avg_return:.2f}%",
        "Profit Factor": f"{profit_factor:.2f}",
    }


def print_metrics(metrics: dict) -> None:
    """Print a formatted metrics summary."""
    print("\n" + "=" * 40)
    print("  BACKTEST PERFORMANCE SUMMARY")
    print("=" * 40)
    for key, value in metrics.items():
        print(f"  {key:<22} {value}")
    print("=" * 40 + "\n")
