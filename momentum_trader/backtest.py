import pandas as pd


def run_backtest(
    df: pd.DataFrame, initial_capital: float = 100_000.0
) -> tuple[pd.DataFrame, list[dict]]:
    """Run a long-only backtest using the Signal column.

    Returns the DataFrame with Portfolio/Cash/Holdings columns and a list of trades.
    """
    cash = initial_capital
    shares = 0
    entry_price = 0.0
    entry_date = None
    trades = []

    portfolio_values = []
    cash_values = []
    holdings_values = []

    for i in range(len(df)):
        price = df["Close"].iloc[i]
        signal = df["Signal"].iloc[i]
        date = df.index[i]

        if signal == 1 and shares == 0:
            # Buy: go all-in
            shares = int(cash // price)
            entry_price = price
            entry_date = date
            cash -= shares * price

        elif signal == -1 and shares > 0:
            # Sell: liquidate
            cash += shares * price
            pnl = (price - entry_price) * shares
            trades.append({
                "entry_date": entry_date,
                "entry_price": entry_price,
                "exit_date": date,
                "exit_price": price,
                "shares": shares,
                "pnl": pnl,
                "return_pct": (price / entry_price - 1) * 100,
            })
            shares = 0

        holdings_val = shares * price
        portfolio_values.append(cash + holdings_val)
        cash_values.append(cash)
        holdings_values.append(holdings_val)

    # Force-close any open position at the last price
    if shares > 0:
        last_price = df["Close"].iloc[-1]
        last_date = df.index[-1]
        cash += shares * last_price
        pnl = (last_price - entry_price) * shares
        trades.append({
            "entry_date": entry_date,
            "entry_price": entry_price,
            "exit_date": last_date,
            "exit_price": last_price,
            "shares": shares,
            "pnl": pnl,
            "return_pct": (last_price / entry_price - 1) * 100,
        })
        portfolio_values[-1] = cash
        cash_values[-1] = cash
        holdings_values[-1] = 0

    df["Portfolio"] = portfolio_values
    df["Cash"] = cash_values
    df["Holdings"] = holdings_values

    return df, trades
