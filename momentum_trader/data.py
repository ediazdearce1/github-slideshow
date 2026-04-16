import numpy as np
import pandas as pd


def fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download historical OHLCV data from Yahoo Finance.

    Falls back to synthetic data if yfinance download fails (e.g., no network).
    """
    try:
        import yfinance as yf

        df = yf.download(ticker, start=start, end=end, progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            if not df.empty:
                return df
    except Exception:
        pass

    print(f"  [!] Live data unavailable — using synthetic data for {ticker}")
    return _generate_synthetic(start, end)


def _generate_synthetic(start: str, end: str) -> pd.DataFrame:
    """Generate realistic synthetic OHLCV data with trends and mean-reversion."""
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)
    rng = np.random.default_rng(42)

    # Geometric Brownian Motion with regime changes
    price = 150.0
    prices = []
    for i in range(n):
        # Add trending and mean-reverting regimes
        drift = 0.0003 * np.sin(2 * np.pi * i / 252)  # seasonal drift
        shock = rng.normal(0, 0.018)
        price *= 1 + drift + shock
        prices.append(price)

    close = np.array(prices)
    high = close * (1 + rng.uniform(0.001, 0.02, n))
    low = close * (1 - rng.uniform(0.001, 0.02, n))
    open_ = low + rng.uniform(0.3, 0.7, n) * (high - low)
    volume = rng.integers(5_000_000, 50_000_000, n)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
