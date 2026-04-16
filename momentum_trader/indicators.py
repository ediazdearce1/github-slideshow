import pandas as pd


def add_roc(df: pd.DataFrame, period: int = 12) -> pd.DataFrame:
    """Add Rate of Change indicator to the DataFrame."""
    df["ROC"] = df["Close"].pct_change(periods=period) * 100
    return df


def add_bollinger_bands(
    df: pd.DataFrame, window: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """Add Bollinger Bands (mid, upper, lower) to the DataFrame."""
    df["BB_Mid"] = df["Close"].rolling(window=window).mean()
    rolling_std = df["Close"].rolling(window=window).std()
    df["BB_Upper"] = df["BB_Mid"] + num_std * rolling_std
    df["BB_Lower"] = df["BB_Mid"] - num_std * rolling_std
    return df
