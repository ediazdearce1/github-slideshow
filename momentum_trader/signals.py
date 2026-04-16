import numpy as np
import pandas as pd


def generate_signals(
    df: pd.DataFrame, roc_threshold: float = 2.0, lookback: int = 10
) -> pd.DataFrame:
    """Generate buy/sell signals based on ROC momentum + Bollinger Band confirmation.

    Strategy logic:
    - BUY: Price recently bounced off lower BB (touched it within lookback bars)
      AND ROC has turned positive (momentum confirming the bounce).
    - SELL: Price recently hit upper BB (touched it within lookback bars)
      AND ROC has turned negative (momentum fading at resistance).

    This captures the classic "mean-reversion entry confirmed by momentum" pattern.
    """
    # Recent BB touches (within lookback window)
    touched_lower = (
        (df["Close"] <= df["BB_Lower"])
        .rolling(window=lookback, min_periods=1)
        .max()
        .astype(bool)
    )
    touched_upper = (
        (df["Close"] >= df["BB_Upper"])
        .rolling(window=lookback, min_periods=1)
        .max()
        .astype(bool)
    )

    # ROC momentum confirmation
    roc_positive = df["ROC"] > roc_threshold
    roc_negative = df["ROC"] < -roc_threshold

    # Raw signals: bounce off lower BB confirmed by positive momentum
    raw_buy = touched_lower & roc_positive
    # Fade at upper BB confirmed by negative momentum
    raw_sell = touched_upper & roc_negative

    df["Signal"] = 0
    df.loc[raw_buy, "Signal"] = 1
    df.loc[raw_sell, "Signal"] = -1

    # Remove redundant signals: only keep first buy after a sell and vice versa
    signals = df["Signal"].values
    filtered = np.zeros(len(signals), dtype=int)
    position = 0  # 0=flat, 1=long
    for i in range(len(signals)):
        if signals[i] == 1 and position == 0:
            filtered[i] = 1
            position = 1
        elif signals[i] == -1 and position == 1:
            filtered[i] = -1
            position = 0

    df["Signal"] = filtered

    # Position state: forward-fill
    pos = pd.Series(np.nan, index=df.index)
    pos.iloc[0] = 0
    pos[df["Signal"] == 1] = 1
    pos[df["Signal"] == -1] = 0
    df["Position"] = pos.ffill().astype(int)

    return df
