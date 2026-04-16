import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_results(df: pd.DataFrame, ticker: str, save_path: str | None = None) -> None:
    """Plot price with Bollinger Bands, ROC, and equity curve."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Panel 1: Price + Bollinger Bands + Signals
    ax1.plot(df.index, df["Close"], label="Close", color="black", linewidth=0.8)
    ax1.plot(df.index, df["BB_Mid"], label="BB Mid", color="blue", linewidth=0.5, alpha=0.7)
    ax1.fill_between(
        df.index, df["BB_Upper"], df["BB_Lower"], alpha=0.15, color="blue", label="BB Band"
    )

    buys = df[df["Signal"] == 1]
    sells = df[df["Signal"] == -1]
    ax1.scatter(buys.index, buys["Close"], marker="^", color="green", s=100, label="Buy", zorder=5)
    ax1.scatter(sells.index, sells["Close"], marker="v", color="red", s=100, label="Sell", zorder=5)

    ax1.set_title(f"{ticker} — Price & Bollinger Bands")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_ylabel("Price ($)")

    # Panel 2: ROC
    ax2.plot(df.index, df["ROC"], color="purple", linewidth=0.8)
    ax2.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax2.axhline(y=2, color="green", linestyle="--", linewidth=0.5, alpha=0.7)
    ax2.axhline(y=-2, color="red", linestyle="--", linewidth=0.5, alpha=0.7)
    ax2.set_title("Rate of Change (ROC)")
    ax2.set_ylabel("ROC (%)")

    # Panel 3: Equity Curve
    ax3.plot(df.index, df["Portfolio"], color="darkgreen", linewidth=1)
    ax3.set_title("Equity Curve")
    ax3.set_ylabel("Portfolio Value ($)")
    ax3.set_xlabel("Date")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Chart saved to {save_path}")
    else:
        plt.savefig("backtest_results.png", dpi=150, bbox_inches="tight")
        print("Chart saved to backtest_results.png")

    plt.close()
