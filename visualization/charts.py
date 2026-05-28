import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REGIME_COLORS = {
    "Bull":     "#2ecc71",
    "Bear":     "#e74c3c",
    "Sideways": "#f39c12"
}

FIG_SIZE = (14, 6)

def plot_equity_curves(backtest: pd.DataFrame, returns: pd.DataFrame, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    common = backtest.index.intersection(returns.index)
    spy_returns = returns.loc[common, "SPY"]
    spy_equity = (1 + spy_returns).cumprod()
    
    ax.plot(backtest["equity"], label="Regime Portfolio", color="#3498db", linewidth=1.5)
    ax.plot(spy_equity, label="SPY Buy & Hold", color="#e74c3c", linewidth=1.5, alpha=0.7)
    ax.set_title("Portfolio vs SPY — Equity Curve")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax

def plot_drawdown(backtest: pd.DataFrame, returns: pd.DataFrame, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    equity = backtest["equity"]
    port_dd = (equity - equity.cummax()) / equity.cummax()
    
    common = backtest.index.intersection(returns.index)
    spy_returns = returns.loc[common, "SPY"]
    spy_equity = (1 + spy_returns).cumprod()
    spy_dd = (spy_equity - spy_equity.cummax()) / spy_equity.cummax()
    
    ax.fill_between(port_dd.index, port_dd, 0, alpha=0.4, color="#3498db", label="Portfolio")
    ax.fill_between(spy_dd.index, spy_dd, 0, alpha=0.4, color="#e74c3c", label="SPY")
    ax.set_title("Drawdown Comparison")
    ax.set_ylabel("Drawdown")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax

def plot_regime_overlay(backtest: pd.DataFrame, regimes: pd.Series, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    ax.plot(backtest["equity"], color="#2c3e50", linewidth=1.5, zorder=3)
    
    common = backtest.index.intersection(regimes.index)
    regimes_aligned = regimes.loc[common]
    
    prev_date = regimes_aligned.index[0]
    prev_regime = regimes_aligned.iloc[0]
    
    for date, regime in regimes_aligned.items():
        if regime != prev_regime:
            ax.axvspan(prev_date, date,
                      alpha=0.15,
                      color=REGIME_COLORS[prev_regime])
            prev_date = date
            prev_regime = regime
    
    ax.axvspan(prev_date, regimes_aligned.index[-1],
              alpha=0.15,
              color=REGIME_COLORS[prev_regime])
    
    patches = [mpatches.Patch(color=v, alpha=0.4, label=k)
               for k, v in REGIME_COLORS.items()]
    ax.legend(handles=patches)
    ax.set_title("Equity Curve with Regime Overlay")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.3)
    return ax

def run():
    backtest = pd.read_parquet("data/backtest_results.parquet")
    returns  = pd.read_parquet("data/processed/returns.parquet")
    regimes  = pd.read_parquet("data/regimes.parquet")["regime"]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 16))
    plot_equity_curves(backtest, returns, ax=axes[0])
    plot_drawdown(backtest, returns, ax=axes[1])
    plot_regime_overlay(backtest, regimes, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig("data/charts.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved to data/charts.png")

if __name__ == "__main__":
    run()