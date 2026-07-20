import numpy as np
import pandas as pd
from optimization.switcher import compute_weights
from config import load_config

CFG = load_config()

def align_data(weights: pd.DataFrame, returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_dates = weights.index.intersection(returns.index)
    return weights.loc[common_dates], returns.loc[common_dates]

def simulate(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    weights, returns = align_data(weights, returns)

    weights_shifted = weights.shift(1).dropna()
    returns_aligned = returns.loc[weights_shifted.index]

    port_returns = (weights_shifted.values * returns_aligned.values).sum(axis=1)

    weight_changes = weights_shifted.diff().abs().sum(axis=1)
    costs = weight_changes * CFG["backtest"]["transaction_cost"]

    port_returns = port_returns - costs.values
    port_returns = pd.Series(port_returns, index=weights_shifted.index, name="portfolio_return")

    equity = (1 + port_returns).cumprod() * CFG["backtest"]["initial_capital"]
    equity.name = "equity"

    return pd.DataFrame({
        "portfolio_return": port_returns,
        "equity": equity
    })

def run(regimes_df: pd.DataFrame | None = None, save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    if regimes_df is None:
        regimes_df = pd.read_parquet(CFG["paths"]["regimes"])

    returns = pd.read_parquet(CFG["paths"]["returns"])

    weights = compute_weights(regimes_df, returns)

    result = simulate(weights, returns)
    if save:
        result.to_parquet(CFG["paths"]["backtest_results"])

    return result, weights

def run_period(
    start: str,
    end: str,
    regimes_df: pd.DataFrame | None = None,
    returns_df: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if regimes_df is None:
        regimes_df = pd.read_parquet(CFG["paths"]["regimes"])

    if returns_df is None:
        returns_df = pd.read_parquet(CFG["paths"]["returns"])

    regimes_slice = regimes_df.loc[start:end]
    returns_slice = returns_df.loc[start:end]

    weights = compute_weights(regimes_slice, returns_df)
    result = simulate(weights, returns_slice)

    return result, weights

if __name__ == "__main__":
    result = run()
    print(result.head(10))
    print(f"\nFinal equity: {result['equity'].iloc[-1]:.4f}")
    print(f"Total return: {(result['equity'].iloc[-1] - 1) * 100:.2f}%")