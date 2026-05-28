import numpy as np
import pandas as pd
from optimization.switcher import compute_weights

INITIAL_CAPITAL = 1.0

def align_data(weights: pd.DataFrame, returns: pd.DataFrame):
    common_dates = weights.index.intersection(returns.index)
    return weights.loc[common_dates], returns.loc[common_dates]

def simulate(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    weights, returns = align_data(weights, returns)
    
    weights_shifted = weights.shift(1).dropna()
    returns_aligned = returns.loc[weights_shifted.index]
    
    port_returns = (weights_shifted.values * returns_aligned.values).sum(axis=1)
    port_returns = pd.Series(port_returns, index=weights_shifted.index, name="portfolio_return")
    
    equity = (1 + port_returns).cumprod() * INITIAL_CAPITAL
    equity.name = "equity"
    
    result = pd.DataFrame({
        "portfolio_return": port_returns,
        "equity": equity
    })
    
    return result

def run() -> pd.DataFrame:
    regimes_df = pd.read_parquet("data/regimes.parquet")
    returns = pd.read_parquet("data/processed/returns.parquet")
    
    regimes = regimes_df["regime"]
    weights = compute_weights(regimes, returns)
    
    result = simulate(weights, returns)
    result.to_parquet("data/backtest_results.parquet")
    
    return result

if __name__ == "__main__":
    result = run()
    print(result.head(10))
    print(f"\nFinal equity: {result['equity'].iloc[-1]:.4f}")
    print(f"Total return: {(result['equity'].iloc[-1] - 1) * 100:.2f}%")