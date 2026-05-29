import pandas as pd
import numpy as np
from backtest.metrics import compute_all

def get_spy_equity(returns: pd.DataFrame) -> pd.Series:
    spy_returns = returns["SPY"]
    equity = (1 + spy_returns).cumprod()
    equity.name = "spy_equity"
    return equity

def run() -> dict:
    returns = pd.read_parquet("data/processed/returns.parquet")
    backtest = pd.read_parquet("data/backtest_results.parquet")
    
    common = returns.index.intersection(backtest.index)
    spy_returns = returns.loc[common, "SPY"]
    spy_equity = (1 + spy_returns).cumprod()
    
    spy_metrics = compute_all(spy_returns, spy_equity)
    port_metrics = compute_all(backtest["portfolio_return"], backtest["equity"])
    
    return {
        "portfolio": port_metrics,
        "spy":       spy_metrics
    }

if __name__ == "__main__":
    results = run()
    
    print(f"{'Metric':<25} {'Portfolio':>12} {'SPY':>12}")
    print("-" * 50)
    for metric in results["portfolio"]:
        port_val = results["portfolio"][metric]
        spy_val  = results["spy"][metric]
        print(f"{metric:<25} {port_val:>12} {spy_val:>12}")