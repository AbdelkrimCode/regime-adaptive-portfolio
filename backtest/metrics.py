import numpy as np
import pandas as pd

RISK_FREE_RATE = 0.04
TRADING_DAYS = 252

def annualized_return(equity: pd.Series) -> float:
    n_days = len(equity)
    total_return = equity.iloc[-1] / equity.iloc[0]
    return total_return ** (TRADING_DAYS / n_days) - 1

def annualized_volatility(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(TRADING_DAYS)

def sharpe_ratio(returns: pd.Series) -> float:
    ann_ret = annualized_return((1 + returns).cumprod())
    ann_vol = annualized_volatility(returns)
    return (ann_ret - RISK_FREE_RATE) / ann_vol

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    return drawdown.min()  # most negative value

def calmar_ratio(equity: pd.Series, returns: pd.Series) -> float:
    ann_ret = annualized_return(equity)
    mdd = abs(max_drawdown(equity))
    return ann_ret / mdd if mdd != 0 else np.nan

def average_turnover(weights: pd.DataFrame) -> float:
    daily_turnover = weights.diff().abs().sum(axis=1)
    return round(daily_turnover.mean(), 4)

def compute_all(returns: pd.Series, equity: pd.Series) -> dict:
    return {
        "annualized_return":     round(annualized_return(equity), 4),
        "annualized_volatility": round(annualized_volatility(returns), 4),
        "sharpe_ratio":          round(sharpe_ratio(returns), 4),
        "max_drawdown":          round(max_drawdown(equity), 4),
        "calmar_ratio":          round(calmar_ratio(equity, returns), 4)
    }

if __name__ == "__main__":
    result = pd.read_parquet("data/backtest_results.parquet")
    metrics = compute_all(result["portfolio_return"], result["equity"])
    for k, v in metrics.items():
        print(f"{k}: {v}")  