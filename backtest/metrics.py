import numpy as np
import pandas as pd
from config import load_config

CFG = load_config()
TRADING_DAYS = CFG["market"]["trading_days"]
RISK_FREE_RATE = CFG["market"]["risk_free_rate"]

def annualized_return(equity: pd.Series) -> float:
    n_periods = len(equity) - 1
    total_return = equity.iloc[-1] / equity.iloc[0]
    return total_return ** (TRADING_DAYS / n_periods) - 1

def annualized_volatility(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(TRADING_DAYS)

def sharpe_ratio(returns: pd.Series, rf: pd.Series | None = None) -> float:
    if rf is not None:
        rf_aligned = rf.reindex(returns.index).ffill().fillna(0)
        excess = returns - rf_aligned
    else:
        excess = returns

    std = excess.std()
    if std == 0:
        return 0.0
    return excess.mean() / std * np.sqrt(TRADING_DAYS)

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

def compute_all(returns: pd.Series, equity: pd.Series, rf: pd.Series | None = None) -> dict:
    return {
        "annualized_return":     round(annualized_return(equity), 4),
        "annualized_volatility": round(annualized_volatility(returns), 4),
        "sharpe_ratio":          round(sharpe_ratio(returns, rf=rf), 4),
        "max_drawdown":          round(max_drawdown(equity), 4),
        "calmar_ratio":          round(calmar_ratio(equity, returns), 4)
    }

if __name__ == "__main__":
    result = pd.read_parquet("data/backtest_results.parquet")
    metrics = compute_all(result["portfolio_return"], result["equity"])
    for k, v in metrics.items():
        print(f"{k}: {v}")  