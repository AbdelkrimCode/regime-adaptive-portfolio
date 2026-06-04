import pandas as pd
import numpy as np
from data.risk_free import fetch_risk_free
from backtest.metrics import compute_all
from config import load_config as _load_config

TRANSACTION_COST = _load_config()["backtest"]["transaction_cost"]

def apply_transaction_costs(weights: pd.DataFrame, returns: pd.Series) -> pd.Series:
    weight_changes = weights.diff().abs().sum(axis=1)
    costs = weight_changes * TRANSACTION_COST
    return returns - costs

def get_spy_equity(returns: pd.DataFrame) -> pd.Series:
    spy_returns = returns["SPY"]
    equity = (1 + spy_returns).cumprod()
    equity.name = "spy_equity"
    return equity

def get_equal_weight_equity(returns: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    n_assets = len(returns.columns)
    monthly_ends = returns.resample("ME").last().index

    weights = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

    for i, month_end in enumerate(monthly_ends):
        next_month_end = monthly_ends[i + 1] if i + 1 < len(monthly_ends) else returns.index[-1]
        mask = (returns.index > month_end) & (returns.index <= next_month_end)
        weights.loc[mask] = 1.0 / n_assets

    weights = weights.dropna()
    aligned_returns = returns.loc[weights.index]
    port_returns = (weights.values * aligned_returns.values).sum(axis=1)
    port_returns = pd.Series(port_returns, index=weights.index, name="ew_return")
    port_returns = apply_transaction_costs(weights, port_returns)
    equity = (1 + port_returns).cumprod()
    equity.name = "ew_equity"
    return port_returns, equity

def get_sixty_forty_equity(returns: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    monthly_ends = returns.resample("ME").last().index

    weights = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

    for i, month_end in enumerate(monthly_ends):
        next_month_end = monthly_ends[i + 1] if i + 1 < len(monthly_ends) else returns.index[-1]
        mask = (returns.index > month_end) & (returns.index <= next_month_end)
        row = pd.Series({col: 0.0 for col in returns.columns})
        row["SPY"] = 0.6
        row["IEF"] = 0.4
        weights.loc[mask] = row.values

    weights = weights.dropna()
    aligned_returns = returns.loc[weights.index]
    port_returns = (weights.values * aligned_returns.values).sum(axis=1)
    port_returns = pd.Series(port_returns, index=weights.index, name="sixty_forty_return")
    port_returns = apply_transaction_costs(weights, port_returns)
    equity = (1 + port_returns).cumprod()
    equity.name = "sixty_forty_equity"
    return port_returns, equity

def get_momentum_equity(returns: pd.DataFrame, lookback: int = 252, top_n: int = 3) -> tuple[pd.Series, pd.Series]:
    monthly_ends = returns.resample("ME").last().index

    weights = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

    for i, month_end in enumerate(monthly_ends):
        next_month_end = monthly_ends[i + 1] if i + 1 < len(monthly_ends) else returns.index[-1]

        history = returns.loc[:month_end]
        if len(history) < lookback:
            continue

        cumulative = (1 + history.tail(lookback)).prod() - 1
        top_assets = cumulative.nlargest(top_n).index.tolist()

        mask = (returns.index > month_end) & (returns.index <= next_month_end)
        row = pd.Series({col: 0.0 for col in returns.columns})
        for asset in top_assets:
            row[asset] = 1.0 / top_n
        weights.loc[mask] = row.values

    weights = weights.dropna()
    aligned_returns = returns.loc[weights.index]
    port_returns = (weights.values * aligned_returns.values).sum(axis=1)
    port_returns = pd.Series(port_returns, index=weights.index, name="momentum_return")
    port_returns = apply_transaction_costs(weights, port_returns)
    equity = (1 + port_returns).cumprod()
    equity.name = "momentum_equity"
    return port_returns, equity

def get_risk_parity_equity(returns: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    from optimization.risk_parity import risk_parity

    monthly_ends = returns.resample("ME").last().index
    weights = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

    for i, month_end in enumerate(monthly_ends):
        next_month_end = monthly_ends[i + 1] if i + 1 < len(monthly_ends) else returns.index[-1]
        history = returns.loc[:month_end]
        if len(history) < 126:
            continue
        raw = risk_parity(history)
        mask = (returns.index > month_end) & (returns.index <= next_month_end)
        weights.loc[mask] = raw

    weights = weights.dropna()
    aligned_returns = returns.loc[weights.index]
    port_returns = (weights.values * aligned_returns.values).sum(axis=1)
    port_returns = pd.Series(port_returns, index=weights.index, name="risk_parity_return")
    port_returns = apply_transaction_costs(weights, port_returns)
    equity = (1 + port_returns).cumprod()
    equity.name = "risk_parity_equity"
    return port_returns, equity

def run() -> dict:
    from config import load_config
    CFG = load_config()

    returns = pd.read_parquet(CFG["paths"]["returns"])
    backtest = pd.read_parquet(CFG["paths"]["backtest_results"])
    rf = fetch_risk_free()

    common = returns.index.intersection(backtest.index)
    returns = returns.loc[common]

    spy_returns = returns["SPY"]
    spy_equity = (1 + spy_returns).cumprod()

    ew_returns, ew_equity = get_equal_weight_equity(returns)
    sf_returns, sf_equity = get_sixty_forty_equity(returns)
    mom_returns, mom_equity = get_momentum_equity(returns)
    rp_returns, rp_equity = get_risk_parity_equity(returns)

    return {
        "portfolio":    compute_all(backtest["portfolio_return"], backtest["equity"], rf=rf),
        "spy":          compute_all(spy_returns, spy_equity, rf=rf),
        "equal_weight": compute_all(ew_returns, ew_equity, rf=rf),
        "sixty_forty":  compute_all(sf_returns, sf_equity, rf=rf),
        "momentum":     compute_all(mom_returns, mom_equity, rf=rf),
        "risk_parity": compute_all(rp_returns, rp_equity, rf=rf),
    }

if __name__ == "__main__":
    results = run()
    
    print(f"{'Metric':<25} {'Portfolio':>12} {'SPY':>12}")
    print("-" * 50)
    for metric in results["portfolio"]:
        port_val = results["portfolio"][metric]
        spy_val  = results["spy"][metric]
        print(f"{metric:<25} {port_val:>12} {spy_val:>12}")