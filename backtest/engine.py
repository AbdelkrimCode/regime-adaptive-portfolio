import numpy as np
import pandas as pd
from optimization.switcher import compute_weights
from backtest.metrics import compute_turnover
from config import load_config

CFG = load_config()

def align_data(weights: pd.DataFrame, returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_dates = weights.index.intersection(returns.index)
    return weights.loc[common_dates], returns.loc[common_dates]

def simulate(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    cost: float | None = None,
    rebalance: pd.Series | None = None,
) -> pd.DataFrame:
    """Simulate a book that trades to target weights and drifts with prices in between.

    weights.loc[t] is the target set by a trade at the close of day t; it earns the
    return of day t+1, which is also charged that trade's cost (cost * turnover).
    Turnover is measured from the drifted weights, not the previous target.
    `rebalance` (bool Series) limits trading to those days; default is every day.
    """
    if cost is None:
        cost = CFG["backtest"]["transaction_cost"]
    weights, returns = align_data(weights, returns)
    targets = weights.values.astype(float)
    rets = returns.values.astype(float)
    if rebalance is None:
        trade_days = np.ones(len(weights), dtype=bool)
    else:
        trade_days = rebalance.reindex(weights.index, fill_value=False).values.astype(bool)

    held = np.zeros(targets.shape[1])
    pending = 0.0
    port, turnover = [], []

    for t in range(len(weights)):
        if t > 0:
            gross = held @ rets[t]
            port.append((1 + gross) * (1 - cost * pending) - 1)
            turnover.append(pending)
            held = held * (1 + rets[t]) / (1 + gross)
        pending = 0.0
        if trade_days[t] and np.isfinite(targets[t]).all():
            pending = np.abs(targets[t] - held).sum()
            held = targets[t].copy()

    idx = weights.index[1:]
    port = pd.Series(port, index=idx, name="portfolio_return")
    equity = (1 + port).cumprod() * CFG["backtest"]["initial_capital"]
    equity.name = "equity"

    return pd.DataFrame({
        "portfolio_return": port,
        "turnover": pd.Series(turnover, index=idx),
        "equity": equity,
    })

def run(
    regimes_df: pd.DataFrame | None = None,
    returns_df: pd.DataFrame | None = None,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if regimes_df is None:
        regimes_df = pd.read_parquet(CFG["paths"]["regimes"])

    if returns_df is None:
        returns_df = pd.read_parquet(CFG["paths"]["returns"])

    weights = compute_weights(regimes_df, returns_df)

    result = simulate(weights, returns_df)
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