import pandas as pd
import numpy as np
from data.risk_free import fetch_risk_free
from backtest.metrics import compute_all
from backtest.engine import simulate
from config import load_config

CFG = load_config()
MIN_HISTORY = CFG["hmm"]["min_train_days"] // 2


def month_end_flags(index: pd.DatetimeIndex) -> pd.Series:
    """True on the last trading day of each month in `index`."""
    dates = index.to_series()
    return dates.groupby(index.to_period("M")).transform("max") == dates


def monthly_rebalanced(returns: pd.DataFrame, weight_fn, name: str, cost: float | None = None) -> tuple[pd.Series, pd.Series]:
    """Trade to weight_fn(history) at each month-end close, then hold and drift.

    weight_fn gets returns up to and including the rebalance day and returns a
    weight vector, or None when there is not enough history yet (book stays put).
    """
    rebalance = month_end_flags(returns.index)
    targets = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    for date in returns.index[rebalance.values]:
        w = weight_fn(returns.loc[:date])
        if w is not None:
            targets.loc[date] = w

    result = simulate(targets, returns, cost=cost, rebalance=rebalance)
    return (
        result["portfolio_return"].rename(f"{name}_return"),
        result["equity"].rename(f"{name}_equity"),
    )


def get_equal_weight_equity(returns: pd.DataFrame, cost: float | None = None):
    n = returns.shape[1]
    return monthly_rebalanced(returns, lambda h: np.full(n, 1.0 / n), "ew", cost)


def get_sixty_forty_equity(returns: pd.DataFrame, cost: float | None = None):
    w = pd.Series(0.0, index=returns.columns)
    w["SPY"], w["IEF"] = 0.6, 0.4
    return monthly_rebalanced(returns, lambda h: w.values, "sixty_forty", cost)


def get_momentum_equity(returns: pd.DataFrame, lookback: int = 252, top_n: int = 3, cost: float | None = None):
    def weights(history):
        if len(history) < lookback:
            return None
        cumulative = (1 + history.tail(lookback)).prod() - 1
        w = pd.Series(0.0, index=history.columns)
        w[cumulative.nlargest(top_n).index] = 1.0 / top_n
        return w.values
    return monthly_rebalanced(returns, weights, "momentum", cost)


def get_risk_parity_equity(returns: pd.DataFrame, cost: float | None = None):
    from optimization.risk_parity import risk_parity

    def weights(history):
        if len(history) < MIN_HISTORY:
            return None
        return risk_parity(history)
    return monthly_rebalanced(returns, weights, "risk_parity", cost)


def run() -> dict:
    # Benchmarks are built on the full history (same inputs the portfolio's
    # optimizers see), then every strategy is scored on the portfolio's dates.
    returns = pd.read_parquet(CFG["paths"]["returns"])
    backtest = pd.read_parquet(CFG["paths"]["backtest_results"])
    rf = fetch_risk_free()
    window = backtest.index

    def score(r: pd.Series) -> dict:
        r = r.reindex(window)
        return compute_all(r, (1 + r).cumprod(), rf=rf)

    return {
        "portfolio":    score(backtest["portfolio_return"]),
        "spy":          score(returns["SPY"]),
        "equal_weight": score(get_equal_weight_equity(returns)[0]),
        "sixty_forty":  score(get_sixty_forty_equity(returns)[0]),
        "momentum":     score(get_momentum_equity(returns)[0]),
        "risk_parity":  score(get_risk_parity_equity(returns)[0]),
    }


if __name__ == "__main__":
    results = run()

    print(f"{'Metric':<25} {'Portfolio':>12} {'SPY':>12}")
    print("-" * 50)
    for metric in results["portfolio"]:
        port_val = results["portfolio"][metric]
        spy_val  = results["spy"][metric]
        print(f"{metric:<25} {port_val:>12} {spy_val:>12}")