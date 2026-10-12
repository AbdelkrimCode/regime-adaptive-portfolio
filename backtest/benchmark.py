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
    """Equal weight across all assets, rebalanced monthly."""
    n = returns.shape[1]
    return monthly_rebalanced(returns, lambda h: np.full(n, 1.0 / n), "ew", cost)


def get_sixty_forty_equity(returns: pd.DataFrame, cost: float | None = None):
    """60% SPY / 40% IEF, rebalanced monthly."""
    w = pd.Series(0.0, index=returns.columns)
    w["SPY"], w["IEF"] = 0.6, 0.4
    return monthly_rebalanced(returns, lambda h: w.values, "sixty_forty", cost)


def get_momentum_equity(returns: pd.DataFrame, lookback: int = 252, top_n: int = 3, cost: float | None = None):
    """Equal weight in the top_n assets by trailing `lookback`-day return, rebalanced monthly."""
    def weights(history):
        if len(history) < lookback:
            return None
        cumulative = (1 + history.tail(lookback)).prod() - 1
        w = pd.Series(0.0, index=history.columns)
        w[cumulative.nlargest(top_n).index] = 1.0 / top_n
        return w.values
    return monthly_rebalanced(returns, weights, "momentum", cost)


def get_risk_parity_equity(returns: pd.DataFrame, cost: float | None = None):
    """Risk parity on all history up to each month-end, rebalanced monthly."""
    from optimization.risk_parity import risk_parity

    def weights(history):
        if len(history) < MIN_HISTORY:
            return None
        return risk_parity(history)
    return monthly_rebalanced(returns, weights, "risk_parity", cost)


COST_GRID_BP = (2, 5, 10)


def benchmark_returns(returns: pd.DataFrame, cost: float | None = None) -> dict[str, pd.Series]:
    """Daily returns of every benchmark, built on the full history of `returns`."""
    return {
        "spy":          returns["SPY"],
        "equal_weight": get_equal_weight_equity(returns, cost)[0],
        "sixty_forty":  get_sixty_forty_equity(returns, cost)[0],
        "momentum":     get_momentum_equity(returns, cost=cost)[0],
        "risk_parity":  get_risk_parity_equity(returns, cost)[0],
    }


def _score(r: pd.Series, window: pd.DatetimeIndex, rf: pd.Series | None) -> dict:
    r = r.reindex(window)
    return compute_all(r, (1 + r).cumprod(), rf=rf)


def cost_sensitivity(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    rf: pd.Series | None = None,
    costs_bp: tuple[int, ...] = COST_GRID_BP,
) -> pd.DataFrame:
    """Sharpe of the portfolio and every benchmark at each one-way cost in bp.

    Every strategy is re-simulated at each cost and scored on the portfolio's
    dates. SPY is buy and hold, so it pays no cost and is the same in every column.
    """
    rows = {}
    for bp in costs_bp:
        cost = bp / 10_000
        port = simulate(weights, returns, cost=cost)["portfolio_return"]
        strategies = {"portfolio": port, **benchmark_returns(returns, cost)}
        rows[f"{bp} bp"] = {
            name: _score(r, port.index, rf)["sharpe_ratio"] for name, r in strategies.items()
        }
    return pd.DataFrame(rows)


def run() -> dict:
    """Metrics of the saved portfolio and every benchmark on the portfolio's dates.

    Benchmarks are built on the full history (the same inputs the portfolio's
    optimizers see), then every strategy is scored on the portfolio's dates.
    """
    returns = pd.read_parquet(CFG["paths"]["returns"])
    backtest = pd.read_parquet(CFG["paths"]["backtest_results"])
    rf = fetch_risk_free()
    strategies = {"portfolio": backtest["portfolio_return"], **benchmark_returns(returns)}
    return {name: _score(r, backtest.index, rf) for name, r in strategies.items()}


if __name__ == "__main__":
    results = run()

    print(f"{'Metric':<25} {'Portfolio':>12} {'SPY':>12}")
    print("-" * 50)
    for metric in results["portfolio"]:
        port_val = results["portfolio"][metric]
        spy_val  = results["spy"][metric]
        print(f"{metric:<25} {port_val:>12} {spy_val:>12}")