import numpy as np
import pandas as pd
from backtest.metrics import compute_all
from data.risk_free import fetch_risk_free
from backtest.sharpe_test import sharpe_difference_test
from config import load_config

CFG          = load_config()

BLOCK_LENGTH = CFG["bootstrap"]["block_length"]
N_ITERATIONS = CFG["bootstrap"]["n_iterations"]
RANDOM_STATE = CFG["bootstrap"]["random_state"]

def block_bootstrap_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Generate block-bootstrap resample indices of length n.

    Reuse the same indices across related series to preserve dependence.
    """
    indices = []
    while len(indices) < n:
        start = rng.integers(0, n - block_length + 1)
        indices.extend(range(int(start), int(start) + block_length))
    return np.array(indices[:n])

def block_resample(returns : pd.Series, block_length: int, rng: np.random.Generator) -> pd.Series:
    """Block-bootstrap resample of a series, keeping its original index."""
    n = len(returns)
    indices = block_bootstrap_indices(n, block_length, rng)
    return pd.Series(returns.values[indices], index=returns.index)

def run_bootstrap(
    port_returns: pd.Series,
    benchmark_returns: pd.Series,
    block_length: int = BLOCK_LENGTH,
    n_iterations: int = N_ITERATIONS,
    random_state: int = RANDOM_STATE,
    rf: pd.Series | None = None
) -> pd.DataFrame:
    """Paired block bootstrap of portfolio and benchmark metrics.

    Both series and rf are resampled with the same block indices, so the
    pairing and each day's rf are preserved."""
    rng = np.random.default_rng(random_state)
    records = []
    n = len(port_returns)
    if rf is not None:
        rf = rf.reindex(port_returns.index).ffill().fillna(0)

    for _ in range(n_iterations):
        indices = block_bootstrap_indices(n, block_length, rng)

        # rf is resampled with the same indices so each day keeps its own rf.
        port_sample  = pd.Series(port_returns.values[indices],  index=port_returns.index)
        bench_sample = pd.Series(benchmark_returns.values[indices], index=port_returns.index)
        rf_sample = None if rf is None else pd.Series(rf.values[indices], index=port_returns.index)

        port_equity = (1 + port_sample).cumprod()
        bench_equity = (1 + bench_sample).cumprod()

        port_metrics = compute_all(port_sample, port_equity, rf=rf_sample)
        bench_metrics = compute_all(bench_sample, bench_equity, rf=rf_sample)

        records.append({
            "port_sharpe":  port_metrics["sharpe_ratio"],
            "bench_sharpe": bench_metrics["sharpe_ratio"],
            "sharpe_diff":  port_metrics["sharpe_ratio"] - bench_metrics["sharpe_ratio"],
            "port_return":  port_metrics["annualized_return"],
            "bench_return": bench_metrics["annualized_return"],
        })

    return pd.DataFrame(records)

def summarize(bootstrap_df: pd.DataFrame) -> dict:
    """Mean, 95% CI and one-sided p-value (share of paths with diff <= 0) of the Sharpe difference."""
    sharpe_diff = bootstrap_df["sharpe_diff"]
    p_value = (sharpe_diff <= 0).mean()

    return {
        "mean_sharpe_diff":   round(sharpe_diff.mean(), 4),
        "std_sharpe_diff":    round(sharpe_diff.std(), 4),
        "ci_lower":           round(sharpe_diff.quantile(0.025), 4),
        "ci_upper":           round(sharpe_diff.quantile(0.975), 4),
        "p_value":            round(p_value, 4),
        "significant_at_95":  p_value < 0.05,
    }

def plot_bootstrap(bootstrap_df: pd.DataFrame, output_path: str | None = None) -> None:
    """Plot bootstrap Sharpe distributions and their difference vs SPY."""
    from config import load_config
    if output_path is None:
        output_path = load_config()["paths"]["bootstrap"]
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(bootstrap_df["port_sharpe"], bins=50, alpha=0.6, label="Portfolio", color="steelblue")
    axes[0].hist(bootstrap_df["bench_sharpe"], bins=50, alpha=0.6, label="SPY", color="darkorange")
    axes[0].set_title("Sharpe Distribution: Portfolio vs SPY")
    axes[0].set_xlabel("Sharpe Ratio")
    axes[0].legend()

    axes[1].hist(bootstrap_df["sharpe_diff"], bins=50, color="steelblue", alpha=0.8)
    axes[1].axvline(0, color="red", linestyle="--", label="No difference")
    axes[1].axvline(bootstrap_df["sharpe_diff"].quantile(0.025), color="gray", linestyle="--", label="95% CI")
    axes[1].axvline(bootstrap_df["sharpe_diff"].quantile(0.975), color="gray", linestyle="--")
    axes[1].set_title("Sharpe Difference Distribution")
    axes[1].set_xlabel("Portfolio Sharpe - SPY Sharpe")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"      Saved to {output_path}")



SIGNIFICANCE_BENCHMARKS = ("spy", "risk_parity", "sixty_forty")


def compare_to_benchmarks(
    port_returns: pd.Series,
    benchmarks: dict[str, pd.Series],
    rf: pd.Series | None = None,
) -> pd.DataFrame:
    """Paired block bootstrap and Ledoit-Wolf test of the portfolio vs each benchmark.

    Both p-values are one-sided (H1: portfolio Sharpe is higher). Benchmarks are
    scored on the portfolio's dates.
    """
    rows = {}
    for name, bench in benchmarks.items():
        bench = bench.reindex(port_returns.index)
        boot = summarize(run_bootstrap(port_returns, bench, rf=rf))
        lw = sharpe_difference_test(port_returns, bench, rf=rf)
        rows[name] = {
            "sharpe_diff":   lw["d_observed"],
            "boot_p":        boot["p_value"],
            "boot_ci_lower": boot["ci_lower"],
            "boot_ci_upper": boot["ci_upper"],
            "lw_se":         lw["se"],
            "lw_p":          lw["p_value"],
        }
    return pd.DataFrame(rows).T


def run(output_path: str | None = None) -> pd.DataFrame:
    """Significance table of the saved portfolio vs SPY, risk parity and 60/40; saves the SPY plot."""
    from backtest.benchmark import benchmark_returns

    if output_path is None:
        output_path = CFG["paths"]["bootstrap"]
    returns = pd.read_parquet(CFG["paths"]["returns"])
    backtest = pd.read_parquet(CFG["paths"]["backtest_results"])
    port_returns = backtest["portfolio_return"]
    rf = fetch_risk_free()

    all_benchmarks = benchmark_returns(returns)
    benchmarks = {name: all_benchmarks[name] for name in SIGNIFICANCE_BENCHMARKS}
    table = compare_to_benchmarks(port_returns, benchmarks, rf=rf)

    spy = benchmarks["spy"].reindex(port_returns.index)
    plot_bootstrap(run_bootstrap(port_returns, spy, rf=rf), output_path)
    return table


if __name__ == "__main__":
    print(run().to_string())