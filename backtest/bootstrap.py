import numpy as np
import pandas as pd
from backtest.metrics import compute_all
from data.risk_free import fetch_risk_free

BLOCK_LENGTH = 20
N_ITERATIONS = 1000
RANDOM_STATE = 42

def block_resample(returns : pd.Series, block_length: int, rng: np.random.Generator) -> pd.Series:
    n = len(returns)
    values = returns.values
    resampled = []

    while len(resampled) < n:
        start = rng.integers(0, n - block_length)
        block = values[start:start + block_length]
        resampled.extend(block)

    return pd.Series(resampled[:n], index=returns.index)

def run_bootstrap(
    port_returns: pd.Series,
    benchmark_returns: pd.Series,
    block_length: int = BLOCK_LENGTH,
    n_iterations: int = N_ITERATIONS,
    random_state: int = RANDOM_STATE,
    rf: pd.Series | None = None
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    records = []

    for _ in range(n_iterations):
        port_sample = block_resample(port_returns, block_length, rng)
        bench_sample = block_resample(benchmark_returns, block_length, rng)

        port_equity = (1 + port_sample).cumprod()
        bench_equity = (1 + bench_sample).cumprod()

        port_metrics = compute_all(port_sample, port_equity, rf=rf)
        bench_metrics = compute_all(bench_sample, bench_equity, rf=rf)

        records.append({
            "port_sharpe":  port_metrics["sharpe_ratio"],
            "bench_sharpe": bench_metrics["sharpe_ratio"],
            "sharpe_diff":  port_metrics["sharpe_ratio"] - bench_metrics["sharpe_ratio"],
            "port_return":  port_metrics["annualized_return"],
            "bench_return": bench_metrics["annualized_return"],
        })

    return pd.DataFrame(records)

def summarize(bootstrap_df: pd.DataFrame) -> dict:
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
    from config import load_config
    if output_path is None:
        output_path = load_config()["paths"]["bootstrap"]
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(bootstrap_df["port_sharpe"], bins=50, alpha=0.6, label="Portfolio", color="steelblue")
    axes[0].hist(bootstrap_df["bench_sharpe"], bins=50, alpha=0.6, label="SPY", color="darkorange")
    axes[0].set_title("Sharpe Distribution — Portfolio vs SPY")
    axes[0].set_xlabel("Sharpe Ratio")
    axes[0].legend()

    axes[1].hist(bootstrap_df["sharpe_diff"], bins=50, color="steelblue", alpha=0.8)
    axes[1].axvline(0, color="red", linestyle="--", label="No difference")
    axes[1].axvline(bootstrap_df["sharpe_diff"].quantile(0.025), color="gray", linestyle="--", label="95% CI")
    axes[1].axvline(bootstrap_df["sharpe_diff"].quantile(0.975), color="gray", linestyle="--")
    axes[1].set_title("Sharpe Difference Distribution")
    axes[1].set_xlabel("Portfolio Sharpe − SPY Sharpe")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"      Saved to {output_path}")



def run(output_path: str | None = None) -> dict:
    from config import load_config
    if output_path is None:
        output_path = load_config()["paths"]["bootstrap"]
    returns = pd.read_parquet("data/processed/returns.parquet")
    backtest = pd.read_parquet("data/backtest_results.parquet")

    common = returns.index.intersection(backtest.index)
    spy_returns = returns.loc[common, "SPY"]
    port_returns = backtest.loc[common, "portfolio_return"]
    rf = fetch_risk_free()

    print("  Running block bootstrap (1000 iterations)...")
    bootstrap_df = run_bootstrap(port_returns, spy_returns, rf=rf)
    summary = summarize(bootstrap_df)
    plot_bootstrap(bootstrap_df, output_path)

    return summary


if __name__ == "__main__":
    summary = run()
    for k, v in summary.items():
        print(f"{k:<25} {v}")