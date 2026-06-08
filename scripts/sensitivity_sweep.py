import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from itertools import product
from data.fetch import fetch_prices
from data.process import compute_returns, compute_features, process_and_save
from models.hmm import run as run_hmm, walk_forward_regimes, load_features
from backtest.engine import run as run_backtest
from backtest.metrics import compute_all
from data.risk_free import fetch_risk_free
from config import load_config

CFG = load_config()

VOL_WINDOWS   = [10, 21, 42]
CORR_WINDOWS  = [42, 63, 126]
RESULTS_PATH  = CFG["paths"]["sensitivity_results"]

def run_single(vol_window: int, corr_window: int, n_jobs: int = -1) -> dict:
    print(f"\n--- VOL_WINDOW={vol_window}, CORR_WINDOW={corr_window} ---")

    prices = fetch_prices()
    returns = compute_returns(prices)
    features = compute_features(returns, vol_window=vol_window, corr_window=corr_window)

    features_path = CFG["paths"]["features"]
    returns_path = CFG["paths"]["returns"]
    features.to_parquet(features_path)
    returns.to_parquet(returns_path)

    rf = fetch_risk_free()
    regimes = walk_forward_regimes(features, n_jobs=n_jobs)

    import os
    os.makedirs(os.path.dirname(CFG["paths"]["regimes"]), exist_ok=True)
    regimes.to_parquet(CFG["paths"]["regimes"])

    backtest, _ = run_backtest(regimes_df=regimes)

    full_metrics = compute_all(backtest["portfolio_return"], backtest["equity"], rf=rf)

    test_start = CFG["evaluation"]["test_start"]
    from backtest.engine import run_period
    test_result, _ = run_period(start=test_start, end="2024-12-31", regimes_df=regimes)
    rf_test = rf.reindex(test_result.index).ffill().fillna(0)
    test_metrics = compute_all(test_result["portfolio_return"], test_result["equity"], rf=rf_test)

    return {
        "vol_window":         vol_window,
        "corr_window":        corr_window,
        "sharpe":             full_metrics["sharpe_ratio"],
        "max_drawdown":       full_metrics["max_drawdown"],
        "calmar":             full_metrics["calmar_ratio"],
        "return":             full_metrics["annualized_return"],
        "held_out_sharpe":    test_metrics["sharpe_ratio"],
        "held_out_drawdown":  test_metrics["max_drawdown"],
    }

def main() -> None:
    from joblib import Parallel, delayed
    combinations = list(product(VOL_WINDOWS, CORR_WINDOWS))
    results = Parallel(n_jobs=-1)(delayed(run_single)(vol_window, corr_window) for vol_window, corr_window in combinations)
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_PATH, index=False)
    print(f"\n=== Sensitivity Sweep Results ===")
    print(df.to_string(index=False))
    print(f"\nFull results saved to {RESULTS_PATH}")

    df = pd.DataFrame(results)
    print("\n=== Sensitivity Sweep Results ===")
    print(df.to_string(index=False))
    print(f"\nFull results saved to {RESULTS_PATH}")

def bootstrap_sweep() -> None:
    from backtest.bootstrap import run_bootstrap, summarize

    block_lengths = [10, 20, 40]

    returns = pd.read_parquet(CFG["paths"]["returns"])
    backtest = pd.read_parquet(CFG["paths"]["backtest_results"])

    common = returns.index.intersection(backtest.index)
    spy_returns = returns.loc[common, "SPY"]
    port_returns = backtest.loc[common, "portfolio_return"]

    print("\n=== Bootstrap Sensitivity (BLOCK_LENGTH) ===")
    print(f"  {'block_length':>12} {'p_value':>10} {'mean_diff':>10} {'ci_lower':>10} {'ci_upper':>10}")
    print(f"  {'-' * 55}")

    for block_length in block_lengths:
        bootstrap_df = run_bootstrap(port_returns, spy_returns, block_length=block_length)
        summary = summarize(bootstrap_df)
        print(f"  {block_length:>12} {summary['p_value']:>10} {summary['mean_sharpe_diff']:>10} {summary['ci_lower']:>10} {summary['ci_upper']:>10}")

if __name__ == "__main__":
    main()
    bootstrap_sweep()