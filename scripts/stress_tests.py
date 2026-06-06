import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from backtest.metrics import compute_all
from data.risk_free import fetch_risk_free
from config import load_config

CFG = load_config()

N_SIMULATIONS = CFG["stress_test"]["n_simulations"]
CONFIDENCE    = CFG["stress_test"]["confidence"]
VIX_THRESHOLD = CFG["stress_test"]["vix_threshold"]
RANDOM_STATE  = CFG["hmm"]["random_state"]
BLOCK_SIZE = CFG["stress_test"]["block_size"]

def compute_var(returns: pd.Series, confidence: float = CONFIDENCE) -> float:
    return float(np.percentile(returns, (1 - confidence) * 100))

def compute_cvar(returns: pd.Series, confidence: float = CONFIDENCE) -> float:
    var = compute_var(returns, confidence)
    return float(returns[returns <= var].mean())

def sharpe_during_high_vix(returns: pd.Series, rf: pd.Series, vix_threshold: float = VIX_THRESHOLD) -> float | None:
    try:
        import yfinance as yf
        vix = yf.download("^VIX", start=returns.index.min(), end=returns.index.max(), progress=False, auto_adjust=True)["Close"].squeeze()
        vix = vix.reindex(returns.index).ffill()
        mask = vix > vix_threshold
        if mask.sum() < 20:
            return None
        high_vix_returns = returns[mask]
        rf_aligned = rf.reindex(high_vix_returns.index).ffill().fillna(0)
        excess = high_vix_returns - rf_aligned
        return float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0
    except Exception:
        return None

def monte_carlo_sharpe(returns: pd.Series, rf: pd.Series, n_sim: int = N_SIMULATIONS) -> dict:
    rng = np.random.default_rng(RANDOM_STATE)
    sharpes = []

    for _ in range(n_sim):
        sampled = returns.sample(n=len(returns), replace=True, random_state=int(rng.integers(0, 1e6)))
        rf_aligned = rf.reindex(returns.index).ffill().fillna(0).values
        excess = sampled.values - rf_aligned
        s = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0.0
        sharpes.append(s)

    sharpes = np.array(sharpes)
    return {
        "mean":  round(float(sharpes.mean()), 4),
        "std":   round(float(sharpes.std()), 4),
        "ci_lower": round(float(np.percentile(sharpes, 2.5)), 4),
        "ci_upper": round(float(np.percentile(sharpes, 97.5)), 4),
    }

def paired_block_bootstrap(
    port_returns: pd.Series,
    spy_returns: pd.Series,
    rf: pd.Series,
    block_size: int = BLOCK_SIZE,
    n_sim: int = N_SIMULATIONS,
) -> dict:
    rf_aligned = rf.reindex(port_returns.index).ffill().fillna(0)

    mean_rf = float(rf_aligned.mean())

    def sharpe(returns: np.ndarray) -> float:
        excess = returns - mean_rf
        return float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0

    d_observed = sharpe(port_returns.values) - sharpe(spy_returns.values)

    rng = np.random.default_rng(RANDOM_STATE)
    n = len(port_returns)
    n_blocks = n // block_size

    d_boot = []
    for _ in range(n_sim):
        block_starts = rng.integers(0, n - block_size, size=n_blocks)
        indices = np.concatenate([
            np.arange(start, start + block_size) for start in block_starts
        ])
        indices = indices[:n]

        port_sample = port_returns.values[indices]
        spy_sample = spy_returns.values[indices]

        d = sharpe(port_sample) - sharpe(spy_sample)
        d_boot.append(d)

    d_boot = np.array(d_boot)
    p_value = float(np.mean(d_boot >= d_observed))

    return {
        "d_observed": round(d_observed, 4),
        "p_value":    round(p_value, 4),
        "ci_lower":   round(float(np.percentile(d_boot, 2.5)), 4),
        "ci_upper":   round(float(np.percentile(d_boot, 97.5)), 4),
    }

def run_stress_tests() -> None:
    print("=" * 50)
    print("  Stress Test Analysis")
    print("=" * 50)

    backtest = pd.read_parquet(CFG["paths"]["backtest_results"])
    returns  = pd.read_parquet(CFG["paths"]["returns"])
    rf       = fetch_risk_free()

    port_returns = backtest["portfolio_return"]
    spy_returns  = returns.loc[port_returns.index, "SPY"]

    print("\n--- VaR / CVaR (95%) ---")
    print(f"{'Metric':<20} {'Portfolio':>12} {'SPY':>12}")
    print("-" * 45)
    print(f"{'VaR(95%)':<20} {compute_var(port_returns):>12.4f} {compute_var(spy_returns):>12.4f}")
    print(f"{'CVaR(95%)':<20} {compute_cvar(port_returns):>12.4f} {compute_cvar(spy_returns):>12.4f}")

    print("\n--- Sharpe during VIX > 30 ---")
    port_vix_sharpe = sharpe_during_high_vix(port_returns, rf)
    spy_vix_sharpe  = sharpe_during_high_vix(spy_returns, rf)
    if port_vix_sharpe is not None:
        print(f"{'Portfolio':<20} {port_vix_sharpe:>12.4f}")
        print(f"{'SPY':<20} {spy_vix_sharpe:>12.4f}")
    else:
        print("  VIX data unavailable")

    print("\n--- Monte Carlo Sharpe (1000 paths) ---")
    mc = monte_carlo_sharpe(port_returns, rf)
    print(f"  Mean Sharpe:  {mc['mean']}")
    print(f"  Std:          {mc['std']}")
    print(f"  95% CI:       [{mc['ci_lower']}, {mc['ci_upper']}]")

    print("\n--- Paired Block Bootstrap (1000 paths) ---")
    pb = paired_block_bootstrap(port_returns, spy_returns, rf)
    print(f"  Observed Difference: {pb['d_observed']}")
    print(f"  P-Value:             {pb['p_value']}")
    print(f"  95% CI:              [{pb['ci_lower']}, {pb['ci_upper']}]")

if __name__ == "__main__":
    run_stress_tests()