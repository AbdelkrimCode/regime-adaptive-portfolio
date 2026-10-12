import numpy as np
import pandas as pd
from data.risk_free import fetch_risk_free
from config import load_config
from backtest.bootstrap import block_resample, run_bootstrap, summarize
from backtest.sharpe_test import sharpe_difference_test

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
    # rf is subtracted before resampling so each day keeps its own rf.
    rng = np.random.default_rng(RANDOM_STATE)
    rf_aligned = rf.reindex(returns.index).ffill().fillna(0)
    excess_returns = returns - rf_aligned
    sharpes = []

    for _ in range(n_sim):
        excess = block_resample(excess_returns, BLOCK_SIZE, rng).values
        s = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0.0
        sharpes.append(s)

    sharpes = np.array(sharpes)
    return {
        "mean":     round(float(sharpes.mean()), 4),
        "std":      round(float(sharpes.std()), 4),
        "ci_lower": round(float(np.percentile(sharpes, 2.5)), 4),
        "ci_upper": round(float(np.percentile(sharpes, 97.5)), 4),
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

    print("\n--- Paired Block Bootstrap vs SPY (1000 paths) ---")
    pb = summarize(run_bootstrap(port_returns, spy_returns, rf=rf))
    print(f"  Mean Difference:     {pb['mean_sharpe_diff']}")
    print(f"  P-Value:             {pb['p_value']}")
    print(f"  95% CI:              [{pb['ci_lower']}, {pb['ci_upper']}]")

    print("\n--- Ledoit-Wolf (2008) HAC Sharpe Difference Test vs SPY ---")
    lw = sharpe_difference_test(port_returns, spy_returns, rf=rf)
    print(f"  Observed Difference: {lw['d_observed']}")
    print(f"  Std. Error:          {lw['se']}")
    print(f"  P-Value (one-sided): {lw['p_value']}")
    print(f"  P-Value (two-sided): {lw['p_value_two_sided']}")

if __name__ == "__main__":
    run_stress_tests()