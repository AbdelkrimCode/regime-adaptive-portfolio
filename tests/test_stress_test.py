import numpy as np
import pandas as pd

from scripts.stress_tests import paired_block_bootstrap, compute_var, compute_cvar


N = 2000
BLOCK_SIZE = 20


def make_pair_with_gap(gap: float, seed: int = 1) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=N, freq="B")
    bench = rng.normal(0.0002, 0.01, N)
    port = rng.normal(0.0002 + gap, 0.01, N)
    return (
        pd.Series(port, index=dates),
        pd.Series(bench, index=dates),
    )


def test_paired_block_bootstrap_detects_large_effect():
    port, bench = make_pair_with_gap(gap=0.0007)  # large, unambiguous edge
    rf = pd.Series(0.0, index=port.index)

    result = paired_block_bootstrap(port, bench, rf, block_size=BLOCK_SIZE, n_sim=500)

    assert result["d_observed"] > 1.0, "sanity check: synthetic gap should be large"
    assert result["p_value"] < 0.05, (
        f"A large, genuine Sharpe advantage should produce a small p-value, "
        f"got p={result['p_value']} (observed diff={result['d_observed']})"
    )


def test_paired_block_bootstrap_no_effect_when_series_identical():
    port, _ = make_pair_with_gap(gap=0.0)
    rf = pd.Series(0.0, index=port.index)

    result = paired_block_bootstrap(port, port, rf, block_size=BLOCK_SIZE, n_sim=500)

    assert result["d_observed"] == 0.0
    assert result["p_value"] > 0.3, (
        "No real difference should not look significant"
    )


def test_compute_var_cvar_ordering():
    returns = pd.Series(np.random.default_rng(0).normal(0, 0.01, 1000))
    var = compute_var(returns)
    cvar = compute_cvar(returns)
    assert cvar <= var, "CVaR should be at least as extreme as VaR"