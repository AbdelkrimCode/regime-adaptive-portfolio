import numpy as np
import pandas as pd
import pytest

from scripts.stress_tests import compute_var, compute_cvar, monte_carlo_sharpe
from backtest.bootstrap import run_bootstrap, summarize
from backtest.sharpe_test import sharpe_difference_test, _hac_covariance


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

    result = summarize(run_bootstrap(port, bench, block_length=BLOCK_SIZE, n_iterations=300, rf=rf))

    assert result["mean_sharpe_diff"] > 1.0, "sanity check: synthetic gap should be large"
    assert result["p_value"] < 0.05


def test_paired_block_bootstrap_no_effect_when_series_identical():
    port, _ = make_pair_with_gap(gap=0.0)
    rf = pd.Series(0.0, index=port.index)

    result = summarize(run_bootstrap(port, port, block_length=BLOCK_SIZE, n_iterations=300, rf=rf))

    assert result["mean_sharpe_diff"] == 0.0
    assert result["p_value"] == 1.0, "a zero difference in every path is no evidence"


def test_sharpe_difference_test_detects_large_effect():
    port, bench = make_pair_with_gap(gap=0.0007)

    result = sharpe_difference_test(port, bench)

    assert result["d_observed"] > 1.0
    assert result["p_value"] < 0.05


def test_sharpe_difference_test_identical_series():
    port, _ = make_pair_with_gap(gap=0.0)

    result = sharpe_difference_test(port, port)

    assert result["d_observed"] == 0.0
    assert result["p_value_two_sided"] == 1.0


def test_sharpe_difference_test_matches_iid_delta_method_on_iid_data():
    # With iid data the HAC standard error should be close to the plain
    # delta-method one (Jobson-Korkie with Memmel's correction).
    port, bench = make_pair_with_gap(gap=0.0002, seed=4)
    r1, r2 = port.values, bench.values
    mu1, mu2, g1, g2 = r1.mean(), r2.mean(), (r1**2).mean(), (r2**2).mean()
    v1, v2 = g1 - mu1**2, g2 - mu2**2
    grad = np.array([g1 / v1**1.5, -g2 / v2**1.5, -mu1 / (2 * v1**1.5), mu2 / (2 * v2**1.5)])
    y = np.column_stack([r1 - mu1, r2 - mu2, r1**2 - g1, r2**2 - g2])
    se_iid = np.sqrt(grad @ np.cov(y.T, bias=True) @ grad / N) * np.sqrt(252)

    result = sharpe_difference_test(port, bench)

    assert result["se"] == pytest.approx(se_iid, rel=0.15)


def test_hac_covariance_widens_for_autocorrelated_data():
    rng = np.random.default_rng(2)
    eps = rng.normal(size=(5000, 2))
    y = np.zeros_like(eps)
    for t in range(1, len(y)):
        y[t] = 0.6 * y[t - 1] + eps[t]
    y -= y.mean(axis=0)

    # Long-run variance of an AR(1) with phi = 0.6 and unit innovations is
    # 1 / (1 - 0.6)^2 = 6.25; the plain variance is only 1 / (1 - 0.36) = 1.56.
    psi = _hac_covariance(y)
    assert psi[0, 0] == pytest.approx(6.25, rel=0.25)
    assert psi[0, 0] > 3 * np.var(y[:, 0])


def test_compute_var_cvar_ordering():
    returns = pd.Series(np.random.default_rng(0).normal(0, 0.01, 1000))
    var = compute_var(returns)
    cvar = compute_cvar(returns)
    assert cvar <= var, "CVaR should be at least as extreme as VaR"

def _returns_tracking_volatile_rf(seed: int = 3) -> tuple[pd.Series, pd.Series]:
    # rf swings between 0 and 10% a year; returns are rf plus a small steady edge,
    # so the true excess Sharpe is high and stable.
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=N, freq="B")
    rf = pd.Series(np.where((np.arange(N) // 250) % 2, 0.10 / 252, 0.0), index=dates)
    returns = rf + rng.normal(0.0002, 0.0005, N)
    return returns, rf


def test_monte_carlo_sharpe_resamples_rf_with_returns():
    returns, rf = _returns_tracking_volatile_rf()
    excess = returns - rf
    observed = excess.mean() / excess.std() * np.sqrt(252)

    mc = monte_carlo_sharpe(returns, rf, n_sim=200)

    assert abs(mc["mean"] - observed) < 0.5, (
        f"Bootstrap Sharpe {mc['mean']} should center on observed {observed:.2f}"
    )


def test_run_bootstrap_resamples_rf_with_returns():
    returns, rf = _returns_tracking_volatile_rf()
    excess = returns - rf
    observed = excess.mean() / excess.std() * np.sqrt(252)

    df = run_bootstrap(returns, returns, n_iterations=100, rf=rf)

    assert abs(df["port_sharpe"].mean() - observed) < 0.5


def test_compare_to_benchmarks_scores_each_benchmark_on_portfolio_dates():
    from backtest.bootstrap import compare_to_benchmarks

    port, bench = make_pair_with_gap(gap=0.0007)
    # Benchmark history starts a year before the portfolio, as in the pipeline.
    early = pd.Series(0.05, index=pd.date_range(end=port.index[0], periods=251, freq="B")[:-1])
    long_bench = pd.concat([early, bench])

    table = compare_to_benchmarks(port, {"risk_parity": long_bench, "sixty_forty": port})

    assert list(table.index) == ["risk_parity", "sixty_forty"]
    expected = sharpe_difference_test(port, bench)
    assert table.loc["risk_parity", "sharpe_diff"] == expected["d_observed"]
    assert table.loc["risk_parity", "lw_p"] < 0.05
    assert table.loc["risk_parity", "boot_p"] < 0.05
    assert table.loc["sixty_forty", "sharpe_diff"] == 0.0
