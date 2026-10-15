import numpy as np
import pandas as pd
import pytest

from backtest.dsr import deflated_sharpe_ratio, expected_max_sharpe, probabilistic_sharpe_ratio


def make_returns(mu=0.0005, sigma=0.01, n=2500, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sigma, n), index=pd.bdate_range("2010-01-01", periods=n))


def test_psr_is_one_half_at_the_observed_sharpe():
    r = make_returns()
    sr = r.mean() / r.std()
    assert probabilistic_sharpe_ratio(r, sr_benchmark=sr) == pytest.approx(0.5)


def test_expected_max_sharpe_matches_monte_carlo():
    rng = np.random.default_rng(1)
    simulated = rng.standard_normal((20000, 100)).max(axis=1).mean()
    assert expected_max_sharpe(100, 1.0) == pytest.approx(simulated, rel=0.03)


def test_expected_max_sharpe_grows_with_trials():
    assert expected_max_sharpe(1, 1.0) == 0.0
    assert expected_max_sharpe(10, 1.0) < expected_max_sharpe(100, 1.0)


def test_dsr_falls_as_more_trials_are_counted():
    r = make_returns()
    rng = np.random.default_rng(2)
    few = deflated_sharpe_ratio(r, list(rng.normal(0, 0.01, 5)))
    many = deflated_sharpe_ratio(r, list(rng.normal(0, 0.01, 500)))
    assert 0 <= many["dsr"] < few["dsr"] <= 1


def test_dsr_n_trials_override_raises_the_bar():
    r = make_returns()
    trials = list(np.random.default_rng(3).normal(0, 0.01, 10))
    default = deflated_sharpe_ratio(r, trials)
    more = deflated_sharpe_ratio(r, trials, n_trials=50)
    assert more["n_trials"] == 50
    assert more["sharpe_threshold"] > default["sharpe_threshold"]
