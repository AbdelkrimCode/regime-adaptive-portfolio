import os
import numpy as np
import pandas as pd
import pytest

from backtest.bootstrap import run_bootstrap

def make_returns_pair(n: int = 300) -> tuple[pd.Series, pd.Series]:
    np.random.seed(42)
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    port = pd.Series(np.random.normal(0.0004, 0.01, n), index=dates)
    bench = pd.Series(np.random.normal(0.0003, 0.01, n), index=dates)
    return port, bench

def test_bootstrap_sharpe_differs_with_rf():
    port, bench = make_returns_pair()
    rf = pd.Series(0.0001, index=port.index)

    df_no_rf = run_bootstrap(port, bench, n_iterations=50)
    df_with_rf = run_bootstrap(port, bench, n_iterations=50, rf=rf)

    assert not np.isclose(
        df_no_rf["port_sharpe"].mean(),
        df_with_rf["port_sharpe"].mean(),
        atol=1e-4
    ), "Sharpe should differ when rf is applied"


from scripts.sensitivity_sweep import run_single as sweep_run_single

def _run_synthetic_sweep(monkeypatch, vol_window: int = 21, corr_window: int = 63) -> dict:
    import scripts.sensitivity_sweep as sweep_mod

    n = 700
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    rng = np.random.default_rng(5)
    log_returns = rng.normal(0.0003, 0.01, (n, 4))
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(log_returns, axis=0)),
        index=dates,
        columns=["SPY", "TLT", "GLD", "EFA"],
    )
    rf = pd.Series(0.0001, index=dates)

    monkeypatch.setattr(sweep_mod, "fetch_prices", lambda: prices)
    monkeypatch.setattr(sweep_mod, "fetch_risk_free", lambda: rf)
    monkeypatch.setitem(sweep_mod.CFG["evaluation"], "test_start", "2020-01-01")
    monkeypatch.setitem(sweep_mod.CFG["evaluation"], "data_end", str(dates[-1].date()))

    return sweep_mod.run_single(vol_window=vol_window, corr_window=corr_window)


def test_sensitivity_sweep_run_single_keys(monkeypatch):
    result = _run_synthetic_sweep(monkeypatch)
    expected_keys = {
        "vol_window", "corr_window", "sharpe", "max_drawdown",
        "calmar", "return", "held_out_sharpe", "held_out_drawdown"
    }
    assert set(result.keys()) == expected_keys


def test_sensitivity_sweep_run_single_valid_values(monkeypatch):
    result = _run_synthetic_sweep(monkeypatch)
    assert -1.0 < result["sharpe"] < 5.0
    assert -1.0 < result["max_drawdown"] <= 0.0
    assert result["vol_window"] == 21



from scripts.feature_ablation import run_single as ablation_run_single, FEATURE_SETS
from data.process import compute_features, compute_returns
from data.fetch import fetch_prices
from config import load_config

# test_ablation_run_single_keys was removed after being superseded.

from models.hmm import _fit_hmm_core, forward_filter
from sklearn.preprocessing import StandardScaler

def test_forward_filter_differs_from_viterbi():
    np.random.seed(42)
    n = 500
    features = np.random.normal(0, 1, (n, 3))
    features[:200, 0] += 0.5
    features[200:350, 0] -= 0.5
    features[350:, 0] += 0.3

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    model = _fit_hmm_core(features_scaled, n_states=3)

    _, ff_posteriors = forward_filter(model, features_scaled)
    viterbi_posteriors = model.predict_proba(features_scaled)

    assert not np.allclose(ff_posteriors, viterbi_posteriors, atol=1e-3), \
        "Forward filter and Viterbi posteriors should differ"

    def entropy(p):
        p = np.clip(p, 1e-10, 1)
        return -np.sum(p * np.log(p), axis=1).mean()

    assert entropy(ff_posteriors) > entropy(viterbi_posteriors), \
        "Forward filter should be more uncertain than Viterbi"

def test_ablation_run_single_does_not_touch_production_paths(monkeypatch):
    import scripts.feature_ablation as ablation_mod

    n = 700
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    rng = np.random.default_rng(3)
    log_returns = rng.normal(0.0003, 0.01, (n, 4))
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(log_returns, axis=0)),
        index=dates,
        columns=["SPY", "TLT", "GLD", "EFA"],
    )
    rf = pd.Series(0.0001, index=dates)

    # Point all writable paths at nonexistent files so accidental writes fail.
    bogus_paths = {
        "features": "/tmp/should_not_be_written_features.parquet",
        "returns": "/tmp/should_not_be_written_returns.parquet",
        "regimes": "/tmp/should_not_be_written_regimes.parquet",
    }
    for key, path in bogus_paths.items():
        if os.path.exists(path):
            os.remove(path)
        monkeypatch.setitem(ablation_mod.CFG["paths"], key, path)

    monkeypatch.setattr(ablation_mod, "fetch_prices", lambda: prices)
    monkeypatch.setattr(ablation_mod, "fetch_risk_free", lambda: rf)
    monkeypatch.setattr(ablation_mod, "TEST_START", "2020-01-01")

    result = ablation_mod.run_single("baseline", ablation_mod.FEATURE_SETS["baseline"])

    expected_keys = {
        "feature_set", "sharpe", "max_drawdown",
        "calmar", "held_out_sharpe", "held_out_drawdown"
    }
    assert set(result.keys()) == expected_keys

    for path in bogus_paths.values():
        assert not os.path.exists(path), f"run_single wrote to {path} - should be in-memory only"