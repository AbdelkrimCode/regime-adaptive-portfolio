import json

import numpy as np
import pandas as pd
import pytest

from optimization.switcher import LABELS
from research.diagnostic import newey_west_lag, rotation_regressions
from research.h4 import CASH, correlation_aware_targets, stock_bond_correlation, with_cash
from research.stats import gap_difference_ci, placebo_p_value, verdict

ASSETS = ["SPY", "TLT", "GLD", "EFA", "IEF", "QQQ", "LQD", "VNQ"]


# --- H1 regression ---

def _posteriors(index, seed=0):
    p = np.random.default_rng(seed).dirichlet(np.ones(4), len(index))
    return pd.DataFrame(p, index=index, columns=LABELS)


def test_rotation_regression_detects_a_planted_relation():
    idx = pd.bdate_range("2010-01-01", periods=2000)
    rng = np.random.default_rng(1)
    post = _posteriors(idx)
    signal = post["Bull"].shift(1).fillna(0.25).values
    books = {
        "max_sharpe": pd.Series(0.004 * signal + rng.normal(0, 0.005, len(idx)), index=idx),
        "risk_parity": pd.Series(rng.normal(0, 0.005, len(idx)), index=idx),
        "min_variance": pd.Series(rng.normal(0, 0.005, len(idx)), index=idx),
        "safe_haven": pd.Series(rng.normal(0, 0.005, len(idx)), index=idx),
    }

    table = rotation_regressions(books, post)

    assert table.loc["max_sharpe", "reject"]
    assert table.loc["max_sharpe", "slope_Bull"] > 0


def test_rotation_regression_finds_nothing_in_noise():
    idx = pd.bdate_range("2010-01-01", periods=2000)
    rng = np.random.default_rng(2)
    books = {n: pd.Series(rng.normal(0, 0.005, len(idx)), index=idx)
             for n in ["max_sharpe", "risk_parity", "min_variance", "safe_haven"]}

    table = rotation_regressions(books, _posteriors(idx, seed=3))

    assert not table["reject"].any()


def test_newey_west_lag_rule():
    assert newey_west_lag(100) == 4
    assert newey_west_lag(4700) == int(np.floor(4 * 47 ** (2 / 9)))


# --- H4 rule ---

def _targets(index):
    w = pd.DataFrame(1 / len(ASSETS), index=index, columns=ASSETS)
    return {"risk_parity": w.copy(), "safe_haven": w.copy(), "max_sharpe": w.copy(), "min_variance": w.copy()}


def test_h4_moves_bonds_to_cash_only_in_defensive_books_when_correlation_positive():
    idx = pd.bdate_range("2020-01-01", periods=4)
    corr = pd.Series([np.nan, -0.2, 0.3, 0.0], index=idx)

    out = correlation_aware_targets(_targets(idx), corr)

    bond_weight = 3 / len(ASSETS)
    rp = out["risk_parity"]
    assert rp.loc[idx[2], CASH] == pytest.approx(bond_weight)
    assert (rp.loc[idx[2], ["TLT", "IEF", "LQD"]] == 0).all()
    for day in (idx[0], idx[1], idx[3]):          # no correlation yet, negative, exactly zero
        assert rp.loc[day, CASH] == 0.0
    assert (out["max_sharpe"][CASH] == 0).all()
    for t in out.values():
        np.testing.assert_allclose(t.sum(axis=1), 1.0)


def test_with_cash_earns_the_risk_free_rate():
    idx = pd.bdate_range("2020-01-01", periods=3)
    returns = pd.DataFrame(0.01, index=idx, columns=ASSETS)
    rf = pd.Series([0.0001, np.nan, 0.0002], index=idx)

    out = with_cash(returns, rf)

    assert list(out[CASH]) == [0.0001, 0.0001, 0.0002]


def test_stock_bond_correlation_uses_only_past_data():
    idx = pd.bdate_range("2020-01-01", periods=200)
    rng = np.random.default_rng(4)
    returns = pd.DataFrame(rng.normal(0, 0.01, (200, len(ASSETS))), index=idx, columns=ASSETS)
    full = stock_bond_correlation(returns)
    truncated = stock_bond_correlation(returns.iloc[:150])
    pd.testing.assert_series_equal(full.iloc[:150], truncated)


# --- study statistics ---

def test_placebo_p_value_counts_the_observed_run():
    assert placebo_p_value(1.0, [0.1, 0.2, 0.3]) == pytest.approx(1 / 4)
    assert placebo_p_value(0.0, [0.1, 0.2, 0.3]) == pytest.approx(1.0)


def test_gap_difference_ci_and_verdict():
    idx = pd.bdate_range("2010-01-01", periods=2500)
    rng = np.random.default_rng(5)
    base = rng.normal(0.0002, 0.01, len(idx))
    c0 = pd.Series(base, index=idx)
    detection = pd.Series(base + rng.normal(0, 0.002, len(idx)), index=idx)
    response = pd.Series(base + 0.001 + rng.normal(0, 0.002, len(idx)), index=idx)

    ci = gap_difference_ci(c0, detection, response, n_iterations=300)

    assert ci["difference"] == pytest.approx(ci["response_value"] - ci["detection_value"])
    assert ci["ci_lower"] > 0
    assert verdict(ci).startswith("supported")
    assert verdict({"ci_lower": -1, "ci_upper": 1}) == "inconclusive"
    assert verdict({"ci_lower": -1, "ci_upper": -0.1}).startswith("refuted")


# --- end-to-end smoke run on synthetic data ---

def test_run_study_end_to_end_on_synthetic_data(tmp_path):
    from research.run_study import run_study, write

    n = 700
    idx = pd.bdate_range("2015-01-01", periods=n)
    rng = np.random.default_rng(6)
    returns = pd.DataFrame(rng.normal(0.0003, 0.01, (n, len(ASSETS))), index=idx, columns=ASSETS)
    rf = pd.Series(0.00005, index=idx)
    features = pd.DataFrame({
        "spy_return": returns["SPY"],
        "spy_vol": returns["SPY"].rolling(21).std().bfill(),
        "mean_corr": rng.normal(0, 1, n),
    }, index=idx)
    reg_idx = idx[150:]
    post = _posteriors(reg_idx, seed=7)
    regimes = post.copy()
    regimes.columns = [f"p_{l.lower()}" for l in LABELS]
    regimes["regime"] = post.idxmax(axis=1)

    summary = run_study(returns, rf, features, regimes, window="dev",
                        placebo_step=50, n_bootstrap=50, prior_trials=[0.5, 0.6, 0.7])
    write(summary, str(tmp_path))

    for cell in ["C0", "C1", "H4", "D1", "D1b", "R1", "F"]:
        assert cell in summary["cells"][2]
    for key in ["H1", "H2", "H3", "H4", "C0_dsr", "yearly_returns"]:
        assert key in summary
    assert set(summary["H3"]["mappings"]["R1"]) == set(LABELS)
    assert json.loads((tmp_path / "summary.json").read_text())["window"] == "dev"
    assert (tmp_path / "cells_10bp.csv").exists()


def test_run_study_refuses_holdout_without_new_data():
    from research.run_study import run_study

    idx = pd.bdate_range("2015-01-01", periods=10)
    regimes = pd.DataFrame(index=idx)
    with pytest.raises(ValueError):
        run_study(pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame(), regimes, window="holdout")
