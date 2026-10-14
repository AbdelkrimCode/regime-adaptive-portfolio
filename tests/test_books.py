import numpy as np
import pandas as pd
import pytest

from optimization.switcher import compute_weights, LABELS
from research.books import (
    blend, constant_posteriors, optimizer_targets, placebo_shifts,
    posteriors_of, run_cell, shifted_posteriors,
)
from tests.conftest import MOCK_RF

ASSETS = ["SPY", "TLT", "GLD", "EFA", "IEF", "QQQ", "LQD", "VNQ"]


def make_inputs(n_hist=200, n_days=40, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_hist)
    returns = pd.DataFrame(rng.normal(0.0003, 0.01, (n_hist, len(ASSETS))), index=dates, columns=ASSETS)
    index = dates[-n_days:]
    p = rng.dirichlet(np.ones(4), n_days)
    regimes = pd.DataFrame(p, index=index, columns=[f"p_{l.lower()}" for l in LABELS])
    regimes["regime"] = [LABELS[i] for i in p.argmax(axis=1)]
    return returns, regimes


def test_blend_matches_compute_weights_with_weekly_refresh():
    returns, regimes = make_inputs()
    targets = optimizer_targets(regimes.index, returns, MOCK_RF)

    fast = blend(targets, posteriors_of(regimes))
    slow = compute_weights(regimes, returns, MOCK_RF, refresh="weekly")

    np.testing.assert_allclose(fast.values, slow.values, atol=1e-10)


def test_static_blend_is_the_average_of_the_four_books():
    returns, regimes = make_inputs()
    targets = optimizer_targets(regimes.index, returns, MOCK_RF)

    static = blend(targets, constant_posteriors(regimes.index))

    expected = sum(t for t in targets.values()) / 4
    np.testing.assert_allclose(static.values, expected.values, atol=1e-12)


def test_shifted_posteriors_keep_values_and_move_timing():
    _, regimes = make_inputs()
    post = posteriors_of(regimes)

    shifted = shifted_posteriors(post, 5)

    assert shifted.index.equals(post.index)
    np.testing.assert_allclose(shifted.iloc[5].values, post.iloc[0].values)
    np.testing.assert_allclose(np.sort(shifted.values, axis=0), np.sort(post.values, axis=0))


def test_placebo_shifts_stay_away_from_zero_in_both_directions():
    shifts = placebo_shifts(1000, min_shift=252, step=21)
    assert min(shifts) >= 252
    assert max(shifts) <= 1000 - 252


def test_run_cell_returns_simulated_returns():
    returns, regimes = make_inputs()
    targets = optimizer_targets(regimes.index, returns, MOCK_RF)

    result = run_cell(targets, posteriors_of(regimes), returns, cost=0.0)

    assert len(result) == len(regimes) - 1
    assert result["portfolio_return"].notna().all()
