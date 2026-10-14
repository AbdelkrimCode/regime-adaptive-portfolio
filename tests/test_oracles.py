import numpy as np
import pandas as pd

from optimization.switcher import LABELS
from research.oracles import (
    LABELS_BY_VOL, argmax_labels, hindsight_mapping, hindsight_posteriors, realized_vol_posteriors,
)


def test_argmax_labels_picks_the_largest_posterior():
    idx = pd.bdate_range("2020-01-01", periods=2)
    post = pd.DataFrame([[0.1, 0.6, 0.2, 0.1], [0.7, 0.1, 0.1, 0.1]], index=idx, columns=LABELS)
    assert list(argmax_labels(post)) == [LABELS[1], LABELS[0]]


def test_hindsight_posteriors_rank_states_by_volatility():
    rng = np.random.default_rng(0)
    n = 900
    vol = np.where((np.arange(n) // 150) % 2 == 0, -1.0, 2.0) + rng.normal(0, 0.3, n)
    ret = rng.normal(0, 1, n) * (1 + (vol > 0))
    corr = rng.normal(0, 1, n)
    idx = pd.bdate_range("2010-01-01", periods=n)
    features = pd.DataFrame({"spy_return": ret, "spy_vol": vol, "mean_corr": corr}, index=idx)

    post = hindsight_posteriors(features, idx[100:])

    assert list(post.columns) == LABELS
    np.testing.assert_allclose(post.sum(axis=1), 1.0)
    labels = argmax_labels(post)
    calm = features.loc[labels.index[labels == "Bull"], "spy_vol"].mean()
    turbulent = features.loc[labels.index[labels.isin(["Bear", "Crash"])], "spy_vol"].mean()
    assert turbulent > calm


def test_realized_vol_posteriors_label_each_day_by_the_volatility_ahead():
    idx = pd.bdate_range("2020-01-01", periods=400)
    t = np.arange(400)
    spy = pd.Series(np.where(t % 2 == 0, 1.0, -1.0) * 0.001 * (1 + t / 50), index=idx)

    post = realized_vol_posteriors(spy, idx, horizon=21)

    np.testing.assert_allclose(post.sum(axis=1), 1.0)
    rank = post.values.argmax(axis=1)
    order = [LABELS.index(label) for label in LABELS_BY_VOL]
    vol_rank = np.array([order.index(r) for r in rank])
    assert (np.diff(vol_rank) >= 0).all(), "volatility rises over time, so labels must too"
    assert post.iloc[0]["Bull"] == 1.0 and post.iloc[-1]["Crash"] == 1.0


def test_realized_vol_posteriors_see_a_shock_before_it_happens():
    idx = pd.bdate_range("2020-01-01", periods=400)
    t = np.arange(400)
    # Calm volatility drifts down over time, so without look-ahead days 279-299 would rank low.
    values = np.where(t % 2 == 0, 1.0, -1.0) * 0.001 * (2 - t / 400)
    values[300] = 0.10
    spy = pd.Series(values, index=idx)

    post = realized_vol_posteriors(spy, idx, horizon=21)

    # Days 279-299 have the shock in their next 21 days, so they are labeled turbulent.
    assert (post.iloc[279:300]["Crash"] == 1.0).all()


def test_hindsight_mapping_picks_the_best_book_per_label():
    idx = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(2)
    labels = pd.Series(np.where(np.arange(400) % 2 == 0, "Bull", "Crash"), index=idx)
    acted_bull = labels.shift(1) == "Bull"
    noise = rng.normal(0, 0.01, 400)
    books = {
        "max_sharpe":   pd.Series(noise + np.where(acted_bull, 0.004, -0.004), index=idx),
        "safe_haven":   pd.Series(noise + np.where(acted_bull, -0.004, 0.004), index=idx),
        "risk_parity":  pd.Series(noise, index=idx),
        "min_variance": pd.Series(noise, index=idx),
    }

    mapping = hindsight_mapping(books, labels)

    assert mapping["Bull"] == "max_sharpe"
    assert mapping["Crash"] == "safe_haven"
