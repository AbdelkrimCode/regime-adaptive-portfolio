import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock
from models.hmm import (
    count_params,
    select_n_states,
    label_states,
    get_transition_matrix,
    get_regime_durations,
)
from backtest.bootstrap import block_resample


def test_count_params_known():
    # n_states=2, n_features=3
    # transition = 2*(2-1) = 2
    # means = 2*3 = 6
    # covariances = 2 * 3*4//2 = 12
    # startprob = 2-1 = 1
    # total = 21
    assert count_params(2, 3) == 21


def test_count_params_four_states():
    # n_states=4, n_features=3
    # transition = 4*3 = 12
    # means = 4*3 = 12
    # covariances = 4 * 3*4//2 = 24
    # startprob = 3
    # total = 51
    assert count_params(4, 3) == 51


def test_count_params_increases_with_states():
    assert count_params(3, 3) > count_params(2, 3)
    assert count_params(4, 3) > count_params(3, 3)


def test_count_params_increases_with_features():
    assert count_params(3, 4) > count_params(3, 3)


def test_block_resample_output_length():
    rng = np.random.default_rng(42)
    returns = pd.Series(np.random.normal(0, 0.01, 500))
    result = block_resample(returns, block_length=20, rng=rng)
    assert len(result) == len(returns)


def test_block_resample_same_index():
    rng = np.random.default_rng(42)
    returns = pd.Series(np.random.normal(0, 0.01, 300))
    result = block_resample(returns, block_length=20, rng=rng)
    assert len(result.index) == len(returns.index)


def test_block_resample_values_from_original():
    rng = np.random.default_rng(42)
    returns = pd.Series(np.random.normal(0, 0.01, 300))
    result = block_resample(returns, block_length=20, rng=rng)
    original_values = set(returns.values.round(10))
    resampled_values = set(result.values.round(10))
    assert resampled_values.issubset(original_values)


def test_block_resample_different_seed():
    returns = pd.Series(np.random.normal(0, 0.01, 300))
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(2)
    r1 = block_resample(returns, block_length=20, rng=rng1)
    r2 = block_resample(returns, block_length=20, rng=rng2)
    assert not r1.equals(r2)


def test_label_states_four_states():
    model = MagicMock()
    model.means_ = np.array([
        [-0.002, 0.03, 0.6],
        [0.001, 0.01, 0.4],
        [-0.0005, 0.02, 0.5],
        [0.002, 0.008, 0.3],
    ])
    feature_df = pd.DataFrame()
    labels = label_states(model, feature_df)
    assert set(labels.values()) == {"Bull", "Bear", "Sideways", "Crash"}


def test_label_states_ordering():
    model = MagicMock()
    model.means_ = np.array([
        [-0.002],
        [0.002],
        [0.0],
        [-0.001],
    ])
    feature_df = pd.DataFrame()
    labels = label_states(model, feature_df)
    means = model.means_[:, 0]
    ranking = np.argsort(means)
    assert labels[ranking[0]] == "Crash"
    assert labels[ranking[3]] == "Bull"


def test_get_transition_matrix_shape():
    model = MagicMock()
    model.n_components = 4
    model.transmat_ = np.array([
        [0.95, 0.03, 0.01, 0.01],
        [0.02, 0.94, 0.03, 0.01],
        [0.01, 0.02, 0.96, 0.01],
        [0.02, 0.01, 0.02, 0.95],
    ])
    state_labels = {0: "Bear", 1: "Sideways", 2: "Bull", 3: "Crash"}
    transmat = get_transition_matrix(model, state_labels)
    assert transmat.shape == (4, 4)
    assert list(transmat.index) == ["Bear", "Sideways", "Bull", "Crash"]


def test_get_transition_matrix_rows_sum_to_one():
    model = MagicMock()
    model.n_components = 4
    model.transmat_ = np.array([
        [0.95, 0.03, 0.01, 0.01],
        [0.02, 0.94, 0.03, 0.01],
        [0.01, 0.02, 0.96, 0.01],
        [0.02, 0.01, 0.02, 0.95],
    ])
    state_labels = {0: "Bear", 1: "Sideways", 2: "Bull", 3: "Crash"}
    transmat = get_transition_matrix(model, state_labels)
    for row_sum in transmat.sum(axis=1):
        assert row_sum == pytest.approx(1.0, abs=1e-6)




def test_get_regime_durations_known():
    transmat = pd.DataFrame(
        [[0.9, 0.1], [0.2, 0.8]],
        index=["Bull", "Bear"],
        columns=["Bull", "Bear"]
    )
    durations = get_regime_durations(transmat)
    assert durations["Bull"] == pytest.approx(10.0, rel=1e-4)
    assert durations["Bear"] == pytest.approx(5.0, rel=1e-4)


def test_get_regime_durations_all_positive():
    transmat = pd.DataFrame(
        np.array([
            [0.95, 0.03, 0.01, 0.01],
            [0.02, 0.94, 0.03, 0.01],
            [0.01, 0.02, 0.96, 0.01],
            [0.02, 0.01, 0.02, 0.95],
        ]),
        index=["Bear", "Sideways", "Bull", "Crash"],
        columns=["Bear", "Sideways", "Bull", "Crash"]
    )
    durations = get_regime_durations(transmat)
    assert (durations > 0).all()