import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock
from models.hmm import (
    count_params,
    select_n_states,
    label_states,
    get_fitted_transition_matrix,
    get_regime_durations,
)
from backtest.bootstrap import block_resample, block_bootstrap_indices


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
    labels = label_states(model)
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
    labels = label_states(model)
    means = model.means_[:, 0]
    ranking = np.argsort(means)
    assert labels[ranking[0]] == "Crash"
    assert labels[ranking[3]] == "Bull"


def test_get_fitted_transition_matrix_shape():
    model = MagicMock()
    model.n_components = 4
    model.transmat_ = np.array([
        [0.95, 0.03, 0.01, 0.01],
        [0.02, 0.94, 0.03, 0.01],
        [0.01, 0.02, 0.96, 0.01],
        [0.02, 0.01, 0.02, 0.95],
    ])
    state_labels = {0: "Bear", 1: "Sideways", 2: "Bull", 3: "Crash"}
    transmat = get_fitted_transition_matrix(model, state_labels)
    assert transmat.shape == (4, 4)
    assert list(transmat.index) == ["Bear", "Sideways", "Bull", "Crash"]


def test_get_fitted_transition_matrix_rows_sum_to_one():
    model = MagicMock()
    model.n_components = 4
    model.transmat_ = np.array([
        [0.95, 0.03, 0.01, 0.01],
        [0.02, 0.94, 0.03, 0.01],
        [0.01, 0.02, 0.96, 0.01],
        [0.02, 0.01, 0.02, 0.95],
    ])
    state_labels = {0: "Bear", 1: "Sideways", 2: "Bull", 3: "Crash"}
    transmat = get_fitted_transition_matrix(model, state_labels)
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

def test_fit_hmm_core_attaches_convergence_diagnostics(monkeypatch):
    import models.hmm as hmm_mod

    monkeypatch.setitem(hmm_mod.CFG["hmm"], "n_init", 2)
    rng = np.random.default_rng(0)
    features = rng.normal(0, 1, (200, 3))

    model = hmm_mod._fit_hmm_core(features, n_states=3)

    assert model is not None
    assert hasattr(model, "converged_")
    assert hasattr(model, "n_failed_restarts_")
    assert isinstance(model.converged_, bool)


def test_fit_hmm_core_flags_non_convergence_with_low_n_iter(monkeypatch, capsys):
    import models.hmm as hmm_mod

    monkeypatch.setitem(hmm_mod.CFG["hmm"], "n_iter", 1)
    monkeypatch.setitem(hmm_mod.CFG["hmm"], "n_init", 2)

    # Well-separated clusters so a real fit needs more than 1 EM step to settle.
    rng = np.random.default_rng(0)
    features = rng.normal(0, 1, (900, 3))
    features[:300, 0] += 3
    features[300:600, 0] -= 3

    model = hmm_mod._fit_hmm_core(features, n_states=3)

    assert model is not None
    assert model.converged_ is False
    captured = capsys.readouterr()
    assert "did not converge" in captured.out


def test_fit_hmm_core_converged_true_with_ample_n_iter(monkeypatch):
    import models.hmm as hmm_mod

    monkeypatch.setitem(hmm_mod.CFG["hmm"], "n_init", 2)

    rng = np.random.default_rng(0)
    features = rng.normal(0, 1, (900, 3))
    features[:300, 0] += 3
    features[300:600, 0] -= 3

    model = hmm_mod._fit_hmm_core(features, n_states=3)

    assert model is not None
    assert model.converged_ is True


def test_label_states_uses_feature_cols_when_columns_reordered():
    # means_ columns are in order [mean_corr, spy_vol, spy_return] here -
    # the opposite of the assumed default [spy_return, spy_vol, mean_corr].
    # Positional fallback (index 0/1) would rank by mean_corr - 0.5*spy_vol,
    # giving the wrong answer. Name-based lookup should get it right regardless.
    model = MagicMock()
    model.means_ = np.array([
        [0.6, 0.03, -0.002],
        [0.4, 0.01, 0.002],
        [0.5, 0.02, 0.0],
        [0.3, 0.008, -0.001],
    ])
    feature_cols = ["mean_corr", "spy_vol", "spy_return"]

    labels = label_states(model, feature_cols=feature_cols)

    return_idx = feature_cols.index("spy_return")
    vol_idx = feature_cols.index("spy_vol")
    expected_ranking = np.argsort(model.means_[:, return_idx] - 0.5 * model.means_[:, vol_idx])

    assert labels[expected_ranking[0]] == "Crash"
    assert labels[expected_ranking[3]] == "Bull"


def test_label_states_skew_kurt_variant_ranks_by_return_only():
    # skew_kurt ablation variant has columns [spy_return, spy_skew, spy_kurt] -
    # no spy_vol at all. Should rank purely by return, NOT silently treat
    # spy_skew as if it were volatility (that was the bug).
    model = MagicMock()
    model.means_ = np.array([
        [-0.002, 0.5, 3.0],
        [0.002, -0.3, 4.0],
        [0.0005, 0.1, 3.5],
        [-0.0008, 0.2, 5.0],
    ])
    feature_cols = ["spy_return", "spy_skew", "spy_kurt"]

    labels = label_states(model, feature_cols=feature_cols)

    ranking_by_return_only = np.argsort(model.means_[:, 0])
    assert labels[ranking_by_return_only[0]] == "Crash"
    assert labels[ranking_by_return_only[3]] == "Bull"


def test_label_states_no_feature_cols_falls_back_to_positional():
    # Backward compatibility: existing callers that don't pass feature_cols
    # keep the original [index 0, index 1] behavior.
    model = MagicMock()
    model.means_ = np.array([
        [-0.002, 0.03],
        [0.002, 0.01],
        [0.0, 0.02],
        [-0.001, 0.008],
    ])
    labels = label_states(model)
    ranking = np.argsort(model.means_[:, 0] - 0.5 * model.means_[:, 1])
    assert labels[ranking[0]] == "Crash"
    assert labels[ranking[3]] == "Bull"


def test_block_bootstrap_indices_reused_gives_paired_consistency():
    # Calling block_bootstrap_indices ONCE and applying the result to two
    # different series (paired resampling) should select the same positions
    # from both - this is what makes a paired block bootstrap valid.
    rng = np.random.default_rng(7)
    indices = block_bootstrap_indices(n=100, block_length=10, rng=rng)

    series_a = np.arange(100)
    series_b = np.arange(100) * -1

    sample_a = series_a[indices]
    sample_b = series_b[indices]

    assert np.array_equal(sample_a, -sample_b)


def test_block_bootstrap_indices_matches_block_resample():
    # block_resample() should produce values consistent with directly
    # indexing via block_bootstrap_indices with the same seed.
    returns = pd.Series(np.arange(200, dtype=float))

    rng1 = np.random.default_rng(99)
    resampled = block_resample(returns, block_length=15, rng=rng1)

    rng2 = np.random.default_rng(99)
    indices = block_bootstrap_indices(n=200, block_length=15, rng=rng2)
    expected = returns.values[indices]

    assert np.array_equal(resampled.values, expected)

def test_select_n_states_default_returns_dataframe_only():
    rng = np.random.default_rng(0)
    features = rng.normal(0, 1, (300, 3))
    result = select_n_states(features, candidate_states=[2, 3])
    assert isinstance(result, pd.DataFrame)


def test_select_n_states_return_models_gives_matching_models():
    rng = np.random.default_rng(0)
    features = rng.normal(0, 1, (300, 3))
    scores_df, models = select_n_states(features, candidate_states=[2, 3], return_models=True)

    assert isinstance(scores_df, pd.DataFrame)
    assert set(models.keys()) == set(scores_df.index)
    for n, model in models.items():
        assert model.n_components == n


def test_select_n_states_returned_model_matches_a_fresh_refit():
    # The whole point of return_models: the returned model for the winning
    # n_states should be usable directly, with no need to refit - confirm it's
    # identical to what a fresh fit_hmm_with_n call would produce.
    from models.hmm import fit_hmm_with_n

    rng = np.random.default_rng(0)
    features = rng.normal(0, 1, (300, 3))
    scores_df, models = select_n_states(features, candidate_states=[2, 3], return_models=True)
    best_n = scores_df["bic"].idxmin()

    reused_model = models[best_n]
    fresh_model, _ = fit_hmm_with_n(features, best_n)

    assert np.isclose(reused_model.score(features), fresh_model.score(features))
    assert np.allclose(reused_model.means_, fresh_model.means_)