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
    # 2 states, 3 features.
    assert count_params(2, 3) == 21


def test_count_params_four_states():
    # 4 states, 3 features.
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
    ranking = np.argsort(model.means_[:, 1])

    assert labels[ranking[0]] == "Bull"
    assert labels[ranking[3]] == "Crash"


def test_label_states_skew_kurt_variant_ranks_by_return_only():

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

    model = MagicMock()
    model.means_ = np.array([
        [-0.002, 0.03],
        [0.002, 0.01],
        [0.0, 0.02],
        [-0.001, 0.008],
    ])
    labels = label_states(model)
    ranking = np.argsort(model.means_[:, 1])
    assert labels[ranking[0]] == "Bull"
    assert labels[ranking[3]] == "Crash"


def test_block_bootstrap_indices_reused_gives_paired_consistency():

    rng = np.random.default_rng(7)
    indices = block_bootstrap_indices(n=100, block_length=10, rng=rng)

    series_a = np.arange(100)
    series_b = np.arange(100) * -1

    sample_a = series_a[indices]
    sample_b = series_b[indices]

    assert np.array_equal(sample_a, -sample_b)


def test_block_bootstrap_indices_matches_block_resample():

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

    from models.hmm import fit_hmm_with_n

    rng = np.random.default_rng(0)
    features = rng.normal(0, 1, (300, 3))
    scores_df, models = select_n_states(features, candidate_states=[2, 3], return_models=True)
    best_n = scores_df["bic"].idxmin()

    reused_model = models[best_n]
    fresh_model, _ = fit_hmm_with_n(features, best_n)

    assert np.isclose(reused_model.score(features), fresh_model.score(features))
    assert np.allclose(reused_model.means_, fresh_model.means_)

def _make_synthetic_features_file(path, n=300, n_features=3):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        rng.normal(0, 1, (n, n_features)),
        index=dates,
        columns=["spy_return", "spy_vol", "mean_corr"][:n_features],
    )
    df.index.name = "date"
    df.to_parquet(path)


def test_run_static_model_reuses_cache_when_config_unchanged(tmp_path, monkeypatch):
    import models.hmm as hmm_mod

    features_path = tmp_path / "features.parquet"
    model_path = tmp_path / "hmm_model.joblib"
    regimes_path = tmp_path / "regimes.parquet"
    _make_synthetic_features_file(features_path)

    monkeypatch.setitem(hmm_mod.CFG["paths"], "features", str(features_path))
    monkeypatch.setitem(hmm_mod.CFG["paths"], "model", str(model_path))
    monkeypatch.setitem(hmm_mod.CFG["paths"], "regimes", str(regimes_path))
    monkeypatch.setitem(hmm_mod.CFG["hmm"], "n_states", 3)
    monkeypatch.setitem(hmm_mod.CFG["hmm"], "n_init", 2)  # keep the test fast

    fit_calls = {"count": 0}
    real_fit_hmm = hmm_mod.fit_hmm

    def counting_fit_hmm(features):
        fit_calls["count"] += 1
        return real_fit_hmm(features)

    monkeypatch.setattr(hmm_mod, "fit_hmm", counting_fit_hmm)

    hmm_mod.run(retrain=False, walk_forward=False)
    assert fit_calls["count"] == 1, "First call with no cache should fit once"

    hmm_mod.run(retrain=False, walk_forward=False)
    assert fit_calls["count"] == 1, "Second call with unchanged config should reuse the cache, not refit"


def test_run_static_model_refits_when_n_states_changes(tmp_path, monkeypatch):
    import models.hmm as hmm_mod

    features_path = tmp_path / "features.parquet"
    model_path = tmp_path / "hmm_model.joblib"
    regimes_path = tmp_path / "regimes.parquet"
    _make_synthetic_features_file(features_path)

    monkeypatch.setitem(hmm_mod.CFG["paths"], "features", str(features_path))
    monkeypatch.setitem(hmm_mod.CFG["paths"], "model", str(model_path))
    monkeypatch.setitem(hmm_mod.CFG["paths"], "regimes", str(regimes_path))
    monkeypatch.setitem(hmm_mod.CFG["hmm"], "n_states", 3)
    monkeypatch.setitem(hmm_mod.CFG["hmm"], "n_init", 2)

    fit_calls = {"count": 0}
    real_fit_hmm = hmm_mod.fit_hmm

    def counting_fit_hmm(features):
        fit_calls["count"] += 1
        return real_fit_hmm(features)

    monkeypatch.setattr(hmm_mod, "fit_hmm", counting_fit_hmm)

    hmm_mod.run(retrain=False, walk_forward=False)
    assert fit_calls["count"] == 1

    # Change n_states without passing retrain=True - old code would have
    # silently reused the stale cache here; fixed code must refit instead.
    monkeypatch.setitem(hmm_mod.CFG["hmm"], "n_states", 4)
    hmm_mod.run(retrain=False, walk_forward=False)
    assert fit_calls["count"] == 2, (
        "Changing n_states without retrain=True must trigger a refit, "
        "not silently reuse the model cached under the old n_states"
    )


def test_run_static_model_refits_when_feature_columns_change(tmp_path, monkeypatch):
    import models.hmm as hmm_mod

    features_path = tmp_path / "features.parquet"
    model_path = tmp_path / "hmm_model.joblib"
    regimes_path = tmp_path / "regimes.parquet"
    _make_synthetic_features_file(features_path, n_features=3)

    monkeypatch.setitem(hmm_mod.CFG["paths"], "features", str(features_path))
    monkeypatch.setitem(hmm_mod.CFG["paths"], "model", str(model_path))
    monkeypatch.setitem(hmm_mod.CFG["paths"], "regimes", str(regimes_path))
    monkeypatch.setitem(hmm_mod.CFG["hmm"], "n_states", 3)
    monkeypatch.setitem(hmm_mod.CFG["hmm"], "n_init", 2)

    fit_calls = {"count": 0}
    real_fit_hmm = hmm_mod.fit_hmm

    def counting_fit_hmm(features):
        fit_calls["count"] += 1
        return real_fit_hmm(features)

    monkeypatch.setattr(hmm_mod, "fit_hmm", counting_fit_hmm)

    hmm_mod.run(retrain=False, walk_forward=False)
    assert fit_calls["count"] == 1

    # Same n_states, but a DIFFERENT feature set (e.g. the skew_kurt ablation
    # variant) - must not silently reuse a model fit on different columns.
    _make_synthetic_features_file(features_path, n_features=2)
    hmm_mod.run(retrain=False, walk_forward=False)
    assert fit_calls["count"] == 2, (
        "Changing feature columns without retrain=True must trigger a refit"
    )

from sklearn.preprocessing import StandardScaler
from models.hmm import _fit_hmm_core, forward_filter


def _fitted_model_and_data():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, (400, 3))
    x[:150, 0] += 0.8
    x[150:280, 0] -= 0.8
    xs = StandardScaler().fit_transform(x)
    return _fit_hmm_core(xs, n_states=3), xs


def test_forward_filter_matches_hmmlearn_on_last_observation():
    # For the last point of a prefix, smoothed = filtered, so hmmlearn is an independent reference.
    model, xs = _fitted_model_and_data()
    _, alpha = forward_filter(model, xs)
    for t in [0, 10, 149, 150, 399]:
        expected = model.predict_proba(xs[: t + 1])[-1]
        np.testing.assert_allclose(alpha[t], expected, atol=1e-8)


def test_forward_filter_survives_extreme_observation():
    model, xs = _fitted_model_and_data()
    extreme = np.vstack([xs, np.full((1, 3), 1e3)])
    _, alpha = forward_filter(model, extreme)
    assert np.all(np.isfinite(alpha))
    np.testing.assert_allclose(alpha.sum(axis=1), 1.0)
    # old code returned uniform (max = 1/3) here
    assert alpha[-1].max() > 0.99

import models.hmm as hmm_mod


def _two_regime_frame(n_train=300, n_test=40, seed=0):
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    ret = np.where(np.arange(n) % 200 < 100, 1.0, -1.0) + rng.normal(0, 0.5, n)
    vol = np.where(np.arange(n) % 200 < 100, -1.0, 1.0) + rng.normal(0, 0.5, n)
    idx = pd.bdate_range("2010-01-01", periods=n)
    df = pd.DataFrame({"spy_return": ret, "spy_vol": vol}, index=idx)
    return df.iloc[:n_train], df.iloc[n_train:]


def test_fit_fold_carries_filter_state_from_training(monkeypatch):
    train_df, test_df = _two_regime_frame()
    cols = ["spy_return", "spy_vol"]

    scaler = StandardScaler().fit(train_df[cols].values)
    train_scaled = scaler.transform(train_df[cols].values)
    model = hmm_mod._fit_hmm_core(train_scaled, n_states=2)

    monkeypatch.setattr(hmm_mod, "_fit_hmm_core", lambda xs, n: model)

    period_df, _ = hmm_mod._fit_fold(train_df, test_df, train_df.index[-1], test_df.index[-1], cols)

    test_scaled = scaler.transform(test_df[cols].values)
    _, carried = hmm_mod.forward_filter(model, np.vstack([train_scaled, test_scaled]))
    _, restarted = hmm_mod.forward_filter(model, test_scaled)

    labels = hmm_mod.label_states(model, feature_cols=cols)
    p_cols = [f"p_{labels[i].lower()}" for i in range(2)]
    got = period_df[p_cols].values

    np.testing.assert_allclose(got, carried[-len(test_df):], atol=1e-10)
    assert not np.allclose(got[0], restarted[0], atol=1e-3)

def test_label_states_ranks_by_volatility_not_return():
    model = MagicMock()
    # state 0: turbulent rebound, highest return AND highest vol
    model.means_ = np.array([
        [ 2.0,  2.5, 1.0],
        [ 0.5, -1.0, -0.5],
        [ 0.0,  0.0, 0.0],
        [-1.0,  1.0, 0.5],
    ])
    labels = label_states(model, feature_cols=["spy_return", "spy_vol", "mean_corr"])
    assert labels == {1: "Bull", 2: "Sideways", 3: "Bear", 0: "Crash"}


def test_fit_fold_uses_fixed_n_states(monkeypatch):
    from config import load_config
    train_df, test_df = _two_regime_frame()
    cols = ["spy_return", "spy_vol"]
    xs = StandardScaler().fit_transform(train_df[cols].values)
    model = hmm_mod._fit_hmm_core(xs, n_states=2)

    seen = {}
    def fake_core(x, n):
        seen["n"] = n
        return model
    monkeypatch.setattr(hmm_mod, "_fit_hmm_core", fake_core)
    monkeypatch.setattr(hmm_mod, "select_n_states",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("BIC selection should not run")))

    hmm_mod._fit_fold(train_df, test_df, train_df.index[-1], test_df.index[-1], cols)
    assert seen["n"] == load_config()["hmm"]["n_states"]

from collections import deque
from types import SimpleNamespace


def _monitor(history, it, n_iter=100, tol=1e-2):
    return SimpleNamespace(history=deque(history, maxlen=2), iter=it, n_iter=n_iter, tol=tol)


def test_em_converged_rejects_likelihood_decrease():
    assert hmm_mod._em_converged(_monitor([-100.0, -150.0], it=10)) is False


def test_em_converged_rejects_hitting_iteration_limit():
    assert hmm_mod._em_converged(_monitor([-100.0, -100.001], it=100)) is False


def test_em_converged_accepts_small_change_before_limit():
    assert hmm_mod._em_converged(_monitor([-100.0, -99.995], it=10)) is True

def _daily_features(n: int = 900) -> pd.DataFrame:
    dates = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame(np.random.default_rng(0).normal(size=(n, 3)), index=dates,
                        columns=["spy_return", "spy_vol", "mean_corr"])


def test_build_folds_tile_test_period_without_overlap():
    from models.hmm import _build_folds
    from config import load_config

    df = _daily_features()
    folds = _build_folds(df)

    assert len(folds[0][1]) >= load_config()["hmm"]["min_train_days"]
    for _, train, test, retrain_date, _ in folds:
        assert train.index[-1] == retrain_date
        assert test.index[0] > retrain_date
    test_dates = pd.DatetimeIndex(np.concatenate([f[2].index.values for f in folds]))
    assert test_dates.is_unique and test_dates.is_monotonic_increasing
    assert test_dates[-1] == df.index[-1]
    first = df.index.get_loc(folds[0][2].index[0])
    assert len(test_dates) == len(df) - first, "no gaps between consecutive test windows"


def test_audit_walk_forward_uses_the_same_folds(monkeypatch, tmp_path):
    import models.hmm as hmm_mod

    df = _daily_features()
    monkeypatch.setattr(hmm_mod, "load_features", lambda: df)

    audit = hmm_mod.audit_walk_forward(output_path=str(tmp_path / "audit.csv"))
    folds = hmm_mod._build_folds(df)

    assert list(audit["fold"]) == [f[0] for f in folds]
    assert list(audit["test_days"]) == [len(f[2]) for f in folds]
    assert not audit["leakage"].any()


class _FlakyHMM:
    """Stand-in for GaussianHMM whose fit raises a chosen error on given restarts."""
    errors: dict = {}

    def __init__(self, n_components, covariance_type, n_iter, random_state):
        self.seed = random_state
        self.n_components = n_components

    def fit(self, X):
        from config import load_config
        exc = self.errors.get(self.seed - load_config()["hmm"]["random_state"])
        if exc is not None:
            raise exc
        self.monitor_ = MagicMock(history=[0.0, 0.0], iter=2, n_iter=10, tol=1e-2)
        return self

    def score(self, X):
        return float(self.seed)


def test_fit_hmm_core_skips_degenerate_restarts_and_logs_them(monkeypatch, capsys):
    import models.hmm as hmm_mod

    _FlakyHMM.errors = {0: ValueError("covars must be positive-definite"),
                        1: np.linalg.LinAlgError("singular matrix")}
    monkeypatch.setattr(hmm_mod, "GaussianHMM", _FlakyHMM)

    model = hmm_mod._fit_hmm_core(np.zeros((10, 3)), 4)

    assert model is not None
    assert model.n_failed_restarts_ == 2
    out = capsys.readouterr().out
    assert "ValueError: covars must be positive-definite" in out
    assert "LinAlgError: singular matrix" in out


def test_fit_hmm_core_does_not_swallow_programming_errors(monkeypatch):
    import models.hmm as hmm_mod

    _FlakyHMM.errors = {0: TypeError("bad argument")}
    monkeypatch.setattr(hmm_mod, "GaussianHMM", _FlakyHMM)

    with pytest.raises(TypeError):
        hmm_mod._fit_hmm_core(np.zeros((10, 3)), 4)
