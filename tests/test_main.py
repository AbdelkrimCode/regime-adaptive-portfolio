import pandas as pd

from main import compute_regime_runs, compute_empirical_transition_matrix


def test_compute_regime_runs_basic():
    idx = pd.date_range("2020-01-01", periods=6)
    regimes_df = pd.DataFrame(
        {"regime": ["Bull", "Bull", "Bear", "Bear", "Bear", "Bull"]},
        index=idx,
    )
    runs_df = compute_regime_runs(regimes_df)

    assert list(runs_df["regime"]) == ["Bull", "Bear", "Bull"]
    assert list(runs_df["duration"]) == [2, 3, 1]


def test_compute_regime_runs_single_regime():
    idx = pd.date_range("2020-01-01", periods=4)
    regimes_df = pd.DataFrame({"regime": ["Sideways"] * 4}, index=idx)
    runs_df = compute_regime_runs(regimes_df)

    assert len(runs_df) == 1
    assert runs_df.iloc[0]["duration"] == 4


def test_compute_empirical_transition_matrix_known_case():
    # Bull -> Bear -> Bull -> Crash: two Bull exits, one to Bear and one to Crash.
    runs_df = pd.DataFrame({
        "regime": ["Bull", "Bear", "Bull", "Crash"],
        "duration": [5, 3, 2, 4],
    })
    labels = ["Bull", "Bear", "Sideways", "Crash"]

    trans_pct = compute_empirical_transition_matrix(runs_df, labels)

    # Bull exits twice: once to Bear, once to Crash -> 50/50
    assert trans_pct.loc["Bull", "Bear"] == 50.0
    assert trans_pct.loc["Bull", "Crash"] == 50.0
    assert trans_pct.loc["Bull", "Sideways"] == 0.0
    # Bear exits once: to Bull -> 100%
    assert trans_pct.loc["Bear", "Bull"] == 100.0
    # Rows should sum to 100 (or be all-NaN for regimes with no exits)
    for row in labels:
        row_sum = trans_pct.loc[row].sum()
        assert row_sum == 100.0 or pd.isna(trans_pct.loc[row]).all()


def test_compute_empirical_transition_matrix_no_exits_for_unvisited_regime():
    runs_df = pd.DataFrame({
        "regime": ["Bull", "Bear", "Bull"],
        "duration": [1, 1, 1],
    })
    labels = ["Bull", "Bear", "Sideways", "Crash"]

    trans_pct = compute_empirical_transition_matrix(runs_df, labels)

    # Sideways and Crash never appear - their rows should be all-NaN, not 0/0 errors
    assert pd.isna(trans_pct.loc["Sideways"]).all()
    assert pd.isna(trans_pct.loc["Crash"]).all()