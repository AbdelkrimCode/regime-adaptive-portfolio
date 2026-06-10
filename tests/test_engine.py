import numpy as np
import pandas as pd
import pytest
from backtest.engine import simulate, align_data


def test_align_data_common_dates():
    idx1 = pd.date_range("2020-01-01", periods=5)
    idx2 = pd.date_range("2020-01-03", periods=5)
    weights = pd.DataFrame({"A": [0.5] * 5, "B": [0.5] * 5}, index=idx1)
    returns = pd.DataFrame({"A": [0.01] * 5, "B": [0.01] * 5}, index=idx2)
    w, r = align_data(weights, returns)
    assert len(w) == len(r)
    assert list(w.index) == list(r.index)


def test_align_data_identical_index():
    idx = pd.date_range("2020-01-01", periods=5)
    weights = pd.DataFrame({"A": [0.5] * 5, "B": [0.5] * 5}, index=idx)
    returns = pd.DataFrame({"A": [0.01] * 5, "B": [0.01] * 5}, index=idx)
    w, r = align_data(weights, returns)
    assert len(w) == 5


def make_inputs(n=10, n_assets=2, daily_return=0.001, weight=0.5):
    idx = pd.date_range("2020-01-01", periods=n)
    returns = pd.DataFrame(
        {f"A{i}": [daily_return] * n for i in range(n_assets)},
        index=idx
    )
    weights = pd.DataFrame(
        {f"A{i}": [weight] * n for i in range(n_assets)},
        index=idx
    )
    return weights, returns


def test_simulate_output_columns():
    weights, returns = make_inputs()
    result = simulate(weights, returns)
    assert "portfolio_return" in result.columns
    assert "equity" in result.columns


def test_simulate_equity_starts_near_one():
    weights, returns = make_inputs()
    result = simulate(weights, returns)
    assert result["equity"].iloc[0] == pytest.approx(1.0, rel=1e-3)


def test_simulate_equity_grows_with_positive_returns():
    weights, returns = make_inputs(n=100, daily_return=0.001)
    result = simulate(weights, returns)
    assert result["equity"].iloc[-1] > result["equity"].iloc[0]


def test_simulate_equity_shrinks_with_negative_returns():
    weights, returns = make_inputs(n=100, daily_return=-0.001)
    result = simulate(weights, returns)
    assert result["equity"].iloc[-1] < result["equity"].iloc[0]


def test_simulate_transaction_costs_reduce_returns():

    weights_stable, returns = make_inputs(n=50, daily_return=0.001)

    idx = pd.date_range("2020-01-01", periods=50)
    weights_volatile = pd.DataFrame(index=idx)
    weights_volatile["A0"] = [0.9 if i % 2 == 0 else 0.1 for i in range(50)]
    weights_volatile["A1"] = [0.1 if i % 2 == 0 else 0.9 for i in range(50)]

    result_stable = simulate(weights_stable, returns)
    result_volatile = simulate(weights_volatile, returns)

    assert result_stable["equity"].iloc[-1] > result_volatile["equity"].iloc[-1]


def test_simulate_weights_shifted_by_one():
    idx = pd.date_range("2020-01-01", periods=3)
    weights = pd.DataFrame({"A": [1.0, 0.0, 0.0], "B": [0.0, 1.0, 1.0]}, index=idx)
    returns = pd.DataFrame({"A": [0.05, 0.05, 0.05], "B": [-0.05, -0.05, -0.05]}, index=idx)
    result = simulate(weights, returns)
    assert result["portfolio_return"].iloc[0] == pytest.approx(0.05, abs=1e-3)


def test_simulate_constant_weights_produce_expected_return():
    idx = pd.date_range("2020-01-01", periods=5)
    weights = pd.DataFrame({"A": [1.0] * 5, "B": [0.0] * 5}, index=idx)
    returns = pd.DataFrame({"A": [0.01] * 5, "B": [0.0] * 5}, index=idx)
    result = simulate(weights, returns)
    for r in result["portfolio_return"]:
        assert r == pytest.approx(0.01, abs=1e-2)