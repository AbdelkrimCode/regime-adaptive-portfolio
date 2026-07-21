import numpy as np
import pandas as pd
import pytest
from data.process import compute_returns, compute_features
from models.hmm import walk_forward_regimes
from backtest.engine import simulate
from optimization.switcher import compute_weights


N_ASSETS = 4
ASSETS = ["SPY", "TLT", "GLD", "EFA"]
N_DAYS = 600

def make_synthetic_prices() -> pd.DataFrame:
    np.random.seed(67)
    dates = pd.date_range("2018-01-01", periods=N_DAYS, freq="B")
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.01, (N_DAYS, N_ASSETS)), axis=0)),
        index=dates,
        columns=ASSETS
    )
    return prices


def test_pipeline_runs_end_to_end():
    prices = make_synthetic_prices()
    returns = compute_returns(prices)
    features = compute_features(returns)

    assert features.shape[1] == 3
    assert len(features) > 0


def test_walk_forward_produces_valid_regimes():
    prices = make_synthetic_prices()
    returns = compute_returns(prices)
    features = compute_features(returns)

    regimes = walk_forward_regimes(features)

    assert len(regimes) > 0
    assert "regime" in regimes.columns
    assert regimes["regime"].notna().all()
    valid_regimes = {"Bull", "Bear", "Sideways", "Crash"}
    assert set(regimes["regime"].unique()).issubset(valid_regimes)


def test_compute_weights_from_regimes():
    prices = make_synthetic_prices()
    returns = compute_returns(prices)
    features = compute_features(returns)

    regimes = walk_forward_regimes(features)

    asset_returns = returns[ASSETS]
    common = regimes.index.intersection(asset_returns.index)
    weights = compute_weights(regimes.loc[common], asset_returns.loc[common])

    assert weights.shape[1] == N_ASSETS
    assert not weights.dropna().empty


def test_simulate_produces_valid_equity():
    prices = make_synthetic_prices()
    returns = compute_returns(prices)
    features = compute_features(returns)

    regimes = walk_forward_regimes(features)

    asset_returns = returns[ASSETS]
    common = regimes.index.intersection(asset_returns.index)
    weights = compute_weights(regimes.loc[common], asset_returns.loc[common])

    result = simulate(weights, asset_returns.loc[common])

    assert "portfolio_return" in result.columns
    assert "equity" in result.columns
    assert result["equity"].iloc[-1] > 0
    assert not result["portfolio_return"].isna().all()