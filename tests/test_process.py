import numpy as np
import pandas as pd
import pytest

from config import load_config
from data.process import compute_returns, compute_features, compute_skew_kurt_features


def test_vol_and_corr_window_defaults_come_from_config():
    import data.process as process_mod

    cfg = load_config()
    assert process_mod.VOL_WINDOW == cfg["features"]["vol_window"]
    assert process_mod.CORR_WINDOW == cfg["features"]["corr_window"]


def make_prices(n=200, n_assets=3):
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, (n, n_assets)), axis=0)),
        index=dates,
        columns=["SPY", "TLT", "GLD"][:n_assets],
    )
    return prices


def test_compute_returns_is_log_returns():
    prices = make_prices(n=5, n_assets=1)
    returns = compute_returns(prices)
    expected = np.log(prices / prices.shift(1)).dropna()
    pd.testing.assert_frame_equal(returns, expected)


def test_compute_features_columns_and_no_nans():
    prices = make_prices(n=200, n_assets=3)
    returns = compute_returns(prices)
    features = compute_features(returns)

    assert list(features.columns) == ["spy_return", "spy_vol", "mean_corr"]
    assert not features.isna().any().any()
    assert len(features) > 0


def test_compute_features_respects_custom_windows():
    prices = make_prices(n=200, n_assets=3)
    returns = compute_returns(prices)

    short = compute_features(returns, vol_window=5, corr_window=10)
    long = compute_features(returns, vol_window=42, corr_window=100)

    # Different windows should start producing valid rows at different points
    # (longer windows need more history before the first non-NaN row).
    assert len(short) > len(long)


def test_compute_skew_kurt_features_columns():
    prices = make_prices(n=200, n_assets=1)
    returns = compute_returns(prices)
    features = compute_skew_kurt_features(returns, window=21)

    assert list(features.columns) == ["spy_return", "spy_skew", "spy_kurt"]
    assert "spy_vol" not in features.columns