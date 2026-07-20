import numpy as np
import pandas as pd
import pytest
from optimization.mean_var import max_sharpe
from optimization.risk_parity import risk_parity
from optimization.min_variance import min_variance
from optimization.crash import crash_weights
from optimization.switcher import compute_weights

ASSETS = ["SPY", "TLT", "GLD", "EFA", "IEF", "QQQ", "LQD", "VNQ"]
N_ASSETS = len(ASSETS)
MAX_POSITION = 0.60


def make_returns(n: int = 500) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    data = np.random.normal(0.0003, 0.01, (n, N_ASSETS))
    return pd.DataFrame(data, index=dates, columns=ASSETS)

# ——— max_sharpe ---

def test_max_sharpe_correct_length():
    assert len(max_sharpe(make_returns())) == N_ASSETS

def test_max_sharpe_weights_sum_to_one():
    assert np.sum(max_sharpe(make_returns())) == pytest.approx(1.0, abs=1e-6)

def test_max_sharpe_non_negative():
    assert np.all(max_sharpe(make_returns()) >= -1e-8)

def test_max_sharpe_no_nan():
    assert not np.any(np.isnan(max_sharpe(make_returns())))

def test_max_sharpe_fallback_on_zero_variance():
    dates = pd.date_range("2015-01-01", periods=300, freq="B")
    returns = pd.DataFrame(0.0, index=dates, columns=ASSETS)
    w = max_sharpe(returns)
    assert len(w) == N_ASSETS
    assert np.sum(w) == pytest.approx(1.0, abs=1e-6)

    # --- risk_parity ---

def test_risk_parity_correct_length():
    assert len(risk_parity(make_returns())) == N_ASSETS

def test_risk_parity_weights_sum_to_one():
    assert np.sum(risk_parity(make_returns())) == pytest.approx(1.0, abs=1e-6)

def test_risk_parity_non_negative():
    assert np.all(risk_parity(make_returns()) >= -1e-8)

def test_risk_parity_no_nan():
    assert not np.any(np.isnan(risk_parity(make_returns())))

def test_risk_parity_respects_floor():
    assert np.all(risk_parity(make_returns()) >= 0.01 - 1e-6)

def test_risk_parity_respects_cap():
    assert np.all(risk_parity(make_returns()) <= MAX_POSITION + 1e-6)
    
def test_risk_parity_achieves_equal_risk_contribution():
    returns = make_returns()
    sigma = np.cov(returns.values, rowvar=False) * 252

    w = risk_parity(returns)
    risk_contrib = w * (sigma @ w)
    risk_contrib_pct = risk_contrib / risk_contrib.sum()

    spread = risk_contrib_pct.max() - risk_contrib_pct.min()
    assert spread < 0.02, (
        f"Risk contributions should be approximately equal, "
        f"got spread of {spread:.4f} across assets: {risk_contrib_pct}"
    )

# --- min_variance ---

def test_min_variance_correct_length():
    assert len(min_variance(make_returns())) == N_ASSETS

def test_min_variance_weights_sum_to_one():
    assert np.sum(min_variance(make_returns())) == pytest.approx(1.0, abs=1e-6)

def test_min_variance_non_negative():
    assert np.all(min_variance(make_returns()) >= -1e-8)

def test_min_variance_no_nan():
    assert not np.any(np.isnan(min_variance(make_returns())))

def test_min_variance_respects_cap():
    assert np.all(min_variance(make_returns()) <= MAX_POSITION + 1e-6)

# --- crash_weights ---

def test_crash_weights_correct_length():
    assert len(crash_weights(make_returns())) == N_ASSETS

def test_crash_weights_sum_to_one():
    assert np.sum(crash_weights(make_returns())) == pytest.approx(1.0, abs=1e-6)

def test_crash_weights_no_nan():
    assert not np.any(np.isnan(crash_weights(make_returns())))

def test_crash_weights_only_safe_havens_nonzero():
    safe_havens = {"IEF", "TLT", "GLD"}
    w = crash_weights(make_returns())
    for i, asset in enumerate(ASSETS):
        if asset not in safe_havens:
            assert w[i] == pytest.approx(0.0, abs=1e-8), f"{asset} should be zero in crash"

def test_crash_weights_fallback_when_no_safe_havens():
    dates = pd.date_range("2015-01-01", periods=300, freq="B")
    returns = pd.DataFrame(
        np.random.normal(0, 0.01, (300, 3)),
        index=dates,
        columns=["SPY", "QQQ", "EFA"]
    )
    w = crash_weights(returns)
    assert np.sum(w) == pytest.approx(1.0, abs=1e-6)
    assert len(w) == 3

    
# --- compute_weights ---

def make_regimes(returns: pd.DataFrame, regime: str) -> pd.DataFrame:
    posteriors = {"p_bull": 0.0, "p_bear": 0.0, "p_sideways": 0.0, "p_crash": 0.0}
    posteriors[f"p_{regime.lower()}"] = 1.0
    df = pd.DataFrame(
        [{"regime": regime, "is_retrain_date": False, **posteriors}] * len(returns),
        index=returns.index
    )
    return df

def test_compute_weights_shape():
    returns = make_returns()
    w = compute_weights(make_regimes(returns, "Bull"), returns)
    assert w.shape == (len(returns), N_ASSETS)

def test_compute_weights_sum_to_one():
    returns = make_returns()
    w = compute_weights(make_regimes(returns, "Bull"), returns).dropna()
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-6)

def test_compute_weights_equal_weight_when_history_too_short():
    returns = make_returns(n=500)
    short_returns = returns.iloc[:50]
    w = compute_weights(make_regimes(short_returns, "Bull"), short_returns).dropna()
    assert np.allclose(w.values, 1.0 / N_ASSETS, atol=1e-6)