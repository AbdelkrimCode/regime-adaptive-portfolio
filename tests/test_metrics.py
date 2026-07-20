import numpy as np
import pandas as pd
import pytest
from backtest.metrics import (
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    max_drawdown,
    calmar_ratio,
    average_turnover,
    compute_turnover,
    compute_all,
)

TRADING_DAYS = 252




def test_annualized_return_flat():
    equity = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0])
    assert annualized_return(equity) == pytest.approx(0.0, abs=1e-6)


def test_annualized_return_known():
    daily = 0.001
    n = TRADING_DAYS
    equity = pd.Series([(1 + daily) ** i for i in range(n + 1)])
    expected = (equity.iloc[-1] / equity.iloc[0]) ** (TRADING_DAYS / (len(equity) - 1)) - 1
    assert annualized_return(equity) == pytest.approx(expected, rel=1e-4)




def test_annualized_volatility_zero():
    returns = pd.Series([0.0] * 100)
    assert annualized_volatility(returns) == pytest.approx(0.0, abs=1e-6)


def test_annualized_volatility_known():
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 0.01, 1000))
    expected = returns.std() * np.sqrt(TRADING_DAYS)
    assert annualized_volatility(returns) == pytest.approx(expected, rel=1e-6)




def test_sharpe_ratio_zero_excess():

    rf_daily = 0.0001
    returns = pd.Series([rf_daily] * 252)
    rf = pd.Series([rf_daily] * 252)
    assert sharpe_ratio(returns, rf=rf) == pytest.approx(0.0, abs=1e-6)


def test_sharpe_ratio_no_rf():

    returns = pd.Series([0.001] * 252)
    result = sharpe_ratio(returns, rf=None)
    assert result > 0




def test_max_drawdown_no_drawdown():
    equity = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4])
    assert max_drawdown(equity) == pytest.approx(0.0, abs=1e-6)


def test_max_drawdown_known():

    equity = pd.Series([1.0, 2.0, 1.0, 1.5])
    assert max_drawdown(equity) == pytest.approx(-0.5, rel=1e-6)


def test_max_drawdown_negative():
    result = max_drawdown(pd.Series([1.0, 0.8, 0.9, 0.7, 0.95]))
    assert result < 0




def test_average_turnover_no_change_after_entry():

    weights = pd.DataFrame(
        {"A": [0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5]}
    )
    turnover = compute_turnover(weights)
    assert turnover.iloc[0] == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(turnover.iloc[1:].values, 0.0, atol=1e-6)


def test_average_turnover_known():

    weights = pd.DataFrame(
        {"A": [0.5, 0.3, 0.3], "B": [0.5, 0.7, 0.7]}
    )
    result = average_turnover(weights)
    assert result == pytest.approx((1.0 + 0.4 + 0.0) / 3, abs=1e-3)


def test_compute_turnover_first_day_reflects_entry_from_cash():

    weights = pd.DataFrame(
        {"A": [1.0, 1.0], "B": [0.0, 0.0]}
    )
    turnover = compute_turnover(weights)
    assert turnover.iloc[0] == pytest.approx(1.0, abs=1e-6), (
        "Entering a 100% position from cash should cost turnover of 1.0, not 0.0"
    )




def test_compute_all_keys():
    returns = pd.Series(np.random.normal(0.0005, 0.01, 500))
    equity = (1 + returns).cumprod()
    result = compute_all(returns, equity)
    expected_keys = {
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "calmar_ratio",
    }
    assert set(result.keys()) == expected_keys


def test_compute_all_types():
    returns = pd.Series(np.random.normal(0.0005, 0.01, 500))
    equity = (1 + returns).cumprod()
    result = compute_all(returns, equity)
    for v in result.values():
        assert isinstance(v, float)


def test_compute_all_max_drawdown_negative():
    np.random.seed(0)
    returns = pd.Series(np.random.normal(0, 0.02, 500))
    equity = (1 + returns).cumprod()
    result = compute_all(returns, equity)
    assert result["max_drawdown"] <= 0