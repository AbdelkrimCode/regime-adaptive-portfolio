import numpy as np
import pandas as pd
import pytest

from backtest.benchmark import month_end_flags, get_sixty_forty_equity, cost_sensitivity
from backtest.engine import simulate
from backtest.metrics import sharpe_ratio


def test_month_end_flags_marks_last_trading_day():
    idx = pd.bdate_range("2020-01-01", "2020-03-31")
    flags = month_end_flags(idx)
    assert list(idx[flags.values].date.astype(str)) == ["2020-01-31", "2020-02-28", "2020-03-31"]


def test_benchmark_holds_and_drifts_within_month():
    idx = pd.bdate_range("2020-01-01", "2020-03-31")
    rng = np.random.default_rng(0)
    cols = ["SPY", "TLT", "IEF"]
    returns = pd.DataFrame(rng.normal(0.0005, 0.01, (len(idx), 3)), index=idx, columns=cols)

    _, equity = get_sixty_forty_equity(returns, cost=0.0)

    feb = returns.loc["2020-02-03":"2020-02-28"]
    expected = 0.6 * (1 + feb["SPY"]).prod() + 0.4 * (1 + feb["IEF"]).prod()
    got = equity.loc["2020-02-28"] / equity.loc["2020-01-31"]
    assert got == pytest.approx(expected, rel=1e-12)

def test_cost_sensitivity_reprices_portfolio_and_benchmarks():
    idx = pd.bdate_range("2019-01-01", "2020-12-31")
    rng = np.random.default_rng(1)
    cols = ["SPY", "TLT", "IEF", "GLD"]
    returns = pd.DataFrame(rng.normal(0.0004, 0.01, (len(idx), 4)), index=idx, columns=cols)
    # Portfolio flips between two books every day, so costs bite hard.
    a, b = np.array([0.7, 0.1, 0.1, 0.1]), np.array([0.1, 0.1, 0.1, 0.7])
    weights = pd.DataFrame([a if i % 2 else b for i in range(len(idx))], index=idx, columns=cols)

    table = cost_sensitivity(weights, returns, costs_bp=(2, 10))

    port_10bp = simulate(weights, returns, cost=0.001)["portfolio_return"]
    assert table.loc["portfolio", "10 bp"] == pytest.approx(round(sharpe_ratio(port_10bp), 4))
    assert table.loc["portfolio", "10 bp"] < table.loc["portfolio", "2 bp"]
    assert table.loc["sixty_forty", "10 bp"] < table.loc["sixty_forty", "2 bp"]
    assert table.loc["spy", "10 bp"] == table.loc["spy", "2 bp"]
