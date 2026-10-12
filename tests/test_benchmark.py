import numpy as np
import pandas as pd
import pytest

from backtest.benchmark import month_end_flags, get_sixty_forty_equity


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