import numpy as np
import pandas as pd
from unittest.mock import patch

import optimization.switcher as switcher_mod


ASSETS = ["SPY", "TLT", "GLD", "EFA", "IEF", "QQQ", "LQD", "VNQ"]


def make_returns(n: int = 200) -> pd.DataFrame:
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    rng = np.random.default_rng(1)
    data = rng.normal(0.0003, 0.01, (n, len(ASSETS)))
    return pd.DataFrame(data, index=dates, columns=ASSETS)


def make_single_day_regime(date, regime: str) -> pd.DataFrame:
    row = {
        "regime": regime,
        "p_bull": 1.0 if regime == "Bull" else 0.0,
        "p_bear": 1.0 if regime == "Bear" else 0.0,
        "p_sideways": 1.0 if regime == "Sideways" else 0.0,
        "p_crash": 1.0 if regime == "Crash" else 0.0,
        "is_retrain_date": True,
    }
    return pd.DataFrame([row], index=[date])


def test_concentration_guard_triggers_on_crash_regime_extreme_weights():
    returns = make_returns()
    date = returns.index[-1]
    regimes_df = make_single_day_regime(date, "Crash")

    extreme = np.zeros(len(ASSETS))
    extreme[ASSETS.index("IEF")] = 1.0  # 100% concentrated in one safe haven

    with patch.dict(switcher_mod.OPTIMIZER_MAP, {"Crash": lambda r: extreme}):
        weights = switcher_mod.compute_weights(regimes_df, returns)

    w = weights.loc[date].values
    assert np.max(w) < switcher_mod.CONCENTRATION_GUARD, (
        "Guard should have replaced the 100%-concentrated Crash allocation"
    )
    # Documents current behavior (see audit §2.6): the fallback is equal-weight
    # across the FULL 8-asset universe, including risky assets - not just
    # within the Crash regime's own safe-haven subset (IEF/TLT/GLD). This is a
    # known design tradeoff, not necessarily the ideal one; this test exists so
    # that changing it is a deliberate decision, not a silent behavior change.
    assert np.allclose(w, 1.0 / len(ASSETS), atol=1e-6)


def test_concentration_guard_triggers_on_sideways_regime_extreme_weights():
    returns = make_returns()
    date = returns.index[-1]
    regimes_df = make_single_day_regime(date, "Sideways")

    extreme = np.zeros(len(ASSETS))
    extreme[ASSETS.index("SPY")] = 1.0

    with patch.dict(switcher_mod.OPTIMIZER_MAP, {"Sideways": lambda r: extreme}):
        weights = switcher_mod.compute_weights(regimes_df, returns)

    w = weights.loc[date].values
    assert np.allclose(w, 1.0 / len(ASSETS), atol=1e-6)


def test_concentration_guard_triggers_on_bull_regime_extreme_weights(monkeypatch):
    returns = make_returns()
    date = returns.index[-1]
    regimes_df = make_single_day_regime(date, "Bull")

    extreme = np.zeros(len(ASSETS))
    extreme[ASSETS.index("QQQ")] = 1.0

    # Bull bypasses OPTIMIZER_MAP and calls max_sharpe directly (see switcher.py).
    monkeypatch.setattr(switcher_mod, "max_sharpe", lambda r, rf: extreme)

    weights = switcher_mod.compute_weights(regimes_df, returns)
    w = weights.loc[date].values
    assert np.allclose(w, 1.0 / len(ASSETS), atol=1e-6)


def test_concentration_guard_does_not_trigger_on_diversified_weights():
    returns = make_returns()
    date = returns.index[-1]
    regimes_df = make_single_day_regime(date, "Sideways")

    diversified = np.ones(len(ASSETS)) / len(ASSETS)

    with patch.dict(switcher_mod.OPTIMIZER_MAP, {"Sideways": lambda r: diversified}):
        weights = switcher_mod.compute_weights(regimes_df, returns)

    w = weights.loc[date].values
    assert np.allclose(w, diversified, atol=1e-6)