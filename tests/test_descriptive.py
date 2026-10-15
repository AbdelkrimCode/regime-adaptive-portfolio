import numpy as np
import pandas as pd
import pytest

from research.descriptive import agreement_rate, confusion, delay_summary, detection_delays


def _series(values):
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values)))


def test_agreement_and_confusion():
    hind = _series(["Bull", "Bull", "Crash", "Crash"])
    real = _series(["Bull", "Crash", "Crash", "Bull"])

    assert agreement_rate(real, hind) == pytest.approx(0.5)
    table = confusion(real, hind)
    assert table.loc["Bull", "Bull"] == pytest.approx(50.0)
    assert table.loc["Crash", "Bull"] == pytest.approx(50.0)
    assert table.loc["Sideways"].isna().all()       # never seen in hindsight


def test_detection_delays_measure_lag_and_misses():
    hind = _series(["Bull"] * 5 + ["Crash"] * 5 + ["Bull"] * 5)
    real = _series(["Bull"] * 7 + ["Crash"] * 8)    # sees Crash 2 days late, never returns to Bull

    delays = detection_delays(real, hind, horizon=3)

    assert list(delays.values[:1]) == [2.0]
    assert np.isnan(delays.values[1])
    summary = delay_summary(delays)
    assert summary["n_changes"] == 2
    assert summary["never_within_horizon"] == 1
    assert summary["share_within_5_days"] == pytest.approx(0.5)
