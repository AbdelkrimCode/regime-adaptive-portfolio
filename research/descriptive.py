"""Descriptive detection diagnostics (exploratory, not part of the pre-registered tests).

They describe how the strategy's causal regimes differ from the hindsight regimes:
how often they agree, how they are confused, and how late the causal filter is.
"""
import numpy as np
import pandas as pd

from optimization.switcher import LABELS
from research.oracles import LABELS_BY_VOL


def agreement_rate(real_labels: pd.Series, hindsight_labels: pd.Series) -> float:
    """Share of days on which the causal and hindsight regimes agree."""
    real, hind = real_labels.align(hindsight_labels, join="inner")
    return float((real == hind).mean())


def confusion(real_labels: pd.Series, hindsight_labels: pd.Series) -> pd.DataFrame:
    """Row-normalized confusion matrix: rows hindsight regime, columns causal regime, in %."""
    real, hind = real_labels.align(hindsight_labels, join="inner")
    table = pd.crosstab(hind, real).reindex(index=LABELS_BY_VOL, columns=LABELS_BY_VOL, fill_value=0)
    return table.div(table.sum(axis=1).replace(0, np.nan), axis=0) * 100


def detection_delays(real_labels: pd.Series, hindsight_labels: pd.Series, horizon: int = 63) -> pd.Series:
    """Trading days until the causal filter first shows each new hindsight regime.

    One value per hindsight regime change; NaN when the causal filter never shows
    that regime within `horizon` days.
    """
    real, hind = real_labels.align(hindsight_labels, join="inner")
    real_values = real.values
    changes = np.flatnonzero(hind.values[1:] != hind.values[:-1]) + 1
    delays = {}
    for i in changes:
        target = hind.values[i]
        window = real_values[i:i + horizon + 1]
        hits = np.flatnonzero(window == target)
        delays[hind.index[i]] = float(hits[0]) if len(hits) else np.nan
    return pd.Series(delays, name="delay_days", dtype=float)


def delay_summary(delays: pd.Series) -> dict:
    """Median, mean and share of regime changes the causal filter shows within 5 and 21 days."""
    found = delays.dropna()
    return {
        "n_changes": int(len(delays)),
        "never_within_horizon": int(delays.isna().sum()),
        "median_days": float(found.median()) if len(found) else np.nan,
        "mean_days": float(found.mean()) if len(found) else np.nan,
        "share_within_5_days": float((delays <= 5).mean()),
        "share_within_21_days": float((delays <= 21).mean()),
    }
