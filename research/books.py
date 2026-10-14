"""Building blocks for the detection vs response study.

Under the weekly refresh rule the optimizer weights do not depend on the regime
posteriors, so they are computed once (optimizer_targets) and every cell of the
study only changes the posteriors or the regime-to-optimizer mapping (blend).
"""
import numpy as np
import pandas as pd

from backtest.engine import simulate
from optimization.switcher import (
    DEFAULT_MAPPING, LABELS, MIN_HISTORY, OPTIMIZERS, TRADING_DAYS,
    optimizer_weights, week_start_flags,
)

POSTERIOR_COLS = [f"p_{label.lower()}" for label in LABELS]


def optimizer_targets(
    index: pd.DatetimeIndex,
    returns: pd.DataFrame,
    rf: pd.Series,
    names: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Daily target weights of each optimizer held alone, refreshed weekly on all history to date.

    Days with less than MIN_HISTORY of history hold equal weight, as in compute_weights.
    """
    names = sorted(OPTIMIZERS) if names is None else names
    assets = returns.columns.tolist()
    week_start = week_start_flags(index)
    targets = {name: pd.DataFrame(index=index, columns=assets, dtype=float) for name in names}
    books = None

    for i, date in enumerate(index):
        history = returns.loc[:date]
        if len(history) < MIN_HISTORY:
            for name in names:
                targets[name].loc[date] = 1.0 / len(assets)
            continue
        if books is None or week_start[i]:
            current_rf = float(rf.reindex([date], method="ffill").iloc[0])
            if np.isnan(current_rf):
                current_rf = float(rf.iloc[0])
            books = optimizer_weights(history, current_rf * TRADING_DAYS, names)
        for name in names:
            targets[name].loc[date] = books[name]
    return targets


def posteriors_of(regimes_df: pd.DataFrame) -> pd.DataFrame:
    """The four regime posteriors as columns named by label."""
    post = regimes_df[POSTERIOR_COLS].copy()
    post.columns = LABELS
    return post


def blend(
    targets: dict[str, pd.DataFrame],
    posteriors: pd.DataFrame,
    mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Posterior-weighted mix of optimizer targets; rows renormalized to sum to 1."""
    mapping = DEFAULT_MAPPING if mapping is None else mapping
    index = posteriors.index
    first = next(iter(targets.values()))
    weights = pd.DataFrame(0.0, index=index, columns=first.columns)
    for label in LABELS:
        if label not in mapping:
            continue
        p = posteriors[label].fillna(0.0)
        weights += targets[mapping[label]].loc[index].mul(p, axis=0)
    total = weights.sum(axis=1)
    return weights.div(total.where(total > 0, 1.0), axis=0)


def constant_posteriors(index: pd.DatetimeIndex, shares: dict[str, float] | None = None) -> pd.DataFrame:
    """The same posterior every day; default 25% per regime (the static blend)."""
    shares = {label: 1.0 / len(LABELS) for label in LABELS} if shares is None else shares
    return pd.DataFrame({label: float(shares.get(label, 0.0)) for label in LABELS}, index=index)


def shifted_posteriors(posteriors: pd.DataFrame, shift: int) -> pd.DataFrame:
    """Placebo: posteriors circularly shifted by `shift` days on the same dates.

    Keeps the distribution and persistence of the signal and destroys its timing.
    """
    return pd.DataFrame(np.roll(posteriors.values, shift, axis=0),
                        index=posteriors.index, columns=posteriors.columns)


def placebo_shifts(n_days: int, min_shift: int = 252, step: int = 21) -> list[int]:
    """Shifts of at least min_shift days in both directions around the circle, every `step` days."""
    return list(range(min_shift, n_days - min_shift + 1, step))


def run_cell(
    targets: dict[str, pd.DataFrame],
    posteriors: pd.DataFrame,
    returns: pd.DataFrame,
    mapping: dict[str, str] | None = None,
    cost: float | None = None,
) -> pd.DataFrame:
    """Simulate one cell: blend the targets with the posteriors and run the shared simulator."""
    weights = blend(targets, posteriors, mapping)
    return simulate(weights, returns, cost=cost)
