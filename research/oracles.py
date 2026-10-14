"""Look-ahead oracles for the detection vs response decomposition.

Everything here uses future data on purpose. The outputs are upper bounds that
measure how much better detection or response could have been; they are not
investable strategies and must never be reported as such.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import load_config
from models.hmm import REGIME_ORDER, _fit_hmm_core, label_states
from optimization.switcher import DEFAULT_MAPPING, LABELS

CFG = load_config()
TRADING_DAYS = CFG["market"]["trading_days"]
LABELS_BY_VOL = REGIME_ORDER[4]          # calmest to most turbulent


def argmax_labels(posteriors: pd.DataFrame) -> pd.Series:
    """Most likely regime label on each day."""
    return posteriors[LABELS].idxmax(axis=1)


def hindsight_posteriors(features: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Smoothed posteriors of one HMM fitted on the whole sample (decision 1a).

    Same model class and labeling rule as the strategy; the only difference is
    information: parameters and states both use the full sample.
    """
    cols = features.columns.tolist()
    scaled = StandardScaler().fit_transform(features.values)
    model = _fit_hmm_core(scaled, CFG["hmm"]["n_states"])
    names = label_states(model, feature_cols=cols)
    smoothed = model.predict_proba(scaled)
    post = pd.DataFrame(0.0, index=features.index, columns=LABELS)
    for state, label in names.items():
        post[label] = smoothed[:, state]
    return post.loc[index]


def realized_vol_posteriors(spy_returns: pd.Series, index: pd.DatetimeIndex, horizon: int = 21) -> pd.DataFrame:
    """Model-free robustness oracle (decision 1b): quartiles of future realized SPY volatility.

    Day t is labeled by the volatility of days t+1..t+horizon, calmest quartile = Bull.
    Quartile cutoffs use the whole index. The last days, with no full window ahead,
    keep the last available label.
    """
    future_vol = (
        spy_returns.rolling(horizon).std().shift(-horizon) * np.sqrt(TRADING_DAYS)
    ).reindex(index).ffill()
    # Ranking first keeps quartiles equal-sized even when vols tie.
    quartile = pd.qcut(future_vol.rank(method="first"), 4, labels=False)
    post = pd.DataFrame(0.0, index=index, columns=LABELS)
    for q, label in enumerate(LABELS_BY_VOL):
        post.loc[quartile == q, label] = 1.0
    return post


def hindsight_mapping(
    book_returns: dict[str, pd.Series],
    labels: pd.Series,
    rf: pd.Series | None = None,
) -> dict[str, str]:
    """For each regime label, the optimizer with the highest realized Sharpe on that label's days.

    labels is the regime the strategy acted on each day; a book's return on day t+1
    belongs to the label of day t, when its weights were set. Labels with no days
    keep the default optimizer.
    """
    acted_on = labels.shift(1)
    mapping = {}
    for label in LABELS:
        best, best_sharpe = DEFAULT_MAPPING[label], -np.inf
        for name, r in book_returns.items():
            x = r[acted_on.reindex(r.index) == label]
            if rf is not None:
                x = x - rf.reindex(x.index).ffill().fillna(0)
            if len(x) < 2 or x.std() == 0:
                continue
            sharpe = x.mean() / x.std()
            if sharpe > best_sharpe:
                best, best_sharpe = name, sharpe
        mapping[label] = best
    return mapping
