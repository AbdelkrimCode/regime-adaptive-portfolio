"""Probabilistic and Deflated Sharpe Ratio.

References: D. Bailey and M. Lopez de Prado, "The Sharpe Ratio Efficient Frontier",
Journal of Risk 15 (2012), and "The Deflated Sharpe Ratio: Correcting for Selection
Bias, Backtest Overfitting and Non-Normality", Journal of Portfolio Management 40 (2014).
All Sharpe ratios here are per period (daily), not annualized.
"""
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

EULER_GAMMA = 0.5772156649015329


def _excess(returns: pd.Series, rf: pd.Series | None) -> np.ndarray:
    r = returns.dropna()
    if rf is not None:
        r = r - rf.reindex(r.index).ffill().fillna(0)
    return r.values.astype(float)


def probabilistic_sharpe_ratio(returns: pd.Series, sr_benchmark: float = 0.0, rf: pd.Series | None = None) -> float:
    """Probability that the true per-period Sharpe exceeds sr_benchmark, allowing for skew and fat tails."""
    x = _excess(returns, rf)
    t = len(x)
    sr = x.mean() / x.std(ddof=1)
    g3 = skew(x)
    g4 = kurtosis(x, fisher=False)
    denom = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)
    return float(norm.cdf((sr - sr_benchmark) * np.sqrt(t - 1) / denom))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum of n_trials Sharpe estimates when every true Sharpe is zero."""
    if n_trials <= 1:
        return 0.0
    return float(np.sqrt(sr_variance) * (
        (1 - EULER_GAMMA) * norm.ppf(1 - 1 / n_trials)
        + EULER_GAMMA * norm.ppf(1 - 1 / (n_trials * np.e))
    ))


def deflated_sharpe_ratio(
    returns: pd.Series,
    trial_sharpes: list[float],
    rf: pd.Series | None = None,
    n_trials: int | None = None,
) -> dict:
    """PSR of `returns` against the Sharpe a best-of-N search would find by luck.

    trial_sharpes are the per-period Sharpe ratios of the variants tried, the chosen one
    included; their variance sets the threshold. n_trials overrides N when some trials
    have no comparable Sharpe (default: len(trial_sharpes)).
    """
    n = len(trial_sharpes) if n_trials is None else n_trials
    variance = float(np.var(trial_sharpes, ddof=1)) if len(trial_sharpes) > 1 else 0.0
    sr0 = expected_max_sharpe(n, variance)
    x = _excess(returns, rf)
    return {
        "n_trials": n,
        "sharpe": float(x.mean() / x.std(ddof=1)),
        "sharpe_threshold": sr0,
        "dsr": probabilistic_sharpe_ratio(returns, sr0, rf),
    }
