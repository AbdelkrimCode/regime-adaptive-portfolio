"""Study statistics: Sharpe, placebo p-value and the bootstrap CI of the H3 gap."""
import numpy as np
import pandas as pd

from backtest.bootstrap import block_bootstrap_indices
from config import load_config

TRADING_DAYS = load_config()["market"]["trading_days"]


def annual_sharpe(returns: pd.Series, rf: pd.Series | None = None) -> float:
    """Annualized Sharpe of daily returns in excess of the daily rf."""
    x = returns.dropna()
    if rf is not None:
        x = x - rf.reindex(x.index).ffill().fillna(0)
    return float(x.mean() / x.std() * np.sqrt(TRADING_DAYS))


def placebo_p_value(observed: float, placebo: list[float]) -> float:
    """Share of placebo Sharpes at least as high as the observed one, counting the observed run."""
    placebo = np.asarray(placebo)
    return float((1 + (placebo >= observed).sum()) / (1 + len(placebo)))


def gap_difference_ci(
    c0: pd.Series,
    detection_oracle: pd.Series,
    response_oracle: pd.Series,
    rf: pd.Series | None = None,
    block_length: int = 20,
    n_iterations: int = 1000,
    random_state: int = 42,
) -> dict:
    """Point estimate and 95% CI of dR - dD, where dX = SR(oracle X) - SR(C0).

    The three series are resampled with the same block indices, so the pairing of
    days is kept. SR(C0) cancels in dR - dD but is kept for readability.
    """
    index = c0.index
    frame = pd.DataFrame({
        "c0": c0, "d": detection_oracle.reindex(index), "r": response_oracle.reindex(index),
    }).dropna()
    rf_aligned = (rf.reindex(frame.index).ffill().fillna(0) if rf is not None
                  else pd.Series(0.0, index=frame.index))
    excess = frame.sub(rf_aligned, axis=0).values

    def gaps(x):
        sr = x.mean(axis=0) / x.std(axis=0, ddof=1) * np.sqrt(TRADING_DAYS)
        return sr[2] - sr[0], sr[1] - sr[0]

    d_r, d_d = gaps(excess)
    rng = np.random.default_rng(random_state)
    diffs = []
    for _ in range(n_iterations):
        idx = block_bootstrap_indices(len(excess), block_length, rng)
        r, d = gaps(excess[idx])
        diffs.append(r - d)
    lower, upper = np.percentile(diffs, [2.5, 97.5])
    return {
        "response_value": float(d_r),
        "detection_value": float(d_d),
        "difference": float(d_r - d_d),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
    }


def verdict(ci: dict) -> str:
    """Pre-registered reading of the H3 CI."""
    if ci["ci_lower"] > 0:
        return "supported (response is the bottleneck)"
    if ci["ci_upper"] < 0:
        return "refuted (detection is the bottleneck)"
    return "inconclusive"
