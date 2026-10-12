"""Ledoit and Wolf (2008) HAC test for the difference of two Sharpe ratios.

Reference: O. Ledoit and M. Wolf, "Robust performance hypothesis testing with
the Sharpe ratio", Journal of Empirical Finance 15 (2008), 850-859, section 3.1.
The long-run covariance uses their HAC choice: Parzen kernel, Andrews (1991)
automatic bandwidth, Andrews and Monahan (1992) VAR(1) prewhitening.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

TRADING_DAYS = 252


def _parzen(x: np.ndarray) -> np.ndarray:
    x = np.abs(x)
    return np.where(
        x <= 0.5,
        1 - 6 * x**2 + 6 * x**3,
        np.where(x <= 1, 2 * (1 - x) ** 3, 0.0),
    )


def _andrews_bandwidth(e: np.ndarray) -> float:
    """Andrews (1991) plug-in bandwidth for the Parzen kernel, AR(1) per column."""
    t = len(e)
    num, den = 0.0, 0.0
    for col in e.T:
        rho = np.dot(col[1:], col[:-1]) / np.dot(col[:-1], col[:-1])
        sigma2 = np.mean((col[1:] - rho * col[:-1]) ** 2)
        num += 4 * rho**2 * sigma2**2 / (1 - rho) ** 8
        den += sigma2**2 / (1 - rho) ** 4
    alpha2 = num / den
    return 2.6614 * (alpha2 * t) ** 0.2


def _hac_covariance(y: np.ndarray) -> np.ndarray:
    """Prewhitened Parzen HAC estimate of the long-run covariance of y (T x k)."""
    t, k = y.shape
    # VAR(1) prewhitening: e_t = y_t - A y_{t-1}
    x, z = y[:-1], y[1:]
    a = np.linalg.lstsq(x, z, rcond=None)[0].T
    e = z - x @ a.T

    bandwidth = _andrews_bandwidth(e)
    n = len(e)
    psi = e.T @ e / n
    for lag in range(1, n):
        w = _parzen(lag / bandwidth)
        if w == 0:
            break
        gamma = e[lag:].T @ e[:-lag] / n
        psi += w * (gamma + gamma.T)

    recolor = np.linalg.inv(np.eye(k) - a)
    psi = recolor @ psi @ recolor.T
    return psi * t / (t - k)


def sharpe_difference_test(
    port_returns: pd.Series,
    bench_returns: pd.Series,
    rf: pd.Series | None = None,
) -> dict:
    """Test H0: equal Sharpe ratios, with a HAC standard error on the difference.

    Inputs are daily returns; rf (daily) is subtracted day by day. The reported
    difference is annualized; the z statistic does not depend on the scaling.
    p_value is one-sided (portfolio better), p_value_two_sided as in the paper.
    """
    r1 = port_returns.values.astype(float)
    r2 = bench_returns.reindex(port_returns.index).values.astype(float)
    if rf is not None:
        rf_aligned = rf.reindex(port_returns.index).ffill().fillna(0).values
        r1, r2 = r1 - rf_aligned, r2 - rf_aligned

    mu1, mu2 = r1.mean(), r2.mean()
    g1, g2 = (r1**2).mean(), (r2**2).mean()
    v1, v2 = g1 - mu1**2, g2 - mu2**2
    diff = mu1 / np.sqrt(v1) - mu2 / np.sqrt(v2)

    grad = np.array([
        g1 / v1**1.5,
        -g2 / v2**1.5,
        -mu1 / (2 * v1**1.5),
        mu2 / (2 * v2**1.5),
    ])
    y = np.column_stack([r1 - mu1, r2 - mu2, r1**2 - g1, r2**2 - g2])
    psi = _hac_covariance(y)
    se = float(np.sqrt(max(grad @ psi @ grad, 0.0) / len(r1)))

    # Identical series: the difference is exactly zero, so there is nothing to reject.
    z = diff / se if se > 0 else 0.0
    scale = np.sqrt(TRADING_DAYS)
    return {
        "d_observed":        round(float(diff * scale), 4),
        "se":                round(float(se * scale), 4),
        "z_stat":            round(float(z), 4),
        "p_value":           round(float(norm.sf(z)), 4),
        "p_value_two_sided": round(float(2 * norm.sf(abs(z))), 4),
    }
