import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.covariance import LedoitWolf
from config import load_config

CFG = load_config()
TRADING_DAYS = CFG["market"]["trading_days"]
RISK_FREE_RATE = CFG["market"]["risk_free_rate"]

def estimate_inputs(returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mu = returns.mean().values * TRADING_DAYS
    sigma = LedoitWolf().fit(returns).covariance_ * TRADING_DAYS
    return mu, sigma

def max_sharpe(returns: pd.DataFrame, rf: float = RISK_FREE_RATE) -> np.ndarray:
    mu, sigma = estimate_inputs(returns)
    n = len(mu)

    y = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(y, sigma))
    constraints = [
        (mu - rf) @ y == 1,
        y >= 0
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    if y.value is None:
        return np.ones(n) / n

    w = y.value / np.sum(y.value)
    return w