import numpy as np
import pandas as pd
import cvxpy as cp

RISK_FREE_RATE = 0.04
TRADING_DAYS = 252

def estimate_inputs(returns: pd.DataFrame):
    mu = returns.mean().values * TRADING_DAYS
    sigma = returns.cov().values * TRADING_DAYS
    return mu, sigma

def max_sharpe(returns: pd.DataFrame) -> np.ndarray:
    mu, sigma = estimate_inputs(returns)
    n = len(mu)
    rf = RISK_FREE_RATE
    
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