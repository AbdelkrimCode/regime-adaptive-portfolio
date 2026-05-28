import numpy as np
import pandas as pd
import cvxpy as cp

TRADING_DAYS = 252

def estimate_covariance(returns: pd.DataFrame) -> np.ndarray:
    return returns.cov().values * TRADING_DAYS

def risk_parity(returns: pd.DataFrame) -> np.ndarray:
    sigma = estimate_covariance(returns)
    n = sigma.shape[0]
    
    w = cp.Variable(n)
    
    objective = cp.Minimize(
        cp.quad_form(w, sigma) - (1/n) * cp.sum(cp.log(w))
    )
    
    constraints = [w >= 0.01]
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    if w.value is None:
        return np.ones(n) / n
    
    weights = w.value / np.sum(w.value)
    return weights
