import numpy as np
import pandas as pd
import cvxpy as cp

TRADING_DAYS = 252

from sklearn.covariance import LedoitWolf

def estimate_covariance(returns: pd.DataFrame) -> np.ndarray:
    return LedoitWolf().fit(returns).covariance_ * TRADING_DAYS

def min_variance(returns: pd.DataFrame) -> np.ndarray:
    sigma = estimate_covariance(returns)
    n = sigma.shape[0]
    
    w = cp.Variable(n)
    
    objective = cp.Minimize(cp.quad_form(w, sigma))
    
    constraints = [
        cp.sum(w) == 1,
        w >= 0
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    if w.value is None:
        return np.ones(n) / n
    
    return w.value