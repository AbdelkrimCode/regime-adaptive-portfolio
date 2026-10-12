import numpy as np
import pandas as pd
import cvxpy as cp
from optimization.covariance import estimate_covariance
from config import load_config

CFG = load_config()

def min_variance(returns: pd.DataFrame) -> np.ndarray:
    sigma = estimate_covariance(returns)
    n = sigma.shape[0]
    
    w = cp.Variable(n)
    
    objective = cp.Minimize(cp.quad_form(w, sigma))
    
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= CFG["optimizer"]["max_position"]
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    if w.value is None:
        return np.ones(n) / n
    
    weights = np.clip(w.value, 0, None)
    weights = weights / np.sum(weights)
    return weights