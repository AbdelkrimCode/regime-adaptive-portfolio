import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.covariance import LedoitWolf
from config import load_config

CFG = load_config()
TRADING_DAYS = CFG["market"]["trading_days"]
MAX_POSITION = CFG["optimizer"]["max_position"]

def estimate_covariance(returns: pd.DataFrame) -> np.ndarray:
    return LedoitWolf().fit(returns).covariance_ * TRADING_DAYS

def risk_parity(returns: pd.DataFrame) -> np.ndarray:
    sigma = estimate_covariance(returns)
    n = sigma.shape[0]
    
    w = cp.Variable(n)
    
    objective = cp.Minimize(
        cp.quad_form(w, sigma) - (1/n) * cp.sum(cp.log(w))
    )
    

    constraints = [
    cp.sum(w) == 1,
    w >= 0.01,
    w <= MAX_POSITION,
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    if w.value is None:
        return np.ones(n) / n
    
    weights = np.clip(w.value, 0, None)
    weights = weights / np.sum(weights)
    return weights
