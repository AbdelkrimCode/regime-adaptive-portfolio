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

def _cap_and_redistribute(weights: np.ndarray, cap: float, max_iter: int = 20) -> np.ndarray:

    w = weights.copy()
    for _ in range(max_iter):
        over = w > cap
        if not np.any(over):
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = ~over
        if not np.any(under):
            return np.ones(len(w)) / len(w)
        w[under] += excess * (w[under] / w[under].sum())
    return w / w.sum()

def risk_parity(returns: pd.DataFrame) -> np.ndarray:
    sigma = estimate_covariance(returns)
    n = sigma.shape[0]
    
    w = cp.Variable(n)
    
    objective = cp.Minimize(
        cp.quad_form(w, sigma) - (1/n) * cp.sum(cp.log(w))
    )
    


    constraints = [
    w >= 0.01,
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    if w.value is None:
        return np.ones(n) / n
    
    weights = np.clip(w.value, 0, None)
    weights = weights / np.sum(weights)
    weights = _cap_and_redistribute(weights, MAX_POSITION)
    return weights