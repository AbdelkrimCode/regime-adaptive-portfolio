import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from config import load_config

TRADING_DAYS = load_config()["market"]["trading_days"]


def estimate_covariance(returns: pd.DataFrame) -> np.ndarray:
    """Annualized Ledoit-Wolf shrunk covariance of daily returns."""
    return LedoitWolf().fit(returns).covariance_ * TRADING_DAYS
