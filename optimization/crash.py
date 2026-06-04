import numpy as np
import pandas as pd
from config import load_config

CFG = load_config()

def crash_weights(returns: pd.DataFrame) -> np.ndarray:
    assets = returns.columns.tolist()
    weights = np.zeros(len(assets))
    safe_havens = [a for a in CFG["optimizer"]["safe_haven_assets"] if a in assets]
    if not safe_havens:
        return np.ones(len(assets)) / len(assets)
    for a in safe_havens:
        weights[assets.index(a)] = 1.0 / len(safe_havens)
    return weights
