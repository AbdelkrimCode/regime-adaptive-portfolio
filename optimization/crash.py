import numpy as np
import pandas as pd

FLIGHT_TO_SAFETY = ["IEF", "TLT", "GLD"]


def crash_weights(returns: pd.DataFrame) -> np.ndarray:
    assets = returns.columns.tolist()
    weights = np.zeros(len(assets))
    safe_havens = [a for a in FLIGHT_TO_SAFETY if a in assets]
    if not safe_havens:
        return np.ones(len(assets)) / len(assets)
    for a in safe_havens:
        weights[assets.index(a)] = 1.0 / len(safe_havens)
    return weights
