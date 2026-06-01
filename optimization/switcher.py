import numpy as np
import pandas as pd
from optimization.mean_var import max_sharpe
from optimization.risk_parity import risk_parity
from optimization.min_variance import min_variance
from config import load_config

CFG = load_config()

OPTIMIZER_MAP = {
    "Bull": max_sharpe,
    "Bear": risk_parity,
    "Sideways": min_variance
}

def get_weights(regime: str, returns: pd.DataFrame) -> np.ndarray:
    optimizer = OPTIMIZER_MAP[regime]
    return optimizer(returns)

def compute_weights(regimes: pd.Series, returns: pd.DataFrame) -> pd.DataFrame:
    assets = returns.columns.tolist()
    weights = pd.DataFrame(index=regimes.index, columns=assets, dtype=float)

    current_regime = None
    current_weights = None
    target_weights = None
    days_since_switch = CFG["optimizer"]["smoothing_days"]

    for date, regime in regimes.items():
        if regime != current_regime:
            target_weights = None
            available_returns = returns.loc[:date]

            if len(available_returns) < 126:
                target_weights = np.ones(len(assets)) / len(assets)
            else:
                target_weights = get_weights(regime, available_returns)
                target_weights = np.clip(target_weights, 0, None)
                target_weights = target_weights / np.sum(target_weights)

                if np.max(target_weights) > 0.99:
                    target_weights = np.ones(len(assets)) / len(assets)

            if current_weights is None:
                current_weights = target_weights

            days_since_switch = 0
            current_regime = regime

        if days_since_switch < CFG["optimizer"]["smoothing_days"]:
            alpha = (days_since_switch + 1) / CFG["optimizer"]["smoothing_days"]
            blended = (1 - alpha) * current_weights + alpha * target_weights
            weights.loc[date] = blended
            if days_since_switch == CFG["optimizer"]["smoothing_days"] - 1:
                current_weights = target_weights
        else:
            weights.loc[date] = current_weights

        days_since_switch += 1

    return weights