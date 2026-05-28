import numpy as np
import pandas as pd
from optimization.mean_var import max_sharpe
from optimization.risk_parity import risk_parity
from optimization.min_variance import min_variance

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
    
    for date, regime in regimes.items():
        if regime != current_regime:
            available_returns = returns.loc[:date]
            
            if len(available_returns) < 126:
                current_weights = np.ones(len(assets)) / len(assets)
            else:
                current_weights = get_weights(regime, available_returns)
                current_weights = np.clip(current_weights, 0, None)
                current_weights = current_weights / np.sum(current_weights)
                
                # Fallback if solver returned degenerate solution
                if np.max(current_weights) > 0.99:
                    current_weights = np.ones(len(assets)) / len(assets)
    
            current_regime = regime
        
        weights.loc[date] = current_weights
    
    return weights
