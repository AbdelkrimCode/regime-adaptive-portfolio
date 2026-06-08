import numpy as np
import pandas as pd
from optimization.mean_var import max_sharpe
from optimization.risk_parity import risk_parity
from optimization.min_variance import min_variance
from optimization.crash import crash_weights
from config import load_config
from data.risk_free import fetch_risk_free
CFG = load_config()


OPTIMIZER_MAP = {
    "Bull": max_sharpe,
    "Bear": risk_parity,
    "Sideways": min_variance,
    "Crash": crash_weights,
}

min_history = CFG["hmm"]["min_train_days"] // 2
CONCENTRATION_GUARD = 0.99

def get_weights(regime: str, returns: pd.DataFrame) -> np.ndarray:
    optimizer = OPTIMIZER_MAP[regime]
    return optimizer(returns)

def compute_weights(regimes_df: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    assets = returns.columns.tolist()
    rf_series = fetch_risk_free()
    weights = pd.DataFrame(index=regimes_df.index, columns=assets, dtype=float)
    cached_weights = {"Bull": None, "Bear": None, "Sideways": None, "Crash": None}
    current_regime = None

    for date, row in regimes_df.iterrows():
        regime = row["regime"]
        available_returns = returns.loc[:date]

        if len(available_returns) < min_history:
            weights.loc[date] = np.ones(len(assets)) / len(assets)
            continue

        is_retrain = row.get("is_retrain_date", False)

        if regime != current_regime or is_retrain:
            current_rf = float(rf_series.reindex([date], method="ffill").iloc[0])
            # All four optimizers recomputed together — blending requires consistent weights across regimes
            for label in ["Bull", "Bear", "Sideways", "Crash"]:
                if label == "Bull":
                    raw = max_sharpe(available_returns, rf=current_rf)
                else:
                    raw = get_weights(label, available_returns)
                raw = np.clip(raw, 0, None)
                raw = raw / np.sum(raw)
                if np.max(raw) > CONCENTRATION_GUARD:
                        raw = np.ones(len(assets)) / len(assets)
                cached_weights[label] = raw
            current_regime = regime

        blended = np.zeros(len(assets))
        for label in ["Bull", "Bear", "Sideways", "Crash"]:
            col = f"p_{label.lower()}"
            if col not in row.index or pd.isna(row[col]):
                continue
            p = row[col]
            if cached_weights[label] is None:
                continue
            blended += p * cached_weights[label]

        total = np.sum(blended)
        if total > 0:
            blended = blended / total

        weights.loc[date] = blended

    return weights