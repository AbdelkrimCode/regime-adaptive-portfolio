import numpy as np
import pandas as pd
from optimization.mean_var import max_sharpe
from optimization.risk_parity import risk_parity
from optimization.min_variance import min_variance
from optimization.crash import crash_weights
from config import load_config

CFG = load_config()
TRADING_DAYS = CFG["market"]["trading_days"]

LABELS = ["Bull", "Bear", "Sideways", "Crash"]

# Each optimizer takes (returns, annual rf). The lambdas look the functions up at
# call time, so tests can monkeypatch them on this module.
OPTIMIZERS = {
    "max_sharpe":   lambda r, rf: max_sharpe(r, rf=rf),
    "risk_parity":  lambda r, rf: risk_parity(r),
    "min_variance": lambda r, rf: min_variance(r),
    "safe_haven":   lambda r, rf: crash_weights(r),
}

DEFAULT_MAPPING = {
    "Bull": "max_sharpe",
    "Bear": "risk_parity",
    "Sideways": "min_variance",
    "Crash": "safe_haven",
}

REFRESH_RULES = ("regime", "weekly")

MIN_HISTORY = CFG["hmm"]["min_train_days"] // 2
CONCENTRATION_GUARD = 0.99
SAFE_HAVEN_ASSETS = CFG["optimizer"]["safe_haven_assets"]


def _concentration_guard_fallback(optimizer: str, assets: list[str]) -> np.ndarray:
    """Equal weight over the safe havens for the safe-haven optimizer, else over all assets."""
    if optimizer == "safe_haven":
        safe_havens = [a for a in SAFE_HAVEN_ASSETS if a in assets]
        if safe_havens:
            fallback = np.zeros(len(assets))
            for a in safe_havens:
                fallback[assets.index(a)] = 1.0 / len(safe_havens)
            return fallback
    return np.ones(len(assets)) / len(assets)


def optimizer_weights(returns: pd.DataFrame, rf_annual: float, optimizers: list[str]) -> dict[str, np.ndarray]:
    """Long-only, normalized weights of each named optimizer on the given history.

    An output with more than CONCENTRATION_GUARD in one asset is replaced by its fallback.
    """
    assets = returns.columns.tolist()
    books = {}
    for name in optimizers:
        raw = np.clip(OPTIMIZERS[name](returns, rf_annual), 0, None)
        total = raw.sum()
        raw = raw / total if total > 0 else np.ones(len(raw)) / len(raw)
        if raw.max() > CONCENTRATION_GUARD:
            raw = _concentration_guard_fallback(name, assets)
        books[name] = raw
    return books


def week_start_flags(index: pd.DatetimeIndex) -> np.ndarray:
    """True on the first trading day of each calendar week in `index`."""
    dates = index.to_series()
    return (dates.groupby(index.to_period("W")).transform("min") == dates).values


def compute_weights(
    regimes_df: pd.DataFrame,
    returns: pd.DataFrame,
    rf: pd.Series,
    mapping: dict[str, str] | None = None,
    refresh: str = "regime",
) -> pd.DataFrame:
    """Posterior-blended target weights per day; rf is the daily risk-free series.

    mapping assigns an optimizer to each regime label (default: DEFAULT_MAPPING).
    refresh decides when optimizer weights are recomputed on all history to date:
    "regime" on every change of the most likely regime and every refit date,
    "weekly" on the first trading day of each week, independent of the regimes.
    """
    if refresh not in REFRESH_RULES:
        raise ValueError(f"refresh must be one of {REFRESH_RULES}, got {refresh!r}")
    mapping = DEFAULT_MAPPING if mapping is None else mapping
    needed = sorted(set(mapping.values()))

    assets = returns.columns.tolist()
    weights = pd.DataFrame(index=regimes_df.index, columns=assets, dtype=float)
    week_start = week_start_flags(regimes_df.index)
    books = None
    current_regime = None

    for i, (date, row) in enumerate(regimes_df.iterrows()):
        regime = row["regime"]
        available_returns = returns.loc[:date]

        if len(available_returns) < MIN_HISTORY:
            weights.loc[date] = np.ones(len(assets)) / len(assets)
            continue

        if refresh == "regime":
            due = regime != current_regime or row.get("is_retrain_date", False)
        else:
            due = books is None or week_start[i]

        if due:
            current_rf = float(rf.reindex([date], method="ffill").iloc[0])
            if np.isnan(current_rf):
                current_rf = float(rf.iloc[0])
            # All optimizers are refreshed together so the blend mixes weights
            # estimated on the same history.
            books = optimizer_weights(available_returns, current_rf * TRADING_DAYS, needed)
            current_regime = regime

        blended = np.zeros(len(assets))
        for label in LABELS:
            col = f"p_{label.lower()}"
            if label not in mapping or col not in row.index or pd.isna(row[col]):
                continue
            blended += row[col] * books[mapping[label]]

        total = blended.sum()
        if total > 0:
            blended = blended / total

        weights.loc[date] = blended

    return weights
