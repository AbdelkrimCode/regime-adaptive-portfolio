
"""
Transforms raw price data into features used by the HMM regime detector:
  - daily log returns
  - rolling 21-day realised volatility (annualised) 
  - rolling 63-day mean pairwise correlation method (average of all pairwise correlations between assets)
"""

import os
import numpy as np
import pandas as pd

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")
RETURNS_PATH  = os.path.join(PROCESSED_DIR, "returns.parquet")
FEATURES_PATH = os.path.join(PROCESSED_DIR, "features.parquet")

VOL_WINDOW  = 21   # ~1 month
CORR_WINDOW = 63   # ~1 quarter


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from adjusted close prices."""
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns


def compute_features(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix for HMM fitting.

    Features:
      - spy_return : daily log return of SPY
      - spy_vol    : rolling realised volatility (annualised)
      - mean_corr  : rolling mean pairwise correlation
    """
    spy = returns["SPY"]

    spy_vol = spy.rolling(VOL_WINDOW).std() * np.sqrt(252)

    mean_corr = (
        returns.rolling(CORR_WINDOW)
        .corr()
        .groupby(level=0)
        .apply(lambda m: (m.values.sum() - m.shape[0]) / (m.shape[0] * (m.shape[0] - 1)))
    )

    features = pd.DataFrame(
        {
            "spy_return": spy,
            "spy_vol":    spy_vol,
            "mean_corr":  mean_corr,
        }
    ).dropna()

    return features


def process_and_save(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full pipeline and save to parquet."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    returns  = compute_returns(prices)
    features = compute_features(returns)

    returns.to_parquet(RETURNS_PATH)
    features.to_parquet(FEATURES_PATH)

    print(f"Returns  : {returns.shape}  → {RETURNS_PATH}")
    print(f"Features : {features.shape} → {FEATURES_PATH}")
    return returns, features


def load_returns() -> pd.DataFrame:
    return pd.read_parquet(RETURNS_PATH)


def load_features() -> pd.DataFrame:
    return pd.read_parquet(FEATURES_PATH)


if __name__ == "__main__":
    from data.fetch import fetch_prices
    prices = fetch_prices()
    process_and_save(prices)