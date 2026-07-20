import os
import numpy as np
import pandas as pd
from config import load_config


CFG = load_config()
RETURNS_PATH  = CFG["paths"]["returns"]
FEATURES_PATH = CFG["paths"]["features"]
PROCESSED_DIR = os.path.dirname(RETURNS_PATH)

VOL_WINDOW  = CFG["features"]["vol_window"]
CORR_WINDOW = CFG["features"]["corr_window"]   


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns


def compute_features(returns: pd.DataFrame, vol_window: int = VOL_WINDOW, corr_window: int = CORR_WINDOW) -> pd.DataFrame:
    """
    Build feature matrix for HMM fitting.

    Features:
      - spy_return : daily log return of SPY
      - spy_vol    : rolling realised volatility (annualised)
      - mean_corr  : rolling mean pairwise correlation
    """
    spy = returns["SPY"]

    spy_vol = spy.rolling(vol_window).std() * np.sqrt(252)

    mean_corr = (
        returns.rolling(corr_window)
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

def compute_skew_kurt_features(returns: pd.DataFrame, window: int = VOL_WINDOW) -> pd.DataFrame:
    spy = returns["SPY"]
    spy_skew = spy.rolling(window).skew()
    spy_kurt = spy.rolling(window).kurt()

    return pd.DataFrame({
        "spy_return": spy,
        "spy_skew":   spy_skew,
        "spy_kurt":   spy_kurt,
    }).dropna()


def process_and_save(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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