import os
import yfinance as yf
import pandas as pd
from config import load_config
from data.cache_utils import is_cache_valid, write_cache_signature

_CFG = load_config()
RF_TICKER = "^IRX"
RF_PATH = os.path.join(os.path.dirname(__file__), "processed", "risk_free.parquet")


def fetch_risk_free(
    start: str = "2005-01-01",
    end: str = _CFG["evaluation"]["data_end"],
    force_refresh: bool = False
) -> pd.Series:
    signature = {"start": start, "end": end}

    if not force_refresh and is_cache_valid(RF_PATH, signature):
        return pd.read_parquet(RF_PATH)["rf_daily"]

    print("Downloading risk-free rate (^IRX)...")
    raw = yf.download(RF_TICKER, start=start, end=end, auto_adjust=True, progress=False)
    rf_annual = raw["Close"].squeeze()
    rf_daily = (rf_annual / 100) / 252
    rf_daily.index = pd.to_datetime(rf_daily.index)
    rf_daily.index.name = "date"
    rf_daily.name = "rf_daily"

    os.makedirs(os.path.dirname(RF_PATH), exist_ok=True)
    rf_daily.to_frame().to_parquet(RF_PATH)
    write_cache_signature(RF_PATH, signature)
    print(f"Saved {len(rf_daily)} risk-free rate observations")
    return rf_daily


if __name__ == "__main__":
    rf = fetch_risk_free(force_refresh=True)
    print(rf.describe())
    print(rf.tail())