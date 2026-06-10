import os
import yfinance as yf
import pandas as pd
from config import load_config

START    = load_config()["evaluation"].get("data_start", "2005-01-01")
END      = load_config()["evaluation"]["data_end"]
TICKERS = ["SPY", "TLT", "GLD", "EFA", "IEF", "QQQ", "LQD", "VNQ"]
RAW_DIR  = os.path.join(os.path.dirname(__file__), "raw")
RAW_PATH = os.path.join(RAW_DIR, "prices.parquet")


def fetch_prices(
    tickers: list[str] = TICKERS,
    start: str = START,
    end: str = END,
    force_refresh: bool = False,
) -> pd.DataFrame:

    if os.path.exists(RAW_PATH) and not force_refresh:
        print("Loading cached prices...")
        return pd.read_parquet(RAW_PATH)

    print(f"Downloading {tickers} from {start} to {end} ...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(how="all")
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"

    os.makedirs(RAW_DIR, exist_ok=True)
    prices.to_parquet(RAW_PATH)
    print(f"Saved {prices.shape[0]} rows x {prices.shape[1]} tickers")
    return prices


if __name__ == "__main__":
    df = fetch_prices()
    print(df.tail())