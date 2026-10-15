import os
import yfinance as yf
import pandas as pd
from config import load_config
from data.cache_utils import is_cache_valid, write_cache_signature

_CFG = load_config()
START    = _CFG["evaluation"]["data_start"]
END      = _CFG["evaluation"]["data_end"]
# Asset name used in the code -> ticker downloaded for it. A config can map an
# asset to a proxy (e.g. SPY -> VFINX for years before the ETF existed).
UNIVERSE = _CFG.get("universe") or {t: t for t in ["SPY", "TLT", "GLD", "EFA", "IEF", "QQQ", "LQD", "VNQ"]}
TICKERS = list(UNIVERSE)
RAW_PATH = _CFG["paths"].get("raw_prices", os.path.join(os.path.dirname(__file__), "raw", "prices.parquet"))
RAW_DIR  = os.path.dirname(RAW_PATH)

def inclusive_end(end: str) -> str:
    """yfinance treats `end` as exclusive; shift one day so `end` itself is included."""
    return (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def fetch_prices(
    tickers: list[str] = TICKERS,
    start: str = START,
    end: str = END,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Adjusted close prices from yfinance, cached by tickers and date range.

    Columns are named by asset; each is downloaded under its UNIVERSE ticker.
    """
    sources = {t: UNIVERSE.get(t, t) for t in tickers}
    signature = {"tickers": sorted(tickers), "start": start, "end": end, "end_inclusive": True}
    if any(asset != source for asset, source in sources.items()):
        signature["sources"] = dict(sorted(sources.items()))

    if not force_refresh and is_cache_valid(RAW_PATH, signature):
        print("Loading cached prices...")
        return pd.read_parquet(RAW_PATH)

    download = list(sources.values())
    print(f"Downloading {download} from {start} to {end} ...")
    raw = yf.download(download, start=start, end=inclusive_end(end), auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": download[0]})
    prices = prices.rename(columns={source: asset for asset, source in sources.items()})[tickers]

    prices = prices.dropna(how="all")
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"

    os.makedirs(RAW_DIR, exist_ok=True)
    prices.to_parquet(RAW_PATH)
    write_cache_signature(RAW_PATH, signature)
    print(f"Saved {prices.shape[0]} rows x {prices.shape[1]} tickers")
    return prices


if __name__ == "__main__":
    df = fetch_prices()
    print(df.tail())