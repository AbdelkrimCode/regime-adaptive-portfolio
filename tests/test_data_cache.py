import json
import os
from unittest.mock import patch

import pandas as pd
import pytest

from data.cache_utils import is_cache_valid, write_cache_signature, meta_path_for


def make_fake_yf_download(n=5):
    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    def _fake(tickers, start, end, auto_adjust, progress):
        cols = pd.MultiIndex.from_product([["Close"], tickers])
        data = pd.DataFrame(100.0, index=dates, columns=cols)
        return data

    return _fake


def test_cache_utils_roundtrip(tmp_path):
    data_path = str(tmp_path / "prices.parquet")
    sig = {"tickers": ["GLD", "SPY"], "start": "2020-01-01", "end": "2020-12-31"}

    assert not is_cache_valid(data_path, sig)  # nothing written yet

    pd.DataFrame({"a": [1]}).to_parquet(data_path)
    write_cache_signature(data_path, sig)

    assert is_cache_valid(data_path, sig)
    assert not is_cache_valid(data_path, {**sig, "end": "2021-12-31"})
    assert os.path.exists(meta_path_for(data_path))


def test_fetch_prices_refetches_on_ticker_change(tmp_path, monkeypatch):
    import data.fetch as fetch_mod

    raw_path = str(tmp_path / "prices.parquet")
    monkeypatch.setattr(fetch_mod, "RAW_PATH", raw_path)
    monkeypatch.setattr(fetch_mod, "RAW_DIR", str(tmp_path))

    with patch.object(fetch_mod.yf, "download", side_effect=make_fake_yf_download()):
        df1 = fetch_mod.fetch_prices(tickers=["SPY", "GLD"], start="2020-01-01", end="2020-12-31")
        assert set(df1.columns) == {"SPY", "GLD"}

        # Different ticker set, same cache file present -> must NOT silently
        # return the SPY/GLD cache; must refetch for the new signature.
        df2 = fetch_mod.fetch_prices(tickers=["TLT", "IEF"], start="2020-01-01", end="2020-12-31")
        assert set(df2.columns) == {"TLT", "IEF"}


def test_fetch_risk_free_refetches_on_date_change(tmp_path, monkeypatch):
    import data.risk_free as rf_mod

    rf_path = str(tmp_path / "risk_free.parquet")
    monkeypatch.setattr(rf_mod, "RF_PATH", rf_path)

    dates_a = pd.date_range("2020-01-01", periods=5, freq="B")
    dates_b = pd.date_range("2021-01-01", periods=5, freq="B")

    def fake_download_a(*a, **k):
        return pd.DataFrame({"Close": [1.0] * 5}, index=dates_a)

    def fake_download_b(*a, **k):
        return pd.DataFrame({"Close": [2.0] * 5}, index=dates_b)

    with patch.object(rf_mod.yf, "download", side_effect=fake_download_a):
        rf1 = rf_mod.fetch_risk_free(start="2020-01-01", end="2020-12-31")
        assert rf1.index[0] == dates_a[0]

    with patch.object(rf_mod.yf, "download", side_effect=fake_download_b):
        rf2 = rf_mod.fetch_risk_free(start="2021-01-01", end="2021-12-31")
        assert rf2.index[0] == dates_b[0], "should have refetched instead of returning 2020 cache"

def test_fetch_start_and_risk_free_start_come_from_config():
    import inspect
    import data.fetch as fetch_mod
    import data.risk_free as rf_mod
    from config import load_config

    cfg = load_config()
    assert fetch_mod.START == cfg["evaluation"]["data_start"]

    rf_default_start = inspect.signature(rf_mod.fetch_risk_free).parameters["start"].default
    assert rf_default_start == cfg["evaluation"]["data_start"]