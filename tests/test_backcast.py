"""Backcast infrastructure: config selection, proxy tickers and the backcast window."""
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from config import CONFIG_ENV, PROJECT_ROOT, load_config
from optimization.switcher import LABELS

ASSETS = ["SPY", "TLT", "EFA", "IEF", "QQQ", "LQD", "VNQ"]


# --- config selection ---

def test_config_env_selects_another_file(tmp_path, monkeypatch):
    cfg = {"paths": {"returns": "data/other/returns.parquet"}, "evaluation": {"data_end": "2004-12-31"}}
    path = tmp_path / "other.yaml"
    path.write_text(yaml.safe_dump(cfg))
    default_end = load_config()["evaluation"]["data_end"]
    monkeypatch.setenv(CONFIG_ENV, str(path))
    try:
        loaded = load_config()
    finally:
        monkeypatch.delenv(CONFIG_ENV)
    assert loaded["evaluation"]["data_end"] == "2004-12-31"
    assert Path(loaded["paths"]["returns"]) == PROJECT_ROOT / "data" / "other" / "returns.parquet"
    assert load_config()["evaluation"]["data_end"] == default_end


def test_backcast_config_never_touches_development_files():
    with open(PROJECT_ROOT / "config_backcast.yaml") as f:
        backcast = yaml.safe_load(f)
    dev = load_config()["paths"]
    for key, value in backcast["paths"].items():
        assert value.startswith(("data/backcast/", "results/backcast/")), key
        assert str(PROJECT_ROOT / value) != dev.get(key), key
    assert set(backcast["universe"]) == set(ASSETS)
    assert backcast["optimizer"]["safe_haven_assets"] == ["IEF", "TLT"]
    assert backcast["evaluation"]["data_end"] == "2004-12-31"


# --- proxy tickers ---

def _fake_download(tickers, start, end, auto_adjust, progress):
    dates = pd.date_range("2000-01-03", periods=5, freq="B")
    cols = pd.MultiIndex.from_product([["Close"], tickers])
    return pd.DataFrame(np.arange(5.0 * len(tickers)).reshape(5, -1) + 100, index=dates, columns=cols)


def test_fetch_downloads_proxies_and_names_columns_by_asset(tmp_path, monkeypatch):
    import data.fetch as fetch_mod

    monkeypatch.setattr(fetch_mod, "RAW_PATH", str(tmp_path / "prices.parquet"))
    monkeypatch.setattr(fetch_mod, "RAW_DIR", str(tmp_path))
    monkeypatch.setattr(fetch_mod, "UNIVERSE", {"SPY": "VFINX", "TLT": "VUSTX"})
    with patch.object(fetch_mod.yf, "download", side_effect=_fake_download) as download:
        prices = fetch_mod.fetch_prices(tickers=["SPY", "TLT"], start="2000-01-01", end="2000-12-31")
    assert download.call_args.args[0] == ["VFINX", "VUSTX"]
    assert list(prices.columns) == ["SPY", "TLT"]


def test_proxy_cache_is_not_reused_for_the_real_tickers(tmp_path, monkeypatch):
    import data.fetch as fetch_mod

    monkeypatch.setattr(fetch_mod, "RAW_PATH", str(tmp_path / "prices.parquet"))
    monkeypatch.setattr(fetch_mod, "RAW_DIR", str(tmp_path))
    args = dict(tickers=["SPY"], start="2000-01-01", end="2000-12-31")
    with patch.object(fetch_mod.yf, "download", side_effect=_fake_download) as download:
        monkeypatch.setattr(fetch_mod, "UNIVERSE", {"SPY": "VFINX"})
        fetch_mod.fetch_prices(**args)
        monkeypatch.setattr(fetch_mod, "UNIVERSE", {"SPY": "SPY"})
        fetch_mod.fetch_prices(**args)
    assert [c.args[0] for c in download.call_args_list] == [["VFINX"], ["SPY"]]


# --- backcast window ---

def _synthetic(start, n, seed=3):
    idx = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(seed)
    returns = pd.DataFrame(rng.normal(0.0003, 0.01, (n, len(ASSETS))), index=idx, columns=ASSETS)
    features = pd.DataFrame({
        "spy_return": returns["SPY"],
        "spy_vol": returns["SPY"].rolling(21).std().bfill(),
        "mean_corr": rng.normal(0, 1, n),
    }, index=idx)
    post = pd.DataFrame(rng.dirichlet(np.ones(4), n - 150), index=idx[150:], columns=LABELS)
    regimes = post.copy()
    regimes.columns = [f"p_{l.lower()}" for l in LABELS]
    regimes["regime"] = post.idxmax(axis=1)
    return returns, pd.Series(0.00005, index=idx), features, regimes


def test_backcast_window_runs_every_cell_and_stops_at_2004():
    from research.run_study import BACKCAST_END, run_study

    returns, rf, features, regimes = _synthetic("2002-06-03", 800)
    assert regimes.index.max() > BACKCAST_END
    summary = run_study(returns, rf, features, regimes, window="backcast",
                        placebo_step=50, n_bootstrap=50, prior_trials=[0.5, 0.6, 0.7])
    assert summary["window"] == "backcast"
    assert pd.Timestamp(summary["score_end"]) <= BACKCAST_END
    for cell in ["C0", "C1", "H4", "D1", "D1b", "R1", "F"]:
        assert cell in summary["cells"][2]
    for key in ["H1", "H2", "H3", "H4", "C0_dsr", "yearly_returns"]:
        assert key in summary


def test_backcast_window_refuses_data_from_the_development_sample():
    from research.run_study import run_study

    returns, rf, features, regimes = _synthetic("2005-01-03", 400)
    with pytest.raises(ValueError):
        run_study(returns, rf, features, regimes, window="backcast")


def test_run_backcast_refuses_when_modules_hold_another_config(monkeypatch):
    from research import run_backcast

    import data.fetch  # noqa: F401  loaded with the development config
    with pytest.raises(SystemExit):
        run_backcast.main()
    import os
    assert CONFIG_ENV not in os.environ
    assert "data.fetch" in sys.modules
