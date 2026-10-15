"""Backcast: the frozen study on a 1996-2004 proxy universe (docs/preregistration_addendum_backcast.md).

    python -m research.run_backcast

Selects config_backcast.yaml before anything reads a config, so every data file,
model and result goes under data/backcast/ and results/study/backcast/. The
development and holdout files are never read or written.
"""
import os
import sys

BACKCAST_CONFIG = "config_backcast.yaml"
PACKAGES = ("backtest", "data", "models", "optimization", "research", "scripts", "visualization")


def loaded_project_modules() -> list[str]:
    """Project modules already imported (they may hold a config read at import time)."""
    return sorted(m for m in sys.modules
                  if m.split(".")[0] in PACKAGES and "." in m and m != __name__)


def main() -> None:
    # Modules read the config when first imported, so none may be loaded yet.
    stale = loaded_project_modules()
    if stale:
        raise SystemExit(f"modules already loaded with another config: {stale}")
    from config import CONFIG_ENV
    os.environ[CONFIG_ENV] = BACKCAST_CONFIG

    from data.fetch import fetch_prices
    from data.process import process_and_save
    from data.risk_free import fetch_risk_free
    from research.run_study import run

    prices = fetch_prices()
    print(f"Prices {prices.index[0].date()} to {prices.index[-1].date()}, rows per asset:")
    print(prices.notna().sum().to_string())
    fetch_risk_free()
    process_and_save(prices)
    run("backcast")


if __name__ == "__main__":
    main()
