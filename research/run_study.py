"""Run the pre-registered detection vs response study (docs/preregistration.md).

    python -m research.run_study --window dev       # development sample, 2006-2024
    python -m research.run_study --window holdout   # once, after data_end is extended
    python -m research.run_backcast                 # 1996-2004 proxy universe (addendum)

Every number written here comes from one call to run_study().
"""
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

from backtest.bootstrap import run_bootstrap, summarize
from backtest.dsr import deflated_sharpe_ratio
from backtest.metrics import compute_all
from backtest.sharpe_test import sharpe_difference_test
from config import PROJECT_ROOT, load_config
from optimization.switcher import LABELS, OPTIMIZERS
from research.books import (
    constant_posteriors, optimizer_targets, placebo_shifts, posteriors_of, run_cell,
    shifted_posteriors,
)
from research.diagnostic import rotation_regressions
from research.h4 import correlation_aware_targets, stock_bond_correlation, with_cash
from research.oracles import (
    argmax_labels, hindsight_mapping, hindsight_posteriors, realized_vol_posteriors,
)
from research.stats import annual_sharpe, gap_difference_ci, placebo_p_value, verdict

CFG = load_config()
TRADING_DAYS = CFG["market"]["trading_days"]
DEV_END = pd.Timestamp("2024-12-31")
HOLDOUT_START = pd.Timestamp("2025-01-02")
# The backcast sample ends before the first day the development sample uses.
BACKCAST_END = pd.Timestamp("2004-12-31")
WINDOWS = ("dev", "holdout", "backcast")
# Windows analysed in full (oracles, H1, H3, DSR); the holdout only gets C0, C1, H4 and the placebo.
FULL_WINDOWS = ("dev", "backcast")
COSTS_BP = (2, 5, 10)
PRIMARY_BP = 2
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "study")


def file_sha256(path: str) -> str | None:
    """SHA-256 of a file, or None if it does not exist."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def prior_trial_sharpes() -> list[float]:
    """Annualized full-period Sharpes of earlier configurations (Section 6).

    The 9-cell window sweep and the one ablation that is not a sweep cell
    (skew_kurt). The frozen model counts as a trial but was only scored on
    2019-2024, so it has no full-period Sharpe to enter the variance.
    """
    sweep = pd.read_csv(os.path.join(PROJECT_ROOT, "results", "sensitivity_results.csv"))
    ablation = pd.read_csv(os.path.join(PROJECT_ROOT, "results", "ablation_results.csv"))
    return list(sweep["sharpe"]) + list(ablation.loc[ablation["feature_set"] == "skew_kurt", "sharpe"])


def _metrics(result: pd.DataFrame, window: pd.DatetimeIndex, rf: pd.Series) -> dict:
    r = result["portfolio_return"].reindex(window).dropna()
    m = compute_all(r, (1 + r).cumprod(), rf=rf)
    m["turnover"] = round(float(result["turnover"].reindex(r.index).mean()), 4)
    return m


def run_study(
    returns: pd.DataFrame,
    rf: pd.Series,
    features: pd.DataFrame,
    regimes: pd.DataFrame,
    window: str = "dev",
    placebo_step: int = 21,
    n_bootstrap: int = 1000,
    prior_trials: list[float] | None = None,
) -> dict:
    """Every cell and test of the pre-registration, for one sample window.

    dev and backcast are analysed in full and scored from their second regime day;
    the holdout is scored from HOLDOUT_START only.
    """
    if window not in WINDOWS:
        raise ValueError(f"window must be one of {WINDOWS}")
    full = window in FULL_WINDOWS
    index = regimes.index
    if full:
        end = DEV_END if window == "dev" else BACKCAST_END
        index = index[index <= end]
        if len(index) < 2:
            raise ValueError(f"{window} window has no regime days on or before {end.date()}")
        returns, features = returns.loc[:end], features.loc[:end]
    elif index.max() <= DEV_END:
        raise ValueError("holdout window needs data after 2024-12-31; extend data_end first")
    regimes = regimes.loc[index]
    score = index[1:] if full else index[index >= HOLDOUT_START]
    primary = PRIMARY_BP / 10_000

    targets = optimizer_targets(index, returns, rf)
    post = posteriors_of(regimes)
    h4_targets = correlation_aware_targets(targets, stock_bond_correlation(returns))
    h4_returns = with_cash(returns, rf)

    def cell(posteriors, mapping=None, cost=primary, tgt=targets, rets=returns):
        return run_cell(tgt, posteriors, rets, mapping=mapping, cost=cost)

    def ret(result):
        return result["portfolio_return"].reindex(score).dropna()

    investable = {
        "C0": lambda c: cell(post, cost=c),
        "C1": lambda c: cell(constant_posteriors(index), cost=c),
        "H4": lambda c: cell(post, cost=c, tgt=h4_targets, rets=h4_returns),
    }
    if full:
        for name in sorted(OPTIMIZERS):
            investable[f"B_{name}"] = (lambda n: lambda c: cell(post, {l: n for l in LABELS}, cost=c))(name)

    results = {bp: {name: run(bp / 10_000) for name, run in investable.items()} for bp in COSTS_BP}

    oracle_maps = {}
    if full:
        d1 = hindsight_posteriors(features, index)
        d1b = realized_vol_posteriors(returns["SPY"], index)
        books = {n: ret(results[PRIMARY_BP][f"B_{n}"]) for n in sorted(OPTIMIZERS)}
        oracle_maps["R1"] = hindsight_mapping(books, argmax_labels(post).reindex(score), rf)
        oracle_maps["F"] = hindsight_mapping(books, argmax_labels(d1).reindex(score), rf)
        oracles = {
            "D1":  lambda c: cell(d1, cost=c),
            "D1b": lambda c: cell(d1b, cost=c),
            "R1":  lambda c: cell(post, oracle_maps["R1"], cost=c),
            "F":   lambda c: cell(d1, oracle_maps["F"], cost=c),
        }
        for bp in COSTS_BP:
            results[bp].update({name: run(bp / 10_000) for name, run in oracles.items()})

    shifts = placebo_shifts(len(index), 252, placebo_step)
    placebo = {bp: [annual_sharpe(ret(cell(shifted_posteriors(post, k), cost=bp / 10_000)), rf)
                    for k in shifts] for bp in COSTS_BP}

    p = {name: ret(r) for name, r in results[PRIMARY_BP].items()}
    sr = {name: annual_sharpe(r, rf) for name, r in p.items()}

    summary = {
        "window": window,
        "score_start": str(score[0].date()),
        "score_end": str(score[-1].date()),
        "n_days": len(score),
        "cells": {bp: {name: _metrics(r, score, rf) for name, r in results[bp].items()} for bp in COSTS_BP},
        "placebo": {bp: {"n_shifts": len(shifts), "median_sharpe": float(np.median(placebo[bp])),
                         "p_value_vs_C0": placebo_p_value(
                             annual_sharpe(ret(results[bp]["C0"]), rf), placebo[bp])}
                    for bp in COSTS_BP},
        "H2": {
            "lw_C0_vs_C1": sharpe_difference_test(p["C0"], p["C1"], rf=rf),
            "bootstrap_C0_vs_C1": summarize(run_bootstrap(p["C0"], p["C1"], n_iterations=n_bootstrap, rf=rf)),
        },
        "H4": {
            "lw_H4_vs_C0": sharpe_difference_test(p["H4"], p["C0"], rf=rf),
            "lw_H4_vs_C1": sharpe_difference_test(p["H4"], p["C1"], rf=rf),
        },
    }
    lw = summary["H2"]["lw_C0_vs_C1"]["p_value"]
    summary["H2"]["supported"] = bool(lw < 0.05 and summary["placebo"][PRIMARY_BP]["p_value_vs_C0"] < 0.05)

    if full:
        prior = prior_trial_sharpes() if prior_trials is None else prior_trials
        study = [sr["C0"], sr["C1"]] + [sr[f"B_{n}"] for n in sorted(OPTIMIZERS)]
        trials = [s / np.sqrt(TRADING_DAYS) for s in prior + study]
        summary["H1"] = rotation_regressions(books, post.reindex(index)).to_dict(orient="index")
        summary["H1"]["supported"] = bool(any(v["reject"] for v in summary["H1"].values()))
        h3 = gap_difference_ci(p["C0"], p["D1"], p["R1"], rf, n_iterations=n_bootstrap)
        h3b = gap_difference_ci(p["C0"], p["D1b"], p["R1"], rf, n_iterations=n_bootstrap)
        summary["H3"] = {"D1": {**h3, "verdict": verdict(h3)}, "D1b": {**h3b, "verdict": verdict(h3b)},
                         "interaction": sr["F"] - sr["D1"] - sr["R1"] + sr["C0"],
                         "mappings": oracle_maps}
        # The frozen model counts in N but has no full-period Sharpe for the variance.
        n = len(trials) + 1
        summary["H4"]["dsr"] = deflated_sharpe_ratio(
            p["H4"], trials + [sr["H4"] / np.sqrt(TRADING_DAYS)], rf, n_trials=n + 1)
        summary["H4"]["supported"] = bool(summary["H4"]["lw_H4_vs_C0"]["p_value"] < 0.05
                                          and summary["H4"]["dsr"]["dsr"] > 0.95)
        summary["C0_dsr"] = deflated_sharpe_ratio(p["C0"], trials, rf, n_trials=n)
        summary["yearly_returns"] = (
            pd.DataFrame(p).groupby(lambda d: d.year).apply(lambda g: (1 + g).prod() - 1)
            .round(4).to_dict(orient="index")
        )
    summary["placebo_sharpes"] = placebo
    return summary


def write(summary: dict, out_dir: str) -> None:
    """Write the summary as JSON plus a cells table per cost."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    for bp, cells in summary["cells"].items():
        pd.DataFrame(cells).T.to_csv(os.path.join(out_dir, f"cells_{bp}bp.csv"))
    print(f"Results written to {out_dir}")


def run(window: str) -> dict:
    """Load the active config's data, run the study for `window` and write the results."""
    from data.fetch import RAW_PATH
    from data.risk_free import RF_PATH, fetch_risk_free
    from models.hmm import load_features, walk_forward_regimes

    data_end = pd.Timestamp(CFG["evaluation"]["data_end"])
    proxies = bool(CFG.get("universe"))
    if window == "dev" and data_end > DEV_END:
        raise SystemExit("dev run with data_end after 2024-12-31 would load holdout data; refusing")
    if window == "backcast" and (data_end > BACKCAST_END or not proxies):
        raise SystemExit("backcast needs the backcast config (proxy universe, data_end 2004-12-31); "
                         "use python -m research.run_backcast")
    if window != "backcast" and proxies:
        raise SystemExit("the active config is the backcast proxy universe; only --window backcast fits it")

    returns = pd.read_parquet(CFG["paths"]["returns"])
    rf = fetch_risk_free()
    features = load_features()
    regimes = walk_forward_regimes(features)

    summary = run_study(returns, rf, features, regimes, window=window)
    summary["data_sha256"] = {"prices": file_sha256(RAW_PATH), "risk_free": file_sha256(RF_PATH)}
    write(summary, os.path.join(OUT_DIR, window))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Detection vs response study")
    parser.add_argument("--window", choices=list(WINDOWS), required=True)
    run(parser.parse_args().window)


if __name__ == "__main__":
    main()
