"""
Temporary diagnostic script - NOT part of the permanent codebase.

Resolves a direct contradiction between the paper and the README about
whether the causal forward filter performs BETTER or WORSE than smoothed
(forward-backward) posteriors. Refits each walk-forward fold once (reusing
the BIC-selected model), then builds two parallel regime dataframes from the
SAME fitted models - one via forward_filter(), one via model.predict_proba()
(hmmlearn's forward-backward smoothed posterior, what the paper/README call
"Viterbi" - see note below) - and runs each through the full backtest to
compare Sharpe and Max Drawdown directly, holding everything else constant.

Note on terminology: hmmlearn's model.predict_proba() returns forward-backward
SMOOTHED posteriors, not the literal Viterbi path (model.predict(), a hard
MAP state sequence). The existing test_forward_filter_differs_from_viterbi
in tests/test_scripts.py already uses predict_proba() as its "Viterbi"
comparison, so this script matches that same convention for consistency.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

from models.hmm import (
    load_features, select_n_states, label_states, forward_filter, CFG
)
from backtest.engine import run_period
from backtest.metrics import compute_all
from data.risk_free import fetch_risk_free


def _fit_and_compare_fold(train_df, test_df, retrain_date, next_date, features_cols):
    train_features = train_df[features_cols].values
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)

    scores_df, candidate_models = select_n_states(
        train_scaled, candidate_states=[2, 3, 4], return_models=True
    )
    if scores_df.empty:
        return None
    best_n = scores_df["bic"].idxmin()
    model = candidate_models[best_n]
    if model is None:
        return None

    test_features = test_df[features_cols].values
    features_scaled = scaler.transform(test_features)
    state_labels = label_states(model, feature_cols=features_cols)

    # Forward filter (causal) - what the pipeline actually uses
    ff_hidden, ff_posteriors = forward_filter(model, features_scaled)

    # Smoothed posteriors (forward-backward) - what the paper/README call "Viterbi"
    smoothed_posteriors = model.predict_proba(features_scaled)
    smoothed_hidden = np.argmax(smoothed_posteriors, axis=1)

    def build_period_df(hidden, posteriors):
        period_df = test_df.copy()
        period_df["state"] = hidden
        period_df["regime"] = period_df["state"].map(state_labels)
        for state_idx, label in state_labels.items():
            period_df[f"p_{label.lower()}"] = posteriors[:, state_idx]
        return period_df

    print(f"  Fold {retrain_date.date()}: n_states={best_n}")
    return build_period_df(ff_hidden, ff_posteriors), build_period_df(smoothed_hidden, smoothed_posteriors)


def build_regimes(method: str, results: list) -> pd.DataFrame:
    idx = 0 if method == "forward_filter" else 1
    frames = [r[idx] for r in results if r is not None]
    combined = pd.concat(frames)
    combined["is_retrain_date"] = False
    return combined


def main():
    df = load_features()
    features_cols = df.columns.tolist()

    raw_dates = pd.date_range(
        start=df.index.min(), end=df.index.max(), freq=CFG["hmm"]["retrain_frequency"]
    )
    retrain_dates = pd.DatetimeIndex([
        df.index[df.index.searchsorted(d, side="left")]
        for d in raw_dates
        if df.index.searchsorted(d, side="left") < len(df.index)
    ])

    folds = []
    for i, retrain_date in enumerate(retrain_dates):
        next_date = retrain_dates[i + 1] if i + 1 < len(retrain_dates) else df.index.max()
        train_df = df.loc[:retrain_date]
        test_df = df.loc[retrain_date:next_date].iloc[1:]
        if len(train_df) < CFG["hmm"]["min_train_days"] or len(test_df) == 0:
            continue
        folds.append((train_df, test_df, retrain_date, next_date))

    print(f"Running {len(folds)} folds, fitting once per fold, comparing both decoding methods...")
    results = Parallel(n_jobs=CFG["hmm"]["n_jobs"])(
        delayed(_fit_and_compare_fold)(train_df, test_df, retrain_date, next_date, features_cols)
        for train_df, test_df, retrain_date, next_date in folds
    )

    returns = pd.read_parquet(CFG["paths"]["returns"])
    rf = fetch_risk_free()
    data_end = CFG["evaluation"]["data_end"]

    print("\n" + "=" * 60)
    print("  Forward Filter vs Smoothed Posteriors - Full Pipeline")
    print("=" * 60)

    rows = []
    for method in ["forward_filter", "smoothed"]:
        regimes = build_regimes(method, results)
        result, _ = run_period(start=str(df.index.min().date()), end=data_end, regimes_df=regimes, returns_df=returns)
        metrics = compute_all(result["portfolio_return"], result["equity"], rf=rf)
        rows.append({"method": method, **metrics})
        print(f"\n--- {method} ---")
        for k, v in metrics.items():
            print(f"  {k:<25} {v}")

    df_out = pd.DataFrame(rows)
    df_out.to_csv("results/forward_filter_vs_smoothed.csv", index=False)
    print(f"\nSaved to results/forward_filter_vs_smoothed.csv")

    ff_sharpe = rows[0]["sharpe_ratio"]
    sm_sharpe = rows[1]["sharpe_ratio"]
    print(f"\nDifference (forward_filter - smoothed): {ff_sharpe - sm_sharpe:+.4f}")
    if ff_sharpe > sm_sharpe:
        print("=> Forward filter OUTPERFORMS smoothed posteriors (matches paper's current claim)")
    else:
        print("=> Forward filter UNDERPERFORMS smoothed posteriors (matches README's current claim)")


if __name__ == "__main__":
    main()