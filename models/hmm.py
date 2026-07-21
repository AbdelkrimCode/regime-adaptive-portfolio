import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import joblib
import os
from sklearn.preprocessing import StandardScaler
from config import load_config


CFG = load_config()



def load_features(path: str | None = None) -> pd.DataFrame:
    if path is None:
        path = CFG["paths"]["features"]
    df = pd.read_parquet(path)
    df = df.dropna()
    return df

def _fit_hmm_core(features_scaled: np.ndarray, n_states: int) -> GaussianHMM | None:
    best_model = None
    best_score = -np.inf
    last_model = None
    n_failed = 0

    for i in range(CFG["hmm"]["n_init"]):
        try:
            model = GaussianHMM(
                n_components=n_states,
                covariance_type="full",
                n_iter=CFG["hmm"]["n_iter"],
                random_state=CFG["hmm"]["random_state"] + i
            )
            model.fit(features_scaled)
            score = model.score(features_scaled)
            last_model = model
            if score > best_score:
                best_score = score
                best_model = model
        except Exception:
            n_failed += 1
            continue

    chosen = best_model if best_model is not None else last_model

    if chosen is not None:
        # hmmlearn's own `monitor_.converged` is True whenever `iter == n_iter`
        # OR the tolerance was met - i.e. it also counts "ran out of iterations"
        # as converged. We only want the genuine signal: did the log-likelihood
        # improvement actually drop below tol before the budget ran out?
        history = chosen.monitor_.history
        tol = chosen.monitor_.tol
        chosen.converged_ = bool(
            len(history) >= 2 and (history[-1] - history[-2] < tol)
        )
        chosen.n_failed_restarts_ = n_failed
        if not chosen.converged_:
            print(
                f"      WARNING: HMM fit (n_states={n_states}) did not converge "
                f"within n_iter={CFG['hmm']['n_iter']} "
                f"({n_failed}/{CFG['hmm']['n_init']} restarts raised an exception)"
            )

    return chosen

def forward_filter(model: GaussianHMM, features_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.stats import multivariate_normal

    n_samples = len(features_scaled)
    n_states = model.n_components

    log_emission = np.zeros((n_samples, n_states))
    for i in range(n_states):
        log_emission[:, i] = multivariate_normal.logpdf(
            features_scaled,
            mean=model.means_[i],
            cov=model.covars_[i]
        )

    alpha = np.zeros((n_samples, n_states))
    alpha[0] = model.startprob_ * np.exp(log_emission[0])
    total = alpha[0].sum()
    if total > 0:
        alpha[0] /= total
    else:
        alpha[0] = np.ones(n_states) / n_states

    for t in range(1, n_samples):
        alpha[t] = alpha[t - 1] @ model.transmat_ * np.exp(log_emission[t])
        total = alpha[t].sum()
        if total > 0:
            alpha[t] /= total
        else:
            alpha[t] = np.ones(n_states) / n_states

    hidden_states = np.argmax(alpha, axis=1)
    return hidden_states, alpha

def fit_hmm(features: np.ndarray) -> tuple[GaussianHMM | None, StandardScaler]:
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    model = _fit_hmm_core(features_scaled, CFG["hmm"]["n_states"])
    return model, scaler


def fit_hmm_with_n(features_scaled: np.ndarray, n_states: int) -> tuple[GaussianHMM | None, None]:
    model = _fit_hmm_core(features_scaled, n_states)
    return model, None

def _fit_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    retrain_date: pd.Timestamp,
    next_date: pd.Timestamp,
    features_cols: list[str]
) -> tuple[pd.DataFrame, dict] | None:
    
    train_features = train_df[features_cols].values
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)

    scores_df = select_n_states(train_scaled, candidate_states=[2, 3, 4])
    if scores_df.empty:
        return None
    best_n = scores_df["bic"].idxmin()

    print(f"  Selected n_states={best_n} for fold {retrain_date.date()}")

    model, _ = fit_hmm_with_n(train_scaled, best_n)
    if model is None:
        return None

    test_features = test_df[features_cols].values
    state_labels = label_states(model, feature_cols=features_cols)
    features_scaled = scaler.transform(test_features)
    hidden_states, posteriors = forward_filter(model, features_scaled)

    period_df = test_df.copy()
    period_df["state"] = hidden_states
    period_df["regime"] = period_df["state"].map(state_labels)

    for state_idx, label in state_labels.items():
        period_df[f"p_{label.lower()}"] = posteriors[:, state_idx]

    n_states_entry = {
        "date": retrain_date,
        "n_states": int(best_n),
        "converged": bool(getattr(model, "converged_", True)),
    }
    print(f"Fitted {retrain_date.date()} -> test through {next_date.date()}")
    return period_df, n_states_entry

def count_params(n_states: int, n_features: int) -> int:
    transition = n_states * (n_states - 1)
    means = n_states * n_features
    covariances = n_states * n_features * (n_features + 1) // 2
    startprob = n_states - 1
    return transition + means + covariances + startprob


def compute_aic_bic(model: GaussianHMM, features_scaled: np.ndarray, n_features: int) -> tuple[float, float]:
    n_samples = len(features_scaled)
    n_states = model.n_components
    log_likelihood = model.score(features_scaled) * n_samples
    n_params = count_params(n_states, n_features)
    aic = -2 * log_likelihood + 2 * n_params
    bic = -2 * log_likelihood + np.log(n_samples) * n_params
    return aic, bic

def select_n_states(features_scaled: np.ndarray, candidate_states: list[int] | None = None) -> pd.DataFrame:
    if candidate_states is None:
        candidate_states = [2, 3, 4, 5]

    n_features = features_scaled.shape[1]

    records = []
    for n in candidate_states:
        print(f"  Fitting HMM with {n} states...")
        model, _ = fit_hmm_with_n(features_scaled, n)
        if model is None:
            continue
        aic, bic = compute_aic_bic(model, features_scaled, n_features)
        records.append({"n_states": n, "aic": round(aic, 2), "bic": round(bic, 2)})

    return pd.DataFrame(records).set_index("n_states")

def plot_state_selection(scores_df: pd.DataFrame, output_path: str | None = None) -> None:
    import matplotlib.pyplot as plt
    if output_path is None:
        output_path = CFG["paths"]["state_selection"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(scores_df.index, scores_df["aic"], marker="o", color="steelblue")
    ax1.set_title("AIC by Number of States")
    ax1.set_xlabel("n_states")
    ax1.set_ylabel("AIC")
    ax1.axvline(scores_df["aic"].idxmin(), color="steelblue", linestyle="--", alpha=0.5)

    ax2.plot(scores_df.index, scores_df["bic"], marker="o", color="darkorange")
    ax2.set_title("BIC by Number of States")
    ax2.set_xlabel("n_states")
    ax2.set_ylabel("BIC")
    ax2.axvline(scores_df["bic"].idxmin(), color="darkorange", linestyle="--", alpha=0.5)

    fig.suptitle("HMM State Selection — AIC / BIC", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"      Saved to {output_path}")

def label_states(model: GaussianHMM, feature_cols: list[str] | None = None) -> dict:
    if feature_cols is not None and "spy_return" in feature_cols:
        return_idx = feature_cols.index("spy_return")
        vol_idx = feature_cols.index("spy_vol") if "spy_vol" in feature_cols else None
    else:
        # Backward-compatible fallback for callers that don't pass feature_cols:
        # assumes the original [spy_return, spy_vol, ...] column order.
        return_idx = 0
        vol_idx = 1 if model.means_.shape[1] > 1 else None

    state_means = model.means_[:, return_idx]
    state_vols = model.means_[:, vol_idx] if vol_idx is not None else np.zeros(len(state_means))

    ranking = np.argsort(state_means - 0.5 * state_vols)
    n = len(ranking)
    if n == 2:
        return {
            ranking[0]: "Bear",
            ranking[1]: "Bull"
        }
    if n == 3:
        return {
            ranking[0]: "Bear",
            ranking[1]: "Sideways",
            ranking[2]: "Bull"
        }
    if n == 4:
        return {
            ranking[0]: "Crash",
            ranking[1]: "Bear",
            ranking[2]: "Sideways",
            ranking[3]: "Bull"
        }
    return {ranking[i]: f"State{i}" for i in range(n)}

def get_fitted_transition_matrix(model: GaussianHMM, state_labels: dict) -> pd.DataFrame:
    """Reads a single fitted model's own theoretical transmat_ parameter directly.

    This is distinct from main.compute_empirical_transition_matrix(), which counts
    actual observed regime-label transitions across a walk-forward run (possibly
    spanning many retrained models). The two are not interchangeable and will not
    generally agree - this one reflects what one specific model learned, the other
    reflects what actually happened across the full walk-forward output."""
    n = model.n_components
    labels = [state_labels[i] for i in range(n)]
    transmat = pd.DataFrame(
        model.transmat_,
        index=labels,
        columns=labels
    )
    return transmat

def get_regime_durations(transmat: pd.DataFrame) -> pd.Series:
    diag = pd.Series({
        regime: transmat.loc[regime, regime]
        for regime in transmat.index
    })
    diag = diag.clip(upper=1 - 1e-10)
    avg_durations = 1 / (1 - diag)
    avg_durations.name = "avg_duration_days"
    return avg_durations.round(1)

def predict_regimes(model: GaussianHMM, features: np.ndarray,
                    feature_df: pd.DataFrame,
                    scaler: StandardScaler) -> pd.DataFrame:
    state_labels = label_states(model, feature_cols=feature_df.columns.tolist())
    features_scaled = scaler.transform(features)
    hidden_states, posteriors = forward_filter(model, features_scaled)

    df = feature_df.copy()
    df["state"] = hidden_states
    df["regime"] = df["state"].map(state_labels)

    for state_idx, label in state_labels.items():
        df[f"p_{label.lower()}"] = posteriors[:, state_idx]

    df["is_retrain_date"] = False
    return df

def walk_forward_regimes(df: pd.DataFrame, n_jobs: int | None = None) -> pd.DataFrame:
    from joblib import Parallel, delayed

    features_cols = df.columns.tolist()

    raw_dates = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=CFG["hmm"]["retrain_frequency"]
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

    effective_n_jobs = n_jobs if n_jobs is not None else CFG["hmm"]["n_jobs"]
    results = Parallel(n_jobs=effective_n_jobs)(
        delayed(_fit_fold)(train_df, test_df, retrain_date, next_date, features_cols)
        for train_df, test_df, retrain_date, next_date in folds
    )

    all_regimes = []
    n_states_log = []
    for result in results:
        if result is None:
            continue
        period_df, n_states_entry = result
        all_regimes.append(period_df)
        n_states_log.append(n_states_entry)

    if not all_regimes:
        raise ValueError(
            "walk_forward_regimes: no folds produced output. "
            "Check min_train_days or data length."
        )
    
    result = pd.concat(all_regimes)
    result["is_retrain_date"] = False
    result.loc[result.index.isin(retrain_dates), "is_retrain_date"] = True

    n_states_df = pd.DataFrame(n_states_log).set_index("date")
    n_states_df.index = pd.DatetimeIndex(n_states_df.index)
    result["n_states_used"] = result.index.map(
        lambda d: n_states_df["n_states"].asof(d) if not n_states_df.empty else CFG["hmm"]["n_states"]
    )
    result["converged"] = result.index.map(
        lambda d: bool(n_states_df["converged"].asof(d)) if not n_states_df.empty else True
    )

    n_folds = len(n_states_log)
    n_converged = sum(1 for entry in n_states_log if entry["converged"])
    print(f"  HMM convergence: {n_converged}/{n_folds} folds converged")

    return result

def audit_walk_forward(output_path: str | None = None) -> pd.DataFrame:
    if output_path is None:
        output_path = CFG["paths"]["walk_forward_audit"]
    df = load_features()

    raw_dates = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=CFG["hmm"]["retrain_frequency"]
    )
    retrain_dates = pd.DatetimeIndex([
        df.index[df.index.searchsorted(d, side="left")]
        for d in raw_dates
        if df.index.searchsorted(d, side="left") < len(df.index)
    ])

    records = []
    for i, retrain_date in enumerate(retrain_dates):
        next_date = retrain_dates[i + 1] if i + 1 < len(retrain_dates) else df.index.max()
        train_df = df.loc[:retrain_date]
        test_df = df.loc[retrain_date:next_date].iloc[1:]

        if len(train_df) < CFG["hmm"]["min_train_days"] or len(test_df) == 0:
            continue

        train_end = train_df.index[-1]
        test_start = test_df.index[0]
        leakage = train_end >= test_start

        records.append({
            "fold": i,
            "retrain_date": retrain_date.date(),
            "train_end": train_end.date(),
            "test_start": test_start.date(),
            "test_end": next_date.date(),
            "train_days": len(train_df),
            "test_days": len(test_df),
            "leakage": leakage
        })

    audit_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    audit_df.to_csv(output_path, index=False)
    print(f"Audit saved to {output_path}")
    print(f"Total folds: {len(audit_df)}")
    print(f"Leakage detected: {audit_df['leakage'].sum()} folds")
    return audit_df


def save_model(model: GaussianHMM, scaler: StandardScaler, path: str | None = None) -> None:
    if path is None:
        path = CFG["paths"]["model"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, path)

def load_model(path: str | None = None) -> tuple[GaussianHMM, StandardScaler]:
    if path is None:
        path = CFG["paths"]["model"]
    data = joblib.load(path)
    return data["model"], data["scaler"]

def run(retrain : bool = False, walk_forward : bool = False) -> pd.DataFrame:
    df = load_features()
    features = df.values

    if walk_forward:
        print("Running walk-forward retraining...")
        result = walk_forward_regimes(df)
    else:
        if retrain or not os.path.exists(CFG["paths"]["model"]):
            model, scaler = fit_hmm(features)
            save_model(model, scaler)
        else:
            model, scaler = load_model()
        result = predict_regimes(model, features, df, scaler)

    os.makedirs(os.path.dirname(CFG["paths"]["regimes"]), exist_ok=True)
    result.to_parquet(CFG["paths"]["regimes"])
    return result

if __name__ == "__main__":
    result = run(retrain=True)
    print(result["regime"].value_counts())
    print(result.groupby("regime")["spy_return"].mean())

    print("\nRunning BIC/AIC state selection...")
    df = load_features()
    train_end = CFG["evaluation"]["train_end"]
    df_train = df.loc[:train_end]
    features = df_train[["spy_return", "spy_vol", "mean_corr"]].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    scores = select_n_states(features_scaled)
    print(scores)
    plot_state_selection(scores)
    print("\nTransition Matrix:")
    model, scaler = load_model()
    df = load_features()
    state_labels = label_states(model, feature_cols=df.columns.tolist())
    transmat = get_fitted_transition_matrix(model, state_labels)
    print(transmat.round(4))
    print("\nAverage Regime Durations:")
    print(get_regime_durations(transmat))