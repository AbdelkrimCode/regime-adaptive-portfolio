import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import joblib
import os
from sklearn.preprocessing import StandardScaler
from config import load_config
import matplotlib.pyplot as plt

CFG = load_config()



def load_features(path: str | None = None) -> pd.DataFrame:
    if path is None:
        path = CFG["paths"]["features"]
    df = pd.read_parquet(path)
    df = df.dropna()
    return df

def fit_hmm(features: np.ndarray) -> tuple[GaussianHMM, StandardScaler]:
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    best_model = None
    best_score = -np.inf
    last_model = None

    for i in range(CFG["hmm"]["n_init"]):
        try:
            model = GaussianHMM(
                n_components=CFG["hmm"]["n_states"],
                covariance_type="full",
                n_iter=CFG["hmm"]["n_iter"],
                random_state=CFG["hmm"]["random_state"] + i
            )
            model.fit(features_scaled)
            last_model = model
            score = model.score(features_scaled)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception:
            continue

    if best_model is None:
        best_model = last_model

    return best_model, scaler

def fit_hmm_with_n(features_scaled: np.ndarray, n_states: int) -> tuple[GaussianHMM | None, None]:
    best_model = None
    best_score = -np.inf
    last_model = None

    for i in range(CFG["hmm"]["n_init"]):
        try:
            model = GaussianHMM(
                n_components=n_states,
                covariance_type="full",
                n_iter=CFG["hmm"]["n_iter"],
                random_state=CFG["hmm"]["random_state"] + i
            )
            model.fit(features_scaled)
            last_model = model
            score = model.score(features_scaled)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception:
            continue

    if best_model is None:
        best_model = last_model

    return best_model, None

def count_params(n_states: int, n_features: int) -> int:
    transition = n_states * (n_states - 1)
    means = n_states * n_features
    covariances = n_states * n_features * (n_features + 1) // 2
    return transition + means + covariances


def compute_aic_bic(model: GaussianHMM, features_scaled: np.ndarray, n_features: int) -> tuple[float, float]:
    n_samples = len(features_scaled)
    n_states = model.n_components
    log_likelihood = model.score(features_scaled) * n_samples
    n_params = count_params(n_states, n_features)
    aic = -2 * log_likelihood + 2 * n_params
    bic = -2 * log_likelihood + np.log(n_samples) * n_params
    return aic, bic

def select_n_states(features: np.ndarray, candidate_states: list[int] | None = None) -> pd.DataFrame:
    if candidate_states is None:
        candidate_states = [2, 3, 4, 5]

    split = int(len(features) * 0.8)
    train_features = features[:split]
    test_features = features[split:]

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    n_features = features.shape[1]

    records = []
    for n in candidate_states:
            print(f"  Fitting HMM with {n} states...")
            model, _ = fit_hmm_with_n(train_scaled, n)
            if model is None:
                continue
            aic, bic = compute_aic_bic(model, test_scaled, n_features)
            records.append({"n_states": n, "aic": round(aic, 2), "bic": round(bic, 2)})

    return pd.DataFrame(records).set_index("n_states")

def plot_state_selection(scores_df: pd.DataFrame, output_path: str | None = None) -> None:
    if output_path is None:
        output_path = "data/state_selection.png"

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

def label_states(model: GaussianHMM, feature_df: pd.DataFrame) -> dict:
    state_means = model.means_[:, 0]
    ranking = np.argsort(state_means)
    labels = ["Crash", "Bear", "Sideways", "Bull"]
    n = model.n_components
    return {ranking[i]: labels[4 - n + i] for i in range(n)}

def get_transition_matrix(model: GaussianHMM, state_labels: dict) -> pd.DataFrame:
    n = model.n_components
    labels = [state_labels[i] for i in range(n)]
    transmat = pd.DataFrame(
        model.transmat_,
        index=labels,
        columns=labels
    )
    return transmat

def get_regime_durations(transmat: pd.DataFrame) -> pd.Series:
    avg_durations = 1 / (1 - pd.Series({
        regime: transmat.loc[regime, regime]
        for regime in transmat.index
    }))
    avg_durations.name = "avg_duration_days"
    return avg_durations.round(1)

def predict_regimes(model: GaussianHMM, features: np.ndarray,
                    feature_df: pd.DataFrame,
                    scaler: StandardScaler) -> pd.DataFrame:
    state_labels = label_states(model, feature_df)
    features_scaled = scaler.transform(features)
    hidden_states = model.predict(features_scaled)
    posteriors = model.predict_proba(features_scaled)

    df = feature_df.copy()
    df["state"] = hidden_states
    df["regime"] = df["state"].map(state_labels)

    for state_idx, label in state_labels.items():
        df[f"p_{label.lower()}"] = posteriors[:, state_idx]

    df["is_retrain_date"] = False
    return df

def walk_forward_regimes(df: pd.DataFrame) -> pd.DataFrame:
    features_cols = ["spy_return", "spy_vol", "mean_corr"]

    retrain_dates = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=CFG["hmm"]["retrain_frequency"]
    )

    all_regimes = []

    for i, retrain_date in enumerate(retrain_dates):
        next_date = retrain_dates[i + 1] if i + 1 < len(retrain_dates) else df.index.max()

        train_df = df.loc[:retrain_date]
        test_df = df.loc[retrain_date:next_date].iloc[1:]

        if len(train_df) < CFG["hmm"]["min_train_days"] or len(test_df) == 0:
            continue

        train_features = train_df[features_cols].values
        model, scaler = fit_hmm(train_features)

        if model is None:
            continue

        test_features = test_df[features_cols].values
        state_labels = label_states(model, train_df)
        features_scaled = scaler.transform(test_features)
        hidden_states = model.predict(features_scaled)

        posteriors = model.predict_proba(features_scaled)

        period_df = test_df.copy()
        period_df["state"] = hidden_states
        period_df["regime"] = period_df["state"].map(state_labels)

        for state_idx, label in state_labels.items():
            period_df[f"p_{label.lower()}"] = posteriors[:, state_idx]

        all_regimes.append(period_df)
        print(f"Fitted {retrain_date.date()} → test through {next_date.date()}")

        result = pd.concat(all_regimes)
        result["is_retrain_date"] = False
        result.loc[result.index.isin(retrain_dates), "is_retrain_date"] = True
    return result


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
    features = df[["spy_return", "spy_vol", "mean_corr"]].values

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
    scores = select_n_states(features)
    print(scores)
    plot_state_selection(scores)
    print("\nTransition Matrix:")
    model, scaler = load_model()
    df = load_features()
    state_labels = label_states(model, df)
    transmat = get_transition_matrix(model, state_labels)
    print(transmat.round(4))
    print("\nAverage Regime Durations:")
    print(get_regime_durations(transmat))