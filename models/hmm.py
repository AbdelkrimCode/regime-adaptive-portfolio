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

def label_states(model: GaussianHMM, feature_df: pd.DataFrame) -> dict:
    state_means = model.means_[:, 0]
    ranking = np.argsort(state_means)
    return {
        ranking[0]: "Bear",
        ranking[1]: "Sideways",
        ranking[2]: "Bull"
    }

def predict_regimes(model: GaussianHMM, features: np.ndarray,
                    feature_df: pd.DataFrame,
                    scaler: StandardScaler) -> pd.DataFrame:
    state_labels = label_states(model, feature_df)
    features_scaled = scaler.transform(features)
    hidden_states = model.predict(features_scaled)
    df = feature_df.copy()
    df["state"] = hidden_states
    df["regime"] = df["state"].map(state_labels)
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

        period_df = test_df.copy()
        period_df["state"] = hidden_states
        period_df["regime"] = period_df["state"].map(state_labels)

        all_regimes.append(period_df)
        print(f"Fitted {retrain_date.date()} → test through {next_date.date()}")

    return pd.concat(all_regimes)


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