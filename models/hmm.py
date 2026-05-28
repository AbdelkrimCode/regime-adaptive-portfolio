import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import joblib
import os

FEATURES_PATH = "data/processed/features.parquet"
MODEL_PATH = "models/hmm_model.joblib"
OUTPUT_PATH = "data/regimes.parquet"

N_STATES = 3
N_ITER = 2000
N_INIT = 10
RANDOM_STATE = 42

def load_features(path=FEATURES_PATH):
    df = pd.read_parquet(path)
    df = df.dropna()
    return df

def fit_hmm(features: np.ndarray) -> GaussianHMM:
    best_model = None
    best_score = -np.inf

    for i in range(N_INIT):
        model = GaussianHMM(
            n_components=N_STATES,
            covariance_type="full",
            n_iter=N_ITER,
            random_state=RANDOM_STATE + i
        )
        model.fit(features)
        score = model.score(features)
        if score > best_score:
            best_score = score
            best_model = model

    return best_model

def label_states(model: GaussianHMM, feature_df: pd.DataFrame) -> dict:
    state_means = model.means_[:, 0]
    ranking = np.argsort(state_means)
    return {
        ranking[0]: "Bear",
        ranking[1]: "Sideways",
        ranking[2]: "Bull"
    }

def predict_regimes(model: GaussianHMM, features: np.ndarray,
                    feature_df: pd.DataFrame) -> pd.DataFrame:
    state_labels = label_states(model, feature_df)
    hidden_states = model.predict(features)
    df = feature_df.copy()
    df["state"] = hidden_states
    df["regime"] = df["state"].map(state_labels)
    return df

def save_model(model: GaussianHMM, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load_model(path=MODEL_PATH) -> GaussianHMM:
    return joblib.load(path)

def run(retrain=False) -> pd.DataFrame:
    df = load_features()
    features = df[["spy_return", "spy_vol", "mean_corr"]].values
    if retrain or not os.path.exists(MODEL_PATH):
        model = fit_hmm(features)
        save_model(model)
    else:
        model = load_model()
    result = predict_regimes(model, features, df)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    result.to_parquet(OUTPUT_PATH)
    return result

if __name__ == "__main__":
    result = run(retrain=True)
    print(result["regime"].value_counts())
    print(result.groupby("regime")["spy_return"].mean())