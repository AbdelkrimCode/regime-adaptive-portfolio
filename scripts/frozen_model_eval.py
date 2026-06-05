import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from models.hmm import load_features, fit_hmm, forward_filter, label_states
from backtest.engine import simulate
from backtest.metrics import compute_all
from backtest.benchmark import run as run_benchmark
from optimization.switcher import compute_weights
from data.risk_free import fetch_risk_free
from config import load_config

CFG = load_config()

FREEZE_DATE = CFG["evaluation"]["train_end"]
TEST_START  = CFG["evaluation"]["test_start"]
TEST_END    = "2024-12-31"
FEATURES_COLS = ["spy_return", "spy_vol", "mean_corr"]

def train_frozen_model() -> tuple:
    df = load_features()
    train_df = df.loc[:FREEZE_DATE]
    features = train_df[FEATURES_COLS].values
    model, scaler = fit_hmm(features)
    print(f"Frozen model trained on {len(train_df)} days (up to {FREEZE_DATE})")
    print(f"  n_states selected: {model.n_components}")
    return model, scaler, df

def build_regimes(model, scaler, df: pd.DataFrame) -> pd.DataFrame:
    test_df = df.loc[TEST_START:TEST_END]
    features_scaled = scaler.transform(test_df[FEATURES_COLS].values)
    hidden_states, posteriors = forward_filter(model, features_scaled)
    state_labels = label_states(model)

    regimes = test_df.copy()
    regimes["state"] = hidden_states
    regimes["regime"] = regimes["state"].map(state_labels)
    regimes["is_retrain_date"] = False

    for state_idx, label in state_labels.items():
        regimes[f"p_{label.lower()}"] = posteriors[:, state_idx]

    return regimes

def run_frozen_eval() -> None:
    print("=" * 50)
    print("  Frozen Model Evaluation")
    print(f"  Train: 2006 → {FREEZE_DATE}")
    print(f"  Test:  {TEST_START} → {TEST_END}")
    print("=" * 50)

    model, scaler, df = train_frozen_model()
    regimes = build_regimes(model, scaler, df)

    returns = pd.read_parquet(CFG["paths"]["returns"])
    weights = compute_weights(regimes, returns)
    result = simulate(weights, returns.loc[TEST_START:TEST_END])

    rf = fetch_risk_free()
    port_metrics = compute_all(result["portfolio_return"], result["equity"], rf=rf)

    spy = returns.loc[TEST_START:TEST_END, "SPY"]
    spy_equity = (1 + spy).cumprod()
    spy_metrics = compute_all(spy, spy_equity, rf=rf)

    print(f"\n{'Metric':<25} {'Frozen Model':>14} {'SPY':>10}")
    print("-" * 52)
    for metric in port_metrics:
        print(f"{metric:<25} {port_metrics[metric]:>14} {spy_metrics[metric]:>10}")

    print("\nRegime distribution (test period):")
    print(regimes["regime"].value_counts().to_dict())

if __name__ == "__main__":
    run_frozen_eval()