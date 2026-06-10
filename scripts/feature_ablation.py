import pandas as pd
from data.fetch import fetch_prices
from data.process import compute_returns, compute_features, compute_skew_kurt_features
from models.hmm import walk_forward_regimes
from backtest.engine import run as run_backtest, run_period
from backtest.metrics import compute_all
from data.risk_free import fetch_risk_free
from config import load_config

CFG = load_config()

RESULTS_PATH = CFG["paths"]["ablation_results"]
TEST_START   = CFG["evaluation"]["test_start"]

FEATURE_SETS = {
    "baseline":  {"vol_window": 21, "corr_window": 63, "type": "standard"},
    "vol_10":    {"vol_window": 10, "corr_window": 63, "type": "standard"},
    "vol_42":    {"vol_window": 42, "corr_window": 63, "type": "standard"},
    "skew_kurt": {"vol_window": 21, "corr_window": 63, "type": "skew_kurt"},
}

def run_single(name: str, config: dict) -> dict:
    print(f"\n--- Feature set: {name} ---")

    prices = fetch_prices()
    returns = compute_returns(prices)

    if config["type"] == "skew_kurt":
        features = compute_skew_kurt_features(returns, window=config["vol_window"])
    else:
        features = compute_features(returns, vol_window=config["vol_window"], corr_window=config["corr_window"])

    features.to_parquet(CFG["paths"]["features"])
    returns.to_parquet(CFG["paths"]["returns"])

    rf = fetch_risk_free()
    regimes = walk_forward_regimes(features)
    regimes.to_parquet(CFG["paths"]["regimes"])

    backtest, _ = run_backtest(regimes_df=regimes)
    full_metrics = compute_all(backtest["portfolio_return"], backtest["equity"], rf=rf)

    test_result, _ = run_period(start=TEST_START, end=CFG["evaluation"]["data_end"], regimes_df=regimes, returns_df=returns)
    test_metrics = compute_all(test_result["portfolio_return"], test_result["equity"], rf=rf)

    return {
        "feature_set":       name,
        "sharpe":            full_metrics["sharpe_ratio"],
        "max_drawdown":      full_metrics["max_drawdown"],
        "calmar":            full_metrics["calmar_ratio"],
        "held_out_sharpe":   test_metrics["sharpe_ratio"],
        "held_out_drawdown": test_metrics["max_drawdown"],
    }

def main() -> None:
    results = []
    try:
        for name, config in FEATURE_SETS.items():
            result = run_single(name, config)
            results.append(result)
            pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
            print(f"  Saved to {RESULTS_PATH}")

        df = pd.DataFrame(results)
        print("\n=== Feature Ablation Results ===")
        print(df.to_string(index=False))
    finally:
        prices = fetch_prices()
        returns = compute_returns(prices)
        baseline = FEATURE_SETS["baseline"]
        features = compute_features(returns,
            vol_window=baseline["vol_window"],
            corr_window=baseline["corr_window"])
        features.to_parquet(CFG["paths"]["features"])
        returns.to_parquet(CFG["paths"]["returns"])
        print("\nBaseline features restored.")

if __name__ == "__main__":
    main()