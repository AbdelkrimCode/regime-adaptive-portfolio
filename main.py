import argparse
import pandas as pd
from models.hmm import run as run_hmm
from backtest.engine import run as run_backtest, run_period
from backtest.metrics import compute_all
from backtest.benchmark import run as run_benchmark
from backtest.metrics import compute_all
from visualization.charts import run as run_charts
from config import load_config

CFG = load_config()

def print_metrics(results: dict):
    print(f"\n{'Metric':<25} {'Portfolio':>12} {'SPY':>12} {'EqualWeight':>12} {'60/40':>12} {'Momentum':>12}")
    print("-" * 80)
    for metric in results["portfolio"]:
        print(f"{metric:<25} "
              f"{results['portfolio'][metric]:>12} "
              f"{results['spy'][metric]:>12} "
              f"{results['equal_weight'][metric]:>12} "
              f"{results['sixty_forty'][metric]:>12} "
              f"{results['momentum'][metric]:>12}")

def main(retrain: bool = False, charts: bool = True, walk_forward: bool = False) -> None:
    print("=" * 50)
    print("  Regime Adaptive Portfolio")
    print("=" * 50)

    print("\n[1/4] Running HMM regime detection...")
    regimes = run_hmm(retrain=retrain, walk_forward=walk_forward)
    print(f"      Regimes found: {regimes['regime'].value_counts().to_dict()}")

    print("\n[2/4] Running backtest...")
    backtest = run_backtest(regimes_df=regimes)
    print(f"      Days simulated: {len(backtest)}")

    print("\n[3/4] Computing metrics...")
    results = run_benchmark()
    print_metrics(results)

    print("\n--- Held-out test period (2019–2024) ---")
    test_start = CFG["evaluation"]["test_start"]
    test_end = CFG["evaluation"]["train_end"]
    test_result = run_period(
        start=test_start,
        end="2024-12-31",
        regimes_df=regimes
    )
    returns = pd.read_parquet(CFG["paths"]["returns"])
    spy_test = returns.loc[test_start:, "SPY"]
    spy_test_equity = (1 + spy_test).cumprod()

    test_metrics = compute_all(test_result["portfolio_return"], test_result["equity"])
    spy_test_metrics = compute_all(spy_test, spy_test_equity)

    print(f"\n{'Metric':<25} {'Portfolio':>12} {'SPY':>12}")
    print("-" * 50)
    for metric in test_metrics:
        print(f"{metric:<25} {test_metrics[metric]:>12} {spy_test_metrics[metric]:>12}")

    if charts:
        print("\n[4/4] Generating charts...")
        run_charts()
        print(f"      Saved to {CFG['paths']['charts']}")

    print("\nDone.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regime Adaptive Portfolio")
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain HMM from scratch")
    parser.add_argument("--no-charts", action="store_true",
                        help="Skip chart generation")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Use walk-forward HMM retraining")
    args = parser.parse_args()

    main(retrain=args.retrain, charts=not args.no_charts, walk_forward=args.walk_forward)