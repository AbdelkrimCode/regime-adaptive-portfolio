import argparse
import pandas as pd
from models.hmm import run as run_hmm
from backtest.engine import run as run_backtest
from backtest.benchmark import run as run_benchmark
from backtest.metrics import compute_all
from visualization.charts import run as run_charts

def print_metrics(results: dict):
    print(f"\n{'Metric':<25} {'Portfolio':>12} {'SPY':>12}")
    print("-" * 50)
    for metric in results["portfolio"]:
        port_val = results["portfolio"][metric]
        spy_val  = results["spy"][metric]
        print(f"{metric:<25} {port_val:>12} {spy_val:>12}")

def main(retrain=False, charts=True):
    print("=" * 50)
    print("  Regime Adaptive Portfolio")
    print("=" * 50)

    print("\n[1/4] Running HMM regime detection...")
    regimes = run_hmm(retrain=retrain)
    print(f"      Regimes found: {regimes['regime'].value_counts().to_dict()}")

    print("\n[2/4] Running backtest...")
    backtest = run_backtest()
    print(f"      Days simulated: {len(backtest)}")

    print("\n[3/4] Computing metrics...")
    results = run_benchmark()
    print_metrics(results)

    if charts:
        print("\n[4/4] Generating charts...")
        run_charts()
        print("      Saved to data/charts.png")

    print("\nDone.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regime Adaptive Portfolio")
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain HMM from scratch")
    parser.add_argument("--no-charts", action="store_true",
                        help="Skip chart generation")
    args = parser.parse_args()

    main(retrain=args.retrain, charts=not args.no_charts)