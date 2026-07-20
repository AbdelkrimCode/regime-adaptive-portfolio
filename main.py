import argparse
import pandas as pd
import numpy as np
from models.hmm import run as run_hmm
from backtest.engine import run as run_backtest, run_period
from backtest.metrics import compute_all, average_turnover
from backtest.benchmark import run as run_benchmark
from visualization.charts import run as run_charts
from config import load_config
from data.risk_free import fetch_risk_free

CFG = load_config()

def print_metrics(results: dict):
    print(f"\n{'Metric':<25} {'Portfolio':>12} {'SPY':>12} {'EqualWeight':>12} {'60/40':>12} {'Momentum':>12} {'RiskParity':>12}")
    print("-" * 92)
    for metric in results["portfolio"]:
        print(f"{metric:<25} "
              f"{results['portfolio'][metric]:>12} "
              f"{results['spy'][metric]:>12} "
              f"{results['equal_weight'][metric]:>12} "
              f"{results['sixty_forty'][metric]:>12} "
              f"{results['momentum'][metric]:>12} "
              f"{results['risk_parity'][metric]:>12}")

def print_subperiod_analysis(regimes_df: pd.DataFrame, returns: pd.DataFrame) -> None:
    rf = fetch_risk_free()
    print("\n--- Subperiod Analysis ---\n")

    for label, period in CFG["subperiods"].items():
        start, end = period["start"], period["end"]
        result, _ = run_period(start=start, end=end, regimes_df=regimes_df)

        spy_slice = returns.loc[start:end, "SPY"]
        spy_equity = (1 + spy_slice).cumprod()

        port_metrics = compute_all(result["portfolio_return"], result["equity"], rf=rf)
        spy_metrics = compute_all(spy_slice, spy_equity, rf=rf)

        print(f"{label}")
        print(f"  {'Metric':<25} {'Portfolio':>12} {'SPY':>12}")
        print(f"  {'-' * 50}")
        for metric in port_metrics:
            print(f"  {metric:<25} {port_metrics[metric]:>12} {spy_metrics[metric]:>12}")
        print()

def compute_regime_runs(regimes_df: pd.DataFrame) -> pd.DataFrame:
    regime_seq = regimes_df["regime"]

    runs = []
    current = regime_seq.iloc[0]
    count = 1
    for label in regime_seq.iloc[1:]:
        if label == current:
            count += 1
        else:
            runs.append({"regime": current, "duration": count})
            current = label
            count = 1
    runs.append({"regime": current, "duration": count})
    return pd.DataFrame(runs)


def compute_empirical_transition_matrix(runs_df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """Empirical, run-based transition matrix: counts actual observed regime-label
    transitions in a walk-forward output (% of exits from each regime).

    This is distinct from models.hmm.get_fitted_transition_matrix(), which reads
    a single HMM model's own theoretical transmat_ parameter directly - the two
    are not interchangeable and will not generally agree."""
    trans = pd.DataFrame(0, index=labels, columns=labels)
    for i in range(len(runs_df) - 1):
        from_r = runs_df.iloc[i]["regime"]
        to_r = runs_df.iloc[i + 1]["regime"]
        if from_r in labels and to_r in labels:
            trans.loc[from_r, to_r] += 1

    return trans.div(trans.sum(axis=1).replace(0, np.nan), axis=0) * 100


def print_regime_diagnostics(regimes_df: pd.DataFrame) -> None:
    print("\n--- Regime Diagnostics ---\n")

    runs_df = compute_regime_runs(regimes_df)

    print("Regime run-length statistics (days):")
    print(f"  {'Regime':<12} {'Count':>8} {'Mean':>8} {'Median':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-' * 52}")
    for regime in ["Bull", "Bear", "Sideways", "Crash"]:
        subset = runs_df[runs_df["regime"] == regime]["duration"]
        if len(subset) == 0:
            continue
        print(f"  {regime:<12} {len(subset):>8} {subset.mean():>8.1f} {subset.median():>8.1f} {subset.min():>8} {subset.max():>8}")

    print("\nEmpirical transition matrix (row → col, % of exits):")
    labels = ["Bull", "Bear", "Sideways", "Crash"]
    trans_pct = compute_empirical_transition_matrix(runs_df, labels)

    print(f"\n  {'':12}", end="")
    for col in labels:
        print(f" {col:>10}", end="")
    print()
    print(f"  {'-' * 55}")
    for row in labels:
        print(f"  {row:<12}", end="")
        for col in labels:
            val = trans_pct.loc[row, col]
            print(f" {val:>9.1f}%" if not pd.isna(val) else f" {'—':>9}", end="")
        print()

def print_jarque_bera(regimes_df: pd.DataFrame) -> None:
    from scipy.stats import jarque_bera

    print("\n--- Jarque-Bera Normality Test per Regime ---\n")
    print(f"  {'Regime':<12} {'N':>6} {'Skew':>8} {'Kurtosis':>10} {'JB Stat':>10} {'p-value':>10} {'Normal?':>8}")
    print(f"  {'-' * 68}")

    for regime in ["Bull", "Bear", "Sideways", "Crash"]:
        subset = regimes_df[regimes_df["regime"] == regime]["spy_return"]
        if len(subset) < 8:
            continue
        jb_stat, p_value = jarque_bera(subset)
        skew = subset.skew()
        kurt = subset.kurtosis()
        normal = "Yes" if p_value > 0.05 else "No"
        print(f"  {regime:<12} {len(subset):>6} {skew:>8.3f} {kurt:>10.3f} {jb_stat:>10.2f} {p_value:>10.4f} {normal:>8}")

def main(retrain: bool = False, charts: bool = True, walk_forward: bool = True) -> None:
    print("=" * 50)
    print("  Regime Adaptive Portfolio")
    print("=" * 50)

    print("\n[1/4] Running HMM regime detection...")
    regimes = run_hmm(retrain=retrain, walk_forward=walk_forward)
    print(f"      Regimes found: {regimes['regime'].value_counts().to_dict()}")

    print("\n[2/4] Running backtest...")
    backtest, weights = run_backtest(regimes_df=regimes)
    print(f"      Days simulated: {len(backtest)}")

    print("\n[3/4] Computing metrics...")
    results = run_benchmark()
    print_metrics(results)
    turnover = average_turnover(weights)
    print(f"\n  Average daily turnover: {turnover:.4f} ({turnover * 100:.2f}% of portfolio per day)")
    print(f"  Implied annual transaction cost: {turnover * CFG['backtest']['transaction_cost'] * 252 * 100:.4f}%")

    print("\n--- Held-out test period (2019–2024) ---")
    test_start = CFG["evaluation"]["test_start"]
    data_end = CFG["evaluation"]["data_end"]
    test_result, _ = run_period(
        start=test_start,
        end=data_end,
        regimes_df=regimes
    )
    returns = pd.read_parquet(CFG["paths"]["returns"])
    spy_test = returns.loc[test_start:data_end, "SPY"]
    spy_test_equity = (1 + spy_test).cumprod()
    
    rf = fetch_risk_free()
    test_metrics = compute_all(test_result["portfolio_return"], test_result["equity"], rf=rf)
    spy_test_metrics = compute_all(spy_test, spy_test_equity, rf=rf)

    print(f"\n{'Metric':<25} {'Portfolio':>12} {'SPY':>12}")
    print("-" * 50)
    for metric in test_metrics:
        print(f"{metric:<25} {test_metrics[metric]:>12} {spy_test_metrics[metric]:>12}")

    print_subperiod_analysis(regimes, returns)
    print_regime_diagnostics(regimes)
    print_jarque_bera(regimes)
    
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
    parser.add_argument("--no-walk-forward", action="store_true",
                        help="Disable walk-forward HMM retraining (use static model)")
    args = parser.parse_args()

    main(retrain=args.retrain, charts=not args.no_charts, walk_forward=not args.no_walk_forward)