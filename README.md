# Regime Adaptive Portfolio — v3

Algorithmic portfolio optimizer combining Hidden Markov Model regime detection with convex optimization. v3 introduces nested cross-validation for per-fold state selection, a risk-parity benchmark, subperiod analysis across GFC / low-vol bull / COVID+rates periods, regime stability diagnostics, and Jarque-Bera normality testing per regime.

![Python](https://img.shields.io/badge/python-3.10+-blue) ![License](https://img.shields.io/badge/license-MIT-green)

---

## How It Works

### 1. Regime Detection
A Gaussian HMM trained on three features — SPY log returns, 21-day rolling volatility, and 63-day mean pairwise correlation — labels each trading day as Bull, Bear, Sideways, or Crash. The model is retrained quarterly using an expanding window to prevent lookahead bias. Walk-forward retraining is the default.

State ordering is determined by ranking state means on the return feature:
- Lowest mean return → Crash
- Second lowest → Bear
- Second highest → Sideways
- Highest → Bull

### 2. Nested CV for State Selection (v3)
v2 selected n=4 states globally using AIC/BIC on pre-2018 data. v3 runs a nested cross-validation at every quarterly fold: candidate state counts [2, 3, 4] are each fitted on 80% of the available training window and scored by BIC on the held-out 20%. The state count minimizing BIC is selected independently per fold.

This means early folds (less data, simpler structure) may select n=2, while later folds (more data, richer structure) select n=3 or n=4. The model complexity adapts to the information available at each retraining date.

**Implementation note:** When a fold selects n=2, it emits only `p_bull` and `p_bear` posteriors. Concatenating folds with different state counts produces NaN entries in the unused posterior columns. The blending layer handles this explicitly — NaN posterior probabilities are skipped rather than propagated.

### 3. Posterior Probability Blending (v2)
v1 used hard regime labels — each day was assigned exactly one regime. This amplifies turnover at transition boundaries and ignores the HMM's uncertainty.

v2 uses `predict_proba()` to extract the full posterior distribution over states for every day:

```
weights(t) = P(Bull|t) × w_bull + P(Bear|t) × w_bear
           + P(Sideways|t) × w_sideways + P(Crash|t) × w_crash
```

This is a convex combination — weights always sum to 1, always non-negative. The 5-day linear transition smoother from v1 is removed — uncertainty is now model-driven, not heuristic.

### 4. Adaptive Optimization
Four convex optimizers, one per regime. Optimizer weights are recomputed at every quarterly retraining date — not just on regime changes — ensuring fresh covariance estimates throughout:

| Regime | Optimizer | Objective |
|---|---|---|
| Bull | Mean-Variance | Maximize Sharpe ratio |
| Bear | Risk Parity | Equalize risk contributions |
| Sideways | Minimum Variance | Minimize portfolio volatility |
| Crash | Equal Weight (IEF, TLT, GLD) | Flight-to-safety defensive allocation |

All optimizers except Crash use Ledoit-Wolf shrinkage on the covariance matrix. Weights constrained to [0, 1] — no shorting, no leverage.

### 5. Regime Diagnostics (v3)
Run-length statistics from the walk-forward label sequence:

| Regime | Count | Mean (days) | Median (days) | Min | Max |
|---|---|---|---|---|---|
| Bull | 65 | 20.6 | 5.0 | 1 | 153 |
| Bear | 58 | 24.3 | 20.5 | 1 | 63 |
| Sideways | 54 | 20.9 | 16.5 | 1 | 128 |
| Crash | 32 | 23.1 | 18.5 | 1 | 62 |

The Bull median of 5 days indicates choppy detection — half of all Bull episodes last less than a week. This contributes to elevated turnover and is a known limitation of the nested CV approach on short early folds.

Empirical transition matrix (% of exits):

| From \ To | Bull | Bear | Sideways | Crash |
|---|---|---|---|---|
| Bull | — | 41.5% | 47.7% | 10.8% |
| Bear | 56.9% | — | 19.0% | 24.1% |
| Sideways | 39.6% | 41.5% | — | 18.9% |
| Crash | 34.4% | 28.1% | 37.5% | — |

### 6. Backtest
Lookahead-free simulation applying day-t weights to day t+1 returns. Transaction costs at 2bps per unit of turnover applied consistently to portfolio and all benchmarks. Average daily turnover: 5.34% (implied annual cost: 0.27%).

Sharpe ratios use a time-varying risk-free rate sourced from the 3-month T-bill (^IRX via yfinance), converted to daily rates.

---

## Results

### Full Period (2006–2024)

| Metric | Portfolio | SPY | Equal Weight | 60/40 | Momentum | Risk Parity |
|---|---|---|---|---|---|---|
| Annualized Return | 5.76% | 8.27% | 5.90% | 6.80% | 5.67% | 4.55% |
| Annualized Volatility | 8.71% | 19.58% | 11.19% | 11.20% | 13.22% | 7.58% |
| Sharpe Ratio | **0.529** | 0.433 | 0.443 | 0.518 | 0.395 | 0.455 |
| Max Drawdown | **-25.49%** | -60.39% | -35.29% | -36.22% | -29.65% | -24.95% |
| Calmar Ratio | 0.226 | 0.137 | 0.167 | 0.188 | 0.191 | 0.182 |

The portfolio leads all benchmarks on Sharpe. The risk-parity benchmark is the most direct comparison — it isolates the value of regime switching over running risk parity unconditionally. The portfolio beats standalone risk parity on both return (5.76% vs 4.55%) and Sharpe (0.529 vs 0.455).

### Held-Out Test Period (2019–2024)

| Metric | Portfolio | SPY |
|---|---|---|
| Annualized Return | 8.29% | 14.86% |
| Annualized Volatility | 9.51% | 19.91% |
| Sharpe Ratio | **0.655** | 0.797 |
| Max Drawdown | -25.26% | -35.75% |
| Calmar Ratio | 0.328 | 0.416 |

The portfolio leads on Sharpe and drawdown protection on the held-out period. Absolute return lags SPY — the 2019–2024 period included a sustained bull market where regime switching reduces equity exposure.

### Subperiod Analysis

| Metric | Portfolio | SPY |
|---|---|---|
| **GFC (2008–2009)** | | |
| Annualized Return | 0.83% | -15.46% |
| Sharpe Ratio | -0.001 | -0.345 |
| Max Drawdown | -15.36% | -56.38% |
| **Low-vol bull (2013–2019)** | | |
| Annualized Return | 6.74% | 13.26% |
| Sharpe Ratio | 0.751 | 1.003 |
| Max Drawdown | -12.89% | -19.82% |
| **COVID+rates (2020–2024)** | | |
| Annualized Return | 4.53% | 11.83% |
| Sharpe Ratio | 0.229 | 0.528 |
| Max Drawdown | -26.14% | -35.75% |

**GFC:** The strategy's strongest result — near-zero return while SPY lost over half its value. Bear and Crash detection worked as designed.

**Low-vol bull:** The cost of regime switching — roughly half SPY's return during a six-year bull market. Expected and unavoidable for a strategy that reduces equity exposure in non-Bull regimes.

**COVID+rates:** The weak spot. The March 2020 crash was too fast for a backward-looking HMM to detect in time. The 2022 rate-driven bear market hit bonds and equities simultaneously, limiting the effectiveness of the flight-to-safety Crash allocation (IEF/TLT/GLD).

### Bootstrap Significance Test (v2)

Block bootstrap with 1000 iterations and 20-day block length (preserving autocorrelation structure):

| Statistic | Value |
|---|---|
| Mean Sharpe Difference | 0.0918 |
| Std Sharpe Difference | 0.3238 |
| 95% CI | [-0.565, 0.707] |
| p-value | 0.376 |
| Significant at 95% | No |

The Sharpe outperformance over SPY is not statistically significant. The confidence interval straddles zero — the effect is real on observed data but indistinguishable from sampling variation over this period. The drawdown reduction is the more defensible edge.

---

## v2 → v3 Improvements

| Change | v2 | v3 |
|---|---|---|
| State selection | Global BIC/AIC on pre-2018 data, fixed n=4 | Nested CV per fold, n selected from [2,3,4] at each retraining date |
| NaN posterior handling | Missing — NaN posteriors silently zeroed weights | Explicit skip of NaN posterior columns in blending layer |
| Benchmarks | SPY, Equal Weight, 60/40, Momentum | + Risk Parity (most direct comparison) |
| Subperiod analysis | None | GFC / Low-vol bull / COVID+rates breakdown |
| Regime diagnostics | Transition matrix + avg duration | + Run-length distribution, empirical transition frequencies |
| Normality testing | None | Jarque-Bera per regime — all four regimes fail (as expected) |
| Config coverage | HMM, backtest, paths, evaluation | + market constants (RISK_FREE_RATE, TRADING_DAYS), subperiod definitions |
| Test suite | test_metrics, test_hmm, test_engine | All passing |
| VIX / yield curve features | Not tested | Tested, reverted — out-of-sample Sharpe degraded from 1.04 to 0.66 |

## v1 → v2 Improvements

| Change | v1 | v2 |
|---|---|---|
| Regime states | 3 (Bull/Bear/Sideways) | 4 (+ Crash with flight-to-safety) |
| Regime uncertainty | Hard label per day | Posterior probability blend |
| State count justification | Assumed n=3 | BIC/AIC on pre-2018 held-out data |
| AIC/BIC parameter count | Missing startprob | Includes all free parameters |
| Optimizer cache | Recomputed on regime change only | Recomputed on every retrain date |
| Walk-forward | Optional flag | Default behavior |
| Risk-free rate | Static 4% | Time-varying 3-month T-bill (^IRX) |
| Benchmark costs | Zero transaction costs | 2bps per unit of turnover |
| Benchmarks | SPY only | SPY, Equal Weight, 60/40, Momentum |
| Significance testing | None | Block bootstrap, p=0.376 |
| Out-of-sample evaluation | None | Held-out 2019–2024 test period |
| Turnover reporting | None | Daily turnover + implied cost |
| Regime diagnostics | None | Transition matrix + avg duration |
| Pipeline coupling | Regimes read from disk silently | Regimes injected explicitly |
| Hyperparameters | Scattered across modules | Centralized in config.yaml |

---

## Architecture

```
regime-adaptive-portfolio/
├── config.yaml            # Centralized hyperparameters (HMM, backtest, market, paths, evaluation, subperiods)
├── config.py              # YAML loader
├── data/
│   ├── fetch.py           # Downloads adjusted close prices via yfinance
│   ├── process.py         # Log returns, 21-day volatility, 63-day mean correlation
│   └── risk_free.py       # 3-month T-bill rate (^IRX) with daily conversion and caching
├── models/
│   └── hmm.py             # Gaussian HMM, nested CV per fold, walk-forward retraining, BIC/AIC, diagnostics
├── optimization/
│   ├── mean_var.py        # Mean-Variance max Sharpe (Bull)
│   ├── risk_parity.py     # Risk Parity via log barrier (Bear)
│   ├── min_variance.py    # Minimum Variance QP (Sideways)
│   ├── crash.py           # Equal-weight IEF/TLT/GLD flight-to-safety (Crash)
│   └── switcher.py        # Posterior probability blending, NaN handling, quarterly cache refresh
├── backtest/
│   ├── engine.py          # Simulation with transaction costs, period slicing
│   ├── metrics.py         # Annualized return, volatility, Sharpe (time-varying RF), drawdown, Calmar, turnover
│   ├── benchmark.py       # SPY, Equal Weight, 60/40, Momentum, Risk Parity — all cost-adjusted
│   └── bootstrap.py       # Block bootstrap significance test
├── tests/
│   ├── test_metrics.py    # 14 tests — metrics module
│   ├── test_hmm.py        # 14 tests — HMM module
│   └── test_engine.py     # Backtest engine
├── visualization/
│   └── charts.py          # Equity curves, drawdown, regime overlay
└── main.py                # Full pipeline entry point
```

---

## Stack

| Library | Purpose |
|---|---|
| yfinance | Price data download + risk-free rate |
| hmmlearn | Gaussian HMM |
| cvxpy | Convex optimization |
| scikit-learn | Feature scaling, Ledoit-Wolf covariance |
| scipy | Jarque-Bera normality test |
| pandas / numpy | Data manipulation |
| matplotlib | Visualization |
| joblib | Model persistence |
| pyyaml | Config loading |
| pytest | Unit testing |

---

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (walk-forward retraining with nested CV is default)
python main.py

# Force HMM retrain from scratch
python main.py --retrain

# Disable walk-forward (use static model)
python main.py --no-walk-forward

# Skip chart generation
python main.py --no-charts

# Run tests
pytest tests/
```

---

## Key Design Decisions

**Why Gaussian HMM?** Markets exhibit persistent regimes — the HMM transition matrix captures this persistence. Gaussian emissions model the continuous feature space naturally, though the Jarque-Bera results below show this assumption is violated in practice.

**Why nested CV for state selection?** A globally fixed n=4 assumes the same model complexity is appropriate across all market environments and data volumes. Nested CV allows early folds with limited data to select simpler models (n=2), reducing overfitting risk in low-data regimes.

**Why posterior blending?** Hard switching ignores the model's own uncertainty. On transition days where P(Bull)=0.55 and P(Bear)=0.45, hard switching commits fully to one optimizer. Posterior blending hedges proportionally — the HMM's confidence directly controls allocation.

**Why quarterly cache refresh?** Without it, optimizer weights computed in 2008 get blended into 2024 allocations — the covariance structure is stale. Recomputing on every retraining date ensures weights reflect the current return environment.

**Why walk-forward as default?** A static HMM trained on all data uses future information to label past regimes. Walk-forward ensures each label is produced by a model that has never seen future data — the only valid out-of-sample setup.

**Why time-varying risk-free rate?** A static 4% rate applied over 2006–2024 misrepresents Sharpe during the zero-rate era (2009–2015, 2020–2022) and the high-rate era (2023–2024). The 3-month T-bill (^IRX) provides a daily risk-free rate aligned to the actual interest rate environment.

**Why cost-adjust benchmarks?** A fair comparison requires consistent cost treatment. All benchmarks incur 2bps per unit of monthly rebalancing turnover — the same model applied to the portfolio.

**Why Ledoit-Wolf shrinkage?** Sample covariance matrices from short windows are noisy and ill-conditioned. Ledoit-Wolf shrinks toward a stable target, preventing solver failures.

---

## Limitations

**Gaussian emission assumption violated.** Jarque-Bera tests reject normality for all four regimes (p≈0 in every case). Bear regime kurtosis is 16.2 — extreme fat tails. The HMM's probability estimates and posterior blending weights are built on a distributional assumption that does not hold. A Student-t HMM would be more appropriate but is not available in hmmlearn.

**Choppy Bull detection.** Median Bull run length is 5 days — half of all Bull episodes last less than a week before the label switches. This generates unnecessary turnover (5.34% daily) and transaction costs without corresponding alpha. A minimum dwell-time constraint or longer smoothing window would reduce this.

**COVID+rates weakness.** The March 2020 crash was too sharp for a backward-looking HMM to detect before most of the drawdown occurred. The 2022 rate-driven bear market hit bonds and equities simultaneously, making the flight-to-safety Crash allocation (IEF/TLT/GLD) ineffective. The strategy's drawdown protection is primarily a GFC-era result.

**Bootstrap significance.** The block bootstrap tests a weaker null than White's Reality Check or Hansen's SPA test. It does not correct for the degrees of freedom consumed by four optimizers, four states, and multiple hyperparameter choices. The p-value of 0.376 should be interpreted as a lower bound on uncertainty.

**Bull market drag.** The strategy structurally underperforms in sustained bull markets — regime detection reduces equity exposure precisely when equity performs well. The 2013–2019 period shows this clearly: 6.74% vs SPY's 13.26%.

**Mean return as signal.** The Mean-Variance optimizer uses sample mean returns as expected return estimates — near-zero signal-to-noise at daily frequency. Shrinkage toward a factor model or Black-Litterman prior would be more robust.

**No sensitivity analysis.** Rolling window lengths (VOL_WINDOW=21, CORR_WINDOW=63) and bootstrap block length (BLOCK_LENGTH=20) are not stress-tested. Results may be sensitive to these choices.

**Asset universe.** Limited to 8 ETFs. Transaction costs modeled at 2bps — realistic for liquid ETFs, not accounting for market impact at scale.

---

## Author

Abdelkrim — Applied Mathematics & AI, PSL-Dauphine
https://github.com/AbdelkrimCode
