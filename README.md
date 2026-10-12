# Regime Adaptive Portfolio

Multi-asset portfolio that blends four convex optimizers using the filtered regime probabilities of a Gaussian Hidden Markov Model.

![Python](https://img.shields.io/badge/python-3.11+-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Tests](https://github.com/AbdelkrimCode/regime-adaptive-portfolio/actions/workflows/tests.yml/badge.svg)
---

## How It Works

### 1. Regime Detection
A 4-state Gaussian HMM (full covariance) is fitted on three daily features: SPY log return, 21-day realized volatility of SPY, and 63-day mean pairwise correlation across the 8 ETFs. Windows live in `config.yaml`.

States are labeled by their mean SPY volatility after every refit, calmest first. In prose they are called **Calm, Normal, Elevated, Turbulent**; the code keeps the historical names:

| Prose name | Code name | Optimizer |
|---|---|---|
| Calm | Bull | Maximum Sharpe |
| Normal | Sideways | Minimum variance |
| Elevated | Bear | Risk parity |
| Turbulent | Crash | Inverse volatility over safe havens (IEF, TLT, GLD) |

Ranking one feature by its mean is invariant to the per-feature standardization, so the order is the same in scaled and raw units. HMM state indices are arbitrary across refits; re-ranking after each refit keeps labels consistent.

The number of states is fixed at 4. An earlier version selected it per fold by BIC, but BIC picked the largest candidate in every fold, so it was not selecting anything. `select_n_states()` remains as a diagnostic.

### 2. Walk-Forward Refits
The HMM is refit quarterly on an expanding window (75 folds, each with at least 252 training days). Each fit uses 10 random restarts and keeps the highest likelihood. Convergence is checked by hand (`_em_converged`), since hmmlearn's own flag is also True when the iteration limit is hit and accepts likelihood decreases. In the current run, **75/75 folds converged**.

`audit_walk_forward()` rebuilds the folds with the same `_build_folds()` used by the walk-forward and checks that every training window ends before its test window starts: **0 leakage in 75 folds** (`results/walk_forward_audit.csv`).

### 3. Causal Forward Filter
Posteriors are `P(state_t | observations_1..t)` from a forward filter computed in log space (`forward_filter()` in `models/hmm.py`). No future observation is used at any date. hmmlearn's `predict_proba()` is not used because it runs forward-backward smoothing, which conditions on the whole sample.

Each fold runs the filter over its training window and then its test window, so the test period starts from the filtered state at the refit date rather than from the model's initial distribution. Restarting the filter at every fold had created about 40% of all regime switches (median Calm run 5 days before the fix, 17 after).

### 4. Posterior Blending
```
weights(t) = P(Calm|t) x w_calm + P(Normal|t) x w_normal
           + P(Elevated|t) x w_elevated + P(Turbulent|t) x w_turbulent
```
A convex combination of long-only weight vectors, so weights are non-negative and sum to 1.

### 5. Optimizers
All four optimizers are refreshed together on every change of the most likely regime and on every refit date, using all return history up to that day, so the blend mixes weights estimated on the same data. Covariances use Ledoit-Wolf shrinkage (`optimization/covariance.py`). Expected returns for max Sharpe are arithmetic means of daily simple returns times 252; its rf is the current 3-month T-bill yield, annualized.

Every asset is capped at 60%. Max Sharpe and min variance include the cap in the convex problem. Risk parity solves the log-barrier problem with only a small floor, normalizes, then caps and redistributes: capping the raw log-barrier solution, whose scale is arbitrary, saturated every asset and silently produced equal weight.

**Concentration guard:** an optimizer output with more than 99% in one asset is replaced by equal weight. For Turbulent the fallback is equal weight over the safe havens only.

### 6. Backtest
`simulate()` trades to the target at the close of day t; the new holdings earn day t+1 and pay that trade's cost on day t+1. Between trades, holdings drift with prices, and turnover is measured from the drifted weights. Costs are 2 bp one-way per unit of turnover by default. Simple returns are used for all P&L and optimizer inputs; log returns only feed the HMM.

Benchmarks (SPY buy and hold, equal weight, 60/40 SPY/IEF, 12-month top-3 momentum, risk parity) trade at each month-end on the same simulator and pay the same costs. Every strategy is scored on the portfolio's dates. Sharpe ratios subtract the daily 3-month T-bill rate (^IRX) day by day.

---

## Results

All numbers below come from one run of the final pipeline (`results/results_after_cleanup.txt`), data 2005-01-04 to 2024-12-31, portfolio scored from 2006-04-05.

**Sample discipline.** 2019-2024 is labeled "held-out" for historical reasons, but it has been looked at many times during development and should be read as in-sample. Data from 2025-01-01 onward has not been loaded and is reserved for a single final test once the research pipeline is frozen.

### Full Period (2006-2024)

| Metric | Portfolio | SPY | Equal Weight | 60/40 | Momentum | Risk Parity |
|---|---|---|---|---|---|---|
| Annualized Return | 6.33% | 10.41% | 7.41% | 7.87% | 8.52% | 5.84% |
| Annualized Volatility | 7.70% | 19.48% | 10.86% | 10.88% | 13.47% | 7.45% |
| Sharpe Ratio | 0.649 | 0.532 | 0.580 | 0.619 | 0.569 | 0.606 |
| Max Drawdown | -26.21% | -55.19% | -30.97% | -32.59% | -30.93% | -23.59% |
| Calmar Ratio | 0.242 | 0.189 | 0.239 | 0.242 | 0.276 | 0.247 |

Average daily turnover is 3.01%, about 0.15% a year in costs at 2 bp.

The portfolio has the highest Sharpe, but its lead over risk parity (0.043) and 60/40 (0.030) is small and not significant (see below). Risk parity has the shallower drawdown.

### Cost Sensitivity (Sharpe)

| Strategy | 2 bp | 5 bp | 10 bp |
|---|---|---|---|
| Portfolio | 0.649 | 0.619 | 0.570 |
| SPY | 0.532 | 0.532 | 0.532 |
| Equal Weight | 0.580 | 0.580 | 0.578 |
| 60/40 | 0.619 | 0.618 | 0.617 |
| Momentum | 0.569 | 0.560 | 0.545 |
| Risk Parity | 0.606 | 0.605 | 0.603 |

The portfolio trades daily as posteriors move, so it is far more cost-sensitive than the monthly benchmarks. At 5 bp it no longer beats 60/40; at 10 bp it trails risk parity and 60/40.

### Significance Tests

Sharpe difference, portfolio minus benchmark, H1: portfolio is better (one-sided). Two genuinely different tests:
- **Paired block bootstrap**: 1000 paths, 20-day blocks, the same indices for both series and rf.
- **Ledoit-Wolf (2008)**: studentized Sharpe difference with a HAC standard error (Parzen kernel, Andrews automatic bandwidth, VAR(1) prewhitening).

| Benchmark | Sharpe Diff | Bootstrap p | Bootstrap 95% CI | LW SE | LW p |
|---|---|---|---|---|---|
| SPY | +0.117 | 0.344 | [-0.422, 0.629] | 0.285 | 0.341 |
| Risk Parity | +0.043 | 0.386 | [-0.236, 0.357] | 0.146 | 0.385 |
| 60/40 | +0.030 | 0.460 | [-0.447, 0.481] | 0.251 | 0.452 |

No difference is significant, and the two tests agree closely.

**Block-length sensitivity** (bootstrap vs SPY):

| Block Length | p-value | Mean Diff | 95% CI |
|---|---|---|---|
| 10 | 0.339 | 0.106 | [-0.445, 0.624] |
| 20 | 0.344 | 0.108 | [-0.422, 0.629] |
| 40 | 0.343 | 0.112 | [-0.479, 0.663] |

### Held-Out Test Period (2019-2024, effectively in-sample)

| Metric | Walk-Forward | Frozen Model | SPY |
|---|---|---|---|
| Annualized Return | 4.79% | 7.83% | 17.11% |
| Annualized Volatility | 9.11% | 9.94% | 19.83% |
| Sharpe Ratio | 0.312 | 0.560 | 0.776 |
| Max Drawdown | -26.21% | -25.69% | -33.72% |
| Calmar Ratio | 0.183 | 0.305 | 0.507 |

The frozen model is fitted once on 2006-2018 and never refit. It beats the walk-forward version on this window, so quarterly refitting did not help here. Both trail SPY.

### Subperiods

| Metric | Portfolio | SPY |
|---|---|---|
| **GFC (2008-2009)** | | |
| Annualized Return | 7.67% | -10.24% |
| Annualized Volatility | 11.12% | 34.78% |
| Sharpe Ratio | 0.661 | -0.171 |
| Max Drawdown | -10.59% | -51.87% |
| **Low-vol bull (2013-2019)** | | |
| Annualized Return | 4.37% | 14.20% |
| Annualized Volatility | 5.10% | 12.82% |
| Sharpe Ratio | 0.695 | 1.069 |
| Max Drawdown | -7.69% | -19.35% |
| **COVID+rates (2020-2024)** | | |
| Annualized Return | 3.04% | 14.28% |
| Annualized Volatility | 9.72% | 20.99% |
| Sharpe Ratio | 0.110 | 0.632 |
| Max Drawdown | -26.21% | -33.72% |

**The worst drawdown is 2022, not the GFC.** Peak 2021-12-27, trough 2022-10-20, -26.2% (GFC: -10.6%). Detection worked: 2022 was classified Elevated on 194 days and Turbulent on 50. The response failed, because both of those optimizers (risk parity and safe havens) assume bonds hedge equities, and in 2022 they fell together. The features are equity-only and cannot see a positive stock-bond correlation regime.

### Regime Diagnostics

| Regime | Runs | Mean (days) | Median (days) | Min | Max |
|---|---|---|---|---|---|
| Calm | 41 | 30.8 | 15.0 | 1 | 171 |
| Normal | 42 | 26.6 | 20.0 | 1 | 153 |
| Elevated | 41 | 34.3 | 23.0 | 1 | 119 |
| Turbulent | 20 | 46.6 | 22.5 | 3 | 386 |

The longest Turbulent run is 2007-06-08 to 2008-12-16 (GFC). The earliest folds train on about 1.5 years of calm data, so early labels may be an artifact of the short sample.

Empirical transition matrix (% of exits):

| From \ To | Calm | Normal | Elevated | Turbulent |
|---|---|---|---|---|
| Calm | - | 65.9% | 34.1% | 0.0% |
| Normal | 58.5% | - | 29.3% | 12.2% |
| Elevated | 41.5% | 22.0% | - | 36.6% |
| Turbulent | 0.0% | 25.0% | 75.0% | - |

### Window Sensitivity Sweep

Full pipeline rerun for every pair of `vol_window` in {10, 21, 42} and `corr_window` in {42, 63, 126} (`results/sensitivity_results.csv`):

| VOL \ CORR | 42 | 63 | 126 |
|---|---|---|---|
| **Full Sharpe** | | | |
| 10 | 0.700 | 0.754 | 0.783 |
| 21 | 0.849 | 0.649 | 0.770 |
| 42 | 0.766 | 0.726 | 0.589 |
| **2019-2024 Sharpe** | | | |
| 10 | 0.232 | 0.378 | 0.468 |
| 21 | 0.537 | 0.312 | 0.508 |
| 42 | 0.444 | 0.199 | 0.280 |

The default (21, 63) was chosen before this sweep and is kept: it is the second-worst cell on full-sample Sharpe, and the spread across cells (0.59 to 0.85) is about as wide as the bootstrap CI of a single Sharpe difference. Picking the best cell now would be tuning to results.

### Feature Ablation

Full pipeline rerun with alternative feature sets (`results/ablation_results.csv`):

| Feature Set | Full Sharpe | Max DD | 2019-2024 Sharpe |
|---|---|---|---|
| baseline (vol 21, corr 63) | 0.649 | -26.21% | 0.312 |
| vol_10 (vol 10, corr 63) | 0.754 | -22.66% | 0.378 |
| vol_42 (vol 42, corr 63) | 0.726 | -25.61% | 0.199 |
| skew_kurt (return, skew, kurtosis) | 0.596 | -22.09% | 0.322 |

The ordering is not stable between the full sample and 2019-2024, consistent with differences of this size being noise.

### Stress Tests

| Test | Portfolio | SPY |
|---|---|---|
| VaR (95%, daily) | -0.76% | -1.82% |
| CVaR (95%, daily) | -1.12% | -3.01% |
| Sharpe on days with VIX > 30 | -0.244 | -1.721 |
| Block-bootstrap Sharpe, mean (1000 paths) | 0.658 | - |
| Block-bootstrap Sharpe, 95% CI | [0.265, 1.097] | - |

---

## Architecture

```
regime-adaptive-portfolio/
├── config.yaml                  # Hyperparameters, paths (relative to the project root), evaluation dates
├── config.py                    # YAML loader; resolves paths against the project root
├── CHANGELOG.md                 # Fix history and earlier versions
├── data/
│   ├── fetch.py                 # yfinance adjusted closes, signature-validated cache, inclusive end date
│   ├── process.py               # Simple returns (P&L), log returns and rolling features (HMM)
│   ├── risk_free.py             # Daily 3-month T-bill rate (^IRX)
│   └── cache_utils.py           # Cache signature helpers
├── models/
│   └── hmm.py                   # Gaussian HMM, log-space forward filter, walk-forward folds, volatility labeling, audit
├── optimization/
│   ├── covariance.py            # Single Ledoit-Wolf covariance estimator
│   ├── mean_var.py              # Max Sharpe (Calm)
│   ├── min_variance.py          # Minimum variance (Normal)
│   ├── risk_parity.py           # Log-barrier risk parity, cap applied after normalization (Elevated)
│   ├── crash.py                 # Inverse-vol safe havens (Turbulent)
│   └── switcher.py              # Posterior blending and concentration guard; rf injected by the caller
├── backtest/
│   ├── engine.py                # Drift-aware simulator, turnover from drifted weights
│   ├── metrics.py               # Return, volatility, Sharpe (daily rf), drawdown, Calmar
│   ├── benchmark.py             # Monthly-rebalanced benchmarks, cost sensitivity
│   ├── bootstrap.py             # Paired block bootstrap, comparison against each benchmark
│   └── sharpe_test.py           # Ledoit-Wolf (2008) HAC Sharpe-difference test
├── scripts/
│   ├── sensitivity_sweep.py     # Feature window sweep and bootstrap block-length sweep
│   ├── frozen_model_eval.py     # Model fitted once on 2006-2018, tested on 2019-2024
│   ├── feature_ablation.py      # Alternative feature sets
│   └── stress_tests.py          # VaR, CVaR, high-VIX Sharpe, Monte Carlo, both significance tests vs SPY
├── results/                     # Run outputs and figures
├── tests/                       # pytest suite (145 tests)
├── visualization/
│   └── charts.py                # Equity curves, drawdowns, regime overlay
└── main.py                      # Full pipeline entry point
```

---

## Usage

```bash
pip install -r requirements.txt          # runtime
pip install -r requirements-dev.txt      # + pytest, jupyter, ruff

python main.py                           # full pipeline, walk-forward refits
python main.py --no-charts               # skip charts
python main.py --no-walk-forward         # single static model (in-sample, for comparison only)

python -m scripts.sensitivity_sweep
python -m scripts.frozen_model_eval
python -m scripts.feature_ablation
python -m scripts.stress_tests
python -m backtest.bootstrap             # significance table and bootstrap figure
python -c "from models.hmm import audit_walk_forward; audit_walk_forward()"

pytest tests/
```

---

## Limitations

**The equity-only features cannot see a stock-bond correlation regime.** This is the cause of the 2022 drawdown described above, and the main thing a next version should address.

**Sharpe outperformance is not significant** against SPY, risk parity or 60/40, under either test, and the edge disappears at 5-10 bp costs.

**Gaussian emissions are rejected.** Jarque-Bera rejects normality in all four regimes (p < 0.0001); Turbulent excess kurtosis is 6.1, the others 1.3-2.1. A Student-t emission model would fit better.

**Expanding covariance window.** Optimizers use all history up to each date (from about 300 days in 2006 to about 5,000 in 2024), so later estimates are dominated by old regimes.

**Mean-return estimates.** Max Sharpe uses sample means, which carry little signal at daily frequency.

**SPY circularity.** All three features are derived from SPY, one of the traded assets.

**Universe and costs.** 8 ETFs, flat proportional costs, no market impact.

**Data revisions.** A fresh yfinance download can shift results slightly because Yahoo backfills price corrections. All numbers here come from a single run.

---

Abdelkrim - Applied Mathematics & AI, PSL-Dauphine
https://github.com/AbdelkrimCode
