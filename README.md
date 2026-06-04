# Regime Adaptive Portfolio — v5

Algorithmic portfolio optimizer combining Hidden Markov Model regime detection with convex optimization. v5 fixes three correctness bugs identified in external review: stale optimizer weights on intra-quarter regime transitions, static risk-free rate in the max-Sharpe QP, and BIC scored on the wrong dataset.

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

### 2. Nested CV for State Selection (v3/v4)
At every quarterly fold, candidate state counts [2, 3, 4] are each fitted on 80% of the available training window and scored by BIC on the full training data. The state count minimizing BIC is selected independently per fold.

**v5 BIC fix:** v4 scored BIC on the held-out 20% of the training window — technically not BIC (which is defined on in-sample log-likelihood) and used the wrong sample size N in the penalty term. v5 correctly scores BIC on the full training data.

**Implementation note:** When a fold selects n=2, it emits only `p_bull` and `p_bear` posteriors. Concatenating folds with different state counts produces NaN entries in the unused posterior columns. The blending layer handles this explicitly — NaN posterior probabilities are skipped rather than propagated.

### 3. Parallelized Fold Execution (v4)
Each quarterly fold is independent — no shared mutable state. Folds execute via `joblib.Parallel`. `n_jobs=1` in `config.yaml` keeps the default deterministic; the sensitivity sweep script overrides to `n_jobs=8` for speed.

### 4. Posterior Probability Blending (v2)
v1 used hard regime labels — each day was assigned exactly one regime. This amplifies turnover at transition boundaries and ignores the HMM's uncertainty.

v2 uses `predict_proba()` to extract the full posterior distribution over states for every day:

```
weights(t) = P(Bull|t) × w_bull + P(Bear|t) × w_bear
           + P(Sideways|t) × w_sideways + P(Crash|t) × w_crash
```

This is a convex combination — weights always sum to 1, always non-negative.

### 5. Adaptive Optimization
Four convex optimizers, one per regime. On every regime transition and every retrain date, all four optimizers are recomputed with current covariance estimates and the current risk-free rate:

| Regime | Optimizer | Objective |
|---|---|---|
| Bull | Mean-Variance | Maximize Sharpe ratio (time-varying RF) |
| Bear | Risk Parity | Equalize risk contributions |
| Sideways | Minimum Variance | Minimize portfolio volatility |
| Crash | Equal Weight (IEF, TLT, GLD) | Flight-to-safety defensive allocation |

All optimizers except Crash use Ledoit-Wolf shrinkage on the covariance matrix. Weights constrained to [0, 1] — no shorting, no leverage.

**v5 optimizer fixes:**
- All four optimizers now recompute on every regime change, not just the active regime. Previously, intra-quarter transitions left three of the four optimizer outputs stale.
- The max-Sharpe QP now uses the time-varying 3-month T-bill rate as the hurdle, not a hardcoded 4%. During 2009–2022 (near-zero rate era), the fixed 4% hurdle systematically distorted allocations.

### 6. Regime Diagnostics (v3)
Run-length statistics from the walk-forward label sequence (v5 results):

| Regime | Count | Mean (days) | Median (days) | Min | Max |
|---|---|---|---|---|---|
| Bull | 65 | 20.6 | 5.0 | 1 | 153 |
| Bear | 58 | 24.3 | 20.5 | 1 | 63 |
| Sideways | 54 | 20.9 | 16.5 | 1 | 128 |
| Crash | 32 | 23.1 | 18.5 | 1 | 62 |

Empirical transition matrix (% of exits):

| From \ To | Bull | Bear | Sideways | Crash |
|---|---|---|---|---|
| Bull | — | 41.5% | 47.7% | 10.8% |
| Bear | 56.9% | — | 19.0% | 24.1% |
| Sideways | 39.6% | 41.5% | — | 18.9% |
| Crash | 34.4% | 28.1% | 37.5% | — |

### 7. Backtest
Lookahead-free simulation applying day-t weights to day t+1 returns. Transaction costs at 2bps per unit of turnover applied consistently to portfolio and all benchmarks. Average daily turnover: 3.90% (implied annual cost: 0.20%).

Sharpe ratios use a time-varying risk-free rate sourced from the 3-month T-bill (^IRX via yfinance), converted to daily rates. The same time-varying rate is now used in the max-Sharpe optimizer.

---

## Results

### Full Period (2006–2024)

| Metric | Portfolio | SPY | Equal Weight | 60/40 | Momentum | Risk Parity |
|---|---|---|---|---|---|---|
| Annualized Return | 5.57% | 8.27% | 5.90% | 6.80% | 5.67% | 4.55% |
| Annualized Volatility | 7.73% | 19.58% | 11.19% | 11.20% | 13.22% | 7.58% |
| Sharpe Ratio | **0.562** | 0.433 | 0.443 | 0.518 | 0.395 | 0.455 |
| Max Drawdown | **-24.11%** | -60.39% | -35.29% | -36.22% | -29.65% | -24.95% |
| Calmar Ratio | 0.231 | 0.137 | 0.167 | 0.188 | 0.191 | 0.182 |

The portfolio leads all benchmarks on Sharpe and achieves near-parity with the standalone risk-parity benchmark on max drawdown. The bug fixes in v5 — particularly the time-varying RF in the optimizer — are directly responsible for the improvement over v4 (Sharpe 0.486 → 0.562).

### Held-Out Test Period (2019–2024)

| Metric | Portfolio | SPY |
|---|---|---|
| Annualized Return | 7.70% | 14.86% |
| Annualized Volatility | 9.18% | 19.91% |
| Sharpe Ratio | 0.614 | 0.677 |
| Max Drawdown | -24.34% | -35.75% |
| Calmar Ratio | 0.316 | 0.416 |

Held-out Sharpe (0.614) is close to SPY (0.677) — a meaningful improvement over v4 (0.579). The portfolio maintains drawdown protection (-24.34% vs -35.75%).

### Subperiod Analysis

| Metric | Portfolio | SPY |
|---|---|---|
| **GFC (2008–2009)** | | |
| Annualized Return | 1.04% | -15.46% |
| Sharpe Ratio | 0.016 | -0.345 |
| Max Drawdown | -14.88% | -56.38% |
| **Low-vol bull (2013–2019)** | | |
| Annualized Return | 6.51% | 13.26% |
| Sharpe Ratio | 0.878 | 1.003 |
| Max Drawdown | -8.96% | -19.82% |
| **COVID+rates (2020–2024)** | | |
| Annualized Return | 3.61% | 11.83% |
| Sharpe Ratio | 0.156 | 0.528 |
| Max Drawdown | -24.86% | -35.75% |

**GFC:** The strategy's strongest result — near-zero return while SPY lost over half its value. Max drawdown -14.88% vs SPY -56.38%.

**Low-vol bull:** The time-varying RF fix significantly improved this period — Sharpe 0.759 → 0.878, max drawdown -16.10% → -8.96%. Near-zero rates mean the optimizer correctly allocated more to equities during 2013–2019.

**COVID+rates:** Still the weak spot. Fast crash, simultaneous bond/equity selloff in 2022.

### Window Sensitivity Sweep (v4)

Full pipeline rerun across `VOL_WINDOW ∈ {10, 21, 42}` × `CORR_WINDOW ∈ {42, 63, 126}`:

| VOL\CORR | 42 | 63 | 126 |
|---|---|---|---|
| **Full Sharpe** | | | |
| 10 | 0.374 | 0.621 | 0.613 |
| 21 | 0.525 | 0.534 | 0.622 |
| 42 | 0.384 | 0.524 | 0.486 |
| **Held-out Sharpe** | | | |
| 10 | 0.323 | 0.799 | 0.638 |
| 21 | 0.488 | 0.760 | 0.711 |
| 42 | 0.271 | 0.768 | 0.579 |

`CORR_WINDOW=63` consistently produces the best or near-best held-out Sharpe. The default parameters sit in a stable region — not cherry-picked.

### Bootstrap Block-Length Sensitivity (v4)

| Block Length | p-value | Mean Diff | 95% CI |
|---|---|---|---|
| 10 | 0.342 | 0.152 | [-0.44, 0.79] |
| 20 | 0.292 | 0.165 | [-0.48, 0.77] |
| 40 | 0.333 | 0.148 | [-0.50, 0.81] |

The conclusion is stable: Sharpe outperformance over SPY is not statistically significant regardless of block length.

### Bootstrap Significance Test (v2)

Block bootstrap with 1000 iterations and 20-day block length:

| Statistic | Value |
|---|---|
| Mean Sharpe Difference | 0.0918 |
| Std Sharpe Difference | 0.3238 |
| 95% CI | [-0.565, 0.707] |
| p-value | 0.376 |
| Significant at 95% | No |

The drawdown reduction is the more defensible edge.

---

## v4 → v5 Improvements

| Change | v4 | v5 |
|---|---|---|
| Optimizer recompute on regime change | Only active regime recomputed on intra-quarter transition | All four optimizers recomputed on every regime change |
| RF in max-Sharpe QP | Hardcoded 4% throughout entire backtest period | Time-varying 3-month T-bill (^IRX) — same rate used in metrics |
| BIC scoring | Computed on held-out 20% of training window (wrong N, not true BIC) | Computed on full training data (correct BIC definition) |
| Min-history threshold | Hardcoded 126 in switcher.py | References `config.yaml` min_train_days // 2 |
| fit_hmm duplication | Two near-identical functions with inconsistent return types | Unified via `_fit_hmm_core()` — thin wrappers preserve call sites |
| Test suite | 37 tests — no optimizer or integration coverage | 55 tests — 14 optimizer/switcher tests + 4 integration tests |
| Performance impact | Full Sharpe 0.486, held-out 0.579 | Full Sharpe 0.562 (+15%), held-out 0.614 (+6%) |

## v3 → v4 Improvements

| Change | v3 | v4 |
|---|---|---|
| Nested CV execution | select_n_states() existed but never called — every fold used fixed n=4 | Genuinely runs per fold — each fold selects n_states by BIC independently |
| Fold execution | Sequential for loop | joblib.Parallel — n_jobs controlled via config.yaml |
| Window sensitivity | Not tested | 9-run sweep over VOL_WINDOW × CORR_WINDOW |
| Bootstrap sensitivity | Not tested | 3-run sweep over BLOCK_LENGTH — p-value stable at 0.29–0.34 |
| SPY RF consistency | Held-out SPY computed without RF | RF applied consistently |

## v2 → v3 Improvements

| Change | v2 | v3 |
|---|---|---|
| State selection | Global BIC/AIC on pre-2018 data, fixed n=4 | Nested CV per fold |
| NaN posterior handling | Missing | Explicit skip of NaN posterior columns |
| Benchmarks | SPY, Equal Weight, 60/40, Momentum | + Risk Parity |
| Subperiod analysis | None | GFC / Low-vol bull / COVID+rates |
| Regime diagnostics | Transition matrix + avg duration | + Run-length distribution, empirical transitions |
| Normality testing | None | Jarque-Bera per regime |

## v1 → v2 Improvements

| Change | v1 | v2 |
|---|---|---|
| Regime states | 3 (Bull/Bear/Sideways) | 4 (+ Crash with flight-to-safety) |
| Regime uncertainty | Hard label per day | Posterior probability blend |
| State count justification | Assumed n=3 | BIC/AIC on pre-2018 held-out data |
| Walk-forward | Optional flag | Default behavior |
| Risk-free rate | Static 4% | Time-varying 3-month T-bill (^IRX) |
| Benchmark costs | Zero transaction costs | 2bps per unit of turnover |
| Benchmarks | SPY only | SPY, Equal Weight, 60/40, Momentum |
| Significance testing | None | Block bootstrap, p=0.376 |
| Out-of-sample evaluation | None | Held-out 2019–2024 test period |

---

## Architecture

```
regime-adaptive-portfolio/
├── config.yaml            # Centralized hyperparameters (HMM, backtest, market, paths, evaluation, subperiods)
├── config.py              # YAML loader
├── data/
│   ├── fetch.py           # Downloads adjusted close prices via yfinance
│   ├── process.py         # Log returns, rolling volatility, rolling correlation (parametric windows)
│   └── risk_free.py       # 3-month T-bill rate (^IRX) with daily conversion and caching
├── models/
│   └── hmm.py             # Gaussian HMM, nested CV per fold, parallelized walk-forward, BIC/AIC, diagnostics
├── optimization/
│   ├── mean_var.py        # Mean-Variance max Sharpe (Bull) — time-varying RF
│   ├── risk_parity.py     # Risk Parity via log barrier (Bear)
│   ├── min_variance.py    # Minimum Variance QP (Sideways)
│   ├── crash.py           # Equal-weight IEF/TLT/GLD flight-to-safety (Crash)
│   └── switcher.py        # Posterior blending, all-optimizer recompute on transition, NaN handling
├── backtest/
│   ├── engine.py          # Simulation with transaction costs, period slicing
│   ├── metrics.py         # Annualized return, volatility, Sharpe (time-varying RF), drawdown, Calmar
│   ├── benchmark.py       # SPY, Equal Weight, 60/40, Momentum, Risk Parity — all cost-adjusted
│   └── bootstrap.py       # Block bootstrap significance test
├── scripts/
│   └── sensitivity_sweep.py  # Window and bootstrap sensitivity analysis
├── tests/
│   ├── test_metrics.py    # 14 tests — metrics module
│   ├── test_hmm.py        # 14 tests — HMM module
│   ├── test_engine.py     # Backtest engine
│   ├── test_optimizers.py # 14 tests — optimizer functions and compute_weights
│   └── test_integration.py # 4 tests — end-to-end pipeline regression
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
| joblib | Model persistence + parallel fold execution |
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

# Run sensitivity sweep (VOL_WINDOW × CORR_WINDOW × BLOCK_LENGTH)
python scripts/sensitivity_sweep.py

# Run tests
pytest tests/
```

---

## Key Design Decisions

**Why Gaussian HMM?** Markets exhibit persistent regimes — the HMM transition matrix captures this persistence. Gaussian emissions model the continuous feature space naturally, though the Jarque-Bera results show this assumption is violated in practice.

**Why nested CV for state selection?** A globally fixed n=4 assumes the same model complexity is appropriate across all market environments and data volumes. Nested CV allows early folds with limited data to select simpler models, reducing overfitting risk.

**Why time-varying RF in both optimizer and metrics?** Consistency. The 4% hurdle used in the max-Sharpe QP during 2009–2022 (near-zero rate era) distorted allocations toward bonds/defensive assets when the true opportunity cost of cash was near zero. Using the same 3-month T-bill rate in both the optimizer and the Sharpe metric ensures the strategy is calibrated to the actual rate environment.

**Why recompute all optimizers on every regime change?** The blended portfolio weights are `sum(p_k * w_k)` across all four regimes. If three of the four weight vectors are stale — computed at a different point in time — the blend is internally inconsistent. All four must reflect the same return and covariance environment.

**Why posterior blending?** Hard switching ignores the model's own uncertainty. Posterior blending hedges proportionally — the HMM's confidence directly controls allocation.

**Why walk-forward as default?** A static HMM trained on all data uses future information to label past regimes. Walk-forward ensures each label is produced by a model that has never seen future data.

**Why cost-adjust benchmarks?** Fair comparison requires consistent cost treatment. All benchmarks incur 2bps per unit of monthly rebalancing turnover.

**Why Ledoit-Wolf shrinkage?** Sample covariance matrices from short windows are noisy and ill-conditioned. Ledoit-Wolf shrinks toward a stable target, preventing solver failures.

---

## Limitations

**Gaussian emission assumption violated.** Jarque-Bera tests reject normality for all four regimes (p≈0). Bear regime kurtosis is 16.2 — extreme fat tails. The HMM's posterior probabilities are built on a distributional assumption that does not hold. A Student-t HMM would be more appropriate but is not available in hmmlearn.

**Choppy Bull detection.** Median Bull run length is 5 days — half of all Bull episodes last less than a week. This generates unnecessary turnover (3.90% daily) and transaction costs. A minimum dwell-time constraint would reduce this.

**COVID+rates weakness.** The March 2020 crash was too sharp for a backward-looking HMM. The 2022 rate-driven bear hit bonds and equities simultaneously, limiting the flight-to-safety allocation. The strategy's drawdown protection is primarily a GFC-era result.

**Bootstrap significance.** The block bootstrap tests a weaker null than White's Reality Check or Hansen's SPA test. The p-value of 0.376 should be interpreted as a lower bound on uncertainty. The bootstrap block-length sweep confirms this conclusion is stable across block lengths 10, 20, and 40.

**Bull market drag.** The strategy structurally underperforms in sustained bull markets. The 2013–2019 period: 6.51% vs SPY's 13.26%.

**Mean return as signal.** The Mean-Variance optimizer uses sample mean returns — near-zero signal-to-noise at daily frequency. Shrinkage toward a factor model or Black-Litterman prior would be more robust.

**Asset universe.** Limited to 8 ETFs. Transaction costs modeled at 2bps — realistic for liquid ETFs, not accounting for market impact at scale.

**State labeling.** `label_states()` ranks states on return mean only. A composite Sharpe-based ranking was tested and reverted — it produced worse out-of-sample results despite sounder theory.

**Bootstrap independence assumption.** The block bootstrap resamples portfolio and SPY returns with independent block offsets, destroying cross-series correlation. A paired bootstrap (same offsets for both series) would be more appropriate and would reduce the variance of the Sharpe difference estimate.

---

## Author

Abdelkrim — Applied Mathematics & AI, PSL-Dauphine
https://github.com/AbdelkrimCode
