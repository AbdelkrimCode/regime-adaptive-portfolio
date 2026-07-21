# Regime Adaptive Portfolio - v6

Algorithmic portfolio optimizer combining Hidden Markov Model regime detection with convex optimization.

![Python](https://img.shields.io/badge/python-3.10+-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Tests](https://github.com/AbdelkrimCode/regime-adaptive-portfolio/actions/workflows/tests.yml/badge.svg)
---

## How It Works

### 1. Regime Detection
A Gaussian HMM trained on three features - SPY log returns, rolling volatility, and rolling mean pairwise correlation - labels each trading day as Bull, Bear, Sideways, or Crash. Feature windows default to 21-day rolling volatility and 63-day rolling correlation.

State ordering is determined by a risk-adjusted score after every retrain:
- Score = mean SPY return - 0.5 × mean rolling volatility
- Lowest score → Crash
- Second lowest → Bear
- Second highest → Sideways
- Highest → Bull

The score's two inputs are located by feature name, not column position, so the scoring rule stays correct even when the feature set changes shape (e.g. the skew/kurtosis ablation variant, which has no volatility column at all - see Audit Fixes below).

**Label stability across retrains:** HMM states have no intrinsic identity - state 0 in Q1 might correspond to "Bear" and state 0 in Q2 might correspond to "Bull". This implementation avoids label permutation by re-ranking states post-hoc.

### 2. BIC State Selection
At every quarterly fold, candidate state counts [2, 3, 4] are each fitted on the full training window and scored by BIC. The state count minimizing BIC is selected independently per fold. The winning fit is reused directly rather than refit from scratch.

**Implementation note:** When a fold selects n=2, it emits only `p_bull` and `p_bear` posteriors. Concatenating folds with different state counts produces NaN entries in the unused posterior columns. These are skipped (not forward-filled) during portfolio construction.

**Convergence diagnostics:** Each fold's HMM fit reports whether the EM algorithm genuinely converged (log-likelihood improvement below tolerance) versus merely exhausting its iteration budget. Across the current 75-fold walk-forward run: **75/75 folds converged.**

### 3. Causal Forward-Filter Posteriors
Prior to v6, regime posteriors were computed using the Viterbi algorithm and `predict_proba()` - both of which use forward and backward passes over the entire observation window. This introduces within-window lookahead bias.

v6 replaces both with a manual forward-filter (`forward_filter()` in `hmm.py`). The filtered posterior `P(state_t | observations_1:t)` is computed using only the forward pass - strictly causal at every time step:

```
alpha_t[i] = P(o_t | state=i) * sum_j( alpha_{t-1}[j] * A[j,i] )
filtered_posterior_t = alpha_t / sum(alpha_t)
```

**Performance impact:** Forward filtering reduces full-sample Sharpe relative to Viterbi because Viterbi's within-window smoothing produced better labels in hindsight. The v6 numbers reflect what a live system would achieve.

### 4. Parallelized Fold Execution
Each quarterly fold is independent - no shared mutable state. Folds execute via `joblib.Parallel`. `n_jobs=1` in `config.yaml` keeps the default deterministic; the sensitivity sweep passes `n_jobs=-1` for speed.

### 5. Posterior Probability Blending
v1 used hard regime labels - each day was assigned exactly one regime. This amplifies turnover at transition boundaries and ignores the HMM's uncertainty.

v2 uses filtered posteriors for every day:

```
weights(t) = P(Bull|t) × w_bull + P(Bear|t) × w_bear
           + P(Sideways|t) × w_sideways + P(Crash|t) × w_crash
```

This is a convex combination - weights always sum to 1, always non-negative.

### 6. Adaptive Optimization
Four convex optimizers, one per regime. On every regime transition and every retrain date, all four optimizers are recomputed with current covariance estimates and the current risk-free rate:

| Regime | Optimizer | Objective |
|---|---|---|
| Bull | Mean-Variance | Maximize Sharpe ratio (time-varying RF) |
| Bear | Risk Parity (log-barrier) | Approximate equal risk contribution |
| Sideways | Minimum Variance | Minimize portfolio volatility |
| Crash | Inverse-Vol Weight (IEF, TLT, GLD) | Flight-to-safety - allocates inversely proportional to current volatility |

All optimizers except Crash use Ledoit-Wolf shrinkage on the covariance matrix. Weights constrained to [0, 1] - no shorting, no leverage. Safe-haven assets configurable via `config.yaml`.

**Concentration guard:** if any regime's optimizer output concentrates more than 99% of weight in a single asset (numerical degeneracy), it's replaced with an equal-weight fallback. For Crash specifically, that fallback is scoped to the regime's own safe-haven assets (IEF/TLT/GLD) rather than the full 8-asset universe - falling back to the full universe would inject equity exposure at exactly the moment the Crash regime exists to avoid it.

### 7. Regime Diagnostics
Run-length statistics from the walk-forward label sequence:

| Regime | Count | Mean (days) | Median (days) | Min | Max |
|---|---|---|---|---|---|
| Bull | 74 | 17.1 | 5.0 | 1 | 171 |
| Bear | 67 | 20.0 | 17.0 | 1 | 62 |
| Sideways | 67 | 18.1 | 11.0 | 1 | 128 |
| Crash | 32 | 28.0 | 24.5 | 3 | 62 |

Empirical transition matrix (% of exits):

| From \ To | Bull | Bear | Sideways | Crash |
|---|---|---|---|---|
| Bull | - | 37.8% | 60.8% | 1.4% |
| Bear | 49.3% | - | 19.4% | 31.3% |
| Sideways | 51.5% | 33.3% | - | 15.2% |
| Crash | 21.9% | 50.0% | 28.1% | - |

### 8. Backtest
Lookahead-free simulation applying day-t weights to day t+1 returns. Transaction costs at 2bps per unit of turnover applied consistently to portfolio and all benchmarks, including the true cost of the first day's entry from cash. Average daily turnover: 4.62%.

Sharpe ratios use a time-varying risk-free rate sourced from the 3-month T-bill (^IRX via yfinance). The same time-varying rate is used in the max-Sharpe optimizer. The `run_period()` function passes the full returns history to each optimizer, not just the in-sample slice.

---

## Results

The primary claim of this strategy is **drawdown control**, not Sharpe outperformance. The observed Sharpe difference vs SPY (0.617 vs 0.435) is positive but not statistically significant (bootstrap p=0.266).

- Max drawdown **-27.03% vs SPY -59.58%** - 55% reduction in peak-to-trough loss
- Calmar ratio **0.233 vs SPY 0.140** - better return per unit of drawdown
- Volatility **8.09% vs SPY 19.51%** - dramatically lower realized risk

### Full Period (2006-2024)

| Metric | Portfolio | SPY | Equal Weight | 60/40 | Momentum | Risk Parity |
|---|---|---|---|---|---|---|
| Annualized Return | 6.28% | 8.34% | 5.96% | 6.88% | 5.63% | 5.75% |
| Annualized Volatility | 8.09% | 19.51% | 11.17% | 11.17% | 13.28% | 10.55% |
| Sharpe Ratio | **0.617** | 0.435 | 0.448 | 0.525 | 0.388 | 0.454 |
| Max Drawdown | **-27.03%** | -59.58% | -35.05% | -35.60% | -27.39% | -32.84% |
| Calmar Ratio | **0.233** | 0.140 | 0.170 | 0.193 | 0.206 | 0.175 |

The observed Sharpe difference over SPY is +0.182 - positive but not statistically significant (bootstrap 95% CI [-0.318, 0.633], p=0.266). The genuine edge is volatility compression (8.09% vs 19.51%).

### Held-Out Test Period (2019-2024)

| Metric | Portfolio | SPY |
|---|---|---|
| Annualized Return | 6.31% | 14.87% |
| Annualized Volatility | 9.55% | 19.91% |
| Sharpe Ratio | 0.453 | 0.677 |
| Max Drawdown | -27.03% | -35.75% |
| Calmar Ratio | 0.234 | 0.416 |

The held-out Sharpe (0.453) lags SPY (0.677). A frozen model - trained once on 2006-2018 and never retrained - produces Sharpe 0.518 on the same period, still lagging SPY but ahead of the walk-forward variant.

| Metric | Walk-Forward | Frozen Model | SPY |
|---|---|---|---|
| Annualized Return | 6.31% | 7.16% | 14.87% |
| Annualized Volatility | 9.55% | 9.54% | 19.91% |
| Sharpe Ratio | 0.453 | 0.518 | 0.677 |
| Max Drawdown | -27.03% | -27.79% | -35.75% |
| Calmar Ratio | 0.234 | 0.258 | 0.416 |

### Subperiod Analysis

| Metric | Portfolio | SPY |
|---|---|---|
| **GFC (2008-2009)** | | |
| Annualized Return | **+4.41%** | -15.49% |
| Annualized Volatility | 11.01% | 34.70% |
| Sharpe Ratio | 0.387 | -0.345 |
| Max Drawdown | **-10.90%** | -56.38% |
| Calmar Ratio | 0.405 | -0.275 |
| **Low-vol bull (2013-2019)** | | |
| Annualized Return | 4.70% | 13.26% |
| Annualized Volatility | 5.33% | 12.83% |
| Sharpe Ratio | 0.726 | 1.003 |
| Max Drawdown | -7.54% | -19.82% |
| Calmar Ratio | **0.622** | 0.669 |
| **COVID+rates (2020-2024)** | | |
| Annualized Return | 4.33% | 11.84% |
| Annualized Volatility | 10.22% | 21.09% |
| Sharpe Ratio | 0.231 | 0.528 |
| Max Drawdown | -27.03% | -35.75% |
| Calmar Ratio | 0.160 | 0.331 |

**GFC:** The strategy's clearest result - portfolio gained +4.41% annualized while SPY lost 15.49%. Max drawdown -10.90% vs SPY -56.38%. The label_states function, which incorporates volatility in the scoring rule and now resolves feature columns by name rather than position, prevented mislabeling the crash as Bull.

**Low-vol bull:** Calmar ratio 0.622 vs SPY 0.669 - the strategy is close to SPY on return-per-drawdown during the bull market. Cost is absolute return (4.70% vs 13.26%).

**COVID+rates:** Weak. The 2020 crash was too fast for a causal HMM to detect in time. The 2022 rate-driven bear hit bonds and equities simultaneously, limiting the flight-to-safety allocation.

### Bootstrap Significance Test

Block bootstrap with 1000 iterations, 20-day block length, time-varying risk-free rate applied consistently:

| Statistic | Value |
|---|---|
| Observed Sharpe Difference | +0.182 |
| 95% CI | [-0.318, 0.633] |
| p-value | 0.266 |
| Significant at 95% | No |

The observed Sharpe difference is positive (+0.182) but the 95% CI straddles zero - not statistically significant. The drawdown reduction remains the defensible claim.

This p-value is computed two independent ways in this codebase - once in `backtest/bootstrap.py` (used by the main pipeline) and once in `scripts/stress_tests.py` (a standalone diagnostic script) - and the two now agree exactly (0.266 both). Prior to an audit-driven fix (see below), they did not: the `stress_tests.py` version had a formula error that made it return approximately 0.5 regardless of the true effect size.

### Window Sensitivity Sweep

Full pipeline rerun across `VOL_WINDOW ∈ {10, 21, 42}` × `CORR_WINDOW ∈ {42, 63, 126}`:

| VOL\CORR | 42 | 63 | 126 |
|---|---|---|---|
| **Full Sharpe** | | | |
| 10 | 0.499 | 0.670 | 0.730 |
| 21 | 0.736 | 0.617 | 0.710 |
| 42 | 0.728 | 0.767 | 0.558 |
| **Held-out Sharpe** | | | |
| 10 | 0.114 | 0.444 | 0.525 |
| 21 | 0.502 | 0.453 | 0.506 |
| 42 | 0.306 | 0.103 | 0.352 |

The default (VOL=21, CORR=63) is not the single best cell on either grid, but it sits in a broadly stable region rather than an isolated peak - several neighboring combinations perform similarly, which is the more relevant robustness property than any one cell's raw score.

### Bootstrap Block-Length Sensitivity

| Block Length | p-value | Mean Diff | 95% CI |
|---|---|---|---|
| 10 | 0.236 | 0.177 | [-0.311, 0.656] |
| 20 | 0.266 | 0.167 | [-0.318, 0.633] |
| 40 | 0.276 | 0.172 | [-0.368, 0.712] |

Conclusion stable across block lengths: not statistically significant.

### Walk-Forward Leakage Audit

`audit_walk_forward()` reconstructs all 75 quarterly folds and verifies train-end strictly precedes test-start for every fold. Result: **0 leakage detected across all 75 folds**. Full audit saved to `results/walk_forward_audit.csv`.

### Feature Ablation Study

Full pipeline rerun across four feature configurations to justify the baseline choice:

| Feature Set | Full Sharpe | Held-Out Sharpe | Max DD |
|---|---|---|---|
| baseline (vol_21, corr_63) | 0.617 | 0.453 | -27.03% |
| vol_10 (vol_10, corr_63) | 0.670 | 0.444 | -23.24% |
| vol_42 (vol_42, corr_63) | 0.767 | 0.103 | -28.71% |
| skew_kurt (skew+kurt, no corr) | 0.521 | 0.260 | -26.11% |

`vol_42` has the best full-sample Sharpe of any variant (0.767) but the worst held-out Sharpe (0.103) - a clear sign of overfitting to the full-sample window rather than a genuinely better feature. `vol_10` is close to baseline both in and out of sample. `skew_kurt` remains the weakest out-of-sample variant, though its held-out Sharpe (0.260) is materially better than earlier reported - an audit-driven fix (see below) found that regime labeling for this variant had been silently using skew as a stand-in for volatility, since it resolved feature columns by position rather than name and this variant has no volatility column at all. Baseline remains the most balanced choice: competitive full-sample Sharpe without `vol_42`'s out-of-sample fragility.

### Stress Tests

| Test | Portfolio | SPY |
|---|---|---|
| VaR (95%) | -0.79% | -1.84% |
| CVaR (95%) | -1.19% | -3.07% |
| Sharpe during VIX > 30 | -1.096 | -1.952 |
| Monte Carlo mean Sharpe (1000 paths) | 0.619 ± 0.228 | - |
| Monte Carlo 95% CI | [0.187, 1.062] | - |

Tail risk is dramatically lower than SPY across all measures. Both strategies produce negative Sharpe during VIX > 30 periods - nobody wins in a crisis - but the portfolio loses far less per unit of volatility.

---

## Audit Fixes (July 2026)

An external multi-agent audit (technical/code-quality, documentation-consistency, and math/methodology passes) was run against this repository and every finding was addressed. The two with genuine impact on reported results:

| Fix | Issue | Resolution |
|---|---|---|
| `risk_parity.py` | Redundant `sum(w)==1` constraint in the same QP as the log-barrier objective was breaking the equal-risk-contribution property it's supposed to guarantee (verified: ~18pp risk-contribution spread with the constraint vs ~0.002pp without, on a controlled test) | Removed the constraint; the existing post-hoc normalization is sufficient and correct |
| `stress_tests.py` paired bootstrap | p-value compared the bootstrap distribution to its own observed point estimate instead of to zero, returning ≈0.5 regardless of true effect size (verified with a synthetic Sharpe gap of 1.2+, still returned p≈0.5) | Corrected to test against the null of no difference, matching `backtest/bootstrap.py`'s existing correct formula |
| `label_states()` | Assumed feature column 0/1 were always `spy_return`/`spy_vol`; confirmed this silently mislabeled the `skew_kurt` ablation variant (no volatility column) by scoring states on `return - 0.5*skew` instead of `return - 0.5*vol` | Now resolves feature columns by name when available, falls back to position only for backward compatibility |
| `data/fetch.py`, `data/risk_free.py` | Cached data returned whenever the file existed, regardless of whether tickers/dates actually matched what was requested | Cache now validated against a signature of the request; auto-refetches on mismatch |
| `models/hmm.py` static model path (`--no-walk-forward`) | Same stale-cache issue - cached model reused regardless of whether `n_states` or the feature set had changed. Confirmed this can outright crash (`StandardScaler` feature-count mismatch), not just silently mislead | Same signature-validation fix as above |
| `backtest/engine.py` | `sensitivity_sweep.py`'s 9 parallel workers all wrote `backtest_results.parquet` through the same code path - a real race condition | `run()` gained a `save` toggle; the sweep and ablation scripts now skip the write entirely |
| `scripts/feature_ablation.py` | Each ablation variant overwrote the shared production `features.parquet`/`returns.parquet`, restored via `try/finally` - safe against normal exceptions but not a hard crash | Removed the shared-file round-trip entirely; `engine.run()` now accepts an in-memory `returns_df`, so nothing needs to touch disk |
| `models/hmm.py` EM fitting | Convergence was never checked - a "best of 10 restarts" could still not have converged, with no signal anywhere | Checks the genuine tolerance-based convergence condition (not hmmlearn's own flag, which conflates "converged" with "ran out of iterations") and reports it |
| `optimization/switcher.py` | The 99%-concentration safety fallback replaced any regime's degenerate output with full-universe equal-weight, including Crash - meaning a legitimate flight-to-safety allocation could get diluted with equity exposure at exactly the wrong moment | Crash's fallback is now scoped to its own safe-haven assets; other regimes unchanged (their optimizers already span the full universe by design) |
| `backtest/engine.py`, `metrics.py`, `benchmark.py` | First day's transaction cost was silently zero: `.diff()`'s all-NaN first row was silently summed to 0.0 by pandas rather than reflecting the true cost of entering from cash | Centralized into one `compute_turnover()` function used by all three, using an explicit zero-filled prior position instead of `.diff()` |
| `backtest/bootstrap.py`, `scripts/stress_tests.py` | Two independent implementations of "paired block bootstrap with shared random indices" - this is exactly how the p-value bug above happened, since the two copies drifted apart | Extracted one shared `block_bootstrap_indices()` helper |
| `models/hmm.py` BIC selection | `select_n_states()` fit and scored models purely to pick a winner, then discarded them and refit the winner from scratch - ~25% wasted computation per fold, confirmed to produce a bit-identical model either way | The already-fitted winning model is now reused directly |

Test coverage was also closed for every gap the audit identified: `switcher.py`'s concentration guard, `stress_tests.py` (previously untested), `data/process.py` (previously untested), and the `sensitivity_sweep.py`/`feature_ablation.py` integration tests, which were previously skipped as "full pipeline - execute manually." All three now run against fast synthetic data with mocked network calls instead. Test suite: **111 tests, 0 skipped.**

**A note on reproducibility discovered during this process:** re-running the full pipeline from a fresh `yfinance` download on a different day produces slightly different regime counts and Sharpe ratios than a previous fresh download, even with identical code, config, and random seeds. The most likely explanation is that Yahoo Finance's historical price history isn't perfectly static between download sessions (dividend/split adjustments and minor data corrections can be backfilled), and since BIC-based state selection is a discrete choice, even small price differences can occasionally flip which state count a fold selects. All numbers in this README come from a single, uninterrupted pipeline run to ensure internal consistency, but they should be read as representative rather than exactly reproducible bit-for-bit on a different day.

## Post-v6 Bug Fixes

| Fix | Issue | Resolution |
|---|---|---|
| feature_ablation.py restore block | KeyError on CFG["data"] - crashed on every completed ablation run | Read from FEATURE_SETS["baseline"] defined in the same file |
| test_optimizers.py | Byte-for-byte duplicate of test_metrics.py - zero optimizer coverage | Replaced with 24 real optimizer tests |
| bootstrap.py rf | Sharpe bootstrap computed with rf=None - inconsistent with pipeline | Thread time-varying rf through run_bootstrap() |
| bootstrap.py + charts.py paths | Hardcoded parquet paths bypassed config.yaml | Replaced with CFG["paths"] throughout |
| sensitivity_sweep n_jobs | CFG mutation never reached hmm.py - always ran with n_jobs=1 | Pass n_jobs explicitly to walk_forward_regimes() |
| stress_tests monte_carlo | IID resampling destroyed autocorrelation structure | Replaced with block_resample() consistent with bootstrap.py |
| main.py test_end | Dead variable set to train_end (wrong value, never used) | Removed |
| min_variance.py | Solver output returned raw - tiny floating point negatives possible | Clip and renormalize before returning |
| switcher.py | Magic number 0.99 hardcoded in concentration guard | Named constant CONCENTRATION_GUARD |
| hmm.py forward_filter | No underflow guard at alpha[0] initialization | Apply same zero-sum guard as t>0 steps |
| main.py transition matrix | Division by zero if a regime never transitions out | Replace 0 row sums with NaN before division |
| benchmark.py | Hardcoded 126 in get_risk_parity_equity() | Read from CFG["hmm"]["min_train_days"] // 2 |


## v5 - v6 Improvements

| Change | v5 | v6 |
|---|---|---|
| Regime posteriors | Viterbi + forward-backward - within-window lookahead | Causal forward-filter - strictly uses only past observations |
| Label permutation | Undocumented | Explicitly documented: labels assigned by mean return rank after every retrain |
| run_period() optimizer | Passed returns_slice - optimizer starved in early test period | Passes full returns - optimizer uses all available history up to each date |
| charts.py | KeyError on Crash regime | Crash added to REGIME_COLORS |
| safe_haven_assets | Hardcoded in crash.py | Configurable via config.yaml |
| select_n_states | Unused train/test split | Removed - BIC on full training data directly |
| sensitivity_sweep | Global config mutation (ineffective) | n_jobs passed explicitly to walk_forward_regimes() |
| smoothing_days | Dead config key | Removed |

## v4 - v5 Improvements

| Change | v4 | v5 |
|---|---|---|
| Optimizer recompute | Only active regime on intra-quarter transition | All four on every regime change |
| RF in max-Sharpe QP | Hardcoded 4% | Time-varying 3-month T-bill |
| BIC scoring | On held-out 20% (wrong N) | On full training data |
| Min-history threshold | Hardcoded 126 | References config |
| fit_hmm duplication | Two near-identical functions | Unified via _fit_hmm_core() |

## v3 - v4 Improvements

| Change | v3 | v4 |
|---|---|---|
| Nested CV execution | Never called - every fold used fixed n=4 | Genuinely runs per fold |
| Fold execution | Sequential | joblib.Parallel |
| Sensitivity analysis | Not tested | Window + bootstrap sweep |

## v2 - v3 Improvements

| Change | v2 | v3 |
|---|---|---|
| NaN posterior handling | Missing | Explicit skip |
| Benchmarks | 4 | + Risk Parity |
| Subperiod analysis | None | GFC / Low-vol bull / COVID+rates |
| Normality testing | None | Jarque-Bera per regime |

## v1 - v2 Improvements

| Change | v1 | v2 |
|---|---|---|
| Regime states | 3 | 4 (+ Crash) |
| Regime uncertainty | Hard label | Posterior probability blend |
| Walk-forward | Optional | Default |
| Risk-free rate | Static 4% | Time-varying ^IRX |
| Benchmark costs | None | 2bps per turnover |
| Significance testing | None | Block bootstrap |
| Out-of-sample evaluation | None | Held-out 2019-2024 |

---

## Architecture

```
regime-adaptive-portfolio/
├── config.yaml            # Centralized hyperparameters (HMM, features, backtest, market, optimizer, bootstrap, paths, evaluation, subperiods)
├── config.py              # YAML loader
├── data/
│   ├── fetch.py           # Downloads adjusted close prices via yfinance, cache validated against request signature
│   ├── process.py         # Log returns, rolling volatility, rolling correlation - window sizes read from config.yaml
│   ├── risk_free.py       # 3-month T-bill rate (^IRX) with daily conversion and signature-validated caching
│   └── cache_utils.py      # Shared cache signature validation, used by fetch.py, risk_free.py, and models/hmm.py
├── models/
│   └── hmm.py             # Gaussian HMM, causal forward-filter, BIC state selection, parallelized walk-forward, convergence diagnostics
├── optimization/
│   ├── mean_var.py        # Mean-Variance max Sharpe (Bull) - time-varying RF
│   ├── risk_parity.py     # Log-barrier risk parity approximation (Bear)
│   ├── min_variance.py    # Minimum Variance QP (Sideways)
│   ├── crash.py           # Inverse-vol flight-to-safety (Crash) - safe-haven assets from config
│   └── switcher.py        # Causal posterior blending, all-optimizer recompute on transition, regime-scoped concentration guard
├── backtest/
│   ├── engine.py          # Simulation with transaction costs (incl. day-1 entry cost), correct period slicing
│   ├── metrics.py         # Annualized return, volatility, Sharpe (time-varying RF), drawdown, Calmar, shared turnover calc
│   ├── benchmark.py       # SPY, Equal Weight, 60/40, Momentum, Risk Parity - all cost-adjusted
│   └── bootstrap.py       # Block bootstrap significance test with time-varying RF, shared index-generation helper
├── scripts/
│   ├── sensitivity_sweep.py    # Window sensitivity analysis - n_jobs=-1 for speed
│   ├── frozen_model_eval.py    # True out-of-sample evaluation - model frozen at 2018-12-31
│   ├── feature_ablation.py     # Ablation study across feature sets and window sizes - no production-file side effects
│   └── stress_tests.py         # VaR, CVaR, VIX-conditional Sharpe, block-resampled Monte Carlo, paired bootstrap
├── results/
│   ├── figures/                # charts.png, bootstrap.png, pca_regimes.png, state_selection.png
│   ├── sensitivity_results.csv
│   ├── walk_forward_audit.csv  # 75 folds, 0 leakage confirmed
│   └── ablation_results.csv
├── tests/
│   ├── conftest.py            # shared fixtures (mocked risk-free rate)
│   ├── test_metrics.py        # metrics module, incl. turnover
│   ├── test_hmm.py            # HMM module: BIC selection, convergence, label_states, transition matrices, model caching
│   ├── test_engine.py         # backtest engine, incl. save toggle and in-memory returns_df
│   ├── test_optimizers.py     # all four optimizers, incl. risk-contribution equality, and compute_weights
│   ├── test_switcher.py       # concentration guard, previously untested
│   ├── test_data_cache.py     # cache signature validation for fetch_prices/fetch_risk_free
│   ├── test_process.py        # feature computation, config-driven windows
│   ├── test_stress_tests.py   # paired bootstrap correctness, VaR/CVaR ordering
│   ├── test_main.py           # regime-run and empirical transition matrix computation
│   ├── test_integration.py    # end-to-end pipeline regression
│   └── test_scripts.py        # sensitivity sweep and ablation integration (synthetic data), bootstrap rf, Viterbi vs forward-filter
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
| scipy | Jarque-Bera normality test + forward-filter emission probabilities |
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

# Run full pipeline (causal forward-filter, BIC state selection, walk-forward default)
python main.py

# Force HMM retrain from scratch
python main.py --retrain

# Disable walk-forward (use static model)
python main.py --no-walk-forward

# Skip chart generation
python main.py --no-charts

# Run sensitivity sweep (VOL_WINDOW × CORR_WINDOW, parallelized)
python -m scripts.sensitivity_sweep

# Run frozen model out-of-sample evaluation (train ≤2018, test 2019-2024)
python -m scripts.frozen_model_eval

# Run feature ablation study
python -m scripts.feature_ablation

# Run stress tests (VaR, CVaR, VIX-conditional, block-resampled Monte Carlo)
python -m scripts.stress_tests

# Audit walk-forward leakage
python -c "from models.hmm import audit_walk_forward; audit_walk_forward()"

# Run tests
pytest tests/
```

---

## Key Design Decisions

**Why Gaussian HMM?** Markets exhibit persistent regimes - the HMM transition matrix captures this persistence. Gaussian emissions model the continuous feature space naturally, though the Jarque-Bera test rejects normality.

**Why forward-filter instead of Viterbi?** Viterbi finds the globally optimal state sequence using both forward and backward passes - this introduces lookahead within the observation window. The forward pass alone introduces zero bias at every timestamp.

**Why label states by mean return rank, resolved by feature name?** HMM state indices are arbitrary - they can permute across retraining windows. Sorting by mean SPY return after every retrain ensures Bull, Bear, Sideways, and Crash maintain consistent interpretations. Resolving `spy_return`/`spy_vol` by column name rather than position means this stays correct even when the feature set itself changes (e.g. ablation variants with different columns).

**Why BIC state selection per fold?** A globally fixed n=4 assumes the same model complexity is appropriate across all market environments. Per-fold BIC selection allows early folds with limited data to choose n=2 or n=3.

**Why time-varying RF in both optimizer and metrics?** The 4% hurdle during 2009-2022 (near-zero rate era) distorted allocations. Using the same 3-month T-bill rate in both the optimizer and Sharpe computation eliminates this distortion.

**Why recompute all optimizers on every regime change?** The blended weights are `sum(p_k * w_k)` across all four regimes. All four weight vectors must reflect the same market environment for the blending to be coherent.

**Why pass full returns to run_period()?** The optimizer needs a full covariance history to compute reliable weights. Slicing returns to the test period starves the optimizer in the early months, producing unstable allocations.

**Why posterior blending?** Hard switching ignores the model's own uncertainty. Posterior blending hedges proportionally.

**Why cost-adjust benchmarks?** Fair comparison requires consistent cost treatment. All benchmarks incur 2bps per unit of monthly rebalancing turnover, including the true cost of their initial entry from cash.

**Why Ledoit-Wolf shrinkage?** Sample covariance matrices from short windows are noisy and ill-conditioned.

**Why inverse-vol weighting for the Crash regime?** Equal-weight over safe havens ignores the fact that TLT is significantly more volatile than IEF. Inverse-vol allocation tilts toward the most stable safe haven available.

**Why scope the concentration guard's fallback for Crash?** Bull/Bear/Sideways optimizers already span the full asset universe, so full-universe equal-weight is the natural fallback for them. Crash is different: it has a well-defined subset (the safe havens), and diluting a genuine flight-to-safety allocation with equity exposure defeats the point of the regime existing.

**Why validate the data/model cache against a signature instead of just file existence?** A cache that's merely "present" isn't necessarily "correct for the current config" - checking only existence meant changing tickers, dates, `n_states`, or the feature set could silently return stale data or even crash on a shape mismatch.

**Why is walk-forward the default if the frozen model performs better out-of-sample?** Walk-forward is the correct methodology for live deployment - a real system would retrain as new data arrives. The frozen variant is a retrospective curiosity.

---

## Limitations

**Regime separation is weak in feature space.** PCA of the three features (SPY return, volatility, mean correlation) projected to 2D shows heavy overlap between Bull, Bear, and Sideways in the dense center.

**Expanding covariance window mixes regimes.** The optimizer uses `returns.loc[:date]` - an expanding window that grows from ~3,260 to ~4,780 days. Early folds use covariance estimates dominated by the 2008 crisis.

**Crash regime uses inverse-vol weighting over safe havens.** The Crash optimizer allocates inversely proportional to current volatility across IEF, TLT, and GLD. In 2022, all three sold off simultaneously, limiting diversification.

**Gaussian emission assumption violated.** Jarque-Bera tests reject normality for all four regimes (p≈0). Crash regime excess kurtosis is 6.09 - the most extreme of the four regimes and clearly fat-tailed relative to a Gaussian (Bull/Bear/Sideways sit closer to 1.8-2.3). A Student-t HMM would be more appropriate, particularly for the Crash regime.

**Sharpe outperformance not significant.** The paired block bootstrap p-value is 0.266 - the observed Sharpe difference (+0.182) is positive but the 95% CI straddles zero. This p-value is now computed identically by two independent code paths (`backtest/bootstrap.py` and `scripts/stress_tests.py`) after an audit-driven fix corrected a formula error in the latter (see Audit Fixes above) - both now report 0.266, which is a useful internal consistency check in itself.

**Forward-filter performance cost.** Replacing Viterbi with forward-filter reduces full-sample Sharpe because Viterbi's within-window smoothing produced better labels in hindsight. Forward-filter is causally correct but empirically weaker.

**Scaler fit twice per fold.** `select_n_states()` fits its own `StandardScaler` internally for BIC scoring, and the main fit path fits a second scaler on the same training data. Both produce identical transformations but represent redundant computation.

**Choppy Bull detection.** Median Bull run length is 5.0 days. BIC selecting n=2 on some folds collapses regimes, causing rapid label switching.

**COVID+rates weakness.** The 2020 crash was too fast for a causal HMM. The 2022 rate-driven bear hit bonds and equities simultaneously, limiting the flight-to-safety allocation.

**SPY circularity.** SPY is both the regime-detection instrument (return and volatility features) and an investable asset in the portfolio. VIX and yield curve slope (5Y-3M spread) were tested as alternatives but performed worse.

**Mean return as signal.** The Mean-Variance optimizer uses sample mean returns - near-zero signal-to-noise at daily frequency. Shrinkage toward a factor model or Black-Litterman prior would be more robust.

**State labeling heuristic.** `label_states()` ranks states by `mean_return - 0.5 × mean_vol` - a risk-adjusted score that prevents mislabeling high-volatility states with similar returns. The 0.5 weight is tuned heuristically, not data-driven. (As of the audit fixes above, the two inputs are now resolved by column name rather than position - this changes *which columns* feed the formula in edge cases, not the formula itself.)

**Risk parity is a tighter approximation than before, but still an approximation.** The Bear regime optimizer minimizes `quad_form(w, sigma) - (1/n)*sum(log(w))` - the Spinu (2013) log-barrier surrogate - subject only to box constraints, normalized post-hoc. An audit-driven fix removed a redundant `sum(w)==1` equality constraint that had been substantially loosening the equal-risk-contribution property this method is meant to guarantee (verified: risk-contribution spread dropped from ~18 percentage points to ~0.002 on a controlled test). It remains an approximation, not an exact solution, but a much closer one.

**Asset universe limited to 8 ETFs.** Transaction costs modeled at 2bps flat - not accounting for market impact at scale. A broader universe or factor-based selection could improve diversification.

**Markov assumption.** HMMs assume the current state depends only on the previous state. Markets exhibit momentum and mean-reversion at multiple horizons, violating this assumption.

**Position limits capped at 60% per asset.** All three optimizers - Mean-Variance, Risk Parity, and Minimum Variance - enforce `w <= 60%` per asset via CVXPY constraints.

**Forward filter initialization uses training startprob.** In test folds starting mid-cycle, the true initial state distribution may differ. This is a minor bias that cannot be corrected without knowing the true regime.

**Time period bias.** The full backtest covers 2006-2024 - a period that includes the GFC, a decade-long bull market, COVID, and a rate shock. Results may not generalize to other macroeconomic regimes.

**Data is not perfectly reproducible session-to-session.** A fresh `yfinance` download on a different day can shift regime counts and Sharpe ratios slightly relative to a previous fresh download, even with identical code and seeds, most likely due to Yahoo Finance backfilling minor historical price corrections between sessions. All numbers reported here come from one uninterrupted pipeline run for internal consistency; treat them as representative rather than exactly bit-reproducible on a different day.

---

Abdelkrim - Applied Mathematics & AI, PSL-Dauphine
https://github.com/AbdelkrimCode
