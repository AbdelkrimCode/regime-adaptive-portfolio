# Regime Adaptive Portfolio - v7

Algorithmic portfolio optimizer combining Hidden Markov Model regime detection with convex optimization.

![Python](https://img.shields.io/badge/python-3.10+-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Tests](https://github.com/AbdelkrimCode/regime-adaptive-portfolio/actions/workflows/tests.yml/badge.svg)
---

## How It Works

### 1. Regime Detection
A Gaussian HMM trained on three features - SPY log returns, rolling volatility, and rolling mean pairwise correlation - labels each trading day as Bull, Bear, Sideways, or Crash. Feature windows default to 21-day rolling volatility and 63-day rolling correlation, both centralized in `config.yaml`.

State ordering is determined by a risk-adjusted score after every retrain:
- Score = mean SPY return - 0.5 x mean rolling volatility
- Lowest score -> Crash
- Second lowest -> Bear
- Second highest -> Sideways
- Highest -> Bull

The score's two inputs are located by feature name, not column position, so the scoring rule stays correct even when the feature set changes shape (e.g. the skew/kurtosis ablation variant, which has no volatility column at all).

**Label stability across retrains:** HMM states have no intrinsic identity - state 0 in Q1 might correspond to "Bear" and state 0 in Q2 might correspond to "Bull". This implementation avoids label permutation by re-ranking states post-hoc.

### 2. BIC State Selection
At every quarterly fold, candidate state counts [2, 3, 4] are each fitted on the full training window and scored by BIC. The state count minimizing BIC is selected independently per fold. The winning fit is reused directly rather than refit from scratch.

**Implementation note:** When a fold selects n=2, it emits only `p_bull` and `p_bear` posteriors. Concatenating folds with different state counts produces NaN entries in the unused posterior columns. These are skipped (not forward-filled) during portfolio construction.

**Convergence diagnostics:** Each fold's HMM fit reports whether the EM algorithm genuinely converged (log-likelihood improvement below tolerance) versus merely exhausting its iteration budget. Across the current 75-fold walk-forward run: **75/75 folds converged.**

### 3. Causal Forward-Filter Posteriors
Prior to v6, regime posteriors were computed using the Viterbi algorithm and `predict_proba()` - both of which use forward and backward passes over the entire observation window. This introduces within-window lookahead bias.

v6 replaced both with a manual forward-filter (`forward_filter()` in `hmm.py`). The filtered posterior `P(state_t | observations_1:t)` is computed using only the forward pass - strictly causal at every time step. A controlled comparison holding the fitted models fixed shows the causal forward filter trailing forward-backward smoothed posteriors by roughly 0.02 Sharpe - the expected direction of effect, since smoothing benefits from hindsight a live system wouldn't have. The forward-filter number is the one reported throughout this README, since it's the one a live system could actually achieve.

### 4. Parallelized Fold Execution
Each quarterly fold is independent - no shared mutable state. Folds execute via `joblib.Parallel`. `n_jobs=1` in `config.yaml` keeps the default deterministic; the sensitivity sweep passes `n_jobs=-1` for speed.

### 5. Posterior Probability Blending
v1 used hard regime labels - each day was assigned exactly one regime. This amplifies turnover at transition boundaries and ignores the HMM's uncertainty.

v2 uses filtered posteriors for every day:

```
weights(t) = P(Bull|t) x w_bull + P(Bear|t) x w_bear
           + P(Sideways|t) x w_sideways + P(Crash|t) x w_crash
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

**Position limits (60% per asset) require care about *when* they're applied.** Min-Variance and Bull/Max-Sharpe both build the 60% cap directly into their convex problem (the latter via a correctly-scaled constraint in the transformed variable space, since it solves for an auxiliary variable `y` rather than `w` directly). **Risk Parity is different, and getting this wrong was the single most consequential bug found in this project's audit** (see "Audit Fixes" below): the log-barrier objective's natural solution scale is arbitrary, unrelated to summing to 1, and routinely exceeds 1.0 per asset against real covariance magnitudes. Applying a 0.6 cap to that raw, pre-normalization scale caused most or all assets to saturate at the cap simultaneously - normalizing N identical capped values just reproduces equal weight. The fix: solve unconstrained (only a small floor to keep the log-barrier finite), normalize to sum to 1, *then* cap-and-redistribute any over-limit asset among the rest.

**Concentration guard:** if any regime's optimizer output still concentrates more than 99% of weight in a single asset (numerical degeneracy), it's replaced with an equal-weight fallback. For Crash specifically, that fallback is scoped to the regime's own safe-haven assets (IEF/TLT/GLD) rather than the full 8-asset universe - falling back to the full universe would inject equity exposure at exactly the moment the Crash regime exists to avoid it.

### 7. Regime Diagnostics
Run-length statistics from the walk-forward label sequence:

| Regime | Count | Mean (days) | Median (days) | Min | Max |
|---|---|---|---|---|---|
| Bull | 73 | 17.3 | 5.0 | 1 | 171 |
| Bear | 67 | 20.1 | 17.0 | 1 | 62 |
| Sideways | 67 | 18.2 | 11.0 | 1 | 128 |
| Crash | 32 | 27.9 | 24.5 | 3 | 62 |

Empirical transition matrix (% of exits):

| From \ To | Bull | Bear | Sideways | Crash |
|---|---|---|---|---|
| Bull | - | 37.0% | 61.6% | 1.4% |
| Bear | 47.8% | - | 19.4% | 32.8% |
| Sideways | 51.5% | 34.8% | - | 13.6% |
| Crash | 21.9% | 50.0% | 28.1% | - |

### 8. Backtest
Lookahead-free simulation applying day-t weights to day t+1 returns. Transaction costs at 2bps per unit of turnover applied consistently to portfolio and all benchmarks, including the true cost of the first day's entry from cash. Average daily turnover: 3.79% - lower than earlier reported figures, since Risk Parity now produces balanced allocations instead of accidentally concentrated ones, which need smaller adjustments as regimes blend.

Sharpe ratios use a time-varying risk-free rate sourced from the 3-month T-bill (^IRX via yfinance). The same time-varying rate is used in the max-Sharpe optimizer. The `run_period()` function passes the full returns history to each optimizer, not just the in-sample slice.

---

## Results

The primary claim of this strategy is **drawdown control**, not Sharpe outperformance. The observed Sharpe difference vs SPY (0.561 vs 0.435) is positive but not statistically significant (bootstrap p=0.344).

- Max drawdown **-25.59% vs SPY -59.58%** - 57% reduction in peak-to-trough loss
- Calmar ratio **0.215 vs SPY 0.140** - better return per unit of drawdown
- Volatility **7.46% vs SPY 19.51%** - dramatically lower realized risk
- Held-out (2019-2024) performance is honestly weaker than earlier reports suggested (see below) - the corrected Risk Parity optimizer trades some out-of-sample resilience for actual risk balance

### Full Period (2006-2024)

| Metric | Portfolio | SPY | Equal Weight | 60/40 | Momentum | Risk Parity |
|---|---|---|---|---|---|---|
| Annualized Return | 5.49% | 8.34% | 5.96% | 6.88% | 5.63% | 4.81% |
| Annualized Volatility | 7.46% | 19.51% | 11.17% | 11.17% | 13.28% | 7.56% |
| Sharpe Ratio | **0.561** | 0.435 | 0.448 | 0.525 | 0.388 | 0.479 |
| Max Drawdown | **-25.59%** | -59.58% | -35.05% | -35.60% | -27.39% | -24.36% |
| Calmar Ratio | **0.215** | 0.140 | 0.170 | 0.193 | 0.206 | 0.197 |

The observed Sharpe difference over SPY is +0.126 - positive but not statistically significant (bootstrap 95% CI [-0.428, 0.616], p=0.344). Note the standalone Risk Parity benchmark: volatility (7.56%) and max drawdown (-24.36%) are now among the best of any strategy in this table, including the main portfolio - a direct, verifiable consequence of the risk-parity optimizer fix described in Audit Fixes below.

### Held-Out Test Period (2019-2024)

| Metric | Portfolio | SPY |
|---|---|---|
| Annualized Return | 4.23% | 14.87% |
| Annualized Volatility | 8.68% | 19.91% |
| Sharpe Ratio | 0.261 | 0.677 |
| Max Drawdown | -25.59% | -35.75% |
| Calmar Ratio | 0.165 | 0.416 |

The held-out Sharpe (0.261) trails SPY (0.677) by a wide margin. This is meaningfully weaker than earlier-reported figures for this same period - see "Net effect of the risk-parity fix" below for why, and an honest account of the tradeoff involved.

A frozen model - trained once on 2006-2018 and never retrained - produces Sharpe 0.339 on the same period, still lagging SPY but ahead of the walk-forward variant, confirming the strategy generalizes rather than depending on retraining within the test window.

| Metric | Walk-Forward | Frozen Model | SPY |
|---|---|---|---|
| Annualized Return | 4.23% | 5.09% | 14.87% |
| Annualized Volatility | 8.68% | 8.57% | 19.91% |
| Sharpe Ratio | 0.261 | 0.339 | 0.677 |
| Max Drawdown | -25.59% | -25.66% | -35.75% |
| Calmar Ratio | 0.165 | 0.198 | 0.416 |

### Subperiod Analysis

| Metric | Portfolio | SPY |
|---|---|---|
| **GFC (2008-2009)** | | |
| Annualized Return | **+4.65%** | -15.49% |
| Annualized Volatility | 11.01% | 34.70% |
| Sharpe Ratio | 0.408 | -0.345 |
| Max Drawdown | **-10.90%** | -56.38% |
| Calmar Ratio | 0.427 | -0.275 |
| **Low-vol bull (2013-2019)** | | |
| Annualized Return | 4.12% | 13.26% |
| Annualized Volatility | 4.89% | 12.83% |
| Sharpe Ratio | 0.673 | 1.003 |
| Max Drawdown | -7.67% | -19.82% |
| Calmar Ratio | 0.536 | 0.669 |
| **COVID+rates (2020-2024)** | | |
| Annualized Return | 2.14% | 11.84% |
| Annualized Volatility | 9.28% | 21.09% |
| Sharpe Ratio | 0.016 | 0.528 |
| Max Drawdown | -25.59% | -35.75% |
| Calmar Ratio | 0.084 | 0.331 |

**GFC:** The strategy's clearest result - portfolio gained +4.65% annualized while SPY lost 15.49%. Max drawdown -10.90% vs SPY -56.38%. This subperiod is essentially unaffected by the risk-parity fix, since 2008-2009's dominant regime was Crash/Bear-heavy in a way that concentrated risk parity would have handled similarly to balanced risk parity.

**Low-vol bull:** Calmar ratio 0.536 vs SPY 0.669. The strategy trails SPY on absolute return during the bull market, as expected for a defensively-positioned strategy in a low-volatility uptrend.

**COVID+rates:** Now the strategy's weakest subperiod by a wide margin (Sharpe 0.016, essentially flat). The 2020 crash was too fast for a causal HMM to detect in time, and the 2022 rate-driven bear hit bonds and equities simultaneously - the corrected, risk-balanced Bear-regime allocation appears to have been more exposed to that specific correlation breakdown than the previous (accidentally equal-weighted) version was. This limitation is reported here directly rather than left out - see Audit Fixes below.

### Bootstrap Significance Test

Block bootstrap with 1000 iterations, 20-day block length, time-varying risk-free rate applied consistently:

| Statistic | Value |
|---|---|
| Observed Sharpe Difference | +0.126 |
| 95% CI | [-0.428, 0.616] |
| p-value | 0.344 |
| Significant at 95% | No |

The observed Sharpe difference is positive (+0.126) but the 95% CI straddles zero - not statistically significant. This p-value is computed two independent ways in this codebase - once in `backtest/bootstrap.py` and once in `scripts/stress_tests.py` - and the two agree exactly (0.344 both), which is itself a useful consistency check following an audit-driven fix to the latter's formula (see below).

### Window Sensitivity Sweep

Full pipeline rerun across `VOL_WINDOW ∈ {10, 21, 42}` x `CORR_WINDOW ∈ {42, 63, 126}`:

| VOL\CORR | 42 | 63 | 126 |
|---|---|---|---|
| **Full Sharpe** | | | |
| 10 | 0.515 | 0.639 | 0.669 |
| 21 | 0.701 | 0.561 | 0.663 |
| 42 | 0.667 | 0.694 | 0.532 |
| **Held-out Sharpe** | | | |
| 10 | 0.075 | 0.333 | 0.332 |
| 21 | 0.322 | 0.261 | 0.316 |
| 42 | 0.202 | 0.150 | 0.242 |

The default (VOL=21, CORR=63) is not the single best cell on either grid, but it sits in a broadly stable region rather than an isolated peak, which is the more relevant robustness property than any one cell's raw score.

### Bootstrap Block-Length Sensitivity

| Block Length | p-value | Mean Diff | 95% CI |
|---|---|---|---|
| 10 | 0.332 | 0.121 | [-0.417, 0.653] |
| 20 | 0.344 | 0.110 | [-0.428, 0.616] |
| 40 | 0.347 | 0.115 | [-0.487, 0.661] |

Conclusion stable across block lengths: not statistically significant.

### Walk-Forward Leakage Audit

`audit_walk_forward()` reconstructs all 75 quarterly folds and verifies train-end strictly precedes test-start for every fold. Result: **0 leakage detected across all 75 folds**. Full audit saved to `results/walk_forward_audit.csv`.

### Feature Ablation Study

Full pipeline rerun across four feature configurations to justify the baseline choice:

| Feature Set | Full Sharpe | Held-Out Sharpe | Max DD |
|---|---|---|---|
| baseline (vol_21, corr_63) | 0.561 | 0.261 | -25.59% |
| vol_10 (vol_10, corr_63) | 0.639 | 0.333 | -22.58% |
| vol_42 (vol_42, corr_63) | 0.694 | 0.150 | -25.98% |
| skew_kurt (skew+kurt, no corr) | 0.495 | 0.098 | -25.72% |

`vol_42` again shows the best full-sample Sharpe of any variant but a much weaker held-out Sharpe - the same overfitting signature seen throughout this study regardless of which version of the risk-parity fix is in place. `vol_10` outperforms baseline on both full-sample and held-out Sharpe this round; still retained as a candidate worth further investigation rather than adopted as the new default, since a single sensitivity sweep run isn't sufficient grounds to change the baseline without further validation. `skew_kurt` remains the weakest performer.

### Stress Tests

| Test | Portfolio | SPY |
|---|---|---|
| VaR (95%) | -0.74% | -1.84% |
| CVaR (95%) | -1.08% | -3.07% |
| Sharpe during VIX > 30 | -0.496 | -1.952 |
| Monte Carlo mean Sharpe (1000 paths) | 0.562 | - |
| Monte Carlo 95% CI | [0.139, 1.009] | - |

Tail risk is dramatically lower than SPY across all measures. Sharpe during VIX > 30 periods also improved substantially versus earlier reports (-0.496 vs a previously-reported -1.096). Both strategies still produce negative Sharpe during acute crisis periods, but the portfolio's loss per unit of risk is now much closer to breakeven than before. This is a separate effect from the 2022 weakness discussed under COVID+rates above - VIX spikes are sharp and short, while 2022 was a slower, grinding correlation breakdown between bonds and equities.

---

## Audit Fixes

An external multi-agent audit (technical/code-quality, documentation-consistency, and math/methodology passes) was run against this repository. Every finding was addressed - full detail, including two fixes that required a second pass after an initial fix turned out to be incomplete, is in **`audit_fixes_report.md`** in the repository root. Summary of the most consequential items:

**Risk Parity (the headline fix, two passes).** The Bear-regime optimizer's log-barrier objective had a redundant `sum(w)==1` equality constraint sitting inside the same optimization problem as its position-limit cap - both individually plausible-looking, but together this broke the equal-risk-contribution property the method is meant to guarantee. A first fix removed the redundant equality constraint and validated the result on synthetic i.i.d. data (18pp -> 0.002pp risk-contribution spread) - correct as far as it went, but the synthetic test data never actually engaged the position-limit cap, since i.i.d. returns keep the optimizer's natural solution scale well under the cap. Against real historical data for the actual 8-asset universe, the cap - still applied to the *raw, pre-normalization* variable - caused most or all assets to saturate simultaneously, silently reproducing equal weight (14-23pp spread across three real historical windows, confirmed). The complete fix moves the position cap to apply *after* normalization, with a cap-and-redistribute procedure for any asset still over the limit post-normalization (spread now 0.0005-0.0014pp on the same real windows). This is now the codebase's Risk Parity implementation, and it has a real, honestly-reported effect on results: the standalone Risk Parity benchmark improved on every metric, while the main strategy's Bear-regime allocation - no longer accidentally equal-weighted - shows reduced resilience specifically to 2022's correlated bond/equity selloff. See `audit_fixes_report.md` for the full verification trail.

**`stress_tests.py`'s bootstrap p-value** compared the bootstrap distribution to its own observed point estimate instead of to zero, returning approximately 0.5 regardless of true effect size - fixed to match the already-correct convention in `backtest/bootstrap.py`; the two now agree to three decimal places on every run.

**`label_states()`** resolved its two scoring inputs by column position rather than name, silently mislabeling the `skew_kurt` ablation variant (which has no volatility column) by scoring on skewness instead - now resolves by name.

**Silent stale-cache pattern** across `data/fetch.py`, `data/risk_free.py`, and the static (non-walk-forward) HMM model path - cached data was returned whenever the file existed, regardless of whether the actual request (tickers, dates, `n_states`, features) matched. Confirmed this could crash outright, not just silently mislead. Cache now validated against a signature of the request.

**A race condition** in `sensitivity_sweep.py` (9 parallel workers writing the same results file) and **shared-state mutation** in `feature_ablation.py` (each variant overwriting production `features.parquet`/`returns.parquet`) - both eliminated, not just mitigated, by passing data in-memory instead of round-tripping through shared disk paths.

**HMM convergence** was never checked - a "best of 10 restarts" could still not have genuinely converged, with no signal anywhere. Now checked against the correct tolerance-based condition (hmmlearn's own flag conflates "converged" with "ran out of iterations").

**The `CONCENTRATION_GUARD` fallback** replaced any regime's degenerate optimizer output with full-universe equal-weight, including Crash - now scoped to Crash's own safe havens specifically, since diluting a flight-to-safety allocation with equity exposure at exactly the wrong moment defeated the point of the regime.

**First-day transaction cost** was silently zero across three files due to a pandas `skipna` quirk on an all-NaN first row - fixed once, centrally, instead of three times independently.

Test coverage was closed for every gap identified, including the three tests previously skipped as "full pipeline - execute manually" (now fast synthetic-data versions that actually run in CI). **Test suite: 114 tests, 0 skipped.**

## Post-v6 Bug Fixes

| Fix | Issue | Resolution |
|---|---|---|
| feature_ablation.py restore block | KeyError on CFG["data"] - crashed on every completed ablation run | Read from FEATURE_SETS["baseline"] defined in the same file |
| test_optimizers.py | Byte-for-byte duplicate of test_metrics.py - zero optimizer coverage | Replaced with real optimizer tests |
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
├── config.yaml                  # Centralized hyperparameters (HMM, features, backtest, market, optimizer, bootstrap, paths, evaluation, subperiods)
├── config.py                    # YAML loader
├── audit_fixes_report.md        # Full record of every audit fix, with before/after verification
├── data/
│   ├── fetch.py                 # Downloads adjusted close prices via yfinance, cache validated against request signature
│   ├── process.py                # Log returns, rolling volatility, rolling correlation - window sizes read from config.yaml
│   ├── risk_free.py             # 3-month T-bill rate (^IRX) with daily conversion and signature-validated caching
│   └── cache_utils.py            # Shared cache signature validation, used by fetch.py, risk_free.py, and models/hmm.py
├── models/
│   └── hmm.py                   # Gaussian HMM, causal forward-filter, BIC state selection, parallelized walk-forward, convergence diagnostics
├── optimization/
│   ├── mean_var.py              # Mean-Variance max Sharpe (Bull) - time-varying RF
│   ├── risk_parity.py           # Log-barrier risk parity - floor-only optimization, cap-and-redistribute applied post-normalization
│   ├── min_variance.py          # Minimum Variance QP (Sideways)
│   ├── crash.py                 # Inverse-vol flight-to-safety (Crash) - safe-haven assets from config
│   └── switcher.py              # Causal posterior blending, all-optimizer recompute on transition, regime-scoped concentration guard
├── backtest/
│   ├── engine.py                 # Simulation with transaction costs (incl. day-1 entry cost), correct period slicing, in-memory returns_df support
│   ├── metrics.py                # Annualized return, volatility, Sharpe (time-varying RF), drawdown, Calmar, shared turnover calc
│   ├── benchmark.py              # SPY, Equal Weight, 60/40, Momentum, Risk Parity - all cost-adjusted
│   └── bootstrap.py              # Block bootstrap significance test with time-varying RF, shared index-generation helper
├── scripts/
│   ├── sensitivity_sweep.py     # Window sensitivity analysis - n_jobs=-1 for speed, no shared-file writes
│   ├── frozen_model_eval.py     # True out-of-sample evaluation - model frozen at 2018-12-31
│   ├── feature_ablation.py      # Ablation study across feature sets and window sizes - no production-file side effects
│   └── stress_tests.py          # VaR, CVaR, VIX-conditional Sharpe, block-resampled Monte Carlo, paired bootstrap
├── results/
│   ├── figures/                 # charts.png, bootstrap.png, pca_regimes.png, state_selection.png
│   ├── sensitivity_results.csv
│   ├── walk_forward_audit.csv   # 75 folds, 0 leakage confirmed
│   └── ablation_results.csv
├── tests/
│   ├── conftest.py              # shared fixtures (mocked risk-free rate)
│   ├── test_metrics.py          # metrics module, incl. turnover
│   ├── test_hmm.py              # HMM module: BIC selection, convergence, label_states, transition matrices, model caching
│   ├── test_engine.py           # backtest engine, incl. save toggle and in-memory returns_df
│   ├── test_optimizers.py       # all four optimizers, incl. risk-contribution equality on both synthetic and correlated-asset fixtures
│   ├── test_switcher.py         # concentration guard
│   ├── test_data_cache.py       # cache signature validation for fetch_prices/fetch_risk_free
│   ├── test_process.py          # feature computation, config-driven windows
│   ├── test_stress_tests.py     # paired bootstrap correctness, VaR/CVaR ordering
│   ├── test_main.py             # regime-run and empirical transition matrix computation
│   ├── test_integration.py      # end-to-end pipeline regression
│   └── test_scripts.py          # sensitivity sweep and ablation integration (synthetic data), bootstrap rf, Viterbi vs forward-filter
├── visualization/
│   └── charts.py                # Equity curves, drawdown, regime overlay
└── main.py                      # Full pipeline entry point
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

# Run sensitivity sweep (VOL_WINDOW x CORR_WINDOW, parallelized)
python -m scripts.sensitivity_sweep

# Run frozen model out-of-sample evaluation (train <=2018, test 2019-2024)
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

**Why apply the Risk Parity position cap after normalization, not inside the optimization?** The log-barrier objective's natural solution scale is set by the covariance matrix's real magnitude, with no inherent relationship to summing to 1. A cap applied to that raw scale is capping an essentially arbitrary number - and empirically, against real market covariances, it caused every asset to saturate simultaneously, silently degenerating into equal weight regardless of the underlying risk structure. Normalizing first, then capping and redistributing any still-over-limit asset, is the only way the 60% limit actually means what it's supposed to: a real ceiling on the final portfolio, not a constraint on an intermediate number with no fixed scale.

**Why scope the concentration guard's fallback for Crash?** Bull/Bear/Sideways optimizers already span the full asset universe, so full-universe equal-weight is the natural fallback for them. Crash is different: it has a well-defined subset (the safe havens), and diluting a flight-to-safety allocation with equity exposure defeats the point of the regime existing.

**Why validate the data/model cache against a signature instead of just file existence?** A cache that's merely "present" isn't necessarily "correct for the current config" - checking only existence meant changing tickers, dates, `n_states`, or the feature set could silently return stale data or even crash on a shape mismatch.

**Why is walk-forward the default if the frozen model performs better out-of-sample?** Walk-forward is the correct methodology for live deployment - a real system would retrain as new data arrives. The frozen variant is a retrospective curiosity.

---

## Limitations

**Regime separation is weak in feature space.** PCA of the three features (SPY return, volatility, mean correlation) projected to 2D shows heavy overlap between Bull, Bear, and Sideways in the dense center.

**Expanding covariance window mixes regimes.** The optimizer uses `returns.loc[:date]` - an expanding window that grows from ~3,260 to ~4,780 days. Early folds use covariance estimates dominated by the 2008 crisis.

**Crash regime uses inverse-vol weighting over safe havens.** The Crash optimizer allocates inversely proportional to current volatility across IEF, TLT, and GLD. In 2022, all three sold off simultaneously, limiting diversification.

**The corrected Risk Parity optimizer trades some resilience to correlated selloffs for real risk balance.** This is the most consequential finding of this audit round, and it's reported here even though it's not flattering. A prior, buggy version of the Bear-regime optimizer was accidentally producing near-equal-weight allocations rather than true risk parity - and that accidental equal-weighting happened to be more robust to 2022's simultaneous bond/equity selloff than the corrected, volatility-driven allocation is. Held-out Sharpe dropped from 0.453 to 0.261 and the COVID+rates subperiod Sharpe dropped to near-zero (0.016) as a direct, verified consequence. This isn't an implementation error in the fixed version - it's a real property of the correct methodology. Risk parity allocates based on realized volatility, and has no way to anticipate a correlation regime change before it happens.

**Gaussian emission assumption violated.** Jarque-Bera tests reject normality for all four regimes (p≈0). Crash regime excess kurtosis is 6.11 - the most extreme of the four regimes and clearly fat-tailed relative to a Gaussian (Bull/Bear/Sideways sit closer to 1.8-2.2). A Student-t HMM would be more appropriate, particularly for the Crash regime.

**Sharpe outperformance not significant.** The paired block bootstrap p-value is 0.344 - the observed Sharpe difference (+0.126) is positive but the 95% CI straddles zero. This p-value is now computed identically by two independent code paths after an audit-driven fix corrected a formula error in one of them - both now report 0.344.

**Forward-filter performance cost.** Replacing Viterbi/smoothed posteriors with the causal forward filter costs roughly 0.02 Sharpe in a controlled comparison holding fitted models fixed - the expected, honest direction of effect, since smoothing benefits from hindsight a live system wouldn't have.

**Choppy Bull detection.** Median Bull run length is 5.0 days. BIC selecting n=2 on some folds collapses regimes, causing rapid label switching.

**COVID+rates weakness.** This is now the strategy's weakest subperiod by a wide margin (Sharpe 0.016). The 2020 crash was too fast for a causal HMM to detect in time, and the 2022 rate-driven bear hit bonds and equities simultaneously - see the Risk Parity note above for why this got worse, not better, after the optimizer was corrected.

**SPY circularity.** All three features are derived from SPY returns. The HMM detects regimes using the same instrument it indirectly trades, introducing circularity in the signal. Macro features (VIX, yield curve slope) were tested as orthogonal alternatives but did not generalize out-of-sample.

**Mean return as signal.** The Mean-Variance optimizer uses sample mean returns - near-zero signal-to-noise at daily frequency. Shrinkage toward a factor model or Black-Litterman prior would be more robust.

**State labeling heuristic.** `label_states()` ranks states by `mean_return - 0.5 x mean_vol` - a risk-adjusted score. The 0.5 weight is tuned heuristically, not data-driven. The two inputs are resolved by column name rather than position as of this audit round, which changes *which columns* feed the formula in edge cases (like the skew_kurt ablation variant), not the formula itself.

**Asset universe limited to 8 ETFs.** Transaction costs modeled at 2bps flat - not accounting for market impact at scale. A broader universe or factor-based selection could improve diversification.

**Markov assumption.** HMMs assume the current state depends only on the previous state. Markets exhibit momentum and mean-reversion at multiple horizons, violating this assumption.

**Position limits capped at 60% per asset.** All three optimizers - Mean-Variance, Risk Parity, and Minimum Variance - enforce `w <= 60%` per asset, though (see Audit Fixes above) *how* this is enforced for Risk Parity specifically required correcting during this audit round.

**Forward filter initialization uses training startprob.** In test folds starting mid-cycle, the true initial state distribution may differ. This is a minor bias that cannot be corrected without knowing the true regime.

**Time period bias.** The full backtest covers 2006-2024 - a period that includes the GFC, a decade-long bull market, COVID, and a rate shock. Results may not generalize to other macroeconomic regimes.

**Data is not perfectly reproducible session-to-session.** A fresh `yfinance` download on a different day can shift regime counts and Sharpe ratios slightly, most likely due to Yahoo Finance backfilling minor historical price corrections between sessions. All numbers reported here come from a single, uninterrupted pipeline run to ensure internal consistency, but they should be read as representative rather than exactly reproducible bit-for-bit on a different day.

---

Abdelkrim - Applied Mathematics & AI, PSL-Dauphine
https://github.com/AbdelkrimCode
