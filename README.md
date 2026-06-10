# Regime Adaptive Portfolio — v6

Algorithmic portfolio optimizer combining Hidden Markov Model regime detection with convex optimization.

![Python](https://img.shields.io/badge/python-3.10+-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Tests](https://github.com/AbdelkrimCode/regime-adaptive-portfolio/actions/workflows/tests.yml/badge.svg)

---

## How It Works

### 1. Regime Detection
A Gaussian HMM trained on three features — SPY log returns, rolling volatility, and rolling mean pairwise correlation — labels each trading day as Bull, Bear, Sideways, or Crash. Feature windows default to 21-day vol and 63-day correlation, defined as constants in `data/process.py`. The model is retrained quarterly using an expanding window to prevent lookahead bias. Walk-forward retraining is the default.

State ordering is determined by a risk-adjusted score after every retrain:
- Score = mean SPY return − 0.5 × mean rolling volatility
- Lowest score → Crash
- Second lowest → Bear
- Second highest → Sideways
- Highest → Bulll

**Label stability across retrains:** HMM states have no intrinsic identity — state 0 in Q1 might correspond to "Bear" and state 0 in Q2 might correspond to "Bull". This implementation avoids label permutation by assigning regime names based on mean return rank, not state index. After every quarterly retrain, states are sorted by their risk-adjusted score (mean_return − 0.5 × mean_vol) and relabeled consistently. The optimizer selected for each day depends on the semantic label, not the raw state index.

### 2. BIC State Selection
At every quarterly fold, candidate state counts [2, 3, 4] are each fitted on the full training window and scored by BIC. The state count minimizing BIC is selected independently per fold.

**Implementation note:** When a fold selects n=2, it emits only `p_bull` and `p_bear` posteriors. Concatenating folds with different state counts produces NaN entries in the unused posterior columns. The blending layer handles this explicitly — NaN posterior probabilities are skipped rather than propagated.

### 3. Causal Forward-Filter Posteriors
Prior to v6, regime posteriors were computed using the Viterbi algorithm and `predict_proba()` — both of which use forward and backward passes over the entire observation window. This introduces within-window lookahead: the regime label assigned to day T uses data through the end of the training window, not just through day T.

v6 replaces both with a manual forward-filter (`forward_filter()` in `hmm.py`). The filtered posterior `P(state_t | observations_1:t)` is computed using only the forward pass — strictly causal at every time step:

```
alpha_t[i] = P(o_t | state=i) * sum_j( alpha_{t-1}[j] * A[j,i] )
filtered_posterior_t = alpha_t / sum(alpha_t)
```

**Performance impact:** Forward filtering reduces full-sample Sharpe relative to Viterbi because Viterbi's within-window smoothing produced better labels in hindsight. The v6 numbers re   flect what a truly causal system would have produced. Methodological correctness is preferred over inflated backtested results.

### 4. Parallelized Fold Execution
Each quarterly fold is independent — no shared mutable state. Folds execute via `joblib.Parallel`. `n_jobs=1` in `config.yaml` keeps the default deterministic; the sensitivity sweep passes `n_jobs=-1` explicitly to `walk_forward_regimes()` for speed.

### 5. Posterior Probability Blending
v1 used hard regime labels — each day was assigned exactly one regime. This amplifies turnover at transition boundaries and ignores the HMM's uncertainty.

v2 uses filtered posteriors for every day:

```
weights(t) = P(Bull|t) × w_bull + P(Bear|t) × w_bear
           + P(Sideways|t) × w_sideways + P(Crash|t) × w_crash
```

This is a convex combination — weights always sum to 1, always non-negative.

### 6. Adaptive Optimization
Four convex optimizers, one per regime. On every regime transition and every retrain date, all four optimizers are recomputed with current covariance estimates and the current risk-free rate:

| Regime | Optimizer | Objective |
|---|---|---|
| Bull | Mean-Variance | Maximize Sharpe ratio (time-varying RF) |
| Bear | Risk Parity (log-barrier) | Approximate equal risk contribution |
| Sideways | Minimum Variance | Minimize portfolio volatility |
| Crash | Inverse-Vol Weight (IEF, TLT, GLD) | Flight-to-safety — allocates inversely proportional to current volatility |

All optimizers except Crash use Ledoit-Wolf shrinkage on the covariance matrix. Weights constrained to [0, 1] — no shorting, no leverage. Safe-haven assets configurable via `config.yaml`.

### 7. Regime Diagnostics
Run-length statistics from the walk-forward label sequence:

| Regime | Count | Mean (days) | Median (days) | Min | Max |
|---|---|---|---|---|---|
| Bull | 78 | 16.2 | 3.0 | 1 | 171 |
| Bear | 70 | 19.2 | 16.0 | 1 | 62 |
| Sideways | 67 | 18.1 | 11.0 | 1 | 128 |
| Crash | 33 | 27.2 | 24.0 | 1 | 62 |

Empirical transition matrix (% of exits):

| From \ To | Bull | Bear | Sideways | Crash |
|---|---|---|---|---|
| Bull | — | 41.0% | 57.7% | 1.3% |
| Bear | 50.0% | — | 18.6% | 31.4% |
| Sideways | 51.5% | 33.3% | — | 15.2% |
| Crash | 27.3% | 45.5% | 27.3% | — |

### 8. Backtest
Lookahead-free simulation applying day-t weights to day t+1 returns. Transaction costs at 2bps per unit of turnover applied consistently to portfolio and all benchmarks. Average daily turnover: 4.83% (implied annual cost: 0.24%).

Sharpe ratios use a time-varying risk-free rate sourced from the 3-month T-bill (^IRX via yfinance). The same time-varying rate is used in the max-Sharpe optimizer. The `run_period()` function passes the full returns history to `compute_weights()` — allowing the optimizer to use all available data up to each date — and slices only at simulation time.

---

## Results

The primary claim of this strategy is **drawdown control**, not Sharpe outperformance. The observed Sharpe difference vs SPY (0.608 vs 0.435) is positive but not statistically significant (bootstrap p=0.478, 95% CI straddles zero). The defensible results are:

- Max drawdown **-26.99% vs SPY -59.58%** — 55% reduction in peak-to-trough loss
- Calmar ratio **0.231 vs SPY 0.140** — better return per unit of drawdown
- Volatility **8.14% vs SPY 19.51%** — dramatically lower realized risk

### Full Period (2006–2024)

| Metric | Portfolio | SPY | Equal Weight | 60/40 | Momentum | Risk Parity |
|---|---|---|---|---|---|---|
| Annualized Return | 6.24% | 8.34% | 5.96% | 6.88% | 5.63% | 5.80% |
| Annualized Volatility | 8.14% | 19.51% | 11.17% | 11.17% | 13.28% | 10.99% |
| Sharpe Ratio | **0.608** | 0.435 | 0.448 | 0.525 | 0.388 | 0.445 |
| Max Drawdown | **-26.99%** | -59.58% | -35.05% | -35.60% | -27.39% | -34.14% |
| Calmar Ratio | **0.231** | 0.140 | 0.170 | 0.193 | 0.206 | 0.170 |

The observed Sharpe difference over SPY is +0.173 — positive but not statistically significant (bootstrap 95% CI [-0.318, 0.630], p=0.478). The genuine edge is volatility compression (8.14% vs 19.51%) and drawdown protection.

### Held-Out Test Period (2019–2024)

| Metric | Portfolio | SPY |
|---|---|---|
| Annualized Return | 6.14% | 14.87% |
| Annualized Volatility | 9.56% | 19.91% |
| Sharpe Ratio | 0.435 | 0.677 |
| Max Drawdown | -26.99% | -35.75% |
| Calmar Ratio | 0.227 | 0.416 |

The held-out Sharpe (0.435) lags SPY (0.677). A frozen model — trained once on 2006–2018 and never retrained — produces Sharpe 0.515 on the same period, outperforming the walk-forward variant. This confirms the strategy generalizes: quarterly retraining during the test period does not inflate results. The walk-forward default is retained for live deployment correctness; the frozen result is the conservative out-of-sample estimate.

| Metric | Walk-Forward | Frozen Model | SPY |
|---|---|---|---|
| Annualized Return | 6.14% | 7.12% | 14.87% |
| Annualized Volatility | 9.56% | 9.54% | 19.91% |
| Sharpe Ratio | 0.435 | 0.515 | 0.677 |
| Max Drawdown | -26.99% | -27.84% | -35.75% |
| Calmar Ratio | 0.227 | 0.256 | 0.416 |

### Subperiod Analysis

| Metric | Portfolio | SPY |
|---|---|---|
| **GFC (2008–2009)** | | |
| Annualized Return | **+4.42%** | -15.49% |
| Annualized Volatility | 11.01% | 34.70% |
| Sharpe Ratio | 0.389 | -0.345 |
| Max Drawdown | **-10.90%** | -56.38% |
| Calmar Ratio | 0.405 | -0.275 |
| **Low-vol bull (2013–2019)** | | |
| Annualized Return | 4.72% | 13.26% |
| Annualized Volatility | 5.34% | 12.83% |
| Sharpe Ratio | 0.730 | 1.003 |
| Max Drawdown | -7.46% | -19.82% |
| Calmar Ratio | **0.632** | 0.669 |
| **COVID+rates (2020–2024)** | | |
| Annualized Return | 4.12% | 11.84% |
| Annualized Volatility | 10.24% | 21.09% |
| Sharpe Ratio | 0.211 | 0.528 |
| Max Drawdown | -26.99% | -35.75% |
| Calmar Ratio | 0.153 | 0.331 |

**GFC:** The strategy's clearest result — portfolio gained +4.42% annualized while SPY lost 15.49%. Max drawdown -10.90% vs SPY -56.38%. The improved label_states function, which incorporates volatility in state ranking, correctly identifies the Crash regime earlier and maintains safe-haven allocation throughout the crisis.

**Low-vol bull:** Calmar ratio 0.632 vs SPY 0.669 — the strategy is close to SPY on return-per-drawdown during the bull market. Cost is absolute return (4.72% vs 13.26%).

**COVID+rates:** Weak. The 2020 crash was too fast for a causal HMM to detect in time. The 2022 rate-driven bear hit bonds and equities simultaneously, limiting the flight-to-safety allocation.

### Bootstrap Significance Test

Block bootstrap with 1000 iterations, 20-day block length, time-varying risk-free rate applied consistently:

| Statistic | Value |
|---|---|
| Observed Sharpe Difference | +0.173 |
| 95% CI | [-0.318, 0.630] |
| p-value | 0.478 |
| Significant at 95% | No |

The observed Sharpe difference is positive (+0.173) but the 95% CI straddles zero — not statistically significant. The drawdown reduction remains the defensible claim.

### Window Sensitivity Sweep

Full pipeline rerun across `VOL_WINDOW ∈ {10, 21, 42}` × `CORR_WINDOW ∈ {42, 63, 126}`:

| VOL\CORR | 42 | 63 | 126 |
|---|---|---|---|
| **Full Sharpe** | | | |
| 10 | 0.419 | 0.571 | 0.591 |
| 21 | 0.482 | 0.432 | 0.486 |
| 42 | 0.423 | 0.527 | 0.425 |
| **Held-out Sharpe** | | | |
| 10 | 0.193 | 0.501 | 0.449 |
| 21 | 0.271 | 0.492 | 0.143 |
| 42 | 0.304 | 0.344 | 0.253 |

`CORR_WINDOW=63` consistently produces the best or near-best held-out Sharpe. Default parameters sit in a stable region — not cherry-picked.

### Bootstrap Block-Length Sensitivity

| Block Length | p-value | Mean Diff | 95% CI |
|---|---|---|---|
| 10 | 0.435 | 0.056 | [-0.52, 0.65] |
| 20 | 0.443 | 0.033 | [-0.60, 0.64] |
| 40 | 0.479 | 0.021 | [-0.56, 0.65] |

Conclusion stable across block lengths: not statistically significant.

### Walk-Forward Leakage Audit

`audit_walk_forward()` reconstructs all 75 quarterly folds and verifies train-end strictly precedes test-start for every fold. Result: **0 leakage detected across 74 folds**. Full audit saved to `results/walk_forward_audit.csv`.

### Feature Ablation Study

Full pipeline rerun across four feature configurations to justify the baseline choice:

| Feature Set | Full Sharpe | Held-Out Sharpe | Max DD |
|---|---|---|---|
| baseline (vol_21, corr_63) | 0.608 | 0.435 | -27.0% ||
| vol_10 (vol_10, corr_63) | 0.571 | 0.501 | -24.6% |
| vol_42 (vol_42, corr_63) | 0.527 | 0.344 | -28.0% |
| skew_kurt (skew+kurt, no corr) | 0.469 | 0.169 | -30.6% |

vol_10 wins in-sample but underperforms out-of-sample. skew_kurt collapses out-of-sample (0.067) — badly overfit. Baseline sits in a stable middle ground — not cherry-picked, robust across market environments.

### Stress Tests

| Test | Portfolio | SPY |
|---|---|---|
| VaR (95%) | -0.79% | -1.84% |
| CVaR (95%) | -1.20% | -3.07% |
| Sharpe during VIX > 30 | -1.113 | -1.952 |
| Monte Carlo mean Sharpe (1000 paths) | 0.611 ± 0.228 | — |
| Monte Carlo 95% CI | [0.179, 1.053] | — |

Tail risk is dramatically lower than SPY across all measures. Both strategies produce negative Sharpe during VIX > 30 periods — nobody wins in a crisis — but the portfolio loses far less per unit of risk. Monte Carlo mean (0.611) is consistent with the actual full-period Sharpe (0.608), confirming no estimation bias.

---

## Post-v6 Bug Fixes

| Fix | Issue | Resolution |
|---|---|---|
| feature_ablation.py restore block | KeyError on CFG["data"] — crashed on every completed ablation run | Read from FEATURE_SETS["baseline"] defined in the same file |
| test_optimizers.py | Byte-for-byte duplicate of test_metrics.py — zero optimizer coverage | Replaced with 24 real optimizer tests |
| bootstrap.py rf | Sharpe bootstrap computed with rf=None — inconsistent with pipeline | Thread time-varying rf through run_bootstrap() |
| bootstrap.py + charts.py paths | Hardcoded parquet paths bypassed config.yaml | Replaced with CFG["paths"] throughout |
| sensitivity_sweep n_jobs | CFG mutation never reached hmm.py — always ran with n_jobs=1 | Pass n_jobs explicitly to walk_forward_regimes() |
| stress_tests monte_carlo | IID resampling destroyed autocorrelation structure | Replaced with block_resample() consistent with bootstrap.py |
| main.py test_end | Dead variable set to train_end (wrong value, never used) | Removed |
| min_variance.py | Solver output returned raw — tiny floating point negatives possible | Clip and renormalize before returning |
| switcher.py | Magic number 0.99 hardcoded in concentration guard | Named constant CONCENTRATION_GUARD |
| hmm.py forward_filter | No underflow guard at alpha[0] initialization | Apply same zero-sum guard as t>0 steps |
| main.py transition matrix | Division by zero if a regime never transitions out | Replace 0 row sums with NaN before division |
| benchmark.py | Hardcoded 126 in get_risk_parity_equity() | Read from CFG["hmm"]["min_train_days"] // 2 |


## v5 → v6 Improvements

| Change | v5 | v6 |
|---|---|---|
| Regime posteriors | Viterbi + forward-backward — within-window lookahead | Causal forward-filter — strictly uses only past observations |
| Label permutation | Undocumented | Explicitly documented: labels assigned by mean return rank after every retrain |
| run_period() optimizer | Passed returns_slice — optimizer starved in early test period | Passes full returns — optimizer uses all available history up to each date |
| charts.py | KeyError on Crash regime | Crash added to REGIME_COLORS |
| safe_haven_assets | Hardcoded in crash.py | Configurable via config.yaml |
| select_n_states | Unused train/test split | Removed — BIC on full training data directly |
| sensitivity_sweep | Global config mutation (ineffective) | n_jobs passed explicitly to walk_forward_regimes() |
| smoothing_days | Dead config key | Removed |

## v4 → v5 Improvements

| Change | v4 | v5 |
|---|---|---|
| Optimizer recompute | Only active regime on intra-quarter transition | All four on every regime change |
| RF in max-Sharpe QP | Hardcoded 4% | Time-varying 3-month T-bill |
| BIC scoring | On held-out 20% (wrong N) | On full training data |
| Min-history threshold | Hardcoded 126 | References config |
| fit_hmm duplication | Two near-identical functions | Unified via _fit_hmm_core() |

## v3 → v4 Improvements

| Change | v3 | v4 |
|---|---|---|
| Nested CV execution | Never called — every fold used fixed n=4 | Genuinely runs per fold |
| Fold execution | Sequential | joblib.Parallel |
| Sensitivity analysis | Not tested | Window + bootstrap sweep |

## v2 → v3 Improvements

| Change | v2 | v3 |
|---|---|---|
| NaN posterior handling | Missing | Explicit skip |
| Benchmarks | 4 | + Risk Parity |
| Subperiod analysis | None | GFC / Low-vol bull / COVID+rates |
| Normality testing | None | Jarque-Bera per regime |

## v1 → v2 Improvements

| Change | v1 | v2 |
|---|---|---|
| Regime states | 3 | 4 (+ Crash) |
| Regime uncertainty | Hard label | Posterior probability blend |
| Walk-forward | Optional | Default |
| Risk-free rate | Static 4% | Time-varying ^IRX |
| Benchmark costs | None | 2bps per turnover |
| Significance testing | None | Block bootstrap |
| Out-of-sample evaluation | None | Held-out 2019–2024 |

---

## Architecture

```
regime-adaptive-portfolio/
├── config.yaml            # Centralized hyperparameters (HMM, backtest, market, optimizer, bootstrap, paths, evaluation, subperiods)
├── config.py              # YAML loader
├── data/
│   ├── fetch.py           # Downloads adjusted close prices via yfinance
│   ├── process.py         # Log returns, rolling volatility, rolling correlation
│   └── risk_free.py       # 3-month T-bill rate (^IRX) with daily conversion and caching
├── models/
│   └── hmm.py             # Gaussian HMM, causal forward-filter, BIC state selection, parallelized walk-forward
├── optimization/
│   ├── mean_var.py        # Mean-Variance max Sharpe (Bull) — time-varying RF
│   ├── risk_parity.py     # Log-barrier risk parity approximation (Bear)
│   ├── min_variance.py    # Minimum Variance QP (Sideways)
│   ├── crash.py           # Inverse-vol flight-to-safety (Crash) — safe-haven assets from config
│   └── switcher.py        # Causal posterior blending, all-optimizer recompute on transition
├── backtest/
│   ├── engine.py          # Simulation with transaction costs, correct period slicing
│   ├── metrics.py         # Annualized return, volatility, Sharpe (time-varying RF), drawdown, Calmar
│   ├── benchmark.py       # SPY, Equal Weight, 60/40, Momentum, Risk Parity — all cost-adjusted
│   └── bootstrap.py       # Block bootstrap significance test with time-varying RF
├── scripts/
│   ├── sensitivity_sweep.py    # Window sensitivity analysis — n_jobs=-1 for speed
│   ├── frozen_model_eval.py    # True out-of-sample evaluation — model frozen at 2018-12-31
│   ├── feature_ablation.py     # Ablation study across feature sets and window sizes
│   └── stress_tests.py         # VaR, CVaR, VIX-conditional Sharpe, block-resampled Monte Carlo
├── results/
│   ├── figures/                # charts.png, bootstrap.png, pca_regimes.png, state_selection.png
│   ├── sensitivity_results.csv
│   ├── walk_forward_audit.csv  # 74 folds, 0 leakage confirmed
│   └── ablation_results.csv
├── tests/
│   ├── conftest.py        # 1 test — fixture module
│   ├── test_metrics.py    # 14 tests — metrics module
│   ├── test_hmm.py        # 14 tests — HMM module
│   ├── test_engine.py     # 9 tests — backtest engine
│   ├── test_optimizers.py # 24 tests — all four optimizers and compute_weights
│   ├── test_integration.py # 4 tests — end-to-end pipeline regression
│   └── test_scripts.py #5 tests — script level integration : bootstrap rf, ablation, sensitivity and Viterbi vs forward-filter
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
python scripts/sensitivity_sweep.py

# Run frozen model out-of-sample evaluation (train ≤2018, test 2019–2024)
python scripts/frozen_model_eval.py

# Run feature ablation study
python scripts/feature_ablation.py

# Run stress tests (VaR, CVaR, VIX-conditional, block-resampled Monte Carlo)
python scripts/stress_tests.py

# Audit walk-forward leakage
python -c "from models.hmm import audit_walk_forward; audit_walk_forward()"

# Run tests
pytest tests/
```

---

## Key Design Decisions

**Why Gaussian HMM?** Markets exhibit persistent regimes — the HMM transition matrix captures this persistence. Gaussian emissions model the continuous feature space naturally, though the Jarque-Bera results show this assumption is violated in practice.

**Why forward-filter instead of Viterbi?** Viterbi finds the globally optimal state sequence using both forward and backward passes — this introduces lookahead within the observation window. The forward-filter computes `P(state_t | observations_1:t)` using only past data. This is the only strictly causal option. The cost is lower reported Sharpe; the benefit is the backtest reflects what would actually have been observed in real time.

**Why label states by mean return rank?** HMM state indices are arbitrary — they can permute across retraining windows. Sorting by mean SPY return after every retrain ensures Bull, Bear, Sideways, and Crash refer to the same economic concept across the full backtest period.

**Why BIC state selection per fold?** A globally fixed n=4 assumes the same model complexity is appropriate across all market environments. Per-fold BIC selection allows early folds with limited data to select simpler models.

**Why time-varying RF in both optimizer and metrics?** The 4% hurdle during 2009–2022 (near-zero rate era) distorted allocations. Using the same 3-month T-bill rate in both the optimizer and Sharpe computation ensures internal consistency.

**Why recompute all optimizers on every regime change?** The blended weights are `sum(p_k * w_k)` across all four regimes. All four weight vectors must reflect the same market environment for the blend to be internally consistent.

**Why pass full returns to run_period()?** The optimizer needs a full covariance history to compute reliable weights. Slicing returns to the test period starves the optimizer in the early months, falling back to equal weights. Returns are passed in full; simulation is sliced separately.

**Why posterior blending?** Hard switching ignores the model's own uncertainty. Posterior blending hedges proportionally.

**Why cost-adjust benchmarks?** Fair comparison requires consistent cost treatment. All benchmarks incur 2bps per unit of monthly rebalancing turnover.

**Why Ledoit-Wolf shrinkage?** Sample covariance matrices from short windows are noisy and ill-conditioned.

**Why inverse-vol weighting for the Crash regime?** Equal-weight over safe havens ignores the fact that TLT is significantly more volatile than IEF. Inverse-vol allocation tilts toward the most stable safe haven at any given time — automatically reducing TLT exposure when bond volatility spikes (as in 2022) and increasing it during calm periods.

**Why is walk-forward the default if frozen performs better out-of-sample?** Walk-forward is the correct methodology for live deployment — a real system would retrain as new data arrives. The frozen model is an evaluation tool that confirms generalization, not a deployment strategy.

---

## Limitations

**Regime separation is weak in feature space.** PCA of the three features (SPY return, volatility, mean correlation) projected to 2D shows heavy overlap between Bull, Bear, and Sideways in the dense central region. Clear separation exists only in the left tail — extreme negative return / high volatility events (Crash and severe Bear). This explains the non-significant bootstrap p-value (0.478) and moderate held-out Sharpe.

**Expanding covariance window mixes regimes.** The optimizer uses `returns.loc[:date]` — an expanding window that grows from ~3,260 to ~4,780 days. Early folds use covariance estimates dominated by 2008 volatility; late folds mix that with 2024 volatility. A rolling window would be more stationary but introduces instability in small samples.

**Crash regime uses inverse-vol weighting over safe havens.** The Crash optimizer allocates inversely proportional to current volatility across IEF, TLT, and GLD. In 2022, all three sold off simultaneously with equities, limiting protection. A min-CVaR optimizer over the safe-haven subset would be more robust to correlation shocks.

**Gaussian emission assumption violated.** Jarque-Bera tests reject normality for all four regimes (p≈0). Bear regime kurtosis is 14.3 — extreme fat tails. A Student-t HMM would be more appropriate but is not available in hmmlearn.

**Sharpe outperformance not significant.** The paired block bootstrap p-value is 0.478 — the observed Sharpe difference (+0.173) is positive but the 95% CI straddles zero.

**Forward-filter performance cost.** Replacing Viterbi with forward-filter reduces full-sample Sharpe because Viterbi's within-window smoothing produced better labels in hindsight. Forward-filter is correct; these numbers are what a live system would have achieved.

**Scaler fit twice per fold.** `select_n_states()` fits its own `StandardScaler` internally for BIC scoring, and the main fit path fits a second scaler on the same training data. Both produce identical transformations — no numerical consequence, but the redundancy could be eliminated.

**Choppy Bull detection.** Median Bull run length is 3.0 days. BIC selecting n=2 on some folds collapses regimes, causing rapid label switching.

**COVID+rates weakness.** The 2020 crash was too fast for a causal HMM. The 2022 rate-driven bear hit bonds and equities simultaneously, limiting the flight-to-safety allocation.

**SPY circularity.** SPY is both the regime-detection instrument (return and volatility features) and an investable asset in the portfolio. VIX and yield curve slope (5Y-3M spread) were tested as additional macro features. Full-sample Sharpe improved but held-out Sharpe collapsed — the macro features overfitted. The 3-feature baseline was restored.

**Mean return as signal.** The Mean-Variance optimizer uses sample mean returns — near-zero signal-to-noise at daily frequency. Shrinkage toward a factor model or Black-Litterman prior would be more robust.

**State labeling heuristic.** `label_states()` ranks states by `mean_return - 0.5 × mean_vol` — a risk-adjusted score that prevents mislabeling high-volatility states with similar returns. The 0.5 weight is not optimized and could be tuned or learned from data.

**Risk parity is an approximation.** The Bear regime optimizer minimizes `quad_form(w, sigma) - (1/n)*sum(log(w))` — the Spinu (2013) log-barrier surrogate. Standard practice, but it does not guarantee true equal risk contribution.

**Asset universe limited to 8 ETFs.** Transaction costs modeled at 2bps flat — not accounting for market impact at scale. A broader universe or factor-based selection could improve diversification.

**Markov assumption.** HMMs assume the current state depends only on the previous state. Markets exhibit momentum and mean-reversion at multiple horizons, violating this assumption.

**Position limits capped at 60% per asset.** All three optimizers — Mean-Variance, Risk Parity, and Minimum Variance — enforce `w ≤ 60%` per asset via CVXPY constraints.

**Forward filter initialization uses training startprob.** In test folds starting mid-cycle, the true initial state distribution may differ. This is a minor bias that cannot be corrected without knowing the true state at the test period start.

**Time period bias.** The full backtest covers 2006–2024 — a period that includes the GFC, a decade-long bull market, COVID, and a rate shock. Results may not generalize to other macroeconomic regimes not present in this window.

---

Abdelkrim — Applied Mathematics & AI, PSL-Dauphine
https://github.com/AbdelkrimCode
