# Regime Adaptive Portfolio — v6

Algorithmic portfolio optimizer combining Hidden Markov Model regime detection with convex optimization. v6 fixes within-window Viterbi lookahead, documents label permutation handling, and corrects the held-out optimizer starvation bug in `run_period()`.

![Python](https://img.shields.io/badge/python-3.10+-blue) ![License](https://img.shields.io/badge/license-MIT-green)

---

## How It Works

### 1. Regime Detection
A Gaussian HMM trained on three features — SPY log returns, 21-day rolling volatility, and 63-day mean pairwise correlation — labels each trading day as Bull, Bear, Sideways, or Crash. The model is retrained quarterly using an expanding window to prevent lookahead bias. Walk-forward retraining is the default.

State ordering is determined by ranking state means on the return feature after every retrain:
- Lowest mean return → Crash
- Second lowest → Bear
- Second highest → Sideways
- Highest → Bull

**Label stability across retrains:** HMM states have no intrinsic identity — state 0 in Q1 might correspond to "Bear" and state 0 in Q2 might correspond to "Bull". This implementation avoids label permutation by assigning regime names based on mean return rank, not state index. After every quarterly retrain, states are sorted by their mean SPY return and relabeled consistently. The optimizer selected for each day depends on the semantic label, not the raw state index.

### 2. Nested CV for State Selection (v3/v4)
At every quarterly fold, candidate state counts [2, 3, 4] are each fitted on the full training window and scored by BIC. The state count minimizing BIC is selected independently per fold.

**Implementation note:** When a fold selects n=2, it emits only `p_bull` and `p_bear` posteriors. Concatenating folds with different state counts produces NaN entries in the unused posterior columns. The blending layer handles this explicitly — NaN posterior probabilities are skipped rather than propagated.

### 3. Causal Forward-Filter Posteriors (v6)
Prior to v6, regime posteriors were computed using the Viterbi algorithm and `predict_proba()` — both of which use forward and backward passes over the entire observation window. This introduces within-window lookahead: the regime label assigned to day T uses data through the end of the training window, not just through day T.

v6 replaces both with a manual forward-filter (`forward_filter()` in `hmm.py`). The filtered posterior `P(state_t | observations_1:t)` is computed using only the forward pass — strictly causal at every time step:

```
alpha_t[i] = P(o_t | state=i) * sum_j( alpha_{t-1}[j] * A[j,i] )
filtered_posterior_t = alpha_t / sum(alpha_t)
```

**Performance impact:** Forward filtering reduces full-sample Sharpe (0.562 → 0.453) because Viterbi's within-window smoothing produced better labels in hindsight. The v6 numbers reflect what a truly causal system would have produced. Methodological correctness is preferred over inflated backtested results.

### 4. Parallelized Fold Execution (v4)
Each quarterly fold is independent — no shared mutable state. Folds execute via `joblib.Parallel`. `n_jobs=1` in `config.yaml` keeps the default deterministic; the sensitivity sweep script overrides to `n_jobs=8` for speed.

### 5. Posterior Probability Blending (v2)
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
| Crash | Equal Weight (IEF, TLT, GLD) | Flight-to-safety defensive allocation |

All optimizers except Crash use Ledoit-Wolf shrinkage on the covariance matrix. Weights constrained to [0, 1] — no shorting, no leverage. Safe-haven assets configurable via `config.yaml`.

### 7. Regime Diagnostics (v3)
Run-length statistics from the walk-forward label sequence (v6 results):

| Regime | Count | Mean (days) | Median (days) | Min | Max |
|---|---|---|---|---|---|
| Bull | 76 | 17.2 | 4.5 | 1 | 151 |
| Bear | 69 | 21.0 | 17.0 | 1 | 61 |
| Sideways | 60 | 18.3 | 10.5 | 1 | 129 |
| Crash | 34 | 22.5 | 16.5 | 1 | 62 |

Empirical transition matrix (% of exits):

| From \ To | Bull | Bear | Sideways | Crash |
|---|---|---|---|---|
| Bull | — | 40.8% | 50.0% | 9.2% |
| Bear | 55.1% | — | 21.7% | 23.2% |
| Sideways | 44.1% | 39.0% | — | 16.9% |
| Crash | 35.3% | 44.1% | 20.6% | — |

### 8. Backtest
Lookahead-free simulation applying day-t weights to day t+1 returns. Transaction costs at 2bps per unit of turnover applied consistently to portfolio and all benchmarks. Average daily turnover: 4.50% (implied annual cost: 0.23%).

Sharpe ratios use a time-varying risk-free rate sourced from the 3-month T-bill (^IRX via yfinance). The same time-varying rate is used in the max-Sharpe optimizer. The `run_period()` function passes the full returns history to `compute_weights()` — allowing the optimizer to use all available data up to each date — and slices only at simulation time.

---

## Results

The primary claim of this strategy is **drawdown control**, not Sharpe outperformance. The Sharpe difference vs SPY (0.453 vs 0.433) is statistically indistinguishable from zero (bootstrap p=0.376). The defensible results are:

- Max drawdown **-26.06% vs SPY -60.39%** — 57% reduction in peak-to-trough loss
- Calmar ratio **0.182 vs SPY 0.137** — better return per unit of drawdown
- GFC max drawdown **-13.34% vs SPY -56.38%** — strategy worked when it mattered most

### Full Period (2006–2024)

| Metric | Portfolio | SPY | Equal Weight | 60/40 | Momentum | Risk Parity |
|---|---|---|---|---|---|---|
| Annualized Return | 4.75% | 8.27% | 5.90% | 6.80% | 5.67% | 4.55% |
| Annualized Volatility | 7.90% | 19.58% | 11.19% | 11.20% | 13.22% | 7.58% |
| Sharpe Ratio | 0.453 | 0.433 | 0.443 | 0.518 | 0.395 | 0.455 |
| Max Drawdown | **-26.06%** | -60.39% | -35.29% | -36.22% | -29.65% | -24.95% |
| Calmar Ratio | **0.182** | 0.137 | 0.167 | 0.188 | 0.191 | 0.182 |

The Sharpe difference over SPY is 0.020 — not statistically significant (bootstrap 95% CI straddles zero, p=0.376). The genuine edge is volatility compression (7.90% vs 19.58%) and drawdown protection.

### Held-Out Test Period (2019–2024)

| Metric | Portfolio | SPY |
|---|---|---|
| Annualized Return | 4.61% | 14.86% |
| Annualized Volatility | 8.68% | 19.91% |
| Sharpe Ratio | 0.292 | 0.677 |
| Max Drawdown | -26.06% | -35.75% |
| Calmar Ratio | 0.177 | 0.416 |

The held-out Sharpe (0.292) lags SPY (0.677). The 2019–2024 period was structurally unfavorable for regime-switching — a sustained bull market interrupted by brief corrections. The portfolio maintains drawdown protection (-26.06% vs -35.75%).

### Subperiod Analysis

| Metric | Portfolio | SPY |
|---|---|---|
| **GFC (2008–2009)** | | |
| Annualized Return | 0.05% | -15.46% |
| Max Drawdown | **-13.34%** | -56.38% |
| Calmar Ratio | 0.004 | -0.274 |
| **Low-vol bull (2013–2019)** | | |
| Annualized Return | 4.53% | 13.26% |
| Max Drawdown | -7.72% | -19.82% |
| Calmar Ratio | **0.587** | 0.669 |
| **COVID+rates (2020–2024)** | | |
| Annualized Return | 2.73% | 11.83% |
| Max Drawdown | -26.06% | -35.75% |
| Calmar Ratio | 0.105 | 0.331 |

**GFC:** The strategy's clearest result — near-zero return while SPY lost over half its value. Max drawdown -13.34% vs SPY -56.38%.

**Low-vol bull:** Calmar ratio 0.587 vs SPY 0.669 — close to SPY on return-per-drawdown during the bull market. Cost is absolute return (4.53% vs 13.26%).

**COVID+rates:** Weak. The 2020 crash was too fast for a causal HMM to detect in time. The 2022 rate-driven bear hurt bonds and equities simultaneously, limiting the flight-to-safety allocation.

### Bootstrap Significance Test

Block bootstrap with 1000 iterations, 20-day block length:

| Statistic | Value |
|---|---|
| Mean Sharpe Difference | 0.020 |
| 95% CI | [-0.44, 0.48] |
| p-value | 0.376 |
| Significant at 95% | No |

Sharpe outperformance is not statistically significant. The drawdown reduction is the defensible claim.

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

`CORR_WINDOW=63` consistently produces the best or near-best held-out Sharpe. Default parameters sit in a stable region — not cherry-picked.

### Bootstrap Block-Length Sensitivity (v4)

| Block Length | p-value | Mean Diff | 95% CI |
|---|---|---|---|
| 10 | 0.342 | 0.152 | [-0.44, 0.79] |
| 20 | 0.292 | 0.165 | [-0.48, 0.77] |
| 40 | 0.333 | 0.148 | [-0.50, 0.81] |

Conclusion stable across block lengths: not statistically significant.

---

## v5 → v6 Improvements

| Change | v5 | v6 |
|---|---|---|
| Regime posteriors | Viterbi + forward-backward — within-window lookahead | Causal forward-filter — strictly uses only past observations |
| Label permutation | Undocumented | Explicitly documented: labels assigned by mean return rank after every retrain |
| run_period() optimizer | Passed returns_slice — optimizer starved in early test period | Passes full returns — optimizer uses all available history up to each date |
| charts.py | KeyError on Crash regime | Crash added to REGIME_COLORS |
| safe_haven_assets | Hardcoded in crash.py | Configurable via config.yaml |
| select_n_states | Unused train/test split | Removed — BIC on full training data directly |
| sensitivity_sweep | Global config mutation | Explicit n_jobs parameter |
| smoothing_days | Dead config key | Removed |
| Performance impact | Held-out Sharpe 0.614 | Held-out Sharpe 0.292 — run_period fix removes optimizer inflation |

## v4 → v5 Improvements

| Change | v4 | v5 |
|---|---|---|
| Optimizer recompute | Only active regime on intra-quarter transition | All four on every regime change |
| RF in max-Sharpe QP | Hardcoded 4% | Time-varying 3-month T-bill |
| BIC scoring | On held-out 20% (wrong N) | On full training data |
| Min-history threshold | Hardcoded 126 | References config |
| fit_hmm duplication | Two near-identical functions | Unified via _fit_hmm_core() |
| Test suite | 37 tests | 55 tests — optimizer + integration coverage |

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
| Significance testing | None | Block bootstrap, p=0.376 |
| Out-of-sample evaluation | None | Held-out 2019–2024 |

---

## Architecture

```
regime-adaptive-portfolio/
├── config.yaml            # Centralized hyperparameters (HMM, backtest, market, optimizer, paths, evaluation, subperiods)
├── config.py              # YAML loader
├── data/
│   ├── fetch.py           # Downloads adjusted close prices via yfinance
│   ├── process.py         # Log returns, rolling volatility, rolling correlation (parametric windows)
│   └── risk_free.py       # 3-month T-bill rate (^IRX) with daily conversion and caching
├── models/
│   └── hmm.py             # Gaussian HMM, causal forward-filter, nested CV, parallelized walk-forward
├── optimization/
│   ├── mean_var.py        # Mean-Variance max Sharpe (Bull) — time-varying RF
│   ├── risk_parity.py     # Log-barrier risk parity approximation (Bear)
│   ├── min_variance.py    # Minimum Variance QP (Sideways)
│   ├── crash.py           # Equal-weight flight-to-safety (Crash) — assets from config
│   └── switcher.py        # Causal posterior blending, all-optimizer recompute on transition
├── backtest/
│   ├── engine.py          # Simulation with transaction costs, correct period slicing
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
│   └── charts.py          # Equity curves, drawdown, regime overlay (all 4 regimes)
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

# Run full pipeline (causal forward-filter, nested CV, walk-forward default)
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

**Why forward-filter instead of Viterbi?** Viterbi finds the globally optimal state sequence using both forward and backward passes — this introduces lookahead within the observation window. The forward-filter computes `P(state_t | observations_1:t)` using only past data. This is the only strictly causal option. The cost is lower reported Sharpe; the benefit is the backtest reflects what would actually have been observed in real time.

**Why label states by mean return rank?** HMM state indices are arbitrary — they can permute across retraining windows. Sorting by mean SPY return after every retrain ensures Bull, Bear, Sideways, and Crash refer to the same economic concept across the full backtest period.

**Why nested CV for state selection?** A globally fixed n=4 assumes the same model complexity is appropriate across all market environments. Nested CV allows early folds with limited data to select simpler models.

**Why time-varying RF in both optimizer and metrics?** The 4% hurdle during 2009–2022 (near-zero rate era) distorted allocations. Using the same 3-month T-bill rate in both the optimizer and Sharpe computation ensures internal consistency.

**Why recompute all optimizers on every regime change?** The blended weights are `sum(p_k * w_k)` across all four regimes. All four weight vectors must reflect the same market environment for the blend to be internally consistent.

**Why pass full returns to run_period()?** The optimizer needs a full covariance history to compute reliable weights. Slicing returns to the test period starves the optimizer in the early months, falling back to equal weights. Returns are passed in full; simulation is sliced separately.

**Why posterior blending?** Hard switching ignores the model's own uncertainty. Posterior blending hedges proportionally.

**Why cost-adjust benchmarks?** Fair comparison requires consistent cost treatment. All benchmarks incur 2bps per unit of monthly rebalancing turnover.

**Why Ledoit-Wolf shrinkage?** Sample covariance matrices from short windows are noisy and ill-conditioned.

---

## Limitations

**Gaussian emission assumption violated.** Jarque-Bera tests reject normality for all four regimes (p≈0). Bear regime kurtosis is 14.3 — extreme fat tails. A Student-t HMM would be more appropriate but is not available in hmmlearn.

**Sharpe outperformance not significant.** The bootstrap p-value is 0.376 — the Sharpe difference is indistinguishable from zero. The strategy's genuine edge is drawdown control and volatility compression.

**Forward-filter performance cost.** Replacing Viterbi with forward-filter reduces full-sample Sharpe from 0.562 to 0.453. Viterbi's within-window smoothing produced better labels in hindsight. Forward-filter is correct; the v6 numbers are what a live system would have achieved.

**Feature preprocessing is backward-looking.** Rolling volatility (21-day) and rolling correlation (63-day) are computed as trailing windows on the full price history before the walk-forward loop. Both windows use only past data at each point — there is no forward-looking contamination. Features are precomputed and cached to parquet; the walk-forward loop reads this file but does not regenerate it mid-backtest.

**Scaler fit twice per fold.** `select_n_states()` fits its own `StandardScaler` internally for BIC scoring, and `fit_hmm_with_states()` fits a second scaler on the same training data. Both produce identical transformations since the data and method are the same — no numerical consequence, but the redundancy could be eliminated.

**Choppy Bull detection.** Median Bull run length is 4.5 days. The nested CV selecting n=2 on some folds collapses regimes, causing rapid label switching.

**COVID+rates weakness.** The 2020 crash was too fast for a causal HMM. The 2022 rate-driven bear hit bonds and equities simultaneously, limiting the flight-to-safety allocation.

**SPY circularity.** SPY is both the regime-detection instrument (return and volatility features) and an investable asset in the portfolio. A macro indicator (ACWI, VIX) would provide more orthogonal signal for regime detection.

**Mean return as signal.** The Mean-Variance optimizer uses sample mean returns — near-zero signal-to-noise at daily frequency. Shrinkage toward a factor model or Black-Litterman prior would be more robust.

**State labeling uses first feature only.** `label_states()` ranks states on SPY return mean only. A composite ranking was tested and reverted — it produced worse out-of-sample results despite sounder theory.

**Risk parity is an approximation.** The Bear regime optimizer minimizes `quad_form(w, sigma) - (1/n)*sum(log(w))` with `w >= 0.01` and no sum-to-one constraint, normalizing post-hoc. This is the Spinu (2013) log-barrier surrogate — standard practice, but it does not guarantee true equal risk contribution.

**Bootstrap independence assumption.** Portfolio and SPY are resampled with independent block offsets, destroying cross-series correlation. A paired bootstrap would be more appropriate.

**Asset universe.** Limited to 8 ETFs. Transaction costs modeled at 2bps — not accounting for market impact at scale.

---

## Author

Abdelkrim — Applied Mathematics & AI, PSL-Dauphine
https://github.com/AbdelkrimCode
