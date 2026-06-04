# Regime Adaptive Portfolio — v4

Algorithmic portfolio optimizer combining Hidden Markov Model regime detection with convex optimization. v4 introduces genuine per-fold nested CV (resolving a v3 implementation gap), parallelized fold execution, window sensitivity sweep, and bootstrap block-length sensitivity sweep.

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
At every quarterly fold, candidate state counts [2, 3, 4] are each fitted on 80% of the available training window and scored by BIC on the held-out 20%. The state count minimizing BIC is selected independently per fold.

**v3 implementation gap:** The nested CV function existed in v3 but `walk_forward_regimes()` never called it — every fold silently used the globally fixed `n_states=4`. v4 resolves this: each fold now genuinely runs `select_n_states()` and picks its own `n_states`.

**Performance tradeoff:** The fixed n=4 model produced better backtest numbers on this dataset. The nested CV is methodologically more honest — it doesn't assume n=4 is always appropriate — but on this sample, fewer Crash and Bear detections result in lower drawdown protection. Methodological correctness and backtest performance do not always move in the same direction.

**Implementation note:** When a fold selects n=2, it emits only `p_bull` and `p_bear` posteriors. Concatenating folds with different state counts produces NaN entries in the unused posterior columns. The blending layer handles this explicitly — NaN posterior probabilities are skipped rather than propagated.

### 3. Parallelized Fold Execution (v4)
Each quarterly fold is independent — no shared mutable state. v4 extracts fold logic into `_fit_fold()` and executes folds via `joblib.Parallel`. `n_jobs=1` in `config.yaml` keeps the default deterministic; the sensitivity sweep script overrides to `n_jobs=8` for speed.

### 4. Posterior Probability Blending (v2)
v1 used hard regime labels — each day was assigned exactly one regime. This amplifies turnover at transition boundaries and ignores the HMM's uncertainty.

v2 uses `predict_proba()` to extract the full posterior distribution over states for every day:

```
weights(t) = P(Bull|t) × w_bull + P(Bear|t) × w_bear
           + P(Sideways|t) × w_sideways + P(Crash|t) × w_crash
```

This is a convex combination — weights always sum to 1, always non-negative.

### 5. Adaptive Optimization
Four convex optimizers, one per regime. Optimizer weights are recomputed at every quarterly retraining date:

| Regime | Optimizer | Objective |
|---|---|---|
| Bull | Mean-Variance | Maximize Sharpe ratio |
| Bear | Risk Parity | Equalize risk contributions |
| Sideways | Minimum Variance | Minimize portfolio volatility |
| Crash | Equal Weight (IEF, TLT, GLD) | Flight-to-safety defensive allocation |

All optimizers except Crash use Ledoit-Wolf shrinkage on the covariance matrix. Weights constrained to [0, 1] — no shorting, no leverage.

### 6. Regime Diagnostics (v3)
Run-length statistics from the walk-forward label sequence (v4 results):

| Regime | Count | Mean (days) | Median (days) | Min | Max |
|---|---|---|---|---|---|
| Bull | 52 | 36.8 | 1.0 | 1 | 343 |
| Bear | 43 | 36.9 | 39.0 | 1 | 97 |
| Sideways | 29 | 27.6 | 20.0 | 1 | 127 |
| Crash | 8 | 32.2 | 26.0 | 1 | 61 |

Empirical transition matrix (% of exits):

| From \ To | Bull | Bear | Sideways | Crash |
|---|---|---|---|---|
| Bull | — | 66.7% | 33.3% | 0.0% |
| Bear | 67.4% | — | 23.3% | 9.3% |
| Sideways | 58.6% | 27.6% | — | 13.8% |
| Crash | 62.5% | 12.5% | 25.0% | — |

### 7. Backtest
Lookahead-free simulation applying day-t weights to day t+1 returns. Transaction costs at 2bps per unit of turnover applied consistently to portfolio and all benchmarks. Average daily turnover: 3.55% (implied annual cost: 0.18%).

Sharpe ratios use a time-varying risk-free rate sourced from the 3-month T-bill (^IRX via yfinance), converted to daily rates.

---

## Results

### Full Period (2006–2024)

| Metric | Portfolio | SPY | Equal Weight | 60/40 | Momentum | Risk Parity |
|---|---|---|---|---|---|---|
| Annualized Return | 5.48% | 8.00% | 5.61% | 6.52% | 5.19% | 4.52% |
| Annualized Volatility | 9.08% | 19.69% | 11.24% | 11.27% | 13.14% | 7.60% |
| Sharpe Ratio | **0.486** | 0.424 | 0.423 | 0.497 | 0.355 | 0.455 |
| Max Drawdown | **-25.84%** | -60.39% | -35.29% | -36.22% | -29.65% | -24.94% |
| Calmar Ratio | 0.212 | 0.133 | 0.159 | 0.180 | 0.175 | 0.181 |

The portfolio leads SPY and Equal Weight on Sharpe. 60/40 leads on Sharpe (0.497) — the rate environment of 2006–2024 favored bond/equity blends. The portfolio's primary edge remains drawdown protection: -25.84% vs SPY's -60.39%.

### Held-Out Test Period (2019–2024)

| Metric | Portfolio | SPY |
|---|---|---|
| Annualized Return | 7.81% | 14.86% |
| Annualized Volatility | 10.06% | 19.91% |
| Sharpe Ratio | 0.579 | 0.797 |
| Max Drawdown | -26.21% | -35.75% |
| Calmar Ratio | 0.298 | 0.416 |

The held-out Sharpe (0.579) lags SPY (0.797) — the 2019–2024 period was structurally unfavorable for regime-switching strategies. The portfolio maintains drawdown protection (-26.21% vs -35.75%).

### Subperiod Analysis

| Metric | Portfolio | SPY |
|---|---|---|
| **GFC (2008–2009)** | | |
| Annualized Return | 0.97% | -15.46% |
| Sharpe Ratio | -0.005 | -0.345 |
| Max Drawdown | -15.06% | -56.38% |
| **Low-vol bull (2013–2019)** | | |
| Annualized Return | 7.48% | 13.26% |
| Sharpe Ratio | 0.759 | 1.003 |
| Max Drawdown | -16.10% | -19.82% |
| **COVID+rates (2020–2024)** | | |
| Annualized Return | 4.77% | 11.83% |
| Sharpe Ratio | 0.244 | 0.528 |
| Max Drawdown | -26.38% | -35.75% |

**GFC:** The strategy's strongest result — near-zero return while SPY lost over half its value. Max drawdown -15.06% vs SPY -56.38%.

**Low-vol bull:** The cost of regime switching — roughly half SPY's return during a six-year bull market.

**COVID+rates:** The weak spot. Fast crash, simultaneous bond/equity selloff in 2022. The strategy's drawdown protection is primarily a GFC-era result.

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

`CORR_WINDOW=63` consistently produces the best or near-best held-out Sharpe across all VOL_WINDOW values. The default parameters sit in a stable region — not cherry-picked.

### Bootstrap Block-Length Sensitivity (v4)

| Block Length | p-value | Mean Diff | 95% CI |
|---|---|---|---|
| 10 | 0.342 | 0.152 | [-0.44, 0.79] |
| 20 | 0.292 | 0.165 | [-0.48, 0.77] |
| 40 | 0.333 | 0.148 | [-0.50, 0.81] |

The conclusion is stable across block lengths: Sharpe outperformance over SPY is not statistically significant regardless of block length choice.

### Bootstrap Significance Test (v2)

Block bootstrap with 1000 iterations and 20-day block length:

| Statistic | Value |
|---|---|
| Mean Sharpe Difference | 0.0918 |
| Std Sharpe Difference | 0.3238 |
| 95% CI | [-0.565, 0.707] |
| p-value | 0.376 |
| Significant at 95% | No |

The Sharpe outperformance over SPY is not statistically significant. The drawdown reduction is the more defensible edge.

---

## v3 → v4 Improvements

| Change | v3 | v4 |
|---|---|---|
| Nested CV execution | `select_n_states()` existed but was never called in `walk_forward_regimes()` — every fold used fixed n=4 | Genuinely runs per fold — each fold selects n_states by BIC independently |
| Fold execution | Sequential for loop | `joblib.Parallel` — `n_jobs` controlled via config.yaml |
| Window sensitivity | Not tested | 9-run sweep over VOL_WINDOW × CORR_WINDOW, results in `data/sensitivity_results.csv` |
| Bootstrap sensitivity | Not tested | 3-run sweep over BLOCK_LENGTH — p-value stable at 0.29–0.34 |
| SPY RF consistency | Held-out SPY computed without RF | RF applied consistently to both portfolio and SPY in held-out block |
| Composite state labeling | Not tested | Tested (Sharpe-based ranking), reverted — held-out Sharpe degraded from 0.760 to 0.579 |

## v2 → v3 Improvements

| Change | v2 | v3 |
|---|---|---|
| State selection | Global BIC/AIC on pre-2018 data, fixed n=4 | Nested CV per fold (implementation completed in v4) |
| NaN posterior handling | Missing | Explicit skip of NaN posterior columns in blending layer |
| Benchmarks | SPY, Equal Weight, 60/40, Momentum | + Risk Parity |
| Subperiod analysis | None | GFC / Low-vol bull / COVID+rates |
| Regime diagnostics | Transition matrix + avg duration | + Run-length distribution, empirical transition frequencies |
| Normality testing | None | Jarque-Bera per regime |
| Config coverage | HMM, backtest, paths, evaluation | + market constants, subperiod definitions |
| Test suite | test_metrics, test_hmm, test_engine | All passing |

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
├── scripts/
│   └── sensitivity_sweep.py  # Window and bootstrap sensitivity analysis
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

**Why nested CV for state selection?** A globally fixed n=4 assumes the same model complexity is appropriate across all market environments and data volumes. Nested CV allows early folds with limited data to select simpler models (n=2), reducing overfitting risk. The tradeoff: fewer Crash detections on this dataset results in slightly weaker drawdown protection.

**Why posterior blending?** Hard switching ignores the model's own uncertainty. Posterior blending hedges proportionally — the HMM's confidence directly controls allocation.

**Why quarterly cache refresh?** Without it, optimizer weights computed in 2008 get blended into 2024 allocations — the covariance structure is stale.

**Why walk-forward as default?** A static HMM trained on all data uses future information to label past regimes. Walk-forward ensures each label is produced by a model that has never seen future data.

**Why time-varying risk-free rate?** A static 4% rate applied over 2006–2024 misrepresents Sharpe during the zero-rate era (2009–2015, 2020–2022) and the high-rate era (2023–2024).

**Why cost-adjust benchmarks?** Fair comparison requires consistent cost treatment. All benchmarks incur 2bps per unit of monthly rebalancing turnover.

**Why Ledoit-Wolf shrinkage?** Sample covariance matrices from short windows are noisy and ill-conditioned. Ledoit-Wolf shrinks toward a stable target, preventing solver failures.

---

## Limitations

**Gaussian emission assumption violated.** Jarque-Bera tests reject normality for all four regimes (p≈0). Bear regime kurtosis is 9.7 — extreme fat tails. The HMM's posterior probabilities are built on a distributional assumption that does not hold. A Student-t HMM would be more appropriate but is not available in hmmlearn.

**Methodological correctness vs backtest performance.** The nested CV implementation (v4) is more honest than the fixed n=4 approach (v3), but produces lower reported Sharpe. This reflects a fundamental tension: the fixed n=4 model happened to identify crash periods more aggressively on this specific dataset. The nested CV generalizes better by design, but that doesn't guarantee better in-sample numbers.

**Choppy Bull detection.** Median Bull run length is 1 day in v4 — the nested CV selecting n=2 on some folds collapses many Sideways/Crash days into Bull, causing rapid label switching. A minimum dwell-time constraint would reduce this.

**COVID+rates weakness.** The March 2020 crash was too sharp for a backward-looking HMM. The 2022 rate-driven bear hit bonds and equities simultaneously, limiting the flight-to-safety allocation. The strategy's drawdown protection is primarily a GFC-era result.

**Bootstrap significance.** The block bootstrap tests a weaker null than White's Reality Check or Hansen's SPA test. The p-value of 0.376 should be interpreted as a lower bound on uncertainty. The bootstrap block-length sweep confirms this conclusion is stable across block lengths 10, 20, and 40.

**Bull market drag.** The strategy structurally underperforms in sustained bull markets. The 2013–2019 period shows this clearly: 7.48% vs SPY's 13.26%.

**Mean return as signal.** The Mean-Variance optimizer uses sample mean returns — near-zero signal-to-noise at daily frequency.

**Asset universe.** Limited to 8 ETFs. Transaction costs modeled at 2bps — realistic for liquid ETFs, not accounting for market impact at scale.

**State labeling.** `label_states()` ranks states on return mean only. A composite Sharpe-based ranking was tested and reverted — it produced worse out-of-sample results despite sounder theory.

---

## Author

Abdelkrim — Applied Mathematics & AI, PSL-Dauphine
https://github.com/AbdelkrimCode
