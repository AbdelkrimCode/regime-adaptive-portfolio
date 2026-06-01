# Regime Adaptive Portfolio — v2

Algorithmic portfolio optimizer combining Hidden Markov Model regime detection with convex optimization. v2 introduces statistically validated regime selection, posterior probability blending, expanded benchmarks, bootstrap significance testing, and a held-out test period.

![Python](https://img.shields.io/badge/python-3.10+-blue) ![License](https://img.shields.io/badge/license-MIT-green)

---

## How It Works

### 1. Regime Detection
A Gaussian HMM trained on three features — SPY log returns, 21-day rolling volatility, and 63-day mean pairwise correlation — labels each trading day as Bull, Bear, or Sideways. The model is retrained quarterly using an expanding window to prevent lookahead bias.

State ordering is determined by ranking state means on the return feature:
- Lowest mean return → Bear
- Middle → Sideways
- Highest → Bull

### 2. State Count Validation (v2)
The choice of 3 states is validated using out-of-sample AIC/BIC. Models with 2–5 states are trained on 80% of the data and scored on the held-out 20%. Both criteria show a clean minimum at n=3 — 4 and 5 states overfit the training period and generalize worse.

![State Selection](data/state_selection.png)

### 3. Posterior Probability Blending (v2)
v1 used hard regime labels — each day was assigned exactly one regime, and the corresponding optimizer was used exclusively. This amplifies turnover at transition boundaries and ignores the HMM's uncertainty.

v2 uses `predict_proba()` to extract the full posterior distribution over states for every day:

```
weights(t) = P(Bull|t) × w_bull + P(Bear|t) × w_bear + P(Sideways|t) × w_sideways
```

This is a convex combination — weights always sum to 1, always non-negative. On high-confidence days the blend approaches a single optimizer. On uncertain days it hedges across all three. The 5-day linear transition smoother from v1 is removed — uncertainty is now model-driven, not heuristic.

### 4. Adaptive Optimization
Three convex optimizers, one per regime:

| Regime | Optimizer | Objective |
|---|---|---|
| Bull | Mean-Variance | Maximize Sharpe ratio |
| Bear | Risk Parity | Equalize risk contributions |
| Sideways | Minimum Variance | Minimize portfolio volatility |

All optimizers use Ledoit-Wolf shrinkage on the covariance matrix. Weights constrained to [0, 1] — no shorting, no leverage.

### 5. Backtest
Lookahead-free simulation applying day-t weights to day t+1 returns. Transaction costs at 2bps per unit of turnover. Walk-forward retraining with quarterly expanding window.

---

## Results

### Full Period (2005–2024)

| Metric | Portfolio | SPY | Equal Weight | 60/40 | Momentum |
|---|---|---|---|---|---|
| Annualized Return | 6.84% | 8.51% | 6.43% | 6.91% | 6.44% |
| Annualized Volatility | 8.43% | 19.14% | 10.98% | 10.97% | 13.28% |
| Sharpe Ratio | **0.337** | 0.236 | 0.221 | 0.266 | 0.184 |
| Max Drawdown | **-23.93%** | -59.58% | -35.05% | -35.60% | -26.98% |
| Calmar Ratio | **0.286** | 0.143 | 0.183 | 0.194 | 0.239 |

The portfolio achieves the highest Sharpe and lowest drawdown across all five strategies. The cost is absolute return — the strategy trades upside capture for volatility compression and downside protection.

### Held-Out Test Period (2019–2024)

| Metric | Portfolio | SPY |
|---|---|---|
| Annualized Return | 5.00% | 14.86% |
| Annualized Volatility | 9.54% | 19.91% |
| Sharpe Ratio | 0.105 | 0.545 |
| Max Drawdown | -23.67% | -35.75% |
| Calmar Ratio | 0.211 | 0.416 |

The 2019–2024 period was structurally unfavorable for regime-switching strategies — a sustained bull market with a sharp but brief COVID correction and a 2022 bear market that recovered quickly. SPY dominates on return and Sharpe. The portfolio's drawdown protection holds out-of-sample (-23.67% vs -35.75%) but is insufficient to compensate for the return gap in a strong bull environment.

### Bootstrap Significance Test (v2)

Block bootstrap with 1000 iterations and 20-day block length (preserving autocorrelation structure):

| Statistic | Value |
|---|---|
| Mean Sharpe Difference | 0.0918 |
| Std Sharpe Difference | 0.3238 |
| 95% CI | [-0.565, 0.707] |
| p-value | 0.376 |
| Significant at 95% | No |

The Sharpe outperformance over SPY is not statistically significant over the sample period. The confidence interval straddles zero widely — the effect size is real on observed data but indistinguishable from sampling variation. The drawdown reduction is the more defensible edge.

![Bootstrap](data/bootstrap.png)

---

## v1 → v2 Improvements

| Change | v1 | v2 |
|---|---|---|
| Regime uncertainty | Hard label per day | Posterior probability blend |
| State count justification | Assumed n=3 | BIC/AIC validated on held-out data |
| Benchmarks | SPY only | SPY, Equal Weight, 60/40, Momentum |
| Significance testing | None | Block bootstrap, p=0.376 |
| Out-of-sample evaluation | None | Held-out 2019–2024 test period |
| Pipeline coupling | Regimes read from disk silently | Regimes injected explicitly |
| Hyperparameters | Scattered across modules | Centralized in config.yaml |

---

## Architecture

```
regime-adaptive-portfolio/
├── config.yaml            # Centralized hyperparameters (HMM, backtest, paths)
├── config.py              # YAML loader
├── data/
│   ├── fetch.py           # Downloads adjusted close prices via yfinance
│   └── process.py         # Log returns, 21-day volatility, 63-day mean correlation
├── models/
│   └── hmm.py             # Gaussian HMM, walk-forward retraining, BIC/AIC selection
├── optimization/
│   ├── mean_var.py        # Mean-Variance max Sharpe (Bull)
│   ├── risk_parity.py     # Risk Parity via log barrier (Bear)
│   ├── min_variance.py    # Minimum Variance QP (Sideways)
│   └── switcher.py        # Posterior probability blending across optimizers
├── backtest/
│   ├── engine.py          # Simulation with transaction costs, period slicing
│   ├── metrics.py         # Annualized return, volatility, Sharpe, drawdown, Calmar
│   ├── benchmark.py       # SPY, Equal Weight, 60/40, Momentum benchmarks
│   └── bootstrap.py       # Block bootstrap significance test
├── visualization/
│   └── charts.py          # Equity curves, drawdown, regime overlay
└── main.py                # Full pipeline entry point
```

---

## Stack

| Library | Purpose |
|---|---|
| yfinance | Price data download |
| hmmlearn | Gaussian HMM |
| cvxpy | Convex optimization |
| scikit-learn | Feature scaling, Ledoit-Wolf covariance |
| pandas / numpy | Data manipulation |
| matplotlib | Visualization |
| joblib | Model persistence |
| pyyaml | Config loading |

---

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python main.py

# Force HMM retrain
python main.py --retrain

# Walk-forward retraining (recommended)
python main.py --walk-forward

# Skip chart generation
python main.py --no-charts
```

---

## Key Design Decisions

**Why Gaussian HMM?** Markets exhibit persistent regimes — the HMM transition matrix captures this persistence. Gaussian emissions model the continuous feature space naturally.

**Why 3 states?** Validated by out-of-sample BIC/AIC — both criteria show a minimum at n=3. Interpretability (Bull/Bear/Sideways) aligns with the statistical result.

**Why posterior blending?** Hard switching amplifies turnover at regime boundaries and ignores the model's own uncertainty. Posterior blending is model-driven — the HMM's confidence directly controls how aggressively we rotate.

**Why walk-forward retraining?** A static HMM trained on all data uses future information to label past regimes. Quarterly expanding-window retraining ensures each label is produced by a model that has never seen future data.

**Why Ledoit-Wolf shrinkage?** Sample covariance matrices from short windows are noisy and ill-conditioned. Ledoit-Wolf shrinks toward a stable target, preventing solver failures.

**Why these benchmarks?** SPY alone is a weak baseline — it carries full equity volatility. Equal Weight, 60/40, and Momentum represent realistic alternatives that also reduce risk. Beating all four on Sharpe and drawdown is a meaningful result.

---

## Limitations

- Sharpe outperformance over SPY is not statistically significant (p=0.376, block bootstrap)
- Strategy structurally underperforms in sustained bull markets — regime switching reduces equity exposure precisely when equity performs well
- HMM features limited to returns, volatility, correlation — macro features (VIX, yield curve slope) would likely improve regime detection
- Asset universe limited to 8 ETFs
- Transaction costs modeled at 2bps — realistic for liquid ETFs but not accounting for market impact at scale

---

## Author

Abdelkrim — Applied Mathematics & AI, PSL-Dauphine
https://github.com/AbdelkrimCode
