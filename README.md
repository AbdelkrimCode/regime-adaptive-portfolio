# Regime Adaptive Portfolio — v2

Algorithmic portfolio optimizer combining Hidden Markov Model regime detection with convex optimization. v2 introduces statistically validated regime selection, posterior probability blending, a 4-state model with Crash regime, expanded benchmarks, bootstrap significance testing, and a held-out test period.

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

### 2. State Count Validation (v2)
The number of states is validated using out-of-sample AIC/BIC on pre-2018 data only — ensuring the choice is made before the backtest period begins. Models with 2–5 states are trained on 80% of the pre-2018 data and scored on the held-out 20%.

Both AIC and BIC prefer n=4 on this clean split. n=4 is adopted — adding a distinct Crash state separate from Bear, capturing extreme negative return environments (2008–2009, 2020, 2022).

![State Selection](data/state_selection.png)

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

### 5. Regime Diagnostics (v2)
The HMM transition matrix confirms regime persistence — a core assumption of the model:

| Regime | Avg Duration (days) |
|---|---|
| Sideways | 55.6 |
| Crash | 56.2 |
| Bull | 54.9 |
| Bear | 41.2 |

Diagonal transition probabilities exceed 0.975 for all regimes — once in a regime, the daily probability of staying exceeds 97.5%.

### 6. Backtest
Lookahead-free simulation applying day-t weights to day t+1 returns. Transaction costs at 2bps per unit of turnover. Average daily turnover: 5.37% (implied annual cost: 0.27%).

---

## Results

### Full Period (2005–2024)

| Metric | Portfolio | SPY | Equal Weight | 60/40 | Momentum |
|---|---|---|---|---|---|
| Annualized Return | 5.79% | 8.27% | 5.90% | 6.80% | 5.76% |
| Annualized Volatility | 8.71% | 19.58% | 11.19% | 11.20% | 13.22% |
| Sharpe Ratio | 0.206 | 0.218 | 0.170 | 0.250 | 0.133 |
| Max Drawdown | **-25.49%** | -60.39% | -35.29% | -36.21% | -29.61% |
| Calmar Ratio | **0.227** | 0.137 | 0.167 | 0.188 | 0.195 |

The portfolio achieves the lowest max drawdown and strong Calmar ratio across all strategies. The cost is absolute return and Sharpe — the strategy trades upside capture for downside protection. On walk-forward evaluation, the Sharpe edge over SPY disappears; the drawdown advantage holds.

### Held-Out Test Period (2019–2024)

| Metric | Portfolio | SPY |
|---|---|---|
| Annualized Return | 8.40% | 14.86% |
| Annualized Volatility | 9.52% | 19.91% |
| Sharpe Ratio | 0.463 | 0.545 |
| Max Drawdown | -25.25% | -35.75% |
| Calmar Ratio | 0.333 | 0.416 |

The 2019–2024 period was structurally unfavorable for regime-switching strategies — a sustained bull market interrupted by a brief COVID correction and a 2022 bear market that recovered quickly. SPY dominates on return. Drawdown protection holds out-of-sample (-25.25% vs -35.75%).

### Bootstrap Significance Test (v2)

Block bootstrap with 1000 iterations and 20-day block length (preserving autocorrelation structure):

| Statistic | Value |
|---|---|
| Mean Sharpe Difference | 0.0918 |
| Std Sharpe Difference | 0.3238 |
| 95% CI | [-0.565, 0.707] |
| p-value | 0.376 |
| Significant at 95% | No |

The Sharpe outperformance over SPY is not statistically significant. The drawdown reduction is the more defensible edge.

![Bootstrap](data/bootstrap.png)

---

## v1 → v2 Improvements

| Change | v1 | v2 |
|---|---|---|
| Regime states | 3 (Bull/Bear/Sideways) | 4 (+ Crash with flight-to-safety) |
| Regime uncertainty | Hard label per day | Posterior probability blend |
| State count justification | Assumed n=3 | BIC/AIC on pre-2018 held-out data |
| Optimizer cache | Recomputed on regime change only | Recomputed on every retrain date |
| Walk-forward | Optional flag | Default behavior |
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
├── config.yaml            # Centralized hyperparameters (HMM, backtest, paths, evaluation)
├── config.py              # YAML loader
├── data/
│   ├── fetch.py           # Downloads adjusted close prices via yfinance
│   └── process.py         # Log returns, 21-day volatility, 63-day mean correlation
├── models/
│   └── hmm.py             # 4-state Gaussian HMM, walk-forward retraining, BIC/AIC, transition diagnostics
├── optimization/
│   ├── mean_var.py        # Mean-Variance max Sharpe (Bull)
│   ├── risk_parity.py     # Risk Parity via log barrier (Bear)
│   ├── min_variance.py    # Minimum Variance QP (Sideways)
│   ├── crash.py           # Equal-weight IEF/TLT/GLD flight-to-safety (Crash)
│   └── switcher.py        # Posterior probability blending, quarterly cache refresh
├── backtest/
│   ├── engine.py          # Simulation with transaction costs, period slicing
│   ├── metrics.py         # Annualized return, volatility, Sharpe, drawdown, Calmar, turnover
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

# Run full pipeline (walk-forward retraining is default)
python main.py

# Force HMM retrain from scratch
python main.py --retrain

# Disable walk-forward (use static model)
python main.py --no-walk-forward

# Skip chart generation
python main.py --no-charts
```

---

## Key Design Decisions

**Why Gaussian HMM?** Markets exhibit persistent regimes — the HMM transition matrix captures this persistence, with diagonal transition probabilities exceeding 0.975. Gaussian emissions model the continuous feature space naturally.

**Why 4 states?** BIC/AIC on pre-2018 held-out data prefers n=4 over n=3. The fourth state (Crash) captures extreme negative return environments distinct from ordinary Bear markets — 2008–2009, March 2020, and 2022 are correctly isolated. Average Crash duration is 56 days, consistent with historical crisis episodes.

**Why posterior blending?** Hard switching ignores the model's own uncertainty. On transition days where P(Bull)=0.55 and P(Bear)=0.45, hard switching commits fully to one optimizer. Posterior blending hedges proportionally — the HMM's confidence directly controls allocation.
