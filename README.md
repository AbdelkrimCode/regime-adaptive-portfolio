# Regime Adaptive Portfolio

Algorithmic portfolio optimizer that detects market regimes using a Hidden Markov Model and dynamically switches between optimization strategies.

![Python](https://img.shields.io/badge/python-3.10+-blue) ![License](https://img.shields.io/badge/license-MIT-green)

## How it works

1. **Regime Detection**   A 3-state Gaussian HMM trained on SPY log returns, rolling volatility, and rolling correlation labels each trading day as Bull, Bear, or Sideways. The model is retrained quarterly using an expanding window to eliminate lookahead bias.
2. **Adaptive Optimization**   Weights are recomputed at each regime change using the appropriate optimizer:
   - Bull → Mean-Variance (maximize Sharpe ratio)
   - Bear → Risk Parity (equalize risk contributions)
   - Sideways → Minimum Variance (minimize portfolio volatility)
3. **Backtest**   Lookahead-free simulation applying t-day weights to t+1 returns across 20 years of data. Transaction costs modeled at 2bps per unit of turnover.

## Results (2006–2024)

| Metric                | Portfolio | SPY Buy & Hold |
|-----------------------|-----------|----------------|
| Annualized Return     | 5.87%     | 8.27%          |
| Annualized Volatility | 8.24%     | 19.58%         |
| Sharpe Ratio          | 0.227     | 0.218          |
| Max Drawdown          | -20.57%   | -60.39%        |
| Calmar Ratio          | 0.285     | 0.137          |

The portfolio outperforms SPY on risk-adjusted return (Sharpe 0.227 vs 0.218) while delivering less than a third of SPY's maximum drawdown. During the 2008 financial crisis the portfolio stayed above water while SPY lost nearly 60%.

## Charts

![charts](data/charts.png)

## Exploration Notebook

An interactive walkthrough of the three core components — regime detection, optimizer comparison, and backtest results.

> GitHub's notebook renderer may fail on this file. View via [nbviewer](https://nbviewer.org/github/AbdelkrimCode/regime-adaptive-portfolio/blob/main/notebooks/exploration.ipynb) or clone the repo and open locally with `jupyter notebook`.


## Architecture

```
regime-adaptive-portfolio/
├── data/
│   ├── fetch.py           # Downloads adjusted close prices via yfinance (SPY, TLT, GLD, EFA, IEF, QQQ, LQD, VNQ)
│   └── process.py         # Computes log returns, 21-day volatility, 63-day mean correlation
├── models/
│   └── hmm.py             # 3-state Gaussian HMM with quarterly walk-forward retraining
├── optimization/
│   ├── mean_var.py        # Mean-Variance max Sharpe via convex QP (Bull regime)
│   ├── risk_parity.py     # Risk Parity via log barrier formulation (Bear regime)
│   ├── min_variance.py    # Minimum Variance QP (Sideways regime)
│   └── switcher.py        # Routes to correct optimizer, enforces lookahead prevention
├── backtest/
│   ├── engine.py          # Simulates portfolio returns with transaction costs
│   ├── metrics.py         # Annualized return, volatility, Sharpe, max drawdown, Calmar
│   └── benchmark.py       # SPY buy-and-hold comparison
├── visualization/
│   └── charts.py          # Equity curves, drawdown comparison, regime overlay
└── main.py                # Full pipeline entry point
```

## Stack

| Library    | Purpose                    |
|------------|----------------------------|
| yfinance   | Price data download        |
| hmmlearn   | Gaussian HMM implementation|
| cvxpy      | Convex optimization solver |
| pandas / numpy | Data manipulation      |
| matplotlib | Visualization              |
| joblib     | Model persistence          |
| scikit-learn   | Feature scaling, Ledoit-Wolf covariance |


## Usage

```bash
# Install dependencies
pip install yfinance hmmlearn cvxpy scikit-learn pandas numpy matplotlib joblib pyarrow

# Run full pipeline (loads cached HMM model)
python main.py

# Force HMM retrain from scratch
python main.py --retrain

# Run without generating charts
python main.py --no-charts

# Run with walk-forward retraining (recommended, a little slower)
python main.py --walk-forward
```

## Key Design Decisions

**Why Gaussian HMM?** Markets exhibit persistent regimes — Bull markets tend to stay bullish, Bear markets tend to stay bearish. The HMM transition matrix captures this persistence. Gaussian emissions model the continuous feature space (returns, volatility, correlation) naturally.

**Why walk-forward retraining?** A static HMM trained on all data uses future information to label past regimes. Quarterly expanding-window retraining ensures each regime label is produced by a model that has never seen future data — the only valid out-of-sample setup.

**Why Ledoit-Wolf shrinkage?** Sample covariance matrices estimated from short windows are noisy. Ledoit-Wolf shrinks toward a stable target, producing more reliable estimates and preventing solver failures from ill-conditioned matrices.

**Why these three optimizers?** Each regime has a different risk-return objective. In Bull markets you want to capture upside — Mean-Variance does this. In Bear markets correlations spike and equities crash — Risk Parity naturally underweights equities. In Sideways markets there is no reliable signal — Minimum Variance just preserves capital.

**Why no short selling?** Weight constraints w >= 0 reflect realistic constraints for most investors and prevent the optimizer from taking leveraged bets based on noisy return estimates.

**Lookahead prevention** — The switcher always slices returns.loc[:date] before passing to any optimizer. Weights on day t are only ever computed using information available on day t.

## Limitations

- Transaction costs modeled at 2bps per unit of turnover (realistic for liquid ETFs)
- Weight transitions smoothed over 5 days to reduce turnover spikes
- Asset universe limited to 8 ETFs
- HMM features limited to returns, volatility, and correlation — adding macro features (VIX, yield curve) would improve regime detection

## Author

Abdelkrim — Applied Mathematics & AI, PSL-Dauphine
https://github.com/AbdelkrimCode
