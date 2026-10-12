# Changelog

## Correctness cleanup (2026)

Each fix was applied in its own commit with a test that fails on the old code. Effect on the full-period headline after each step:

| Checkpoint | Portfolio Sharpe | Risk Parity | 60/40 | Turnover/day |
|---|---|---|---|---|
| after fixes 1-2 (returns, rf) | 0.641 | 0.635 | 0.637 | 4.26% |
| after fixes 3-4 (filter) | 0.634 | 0.635 | 0.637 | 2.79% |
| after fix 5 (labels) | 0.635 | 0.635 | 0.637 | 2.81% |
| after fixes 8-9 (drift) | 0.649 | 0.606 | 0.619 | 3.01% |

The final numbers after the full cleanup are in the README and `results/results_after_cleanup.txt`.

| # | Commit | Issue | Resolution |
|---|---|---|---|
| 1 | `fix(returns)` | Log returns were used for P&L and optimizer inputs | Simple returns for P&L and optimizers; log returns only for HMM features. mean_var mu is the arithmetic mean x 252 |
| 2 | `fix(rf)` | rf units mixed between daily and annual | rf stored daily; annualized only when passed to `max_sharpe` |
| 3 | `fix(hmm)` | Forward filter could underflow | Filter computed in log space (logsumexp) |
| 4 | `fix(hmm)` | Filter restarted from `startprob_` at every fold, creating ~40% of regime switches | Filter runs over train+test so each fold starts from its filtered state |
| 5 | `fix(hmm)` | Labels ranked by a heuristic return/vol score; BIC always picked the largest candidate | States labeled by mean `spy_vol` (calmest = Bull ... most turbulent = Crash); n_states fixed at 4 |
| 6 | `fix(hmm)` | hmmlearn reports convergence when n_iter is hit and accepts likelihood decreases | `_em_converged` helper |
| 7 | `fix(data)` | yfinance `end` is exclusive, so data_end was dropped | `inclusive_end()`; cache signature records it |
| 8 | `fix(engine)` | Holdings did not drift between trades; turnover measured target to target | `simulate()` tracks drifting holdings; turnover from drifted weights; cost charged on the day the trade starts earning |
| 9 | `fix(benchmark)` | Benchmarks rebuilt with their own simulator and scored on different dates | Month-end rebalancing with drift on the same simulator; all strategies scored on the portfolio's dates |
| 10 | `feat(benchmark)` | No cost sensitivity | Sharpe of every strategy at 2, 5 and 10 bp one-way |
| 11 | `fix(stats)`, `feat(stats)` | Bootstrap and Monte Carlo resampled returns but not rf; the "two independent p-values" were the same test (same seed, block length and index generator); only SPY was tested | rf resampled with returns; second test is the Ledoit-Wolf (2008) HAC Sharpe-difference test; portfolio tested against SPY, risk parity and 60/40 |

Hygiene: shared `_build_folds()` for walk-forward and its audit; one Ledoit-Wolf covariance estimator (`optimization/covariance.py`); rf injected into `compute_weights`; only degenerate-fit errors caught in `_fit_hmm_core`, and logged; config paths resolved against the project root; dead imports, functions and config keys removed; consistent docstrings; runtime and dev requirements split; CI on Python 3.11 to 3.13.

## Post-v6 bug fixes

| Fix | Issue | Resolution |
|---|---|---|
| feature_ablation.py restore block | KeyError on CFG["data"] crashed every completed ablation run | Read from FEATURE_SETS["baseline"] defined in the same file |
| test_optimizers.py | Byte-for-byte duplicate of test_metrics.py, zero optimizer coverage | Replaced with real optimizer tests |
| bootstrap.py rf | Sharpe bootstrap computed with rf=None | Thread time-varying rf through run_bootstrap() |
| bootstrap.py + charts.py paths | Hardcoded parquet paths bypassed config.yaml | Replaced with CFG["paths"] throughout |
| sensitivity_sweep n_jobs | CFG mutation never reached hmm.py | Pass n_jobs explicitly to walk_forward_regimes() |
| stress_tests monte_carlo | IID resampling destroyed autocorrelation structure | Replaced with block_resample() |
| main.py test_end | Dead variable set to train_end | Removed |
| min_variance.py | Tiny floating point negatives possible | Clip and renormalize before returning |
| switcher.py | Magic number 0.99 in concentration guard | Named constant CONCENTRATION_GUARD |
| hmm.py forward_filter | No underflow guard at alpha[0] | Same guard as t>0 steps (superseded by the log-space filter) |
| main.py transition matrix | Division by zero if a regime never transitions out | Replace 0 row sums with NaN before division |
| benchmark.py | Hardcoded 126 in get_risk_parity_equity() | Read from CFG["hmm"]["min_train_days"] // 2 |
| risk_parity.py | The 60% cap was applied to the raw log-barrier solution, saturating every asset into equal weight | Solve with a small floor only, normalize, then cap and redistribute |

## v5 to v6

| Change | v5 | v6 |
|---|---|---|
| Regime posteriors | Forward-backward smoothing, within-window lookahead | Causal forward filter |
| Label permutation | Undocumented | Labels re-ranked after every retrain |
| run_period() optimizer | Passed returns_slice, starving the optimizer early in the test period | Passes full returns history |
| charts.py | KeyError on Crash regime | Crash added to REGIME_COLORS |
| safe_haven_assets | Hardcoded in crash.py | Configurable via config.yaml |
| select_n_states | Unused train/test split | Removed |
| sensitivity_sweep | Global config mutation (ineffective) | n_jobs passed explicitly |
| smoothing_days | Dead config key | Removed |

## v4 to v5

| Change | v4 | v5 |
|---|---|---|
| Optimizer recompute | Only active regime on intra-quarter transition | All four on every regime change |
| RF in max-Sharpe QP | Hardcoded 4% | Time-varying 3-month T-bill |
| BIC scoring | On held-out 20% (wrong N) | On full training data |
| Min-history threshold | Hardcoded 126 | References config |
| fit_hmm duplication | Two near-identical functions | Unified via _fit_hmm_core() |

## v3 to v4

| Change | v3 | v4 |
|---|---|---|
| Nested CV execution | Never called | Runs per fold |
| Fold execution | Sequential | joblib.Parallel |
| Sensitivity analysis | Not tested | Window + bootstrap sweep |

## v2 to v3

| Change | v2 | v3 |
|---|---|---|
| NaN posterior handling | Missing | Explicit skip |
| Benchmarks | 4 | + Risk Parity |
| Subperiod analysis | None | GFC / Low-vol bull / COVID+rates |
| Normality testing | None | Jarque-Bera per regime |

## v1 to v2

| Change | v1 | v2 |
|---|---|---|
| Regime states | 3 | 4 (+ Crash) |
| Regime uncertainty | Hard label | Posterior probability blend |
| Walk-forward | Optional | Default |
| Risk-free rate | Static 4% | Time-varying ^IRX |
| Benchmark costs | None | 2bps per turnover |
| Significance testing | None | Block bootstrap |
| Out-of-sample evaluation | None | Held-out 2019-2024 |
