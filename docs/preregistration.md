# Pre-registration: detection vs response in a regime allocator

Status: frozen at git tag `prereg-v1`, together with the code that runs it (`research/run_study.py` and the modules it imports). Any later change is listed under Deviations with its reason and date.

This is a self pre-registration: the git tag and its timestamp are the record. Nothing in Sections 4 to 7 was run on real data before the tag.

## 1. Question

When a hidden Markov regime allocator fails to beat a static allocation, is the cause the detection (the regimes are wrong or late) or the response (the allocation chosen for each regime is wrong)?

Prior work concludes the bottleneck is the decision rule rather than detection, but infers it without measuring how much better detection would help (Grube Martin-Lunas, Lazcano and Sandubete, Economies 14(7), 2026). This study measures both with look-ahead oracles.

## 2. Data and samples

- Universe: SPY, TLT, GLD, EFA, IEF, QQQ, LQD, VNQ (yfinance adjusted closes). Risk-free: 3-month T-bill (^IRX), daily.
- Development sample: data 2005-01-04 to 2024-12-31; evaluation on the portfolio's dates, 2006-04-05 to 2024-12-31. This sample has been inspected many times during development and is treated as in-sample.
- Holdout: 2025-01-02 to the last close available on the run date. Never loaded before the holdout run (Section 7).
- The SHA-256 of `data/raw/prices.parquet` and `data/processed/risk_free.parquet` is recorded with every result file.

## 3. Fixed components (not tuned in this study)

- Regime model: 4-state Gaussian HMM, full covariance, features SPY log return, 21-day realized vol, 63-day mean pairwise correlation; quarterly expanding-window refits; causal forward filter carried across folds; states labeled by mean `spy_vol` (Bull calmest, then Sideways, Bear, Crash).
- Optimizers: Bull max Sharpe, Sideways min variance, Bear risk parity, Crash inverse-vol safe havens (IEF, TLT, GLD); Ledoit-Wolf covariance on all history to date; 60% cap; concentration guard.
- Refresh: optimizer weights recomputed on the first trading day of each week, identical in every cell (`research.books.optimizer_targets`).
- Execution: `backtest.engine.simulate`, trade at close t, earn from t+1, drifting holdings, turnover from drifted weights.
- Costs: 2 bp one-way (primary); 5 and 10 bp reported for every cell.
- Sharpe: daily excess over rf, annualized by sqrt(252).

## 4. Cells

Investable:
- **C0 baseline:** real filtered posteriors, default mapping.
- **C1 static blend:** 25% per regime every day (decision 2a).
- **C2 placebo:** real posteriors circularly shifted by every k in `placebo_shifts(n, 252, 21)`; one run per k.
- **B1-B4 single books:** each optimizer held alone (descriptive).
- **H4 variant:** Section 5, H4.

Oracles (look-ahead on purpose, upper bounds, never reported as strategies):
- **D1 detection oracle:** HMM fitted on the whole development sample, smoothed posteriors (decision 1a); default mapping.
- **D1b detection oracle, model-free:** quartiles of next-21-day realized SPY volatility (decision 1b); default mapping. Robustness for D1.
- **R1 response oracle:** real posteriors; mapping = `hindsight_mapping` chosen on C0's argmax labels.
- **F full oracle:** D1 posteriors; mapping chosen on D1's argmax labels.

## 5. Hypotheses and tests

**H1: detection carries information about which optimizer will do better.**
For each optimizer k, regress its next-day return minus the average of the four books on the day-t posteriors (Bull, Sideways, Bear; Crash omitted since they sum to 1). Newey-West standard errors, lag floor(4 (T/100)^(2/9)). Joint Wald test that the three slopes are zero; Holm correction across the four optimizers.
Supported if at least one optimizer rejects at 5% after correction.

**H2: the strategy beats its own controls.**
- C0 vs C1: Ledoit-Wolf (2008) HAC Sharpe-difference test, one-sided, alpha 5%. Paired block bootstrap (20-day blocks, 1000 paths) reported alongside.
- C0 vs C2: p = (1 + #{placebo Sharpe >= C0 Sharpe}) / (1 + number of shifts), alpha 5%.
Supported only if both reject.

**H3: the response, not the detection, is the bottleneck.**
- Detection value: dD = SR(D1) - SR(C0). Response value: dR = SR(R1) - SR(C0). Interaction: SR(F) - SR(D1) - SR(R1) + SR(C0).
- Stated prediction, based on the 2022 drawdown: dR > dD.
- 95% CI of dR - dD by paired block bootstrap over days (C0, D1 and R1 resampled with the same indices; mappings and posteriors held fixed). Supported if the CI excludes 0 and dR > dD; refuted if it excludes 0 and dR < dD; inconclusive otherwise.
- Same comparison with D1b in place of D1 as robustness.
- Caveat stated in advance: oracles chosen on the same sample they are scored on are optimistic by construction. The comparison of dR and dD is the result; their levels are upper bounds.
- Descriptive: every cell by calendar year, with 2008 and 2022 discussed.

**H4: a correlation-aware response improves the strategy.**
Rule, fixed here: when the 63-day correlation of SPY and IEF daily returns at the close of day t is above 0, the two defensive books (risk parity and safe haven) move their weight in TLT, IEF and LQD into cash earning the daily rf. Otherwise nothing changes. Everything else as in C0.
- Test: H4 vs C0, Ledoit-Wolf one-sided, alpha 5%; also reported vs C1.
- Deflated Sharpe Ratio of H4 with the trial count of Section 6.
Supported if the Ledoit-Wolf test rejects and DSR > 0.95.

## 6. Multiple testing and trial count

- Confirmatory tests: H1 (Holm across 4), H2 (two tests, both required), H3 (one CI), H4 (one test plus DSR). Everything else is descriptive.
- Trial count for every Deflated Sharpe Ratio: 17 distinct investable configurations evaluated in this project: the 9-cell feature window sweep (its default cell is the original regime-refresh strategy), the skew/kurtosis ablation (the other two ablations are sweep cells), the frozen model, and in this study C0, C1 and B1-B4. H4 is added as trial 18 when its own DSR is computed.
- The DSR threshold uses N = 17 (18 for H4) in the expected-maximum term, and the variance of the daily Sharpe ratios of the 16 trials that have a full-period Sharpe (the frozen model was only scored on 2019-2024).

## 7. Holdout

Run once, after every Section 5 result on the development sample is committed:
1. One commit changes `data_end` and nothing else.
2. Walk-forward refits continue quarterly into the holdout; nothing else is refit or re-chosen. The R1 and F mappings are not re-estimated.
3. Report C0, C1, the C2 distribution (shifts computed on the full series, scored on holdout days), and H4, at all three cost levels.
4. Results are reported whatever they show. With about 1.8 years of data the Sharpe standard error is around 0.75, so the holdout can reveal a failure but cannot confirm an edge of the size seen in development.

## 8. What would change the conclusions

- A bug found after the tag: fixed in its own commit, every result rerun, and the fix listed below.
- No parameter in Sections 3 to 5 is changed after seeing results. If one has to change, the original and the changed results are both reported.

## Deviations

None yet.

## Addenda

- `docs/preregistration_addendum_backcast.md` (tag `prereg-v2`): the same study on a 1996-2004 proxy universe, as a second out-of-sample test.
