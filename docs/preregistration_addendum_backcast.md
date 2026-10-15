# Pre-registration addendum: backcast on 1996-2004

Status: frozen at git tag `prereg-v2`, together with the code that runs it (`research/run_backcast.py`, `config_backcast.yaml` and the modules they import). Nothing below was run on real data before the tag. Any later change is listed under Deviations.

## 1. Why

The 2025-2026 holdout has about 1.8 years of data, so it can reveal a failure but cannot confirm an edge (Section 7 of `docs/preregistration.md`). The only other data the project has never looked at lies before the development sample. This addendum runs the frozen study, unchanged, on 1996-2004 as a second out-of-sample test.

Disclosed timing: the backcast was decided on 2026-10-15, after the holdout results were known. The hypotheses and tests below are copied from the original pre-registration, so the holdout result could not shape them.

## 2. Universe

The ETFs did not exist before 2002 (QQQ 1999, EFA 2001, IEF, TLT and LQD 2002, VNQ 2004). Each asset is replaced by an index fund or index with the same exposure. Columns keep the ETF names, so the code is unchanged.

| Asset | Proxy | Proxy description | First date |
|---|---|---|---|
| SPY | VFINX | Vanguard 500 Index | 1980 |
| QQQ | ^NDX | Nasdaq-100 index, price only | 1985 |
| TLT | VUSTX | Vanguard Long-Term Treasury | 1986 |
| IEF | VFITX | Vanguard Intermediate-Term Treasury | 1991 |
| LQD | VFICX | Vanguard Intermediate-Term Investment-Grade | 1993 |
| EFA | VGTSX | Vanguard Total International Stock Index | 1996-04-29 |
| VNQ | VGSIX | Vanguard REIT Index | 1996-05-13 |
| GLD | dropped | no daily gold series before 2000 | |

Risk-free: ^IRX, as in the development sample.

Gold is dropped rather than proxied, so the universe has 7 assets and the safe havens (Crash book) are IEF and TLT. This also changes the `mean_corr` feature, which averages over 7 assets instead of 8.

## 3. Proxy caveats (stated before the run)

- Fund returns are net of expense ratios (roughly 0.2% to 0.3% a year), so the proxies slightly understate the ETFs' gross returns. The effect on Sharpe differences between cells is negligible because every cell holds the same assets.
- ^NDX is a price index: it omits dividends, which were small for the Nasdaq-100 in this period.
- VGTSX includes emerging markets, unlike EFA. Its daily NAV is struck at the US close on foreign prices set hours earlier, so its returns are partly stale and autocorrelated. This can make the international asset look less correlated and less volatile than it was.
- VFICX holds intermediate corporate bonds; LQD has a longer duration.
- Mutual-fund adjusted closes from yfinance include reinvested distributions. Their quality before 2000 has not been checked and will not be checked before the run; any data problem found after the run is reported, not fixed silently.

## 4. Sample

- Data: 1996-05-01 to 2004-12-31 (first full day of all proxies is about 1996-05-14). The sample ends before the first day the development sample uses (2005-01-04), so no day appears in both.
- Evaluation: from the second day with a walk-forward regime (expected around October 1997, after the 63-day feature window and the 252-day minimum training sample) to 2004-12-31, about 7.2 years.
- It contains the 1998 LTCM crisis, the 2000-2002 dot-com bust and the 2003-2004 recovery. Stocks and bonds were mostly negatively correlated, unlike 2022.

## 5. What is identical

Everything in Sections 3 to 6 of `docs/preregistration.md`: the regime model, its walk-forward schedule and labeling, the optimizers, the weekly refresh, execution, costs (2 bp primary, 5 and 10 bp reported), the cells (C0, C1, C2 placebo, B1-B4, H4), the oracles, hypotheses H1 to H4, their tests and decision rules, and the Sharpe definition.

Applied to this sample:
- All models are fitted on backcast data only. The walk-forward starts afresh in 1996. The D1 oracle fits one HMM on the whole backcast sample; the R1 and F mappings are chosen on backcast labels. Nothing estimated on 2005-2024 is reused.
- The Deflated Sharpe Ratio uses the same trial count as the original (17, and 18 for H4), with the variance taken from the development trial Sharpes plus the six study cells of this sample. This is conservative: no configuration was selected on this sample.

## 6. How it runs

1. Tag `prereg-v2` on the commit that adds this file and the backcast code.
2. `python -m research.run_backcast`, once. It downloads the proxies into `data/backcast/`, recomputes returns and features, runs the walk-forward and the study, and writes `results/study/backcast/`. The development files (`data/raw/`, `data/processed/`) are neither read nor written; the SHA-256 of the backcast prices and risk-free file are recorded in the summary.
3. Results are reported whatever they show, next to the development and holdout results, with the caveats above.

Before the tag, the only contact with this data was a check of each proxy's first available date and row count (2026-10-15). No prices or returns were looked at. The pipeline was tested on synthetic prices only.

## 7. Reading the result

- With about 7 years, the standard error of a Sharpe difference between two highly correlated cells is far smaller than that of a single Sharpe, so H2 and H4 have more power here than in the holdout.
- If the backcast agrees with the development sample (no edge over the static blend, detection as the bottleneck), the conclusion holds across three samples. If it disagrees, both results are reported and the disagreement is the finding.

## Deviations

None yet.
