# 05: The 1-day pop with the SMA screen alone, 2010–2025

*Run 2026-11-20 with `python research/long_history.py`. Outputs are `research/output/long_*.csv`.*

## Question

Report 04 found a borderline result: on 15 unseen weeks, SMA-screened stocks beat their sector
on day 1. Does the same rule, without the Barchart signal, hold up over 16 years?

## Method

- **Rule:** frozen in report 04 and unchanged here.
  - Earnings with a known time of day: before 09:30 ET = BMO, from 16:00 = AMC, anything in
    between dropped.
  - At the close before the entry day: SMA20 > SMA50, having crossed above within the last
    10 trading days.
  - Buy at the close before the announcement, sell at the next close.
- **Universe:** current S&P 500, 400 and 600 members (1,502 companies, 1,501 with prices),
  with their GICS sectors.
  - 84,511 earnings events from 2010 to Nov 2026.
  - The screen fired on 10,004 of them.
- **Benchmark:** the GICS sector SPDR ETF, or SPY for a sector before its ETF existed.
- **Pre-registered tests** (written into the script before it ran):
  - **Window:** 2010-01-01 to 2025-10-31, which no earlier report had looked at.
  - **H1:** screened 1-day sector excess > 0.
  - **H2:** screened minus control > 0.
  - Both use weekly-portfolio t-tests, one-sided, 5%.
- **Data check:** if the dates and times are right, the abnormal move should spike on the
  assumed reaction day. The median absolute sector-excess move was:

| Day | Median absolute move |
|---|---|
| Day before | 0.85% |
| **Reaction day** | **3.10%** |
| Day after | 1.27% |

  The dates and times are reliable.

## Results (2010-01 to 2025-10)

| Group | Events | Weeks | Mean | Median | Hit | Weekly mean | Weekly t | One-sided p |
|---|---|---|---|---|---|---|---|---|
| **Screened** | 9,095 | 658 | +0.27% | +0.21% | 52.0% | +0.11% | **0.61** | **0.27** |
| Control (screen didn't fire) | 69,043 | 813 | +0.32% | +0.16% | 51.7% | +0.27% | 4.36 | <0.001 |
| Uptrend, cross older than 10 days | 36,555 | 785 | +0.26% | +0.09% | 51.0% | +0.09% | 0.90 | 0.18 |
| Below trend (SMA20 ≤ SMA50) | 32,488 | 784 | +0.39% | +0.26% | 52.4% | +0.48% | 4.19 | <0.001 |

- **Screened minus control:** −0.13% per week, t −0.73, p 0.77.
- **H1 and H2 both fail.**
- **After assumed costs** (10/20/40 bps round trip by index), screened events earn +0.04% per
  trade (weekly t −0.73).

### Robustness (all reported, none used to select a rule)

- **Cross window** (weekly t): within 5 days 0.93, within 10 days 0.61, within 20 days 0.66.
- **Benchmarks:** SPY t 0.48, market model t 0.52, raw return t 1.03.
- **Size** (weekly t): S&P 500 0.53, S&P 400 1.11, S&P 600 1.17.
- **Timing** (weekly t): BMO 1.28, AMC 0.38.
- **5-day hold:** screened t 0.37, control t 3.72.
- **By year:** screened beat control in 6 of 16 years. The yearly difference ranges from
  −0.59% (2017) to +0.66% (2012), with no trend.
- **By sector:** Information Technology was the only sector with p < 0.05 (+0.70%, t 2.38,
  p 0.009). With 11 sectors tested, that's within what chance produces (Bonferroni threshold
  0.0045), so it's not a finding. Consumer Staples and Utilities were negative.
- **The tracker's own companies** (253 of them are in the S&P 1500):
  - 2010–2025: screened +0.06% (t −0.67), control +0.21% (t 1.15).
  - May–Nov 2026: screened +0.95% (73 events, t 1.05).
  - Report 04's +1.31% looks like a favorable 15-week stretch for this group of names.
- **The window seen in reports 01–04** (Nov 2025 – Nov 2026, S&P 1500): screened +0.15%
  (weekly t −0.81), control +0.28%.

## Conclusion

- **The SMA crossover screen with a 1-day hold has no edge over 2010–2025.** That's 9,095
  events in 658 weeks.
  - Screened earnings reports did no better than the same companies' unscreened reports.
  - They earned nothing after costs.
  - The result doesn't depend on the cross window, benchmark, size, timing or year.
- **The 15-week result in report 04 is best read as noise in a small sample.** Short windows
  of tail-heavy event returns routinely produce p ≈ 0.05.
- **What the data does show:** S&P 1500 companies as a whole beat their sector by about
  +0.3% on the reaction day (t 4.4). Stocks *below* their trend did a little better (+0.39%).
  This matches the documented earnings-announcement premium. It's also inflated by
  survivorship bias, because the universe is today's index members. It's not a tradeable
  conclusion without a survivorship-free universe.
- **The Barchart question is still open.** The live tracked trades, which required the
  Barchart "Buy", earned +1.79% on day 1. The SMA screen alone, in the same period and
  universe, earned +0.54% (report 04), and +0.15% across the S&P 1500.
  - If the live trades are genuine, the Barchart filter (or something else in the live
    process) is where any edge came from, not the moving-average cross.
  - That can only be tested going forward: Barchart opinions have no history, and the
  Nov–Dec 2025 rows can't be verified.
