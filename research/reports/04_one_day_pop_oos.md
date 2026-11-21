# 04: The 1-day pop: in-sample checks, and an out-of-sample test on unseen seasons

*Run 2026-11-20 with `python research/one_day_pop.py` and `python research/oos_rebuild.py` (both
after `excess_returns.py`). Outputs are in `research/output/` (`pop_*.csv`, `oos_*.csv`).*

## Question

Reports 01–02 found that about 90% of the excess return arrives on day 1. Is a **1-day hold**
(buy at the pre-announcement close, sell at the next close) a strategy that works? And does
skipping Technology help?

## Part A: on the tracked trades (in-sample, so a hypothesis only)

**1-day return in excess of the sector ETF:**

| Universe | Period | Trades | Mean | Median | Weekly t |
|---|---|---|---|---|---|
| All sectors | Full sample | 541 | +1.79% | +0.61% | 2.79 |
| All sectors | From 2026-01-09 | 379 | +0.93% | +0.37% | 1.85 |
| Ex-Technology | Full sample | 465 | +1.60% | +0.66% | 3.78 |
| Ex-Technology | From 2026-01-09 | 332 | +0.75% | +0.42% | 2.35 |

**Technology isn't a drag.**
- It's one of the best sectors on day 1: +2.96% (+2.19% from 2026-01-09).
- Dropping it *lowers* the mean. From 2026-01-09, dropping Technology gives the 10th-best mean
  of the 11 possible one-sector exclusions.
- The t-stat rises only because Technology trades are the most volatile.
- The sectors with negative day-1 excess are Basic Materials, Energy and Real Estate (−1.7%,
  −0.8%, −0.4%). That's a hindsight pick, so it wasn't adopted.

**Don't sell early on day 1** (ex-Technology, from 2026-01-09, sector-adjusted with hourly ETF
prices):

| Exit | Mean |
|---|---|
| 10:30 (first hourly bar) | +0.44% |
| 12:30 | +0.79% |
| 15:30 | +0.77% |
| 16:00 close | +0.83% |

The reaction holds or builds through the session.

## Part B: out-of-sample test (pre-registered)

The live scraper stopped adding trades around 2026-05-01. The May–Nov 2026 earnings seasons
have never been looked at by any rule in this research. `oos_rebuild.py` rebuilds the screen
point-in-time on those seasons.

### Pre-registered plan

The hypotheses were written into the script's docstring before any out-of-sample number was
computed.

- **H1:** screened trades, all sectors, 1-day hold, have mean sector excess > 0 (weekly
  t-test, one-sided, 5%).
- **H2:** screened trades beat a control group on the same measure.
- **Secondary, reported only:** ex-Technology, the 5-day hold, and results after costs.

### Rebuilt screen

- **Earnings dates:** from yfinance. BMO if the ET hour is before 12, AMC if 16 or later.
- **Signal:** at the close *before* the entry day, SMA20 > SMA50, with SMA20 having crossed
  above SMA50 within the last 10 trading days.
- **Calibration:** among the 169 tracked trades with a known `Date Added`, 91% had crossed
  within 10 days and 96% were above.
- **Not included:** the Barchart "Buy" opinion. It has no history, so this tests the
  moving-average screen alone.

### Universe and control group

- **Universe:** the 538 tickers that ever appeared in the tracker.
- **Control group:** the same companies' earnings in the same seasons when the screen did
  *not* fire.
- Neither group is a random sample of the market.

### Results (May 4 – Nov 18, 2026)

| Group | Trades | Weeks | Mean 1-day excess | Median | Hit rate | Weekly t | One-sided p | 95% CI (week bootstrap) |
|---|---|---|---|---|---|---|---|---|
| **Screened, all sectors** | 141 | 15 | **+1.31%** | +1.09% | 56.7% | 1.76 | 0.051 | +0.34% to +2.56% |
| Screened, after assumed costs | 141 | 15 | +1.04% | +0.99% | 55.3% | 1.57 | 0.070 | +0.04% to +2.30% |
| Control (screen didn't fire) | 1,053 | 29 | −0.06% | −0.09% | 49.4% | −0.78 | – | −0.69% to +0.41% |
| Screened, ex-Technology | 128 | 13 | +1.12% | +0.83% | 55.5% | 1.83 | 0.046 | −0.10% to +2.24% |
| Screened, 5-day hold | 141 | 15 | +0.38% | +0.88% | 56.0% | 1.21 | 0.122 | −1.09% to +2.17% |

**Screened minus control, weekly difference in 1-day excess:** +2.54%, t 1.79, one-sided
p 0.048, over 15 weeks.

**Descriptive checks (not used to pick a rule):**
- **By season:** May–Aug +1.74% (75 trades); Oct–Nov +0.82% (66 trades).
- **By timing:** AMC +1.62%; BMO +1.01%.
- **Outliers:**
  - Dropping the 5 best trades: +0.49%.
  - Dropping the 5 best and 5 worst: +1.60%.
  - Winsorized at 1%/99%: +1.36%.
- **Sectors** (small samples): Technology was again among the best, at +3.18% on 13 trades.
  Energy was again negative (−1.7%, 10 trades). Healthcare flipped from +3.6% in-sample to
  −1.5%. Sector patterns aren't stable enough to filter on.

### In-sample caveat on the rebuilt screen

Applied to Nov 2025–Apr 2026, the rebuilt screen fired on 380 of the 548 tracked trades. Its
screened trades earned only +0.54% on day 1, against +1.79% for the live trades. Part of the
live in-sample edge therefore came from trades the SMA rule alone doesn't reproduce: the
Barchart filter, date differences, or the unverifiable Nov–Dec 2025 rows.

The in-sample control group is also contaminated. It contains the 124 live trades the rebuild
missed. Use the out-of-sample comparison, not the in-sample one.

## Conclusion

- **The out-of-sample results favor the 1-day pop, but they're borderline.**
  - On 141 trades in two seasons no rule had seen, screened stocks beat their sector by
    +1.31% on day 1, with a positive median (+1.09%) and a 56.7% hit rate.
  - The same companies on reports where the screen didn't fire earned nothing (−0.06%).
  - H2 (screened beats control) passes at one-sided 5% (p 0.048). H1 just misses (p 0.051).
  - With two primary tests, neither clears a multiple-testing correction (2.5% each).
- **What survives is the screen plus the 1-day hold. Skipping Technology doesn't survive.**
  Technology was among the best sectors in both samples, and ex-Technology was lower than all
  sectors out of sample too. The 5-day hold gave back most of the day-1 gain, consistent with
  reports 01–02.
- **The evidence is thin and tail-heavy.** It's 15 weeks, and dropping the 5 best trades cuts
  the mean to +0.49%.

**A defensible, frozen rule to forward-test:**
- **Universe:** stocks reporting earnings this week.
- **Signal:** SMA20 crossed above SMA50 within the last 10 trading days and is still above,
  checked at the prior close.
- **Entry:** buy at the close before the announcement (BMO: the day before; AMC: the earnings
  day).
- **Exit:** sell at the next session's close.
- **Scope:** all sectors, no stop, equal weight.

The forward test should log each signal before the entry close and never revise it.
