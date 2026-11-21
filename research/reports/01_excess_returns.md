# 01: Excess returns and trading costs

*Run 2026-11-20 with `python research/excess_returns.py`. Outputs are in `research/output/`.*

## Question

The tracker shows earnings-week trades averaging a positive 5-day return. Is that a return
**in excess of** the market and the stock's own sector, and does it survive trading costs?

## Method

- **Trades:** every tracker row with `Date Check = OK` and a BMO/AMC timing (562). 548 have
  Yahoo price history, and 541 have a complete window at every holding length. Entries run from
  2025-10-31 to 2026-04-30 and fall in **24 distinct weeks**.
- **Timing (README rule):** BMO trades buy at the prior day's close and AMC trades at the
  earnings-day close. Each holding length h exits at the close of trading day h. Returns are
  recomputed from daily adjusted closes rather than read from the tracker (see "Tracker timing
  bug" below).
- **Benchmarks over the same window:**
  - SPY.
  - The sector SPDR ETF.
  - A market model: beta estimated from trading days −260 to −21 before entry (at least 120
    days), and abnormal return = r − β·r_SPY, summed daily.
- **Statistics:** trades in the same week share market moves, so they aren't independent. The
  main test averages all trades entered in a week into one weekly return and runs a t-test
  across the 24 weeks. A bootstrap that resamples whole weeks (10,000 draws) gives the 95%
  confidence interval for the trade-level mean.
- **Costs (assumed):** no quote data was available, so round-trip costs are assumed by market
  cap: 10 bps for ≥$10B, 20 bps for $2–10B, 40 bps for $300M–2B, and 100 bps below $300M.
  Results also report the **break-even cost**, the round-trip cost that takes the mean to zero.

## Results

5-day hold (the strategy's exit), no stop-loss:

| Benchmark | Mean | Median | Hit rate | Weekly t (24 wks) | 95% CI (week bootstrap) | Mean after costs | Break-even |
|---|---|---|---|---|---|---|---|
| None (raw) | +2.59% | +1.19% | 56.9% | 2.50 | +1.23% to +4.12% | +2.26% | 259 bps |
| SPY | +2.26% | +0.80% | 53.0% | 2.21 | +0.80% to +3.98% | +1.94% | 226 bps |
| Sector ETF | **+1.97%** | +0.48% | 52.5% | **2.00** | +0.71% to +3.36% | +1.64% | 197 bps |
| Market model | +2.24% | +0.83% | 54.9% | 2.21 | +0.85% to +3.92% | +1.91% | 224 bps |

**Robustness (5-day, sector excess):**
- Capping returns at the 1st and 99th percentiles (winsorizing): +1.82%.
- Dropping the 5 best trades: +1.32%.

**By holding length (sector excess):**

| Hold | 1 day | 3 days | 5 days | 10 days |
|---|---|---|---|---|
| Mean | +1.79% | +1.79% | +1.97% | +2.25% |

**By market cap (5-day sector excess):**

| Bucket | Trades | Mean | Weekly t |
|---|---|---|---|
| Large (≥$10B) | 149 | +0.71% | 1.05 |
| Mid ($2–10B) | 161 | +2.85% | 2.04 |
| Small ($300M–2B) | 170 | +1.11% | 0.13 |
| Micro (<$300M) | 61 | +5.09% | 1.13 |

## What this says

1. **There is excess return over the market and the sector, and it's not just beta.** The
   market-model and sector-adjusted means are close to the raw mean minus the market's move.
2. **Costs don't explain it.** The break-even is about 2% round trip, far above any realistic
   cost for closing-auction trades in stocks above $300M.
3. **The return comes almost entirely from the earnings reaction itself.** About 90% of the
   5-day sector excess (+1.79% of +1.97%) arrives on day 1, the first session after the
   announcement. This is a bet on the announcement, not post-earnings drift, so the stop-loss
   and 5-day exit change the outcome far less than the entry does.
4. **It's skewed and concentrated.** The median trade beats its sector by only +0.48%, and
   the hit rate is 52.5%. The mean depends on a right tail of big winners. Large caps, the
   names you could trade in size, show +0.71% with no statistical significance.
5. **The evidence is promising, not proven.** A weekly t of about 2.0 comes from 24 weeks in
   one six-month market regime, and this is the same sample the rules were designed on.
   Anything chosen in later sessions also needs testing on data it wasn't chosen on.

## Problems found

- **Tracker timing bug.** For BMO trades the tracker's `nD Return` columns start counting the
  day *after* earnings. The tracker's BMO `5D Return` matches a recomputed 6-day hold on 261 of
  261 trades, and a 5-day hold on only 21%. AMC trades match the 5-day hold exactly. The bug is
  in `get_ret` in `Earnings Momentum Model.py`: the earnings date is a timestamp, and the code
  takes prices strictly after it, which drops the earnings day itself.
  Not fixed yet: changing it would rewrite the historical tracker columns.
- **Signal timing is unverifiable for most rows.** 373 of the 541 trades have no
  `Date Added`, so there's no proof the screen fired before the entry close. The 168 trades
  with a known add time show +1.62% (weekly t 2.14). The effect doesn't come only from the
  unverifiable rows, but the 373 can't be confirmed free of look-ahead bias (using information
  that wasn't available yet).
- **Missing prices.** 14 tickers (2.5% of events) return no Yahoo data. Their outcomes are
  missing, which can bias results if they were delisted after bad news.

## Open questions for the next sessions

- **Control group:** do *all* stocks reporting that week show the same reaction, or only
  those that pass the SMA and Barchart screen? Without a control group, these results can't
  separate the signal from a general earnings-week premium.
- **Walk-forward tests:** check the stop level, exit day and classifier on data they weren't
  fitted to (session 2).
- **Classifier evaluation:** measure accuracy and calibration, and check whether the win
  probability ranks trades better than chance (session 3).
