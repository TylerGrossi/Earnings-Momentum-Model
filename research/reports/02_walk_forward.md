# 02: Walk-forward test of the stop-loss and exit rule, and edge stability

*Run 2026-11-20 with `python research/walk_forward.py` (after `excess_returns.py`). Outputs are in
`research/output/`.*

## Question

The README rule (−10% stop, exit at the day-5 close) was chosen by looking at the data. How
much of a chosen rule's backtest holds up on weeks it wasn't chosen on? And is the edge stable
over time?

## Method

- **Rules:** 75 in total. Stops are none, −2% to −10% in 1-point steps, and −12% to −20% in
  2-point steps, each combined with an exit at the close of day 1 to 5.
- **Data:** hourly paths for 548 trades over 24 entry weeks.
- **Stop fills:** stops are checked on hourly bar closes, because the data has no intraday
  lows. If the first bar is already below the stop, the trade takes that bar's loss (a gap
  down). Otherwise it fills at the stop. This is the same convention as the app's stop-loss
  tab.
- **Measure:** return in excess of the sector ETF over the same days. For a stopped trade, the
  benchmark runs to the close of the stop day.
- **Three tests:**
  1. **In-sample:** every rule on the full sample.
  2. **Walk-forward:** after an 8-week warm-up, each week uses the rule with the best mean on
     all *earlier* weeks (16 test weeks, 388 trades).
  3. **Probability of backtest overfitting** (CSCV, Bailey et al. 2017): the 24 weeks are split
     into 8 blocks, giving 70 ways to choose 4 blocks as in-sample. For each split, find the
     in-sample best rule and rank it on the other 4 blocks. PBO is the share of splits where
     that rule ranks in the bottom half.

## Results

### The rule choice barely matters, and choosing it doesn't carry over

- **In-sample spread:** all 75 rules fall between +1.74% and +2.27% mean sector excess. The best
  in hindsight is **SL −9% / day 4** (+2.27%). The README rule ranks 19th (+2.11%), and no stop
  with a day-5 exit ranks 30th (+1.98%).
- **Walk-forward, same 16 test weeks:**

| Rule | Mean excess | After assumed costs | Weekly t |
|---|---|---|---|
| Walk-forward pick (chosen on past weeks only) | +0.85% | +0.54% | 1.43 |
| README rule, SL −10% / day 5 | +0.93% | +0.63% | 1.33 |
| No stop / day 5 | +1.05% | +0.75% | 1.47 |
| SL −9% / day 4 (full-sample best, uses hindsight) | +1.10% | +0.80% | 1.66 |

- **PBO = 0.64.** In 64% of the splits, the in-sample best rule ranked in the bottom half out
  of sample. Its average rank was the 40th percentile. Choosing a stop level and exit day from
  this data does slightly *worse* than picking at random.

So the stop-loss and exit-day tuning adds nothing that survives. That fits session 1: about 90%
of the excess arrives on day 1, before most stops or exits come into play. The simplest rule
(no stop, day-5 exit) did as well as anything chosen. A stop may still be worth keeping for
risk control, but this data can't justify a particular level.

### The edge is not stable: it's concentrated in the first weeks

No stop / day 5, sector excess:

| Period | Weeks | Trades | Mean | Median | Weekly t |
|---|---|---|---|---|---|
| Before 2026-01-09 (no `Date Added`) | 9 | 165 | +4.16% | +1.51% | 2.61 |
| From 2026-01-09 | 16 | 383 | +1.05% | −0.20% | 1.25 |

By month: Nov 2025 +3.95%, Dec +6.09%, Jan 2026 +2.55%, Feb +0.24%, Mar +2.44%, Apr +0.92%.
The medians are negative in Feb, Mar and Apr.

**Why the split date matters:**
- `returns_tracker.csv` was first committed on 2025-12-24.
- No trade entered before 2026-01-09 has a `Date Added`.
- So the strongest period is the one where the repo can't show the trades were recorded
  before they happened.
- If those rows were added afterwards, and the screen (SMA20 > SMA50, Barchart buy) was
  evaluated after the price move, the early returns could include look-ahead bias.
- **This needs Tyler's confirmation:** were the Nov–Dec 2025 trades logged live by the nightly
  scraper, or backfilled?

**In the verifiable period (from 2026-01-09: 379 trades, 16 weeks, 5-day hold):**

| Benchmark | Mean | Median | Weekly t |
|---|---|---|---|
| Raw | +1.50% | +0.68% | 1.48 |
| SPY | +1.14% | −0.21% | 1.24 |
| Sector ETF | +1.10% | −0.04% | 1.27 |
| Market model | +1.23% | +0.23% | 1.44 |

None of these is statistically different from zero at conventional levels.

## Conclusion

- **The stop-loss and exit-day tuning is overfitting.** PBO is 0.64, and the walk-forward pick
  did no better than the untuned rule.
- **The excess return rests on the Nov–Dec 2025 trades, which can't be verified.** In the
  verifiable period it's about +1.1% on average but zero for the median trade, with t ≈ 1.3.
- **What can honestly be said:** the strategy showed positive excess returns early on, but
  they haven't been confirmed in the period where trades were recorded live. Tests in the
  verifiable period don't distinguish it from zero.

## Next

- **Session 3:** test the win-probability classifier the same way, training on past weeks
  only.
- **If the Nov–Dec rows were logged live**, the full-sample results stand, with the stability
  caveat. **If they were backfilled**, report only the verifiable period.
