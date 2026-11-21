# 06: Alternative buy signals that can be tracked historically

*Run 2026-11-20 with `python research/alt_signals.py` (after `long_history.py`). Outputs are
`research/output/alt_signals_*.csv`.*

## Question

The Barchart opinion has no history. Can a buy signal that does have history pick earnings
reports with a better day-1 reaction?

## Method

- **Signals** (pre-registered in the script docstring before running). Each uses only data
  available before the entry day.

| Signal | Definition | Source |
|---|---|---|
| S1 prior beat | The previous report beat the EPS estimate | Yahoo earnings history |
| S2 analyst momentum | More upgrades than downgrades in the prior 90 days | yfinance `upgrades_downgrades` (from 2012) |
| S3 technical opinion | Barchart-style composite of 10 trend votes (price vs SMA20/50/100; SMA20>50, 20>100, 50>100, 50>150, 50>200; CCI 40/60 > 0); Buy = 8+ of 10 bullish | Prices |
| S4 pre-event strength | Stock beat its sector ETF over the 20 days before entry | Prices |

- **Trade:** buy at the close before the announcement, sell at the next close.
- **Outcome:** 1-day return in excess of the sector ETF.
- **Universe:** the S&P 1500 events from report 05.
- **Split:** development 2012–2018 and holdout 2019 to Oct 2025.
- **Pass rule:** signal-on events must beat signal-off events (weekly difference > 0) in both
  periods, and the holdout difference must be significant after a Holm correction for 4 tests.

## Results

Weekly difference between signal-on and signal-off events, 1-day sector excess:

| Signal | Development diff (t) | Holdout diff (t) | Signal-on mean, holdout | Signal-off mean, holdout | Pass |
|---|---|---|---|---|---|
| S1 prior beat | −0.14% (−0.70) | −0.14% (−0.55) | +0.27% | +0.39% | No |
| S2 analyst momentum | −0.02% (−0.09) | −0.55% (−2.10) | +0.06% | +0.36% | No |
| S3 technical opinion | −0.45% (−2.24) | −0.41% (−1.83) | +0.20% | +0.34% | No |
| S4 pre-event strength | −0.40% (−1.96) | −0.27% (−1.38) | +0.27% | +0.33% | No |

- **No signal passes.** Every one does *worse* than its "off" group in both periods.
- **After assumed costs,** no signal-on group earns more than +0.05% per trade in either
  period.
- **Combined with the SMA cross screen,** no signal produced a consistently positive result
  either.

**The window already seen (Nov 2025 – Nov 2026), reference only.** All four signals flip
*positive*:

| Signal | Signal-on minus signal-off | t |
|---|---|---|
| S1 prior beat | +0.91% | 1.91 |
| S2 analyst momentum | −0.70% | −1.61 |
| S3 technical opinion | +1.17% | 1.81 |
| S4 pre-event strength | +1.35% | 2.31 |

## What this says

1. **No historically trackable bullish signal improves the day-1 earnings reaction in
   2012–2025.** Strength going into the report (trend, technical opinion, run-up, analyst
   upgrades, a prior beat) tends to come with a slightly *weaker* reaction. That fits an
   expectations story: good news is already priced into strong stocks.
2. **2025–26 was an unusual period.** In the same window as the live tracker, momentum-type
   signals worked on earnings day (S3 and S4 at t ≈ 1.8–2.3), the opposite of the prior 14
   years. This is the most likely explanation for the live strategy's strong early results.
   It traded a momentum-into-earnings pattern during the one stretch where that pattern paid,
   which says nothing about whether the pattern persists.
3. **This is evidence against the Barchart "Buy" as a lasting edge too, though indirect.** S3
   is built from the same kind of trend indicators Barchart uses, and its "Buy" lagged in both
   historical periods. It's only an approximation of Barchart's proprietary opinion, so it
   can't rule Barchart out.
4. **Don't flip to a contrarian rule on this evidence.** Weak stocks doing better is partly
   survivorship bias. This universe is *today's* index members, so the stocks that were weak
   in the past and are still here are mostly the ones that recovered. A contrarian
   earnings-day test needs a universe that includes delisted and dropped companies.

## Conclusion

Neither the Barchart signal's stand-ins nor any other trackable signal turns the earnings-day
pop into a tested edge. The live results look like a regime effect from 2025–26, not a durable
signal.
