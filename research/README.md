# Research: does the earnings-momentum edge hold up?

Out-of-sample tests of the strategy. The scripts only read `returns_tracker.csv` and
`hourly_prices.csv`; they never write to them.

Run in order from the repo root:

```bash
python research/excess_returns.py    # 01: benchmarks, costs, timing reconciliation
python research/walk_forward.py      # 02: stop/exit walk-forward, overfitting, stability
python research/classifier_eval.py   # 03: win-probability classifier, walk-forward
python research/one_day_pop.py       # 04a: 1-day hold, sector exclusion, day-1 exit time
python research/oos_rebuild.py       # 04b: rebuilt screen on unseen May-Nov 2026 seasons
python research/long_history.py      # 05: same rule on the S&P 1500, 2010-2026
python research/alt_signals.py       # 06: trackable buy signals (beats, analysts, technicals)
```

Prices are cached in `research/cache/` (gitignored); add `--refresh` to `excess_returns.py` to
re-download them. Results go to `research/output/`.

| Report | Finding |
|---|---|
| [01 Excess returns](reports/01_excess_returns.md) | Full sample: +1.97% mean 5-day return over the sector ETF (weekly t 2.00, 24 weeks), break-even cost about 197 bps. About 90% arrives on day 1. The tracker's BMO returns are one day too long. |
| [02 Walk-forward](reports/02_walk_forward.md) | Stop/exit tuning doesn't survive out of sample (PBO 0.64). The edge sits in Nov–Dec 2025, before any row has a `Date Added`. From 2026-01-09: +1.10% mean, −0.04% median, t 1.27. |
| [03 Classifier](reports/03_classifier.md) | Stored win probabilities are in-sample fits (AUC 0.937). Walk-forward AUC is 0.462 (CI 0.40–0.51): no skill. |
| [04 1-day pop + out-of-sample](reports/04_one_day_pop_oos.md) | Pre-registered test on May–Nov 2026 (141 trades, 15 weeks): screened stocks +1.31% 1-day sector excess (p 0.051), control −0.06%, difference p 0.048. Excluding Technology doesn't help. |
| [05 Long history](reports/05_long_history.md) | Same rule, S&P 1500, 2010–2025 (9,095 screened events, 658 weeks): +0.27% mean, weekly t 0.61; screened minus control −0.13%/week. **No edge.** Report 04's result was a lucky window. |
| [06 Alternative signals](reports/06_alt_signals.md) | Prior beat, analyst upgrades, a Barchart-style technical composite and pre-event strength all *lag* their "off" groups in 2012–18 and 2019–25; none passes. All four flip positive only in Nov 2025–Nov 2026, a momentum regime that likely explains the live results. |

## Summary

The earnings-week trades beat their sector on average, and costs don't explain it. But the
excess comes mostly from the first two months, which can't be verified as recorded live, and
the median trade in the verifiable period doesn't beat its sector. Neither the stop/exit
tuning nor the classifier adds value out of sample.

The 1-day hold with the SMA screen alone looked borderline-positive on 15 unseen weeks (report
04), but failed over 16 years of S&P 1500 data (report 05): screened earnings reports did no
better than unscreened ones. Four other trackable buy signals failed the same way (report 06).
In 2012–2025, bullish signals went with slightly weaker earnings-day reactions, but in
2025–26 they briefly worked, which is the likeliest explanation for the live results.
