# Pre-registration

Written 2026-11-20, **before** running any of the tests below. Signal definitions, samples,
variants and success criteria are fixed here. Results go in `reports/` and are reported whatever
they show. Any change after this date is listed in the change log at the bottom with its reason.

Background: the first research pass ([reports/RESEARCH_NOTE.md](reports/RESEARCH_NOTE.md)) found
no combination of nine classic signals beats costs inside the S&P 500 (2014–2026).

---

## Test A: earnings surprise (post-earnings drift), S&P 500, free data

**Hypothesis.** Stocks with large positive earnings surprises keep outperforming those with
large negative surprises in the following months (Bernard and Thomas, 1989).

**Signal (SUE).** Standardized unexpected earnings from quarterly net income in SEC XBRL:
`(NI_q − NI_{q−4}) / std(NI_{q−k} − NI_{q−k−4}, k = 1..8)`, requiring at least 6 of the 8
changes. Q4 net income = fiscal-year net income minus the three reported quarters. Each quarter's
value counts from the date it was **first filed** (10-Q, or 10-K for Q4). Companies usually
announce earnings 2–4 weeks before filing, so this signal is late and the test is conservative.
At month end t a stock's signal is the SUE of its latest quarter filed in the previous 92 days,
otherwise missing.

**Evaluation.** Point-in-time S&P 500 members; formation months from Jan 2012 (8 quarters of
history needed) to the latest; winsorize, z-score, sector-neutralize as in the first pass;
equal-weighted top-minus-bottom quintile, monthly, 10 bps per unit traded; Newey-West t-stats
(6 lags); attribution on Fama-French 5 + momentum. One variant, no tuning.

**Criteria.**
- **Finding:** mean rank IC t ≥ 2 **and** net long-short Newey-West t ≥ 2.
- **Weak / inconclusive:** either t between 1 and 2.
- **No effect:** both t below 1.
- Reported but not used for the verdict: SUE added to the equal-weight composite.

## Test B: lower-turnover construction, S&P 500, free data

**Hypothesis.** The equal-weight composite of the nine signals has a small edge (IC 0.011,
+0.8%/yr gross) that 76% monthly turnover destroys. Slower trading keeps more of it.

**Fixed inputs.** The same equal-weight composite and out-of-sample window (Jan 2014 onward) as
the first pass, 10 bps per unit traded. Exactly two variants:
1. **3-month holding:** each month form the top/bottom-quintile book and hold it 3 months; the
   portfolio is the average of the last 3 books (Jegadeesh-Titman overlapping portfolios).
2. **Banding:** a stock enters the long side in the top 20% and leaves only when it falls out of
   the top 40% (mirror rule for shorts).

**Criteria.** Finding: net Sharpe > 0 with Newey-West t ≥ 2. **Caveat set in advance:** this
composite's results were already seen, so even a positive result here is weaker evidence than
Test A or Test C.

## Test C: broad US universe on CRSP/Compustat (runs once WRDS access is approved)

**Universe.** US common stocks on NYSE, AMEX and NASDAQ (CRSP share codes 10/11, or the CIZ
equivalent), market cap above the NYSE 20th percentile each month (excludes microcaps, following
Hou, Xue and Zhang, 2020). Delisting returns included.

**Signals.** The same nine signals, with Compustat versions of the fundamentals (Fama-French book
equity and operating profitability), plus SUE dated by Compustat's actual announcement date
(`rdq`). Annual Compustat data counts from 6 months after fiscal year end.

**Samples.** Exploration: formation months Jan 1990 – Dec 2017. **Holdout: Jan 2018 onward,
run once, only after every choice is frozen.** Walk-forward models start scoring in Jan 1995.

**Criteria.**
- A single signal counts as replicated if its exploration-sample long-short t ≥ 2 **and** the
  holdout return has the same sign.
- A combination counts as a finding only on the holdout: net Sharpe > 0 with Newey-West t ≥ 2.
- Anything new (not a published anomaly) needs t ≥ 3 (Harvey, Liu and Zhu, 2016).

---

## Change log

**2026-11-20, after Tests A and B ran (results in reports/test_results.md).** Test A's verdict
stands: NO EFFECT. A diagnostic event study run afterwards to check that the SUE signal is valid
(reports/sue_event_check.txt) found: SUE correlates 0.10 (rank) with the market-adjusted return in
the 30 trading days before filing, so the signal captures real surprises; and the top-minus-bottom
**decile** return over the 60 trading days after filing was +1.15% (t 2.07, 59 quarters). This
was not pre-registered (decile cut and window chosen during the diagnostic), so it is a
hypothesis, not a finding. It is added to Test C as an extra pre-registered test before any CRSP
data is seen:

> **Test C2: SUE event-time drift.** Events: Compustat quarterly announcements (`rdq`) in the
> Test C universe. SUE as in Test A using `ibq`. Each calendar quarter, sort events into SUE
> deciles; measure the market-adjusted return from trading day +2 to +61 after `rdq`.
> Statistic: quarterly top-minus-bottom decile mean, t-stat across quarters. Exploration
> 1990–2017; holdout 2018 on, run once. Replicated if exploration t ≥ 2 and holdout mean > 0.
> Also report the monthly-portfolio version (top-minus-bottom decile, held 3 months, 10 bps).
