# Factor Research Platform

Cross-sectional equity factor research on the point-in-time S&P 500: build signals, test whether
they predict next-month returns, turn them into long-short portfolios, and check the results
against the published Ken French factors.

This folder also holds a second, self-contained study in [trend-following/](trend-following/README.md):
a time-series momentum replication across 20 asset-class ETFs (2007–2026), validated against live
managed-futures funds. It has its own code, data and reports, and shares nothing with the equity
pipeline.

## Status (2026-11-20)

- **Data layer:** point-in-time S&P 500 membership (1996 to Aug 2026, 1,204 tickers), daily
  prices from yfinance, monthly returns, Ken French factors (FF3, FF5, momentum, reversal).
- **Fundamentals:** annual SEC EDGAR XBRL facts for 623 of 860 members since 2009, used only
  from their first filing date (no lookahead); split-adjusted share counts for market cap; filter
  for XBRL unit errors. See [reports/fundamentals_coverage.md](reports/fundamentals_coverage.md).
- **Signals:** 12-1 momentum, 1-month reversal, 12-month volatility, book-to-market, earnings
  yield, operating and gross profitability, asset growth, size.
- **Pipeline validation passed:** our equal-weighted long-short portfolios track the published
  Ken French factors: momentum 0.93 with UMD and reversal 0.84 with ST_Rev (2000-2026);
  book-to-market 0.81 with HML, asset growth 0.65 with CMA, size 0.53 with SMB, operating
  profitability 0.46 with RMW (2011-2026). See [reports/factor_validation.md](reports/factor_validation.md).
- **No signal is significant on its own in this sample** (all long-short t-stats below 1.1), which
  matches French's own factors over the same years (HML -1.1%/yr since 2011).
- **Research layer done:** sector neutralization (FF12 from SEC SIC), IC decay, Fama-MacBeth,
  four walk-forward combinations, costs and FF5+momentum attribution. Result: no combination beats
  costs inside the S&P 500 from 2014 to 2026 (best: equal weight, IC 0.011, −1.0%/yr net).
  Read the one-page [research note](reports/RESEARCH_NOTE.md).
- **Pre-registered follow-ups** ([PREREGISTRATION.md](PREREGISTRATION.md),
  [test_results.md](reports/test_results.md)): earnings surprise showed no effect as a monthly
  quintile portfolio; lower-turnover construction didn't rescue the composite. A diagnostic found
  extreme-decile post-earnings drift (+1.15% over 60 trading days, t 2.07), now pre-registered as
  Test C2 for CRSP.
- **CRSP/Compustat loader written, untested on real data** (WRDS account pending). Smoke-tested
  on a fake connection; Test C (broad universe, holdout 2018 on) runs once access arrives.
- **Known data gap:** free prices cover 67% of member-months, about half before 2005 and 90%+
  after 2021, because Yahoo drops delisted stocks. See
  [reports/data_coverage.md](reports/data_coverage.md). CRSP via WRDS would close this gap.

## Run

```
pip install -r requirements.txt
python scripts/build_data.py        # universe, prices (cached), French factors, coverage report
python scripts/build_fundamentals.py  # SEC XBRL fundamentals (cached) and fundamental signals
python scripts/validate_factors.py  # factor check against Ken French
python scripts/run_research.py      # research layer -> reports/research_results.md
```

Pre-registered tests on free data: `python scripts/run_tests.py` (Tests A and B).

### CRSP / Compustat (Test C)

```
pip install wrds
python scripts/build_crsp.py --probe --user <wrds username>   # first run: password + Duo, save .pgpass
python scripts/build_crsp.py                                  # builds data/crsp/ (--legacy for old msf tables)
FACTORLAB_SOURCE=crsp python scripts/validate_factors.py
FACTORLAB_SOURCE=crsp python scripts/run_research.py          # exploration only, 1990-2017
FACTORLAB_SOURCE=crsp python scripts/run_test_c2.py           # SUE event drift, exploration
# only after every choice is frozen, once:
FACTORLAB_SOURCE=crsp python scripts/run_research.py --holdout
FACTORLAB_SOURCE=crsp python scripts/run_test_c2.py --holdout
```

Reports for CRSP runs go to `reports/crsp/`.

SEC asks automated clients for a contact in the User-Agent. Set `SEC_USER_AGENT="Name email"`
to use your own; otherwise a generic label is sent.

## Layout

| Path | What it does |
|---|---|
| `src/factorlab/universe.py` | Point-in-time S&P 500 membership from github.com/fja05680/sp500 |
| `src/factorlab/prices.py` | yfinance download (cached per chunk), monthly panel, bad-ticker list |
| `src/factorlab/french.py` | Ken French Data Library download and parsing |
| `src/factorlab/signals.py` | Signals formed at month end t from data up to t |
| `src/factorlab/evaluate.py` | Forward returns, rank IC, quantile portfolios, performance stats |
| `src/factorlab/fundamentals.py` | SEC XBRL companyfacts, first-filed values, share counts |
| `src/factorlab/sectors.py` | SEC SIC code to Fama-French 12 industries |
| `src/factorlab/preprocess.py` | Signal panel, winsorize / z-score / sector-neutralize |
| `src/factorlab/research.py` | IC decay, Fama-MacBeth, walk-forward combos, long-short with costs, attribution |
| `src/factorlab/portfolio.py` | Quintile, 3-month overlapping and banded portfolio construction |
| `src/factorlab/crsp.py` | CRSP/Compustat via WRDS (CIZ or legacy), same output files as the free source |
| `PREREGISTRATION.md` | Tests, samples and success criteria fixed before running, with change log |
| `scripts/` | Entry points; `data/` is rebuildable and gitignored |
| `reports/` | Generated markdown reports and charts |

## Roadmap

1. ~~Data layer and validation against French~~
2. ~~Fundamentals with filing-date lag (SEC EDGAR): value, profitability, investment, size~~
   Next: Compustat via WRDS (account pending); French-style operating profitability
   (revenue - COGS - SG&A - interest) / book equity to lift the 0.46 RMW correlation; quarterly TTM
3. ~~Cleaning: winsorize, z-score, sector-neutralize~~
4. ~~Evaluation: IC decay, Fama-MacBeth regressions, turnover~~
5. ~~Portfolio: dollar-neutral long-short with transaction costs~~
6. ~~Signal combination: equal weight, IC-weighted, ridge / LightGBM, walk-forward only~~
7. ~~Attribution: alpha vs Fama-French 5 + momentum~~
8. ~~Research write-up~~ ([RESEARCH_NOTE.md](reports/RESEARCH_NOTE.md))
9. Next (fixed before further testing): CRSP/Compustat; universe beyond the S&P 500;
   lower-turnover construction; earnings-surprise signal from the Earnings Momentum Model
