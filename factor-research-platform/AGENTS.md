# Agent guide: Factor Research Platform

Read [README.md](README.md) first. Root rules in `../../../AGENTS.md` apply (never push, commit only
when asked).

This folder lives inside the Earnings Momentum Model repo (`../`), which is on GitHub; the repo's
own rules in `../` apply too. `trend-following/` is a separate study in this folder, with its own `src/`, `scripts/`,
`data/` and `reports/`. Run its scripts from inside that folder, and read its README before
editing it. Its holdout has already been run once; don't rerun it.

## Research rules (these are the point of the project)

- **No lookahead.** A signal for month end t uses only data available at t. Fundamentals must be
  lagged to their filing date, not their fiscal period end.
- **Point-in-time universe only.** Never filter to today's S&P 500 members.
- **Validate before claiming.** New factors that have a published counterpart are checked against
  Ken French first (`scripts/validate_factors.py`).
- **Report honestly.** Report the data-coverage caveat with any result from yfinance data. No costs
  means say "before costs". Count every variant tested; don't report only the best.
- **Bad data:** add tickers whose Yahoo series is the wrong company to `prices.BAD_TICKERS` with
  a reason, and never silently drop extreme returns that are real (AIG 2008, Ford 2009).

## Data

- `data/` is gitignored and rebuilt by `scripts/build_data.py`. The price download is cached per
  chunk in `data/raw/prices/`; delete a chunk file to refetch it.
- yfinance prints "possibly delisted" for about 37% of tickers; that's expected (see coverage report).
- Run downloaded-data scripts from the project root, not from inside `data/`.

## Pre-registration

- New tests go in `PREREGISTRATION.md` before they are run. Changes after a test has run go in
  its change log with the reason. Never rerun a CRSP `--holdout` after changing anything it
  depends on; that is the one-shot test.
