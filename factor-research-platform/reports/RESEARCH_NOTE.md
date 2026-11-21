# Do classic equity factors still work inside the S&P 500? (2011–2026)

## Question

Nine published anomaly signals (momentum, short-term reversal, low volatility, book-to-market,
earnings yield, operating and gross profitability, asset growth, size): do they predict
next-month returns among S&P 500 members, alone or combined, after trading costs?

## Data and method

- **Universe:** stocks in the S&P 500 at each month end (point-in-time membership, 1,204 tickers
  since 1996), so removed and delisted stocks stay in the sample while they were members.
- **Prices:** yfinance, split- and dividend-adjusted. **Fundamentals:** SEC EDGAR XBRL, each
  value used only from the date it was first filed. Market cap uses split-adjusted share counts.
- **Pipeline validation:** our long-short portfolios correlate with the published Ken French
  factors: momentum 0.93, value 0.81, investment 0.65, size 0.53, profitability 0.46
  ([factor_validation.md](factor_validation.md)).
- **Cleaning:** each month, winsorize at 1/99%, z-score, and neutralize against the Fama-French
  12 industries (from SEC SIC codes).
- **Portfolios:** equal-weighted long top quintile / short bottom quintile, monthly, dollar
  neutral, 10 bps per unit traded. Newey-West t-stats.
- **Combinations:** equal weight (nothing fitted), trailing-IC weights, ridge and LightGBM. The
  fitted models are refit every January on data whose outcome was already known, then tested on
  the next 12 months. Out-of-sample period: Jan 2014 to Sep 2026 (151 months).
- **Plumbing checks:** a signal equal to next month's return earns +710%/yr (IC 1.0); a random
  signal earns −0.3%/yr (t −0.23); the same "cheat" lagged one month earns −1.4%/yr (t −0.11).
  The backtest finds information when it exists and doesn't leak future data.

## Findings

1. **No single signal is significant on its own.** The strongest IC is gross profitability
   (0.012, t 1.41). The best long-short after costs is low asset growth, +1.3%/yr (Sharpe 0.24).
2. **Operating profitability is the only signal significant given the others** (Fama-MacBeth
   +7.8 bp/month per standard deviation, t 2.36). With nine signals tested it doesn't clear a
   multiple-testing bar (Bonferroni about t 2.8), so it is suggestive, not established.
3. **Combining helps prediction a little but not enough to pay for trading.** The equal-weight
   composite has the best IC (0.011, t 1.89) but earns +0.8%/yr gross and −1.0%/yr after costs.
   Turnover is 76% of the book per month, driven by 1-month reversal (156% per month on its own).
4. **Fitting weights made results worse, not better.** Trailing-IC weights −6.3%/yr net, ridge
   −4.3%, LightGBM −1.9%. Signals that worked recently tended to stop working, so models that
   chased them lost.
5. **What return there is comes from known factors.** Regressed on Fama-French 5 + momentum, the
   equal-weight portfolio loads on profitability (0.41), value (0.26) and momentum (0.18); its
   alpha is −2.8%/yr net (t −2.2).
6. **Sector neutralization helped.** Equal-weight net return improved from −2.4%/yr to −1.0%/yr
   (Sharpe −0.20 to −0.09) when industry bets were removed.

## Interpretation

This matches the literature: anomaly returns shrink after publication (McLean and Pontiff, 2016)
and are weakest in large, liquid stocks, which is all the S&P 500 is. The pipeline's value is
that it reaches this answer honestly, with no lookahead, a point-in-time universe, costs and an
out-of-sample test, instead of producing a flattering backtest.

## Caveats

- Free data misses many later-delisted stocks (67% of member-months covered overall; fundamentals
  about 60% in 2012–15, 80% from 2021). CRSP/Compustat via WRDS will close this.
- All 14 variants tested are reported. Model settings were fixed in advance.
- Attribution t-stats ignore estimation error in the factor loadings.

## Follow-up: pre-registered Tests A and B (2026-11-20)

Fixed in advance in [PREREGISTRATION.md](../PREREGISTRATION.md); full results in
[test_results.md](test_results.md).

- **Test A, earnings surprise (SUE), S&P 500: no effect.** IC 0.0007 (t 0.11); quintile
  long-short −2.2%/yr net (t −1.15). A follow-up check confirmed the signal is valid: it
  correlates 0.10 with the stock's market-adjusted move in the 30 trading days before filing,
  when earnings are announced. That check also found top-minus-bottom **decile** drift of +1.15%
  over the 60 trading days after filing (t 2.07, 59 quarters). Because that cut was chosen after
  seeing data, it is recorded as a hypothesis and pre-registered as Test C2 for CRSP/Compustat,
  where Compustat's actual announcement dates make the test sharper.
- **Test B, lower turnover: not a finding.** 3-month overlapping holdings cut turnover from 76%
  to 31% a month and banding to 38%, but the gross edge fell with it: net −0.65%/yr and
  −0.94%/yr (Sharpe −0.06 and −0.12). The composite has no edge to protect inside the S&P 500.

## Next steps (decided before looking further)

1. Rerun on CRSP/Compustat once WRDS access is approved.
2. Broaden the universe beyond the S&P 500 (Russell 1000/3000 equivalents via CRSP), where
   anomalies are documented to be stronger.
3. Lower-turnover construction: hold positions for 3 months or trade only when a stock leaves a
   wider band, so reversal's churn doesn't eat the edge.
4. Add the earnings-surprise (post-earnings drift) signal from the Earnings Momentum Model.

Full tables: [research_results.md](research_results.md).
