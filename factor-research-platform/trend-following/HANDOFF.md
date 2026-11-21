# Handoff: trend-following (time-series momentum) replication

From the session that ran the Earnings Momentum research (2026-11-20). Tyler wants this built
next as a new resume project with real, honest, positive numbers.

## Why this project

- The Earnings Momentum audit (see `../../research/README.md`, reports
  01–06) found no durable edge. The SMA-screen 1-day pop failed over 2010–2025 on the S&P 1500
  (t 0.61), and four alternative buy signals also failed. Tyler wants a project with real
  results instead.
- Market research picked **trend following / time-series momentum** across asset classes:
  - It's the most robust published strategy.
  - It doesn't overlap the equity factor platform (stock selection).
  - It can be validated against a real fund.

## Market research (cite these in the README)

- **Hurst, Ooi & Pedersen, "A Century of Evidence on Trend-Following Investing"** (AQR, JPM
  2017):
  - 1880–2013: 11.2% a year net of 2/20 fees, 9.7% volatility, **Sharpe 0.77 net**, 0.00
    correlation to US stocks.
  - 2000–2013 sub-period: Sharpe 0.62.
  - Positive in every decade.
  - Gains in 60/40 crises: 2007–09 +22% while 60/40 lost 31%; 2000–02 +26% vs −22%.
  - 80% 60/40 + 20% trend: Sharpe 0.38 → 0.54, max drawdown −62% → −50%.
- **Moskowitz, Ooi & Pedersen (2012), "Time Series Momentum"**: the original paper. 12-month
  return sign, volatility-scaled positions, positive for nearly all futures since 1985.
- **SG Trend Index** (10 largest trend CTAs, since 2000):
  - 5.33% CAGR, max drawdown −20.61% (TopTradersUnplugged, Dec 2025 report).
  - 2022 median manager +34%; 2023 about −3%; 2025 down about 9.5% by May, then finished
    +2.39% for the year.
  - S&P 500 TR since 2000: 7.98% CAGR, −50.95% max drawdown.
- **SG research (2025):** CTAs have shifted to slower models since 2007 (slow moving averages of
  60–300 days); longer-term models performed better on average since 2008.
- **Expectations for any replication:** published strategies decay after publication. McLean &
  Pontiff: −26% out of sample, −58% after publication. Hou, Xue & Zhang: most of 447 anomalies
  fail a t > 2.78 hurdle. So expect a modest Sharpe (≈0.3–0.7) in 2007–2026, with a weak
  2010s. Report whatever comes out honestly.

## What to build

New project in `Quantitative Finance/Trend-Following/` (git init it; don't commit or push
unless Tyler asks). *Moved on 2026-11-20 at Tyler's request, first into the Factor Research Platform and then,
with it, into the Earnings Momentum repo: now `Earnings-Momentum-Model/factor-research-platform/trend-following/`.*

- **Universe:** about 20 liquid ETFs, total-return adjusted from yfinance (auto_adjust).
  - US/intl/EM equity: SPY, IWM, EFA, EEM, plus EWJ or VGK.
  - Bonds: SHY, IEF, TLT, TIP, LQD, HYG.
  - Commodities: GLD, SLV, DBC, USO, DBA.
  - FX: UUP, FXE, FXY.
  - Real estate: VNQ.
  - Start each ETF at its inception and keep the universe fixed (no picking on results).
  - Note: ETFs instead of futures means some carry is embedded and there are fewer markets.
    Say so.
- **Signal** (fix from the papers BEFORE looking at results; write it in the README first):
  - Sign of the 12-month excess return (MOP 2012) as the primary signal.
  - A blend of 1-, 3- and 12-month signals as the one secondary variant.
  - Monthly rebalance; long and short.
- **Sizing:** each position scaled to a 40% annualized volatility target per asset using an
  ex-ante EWMA volatility (MOP use a 60-day center of mass). Then scale the portfolio to about
  10% annualized volatility. Cap leverage and say how.
- **Costs:** apply them on turnover (assume about 5–10 bps per trade for these ETFs). Also
  charge a short-borrow cost or a cash/T-bill adjustment. Use excess returns over T-bills
  (^IRX or BIL).
- **Validation (key resume point):**
  - Correlate monthly returns with **AQR Managed Futures fund (AQMIX, since 2010)** and other
    live managed-futures funds/ETFs on yfinance (e.g. DBMF 2019+, KMLM 2020+).
  - A high correlation shows the replication is right, in the same way the NHL site was
    validated against MoneyPuck.
- **Results to report:**
  - CAGR, volatility, Sharpe with a t-stat, max drawdown, worst year.
  - Calendar-year table.
  - Crisis performance (2008, 2020 Q1, 2022) against SPY and 60/40.
  - Correlation to SPY and to 60/40.
  - The 80/20 blend vs 60/40.
  - Returns by asset class.
  - A sub-period split (2007–2014 / 2015–2026).
  - A held-out final stretch (e.g. 2023–2026) evaluated only once, at the end.
- **Overfitting guard:** no parameter search. If any variant beyond the two above is tried,
  list every one and report all of them.

## Rules (from Tyler's AGENTS.md and this session)

- Every number on the resume must come from code output. Update
  `../../../../RESUME-BRIEF.md` with results and "Do not claim" notes (e.g. backtest only, ETF
  proxies, no live trading). Never round up.
- Read Excel/Power BI files only; never write to them.
- Don't move to a parameter that wasn't pre-specified because it looked better.
- If the factor-platform work in this session isn't at a stopping point, tell Tyler and ask
  which comes first.
