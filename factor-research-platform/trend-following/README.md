# Trend Following: time-series momentum replication

A replication of cross-asset trend following (time-series momentum) on 20 liquid ETFs, validated
against live managed-futures funds.

## Pre-specified design (written 2026-11-20, before any data was downloaded)

Everything below was fixed from the published papers before running anything. There is **no
parameter search**. Any variant beyond the primary and the one secondary listed here goes in the
variant log at the bottom, and every one is reported.

### Universe (fixed, 20 ETFs)

Total-return (dividend-adjusted) daily prices from yfinance. Each ETF enters once it has the
history the signal and volatility estimate need. The list is never changed based on results.

| Class | ETFs |
|---|---|
| Equities | SPY, IWM, EFA, EEM, EWJ |
| Bonds | SHY, IEF, TLT, TIP, LQD, HYG |
| Commodities | GLD, SLV, DBC, USO, DBA |
| Currencies | UUP, FXE, FXY |
| Real estate | VNQ |

ETFs stand in for the futures the papers use. That means fewer markets (20 vs 58 in MOP), some
carry embedded in ETF returns (bond coupons, commodity roll), and higher costs. Financing is
modelled as T-bill (^IRX) rates on the excess returns, as a futures position would earn.

### Signal (monthly, long and short)

- **Primary (Moskowitz, Ooi and Pedersen 2012):** at each month end, the sign of the asset's
  12-month excess return over T-bills, i.e. +1 (long) or −1 (short).
- **Secondary (the only other variant):** the average of the signs of the 1-, 3- and 12-month
  excess returns, so the signal is one of −1, −1/3, +1/3, +1 (in the spirit of Hurst, Ooi and
  Pedersen 2017, who combine 1-, 3- and 12-month trends).

### Sizing

1. **Per asset:** position = signal × 40% / σ, where σ is the ex-ante annualized volatility from
   an exponentially weighted average of squared daily excess returns, center of mass 60 trading
   days, annualized with 261 days (MOP 2012). Positions are averaged across the N assets active
   that month (weight = position / N).
2. **Portfolio:** scaled to a 10% annualized volatility target. The scale factor is 10% divided
   by the ex-ante volatility of that month's weights applied to the previous 252 trading days of
   asset returns.
3. **Leverage caps:** each asset's final weight is capped at ±100% of capital, then gross
   exposure (sum of absolute weights) is capped at 400% by scaling all weights down
   proportionally. Without caps, a 40%-volatility target on a 1.5%-volatility ETF like SHY would
   need extreme leverage.
4. An asset becomes active once it has 12 months of returns and 120 trading days for volatility.

### Returns and costs

- Monthly rebalance at month end; weights held through the next month.
- Strategy return = T-bill return + Σ weight × (asset return − T-bill return) − costs.
- **Trading cost:** 10 bps per unit of turnover (Σ |w_t − w_{t−1}|), the conservative end of
  5–10 bps for these ETFs. Sensitivity at 5 and 20 bps is reported, not used for selection.
- **Short borrow:** 0.5% a year on short notional.
- Sharpe ratios use excess returns over T-bills. t-stats are Newey-West (6 lags).

### Samples

- **Development:** Jan 2007 – Dec 2022 (2007 is the first year the full commodity and currency
  set exists).
- **Holdout:** Jan 2023 onward, **evaluated once**, after the development results are written up.
- Reported sub-periods: 2007–2014 and 2015–latest. Crisis windows: 2008, 2020 Q1, 2022.

### Benchmarks and validation

- **SPY**, and **60/40** = 60% SPY + 40% IEF, rebalanced monthly.
- **80/20 blend** = 80% 60/40 + 20% trend strategy, rebalanced monthly.
- **Validation:** correlation of monthly returns with live managed-futures funds: AQR Managed
  Futures (AQMIX, 2010+), iMGP DBi Managed Futures ETF (DBMF, 2019+) and KraneShares Mount Lucas
  (KMLM, 2020+). A high correlation means the replication captures the same strategy.

### Expectations (from the literature, before running)

- Hurst, Ooi and Pedersen (2017), "A Century of Evidence on Trend-Following Investing": Sharpe
  0.77 net of 2/20 fees over 1880–2013, 0.62 over 2000–2013, positive in every decade; +22% in
  2007–09 while 60/40 lost 31%; adding 20% trend to 60/40 raised Sharpe from 0.38 to 0.54.
- Moskowitz, Ooi and Pedersen (2012), "Time Series Momentum", Journal of Financial Economics.
- SG Trend Index (10 largest trend CTAs): 5.33% CAGR since 2000, max drawdown −20.61%; 2022 a
  strong year, 2023 roughly −3%.
- Published anomalies weaken out of sample (McLean and Pontiff 2016: −26% out of sample, −58%
  after publication). **Expected: Sharpe roughly 0.3–0.7 for 2007–2026, with a weak 2010s.**
  Whatever comes out is reported.

## Development results (Jan 2007 – Dec 2022), written before the holdout was run

Full tables: [reports/development.md](reports/development.md).

- **Primary (12-month sign):** CAGR +5.9%, volatility 10.7%, Sharpe 0.51 (t 2.01), max drawdown
  −20.1%, worst year 2016 (−14.9%). Secondary blend: Sharpe 0.40 (t 1.68).
- **Validation:** monthly correlation 0.71 with AQR Managed Futures (AQMIX, 155 months), 0.83 with
  DBMF (43 months), 0.71 with KMLM (24 months).
- **Crises:** 2008 +18.8% (SPY −36.8%, 60/40 −18.0%); 2020 Q1 +5.1% (SPY −19.4%); 2022 +26.2%
  (SPY −18.2%, 60/40 −16.7%). Correlation with SPY −0.17.
- **80/20 blend:** Sharpe 0.80 vs 0.65 for 60/40; max drawdown −20.2% vs −29.5%.
- **Decay, as expected:** Sharpe 0.72 in 2007–2014, 0.33 in 2015–2022.
- **Costs:** Sharpe 0.56 / 0.51 / 0.42 at 5 / 10 / 20 bps. Average gross exposure 310% (cap 400%),
  turnover 86% a month, cost drag 1.03%/yr plus 0.55%/yr borrow.
- Bonds contributed most (+3.1%/yr before costs), helped by the long bond rally; the strategy
  averaged +89% net long.

## Holdout (Jan 2023 – Oct 2026, run once) and full-period results

Full tables: [reports/holdout.md](reports/holdout.md). Live-fund context:
[reports/funds_holdout_context.txt](reports/funds_holdout_context.txt).

- **Holdout: the strategy did not work.** Primary Sharpe −0.16 (t −0.34): +2.8% a year, below
  T-bills; 2023 −5.7%, 2024 +4.8%, 2025 +4.0%. Secondary blend Sharpe −0.53. Over the same months
  SPY returned +20.9% a year and 60/40 +13.3%.
- **Live funds over the same months were mixed:** AQMIX Sharpe 0.56, DBMF 0.28, KMLM −0.22.
  KMLM, the simplest pure-trend fund of the three, was also negative.
- **Full Jan 2007 – Oct 2026:** primary CAGR +5.3%, volatility 10.3%, Sharpe 0.40 (t 1.75), max
  drawdown −20.1%. Inside the 0.3–0.7 range expected from the literature, but not significant at
  t ≥ 2. Sharpe by sub-period: 0.72 (2007–2014), 0.18 (2015–Oct 2026).
- **As a diversifier, full period:** 80% 60/40 + 20% trend had Sharpe 0.81 vs 0.71 for 60/40,
  and max drawdown −20.2% vs −29.5%.
- **Validation, full overlap:** monthly correlation 0.68 with AQMIX (201 months), 0.77 with DBMF
  (89 months), 0.57 with KMLM (70 months).

**Bottom line.** The replication tracks live managed-futures funds and delivered what trend
following is known for: large gains in 2008 and 2022, negative correlation to stocks, and lower
drawdowns when added to 60/40. As a stand-alone return source it was weak after 2015 and lost
to T-bills in the 2023–2026 holdout, in line with the published decay of the strategy and the
weak recent record of simple trend funds.

## Variant log

| # | Variant | Why | Reported in |
|---|---|---|---|
| 1 | Primary: 12-month sign | Pre-specified | results |
| 2 | Secondary: 1/3/12-month blend | Pre-specified | results |

**Correction (2026-11-20, after the first holdout run).** The data ended Nov 20, 2026, and the
month-to-date return for November 2026 was counted as a full month. The fix drops an incomplete
final month. It doesn't change the design, and development numbers are unchanged. The holdout
was rerun once with the fix. First run (Jan 2023 – Nov 2026, partial month included): primary
Sharpe −0.24 (t −0.52). Corrected run (Jan 2023 – Oct 2026): Sharpe −0.16 (t −0.34).

## Run

```
pip install -r requirements.txt
python scripts/run_backtest.py             # development sample -> reports/development.md
python scripts/run_backtest.py --holdout   # holdout + full period -> reports/holdout.md
```

Prices are cached in `data/prices.parquet` (gitignored); delete it to re-download.

## Layout

| Path | What it does |
|---|---|
| `src/trend/data.py` | Universe, yfinance download, T-bill rate |
| `src/trend/strategy.py` | Signals, EWMA volatility, sizing, caps, backtest with costs |
| `src/trend/stats.py` | Sharpe, Newey-West t, drawdowns, calendar years |
| `scripts/run_backtest.py` | Development or holdout run and report |
| `HANDOFF.md` | The brief this project was built from, with the market research |
