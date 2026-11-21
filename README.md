
# Earnings Momentum Strategy

A quantitative trading strategy that identifies stocks with positive technical momentum ahead of earnings announcements.

## Research in this repo

| Folder | What it is |
|---|---|
| [research/](research/README.md) | Out-of-sample audit of this strategy (reports 01–06): sector benchmarks, walk-forward tests, a classifier leakage audit, and a test of the screen on 84,511 S&P 1500 earnings events (2010–2026). |
| [factor-research-platform/](factor-research-platform/README.md) | Point-in-time S&P 500 equity factor research, validated against Ken French's published factors. |
| [factor-research-platform/trend-following/](factor-research-platform/trend-following/README.md) | Trend-following (time-series momentum) replication across 20 asset-class ETFs, validated against live managed-futures funds. |

## Strategy Overview

### Entry Criteria (All must be met)

1. **Earnings This Week** - Stock has earnings announcement scheduled for the current week
2. **SMA20 > SMA50** - 20-day simple moving average has crossed above the 50-day SMA (golden cross)
3. **Barchart Buy Signal** - Stock shows a "Buy" opinion signal on Barchart.com

### Entry Timing

* **BMO (Before Market Open)** : Buy at previous day's close (~4pm day before earnings)
* **AMC (After Market Close)** : Buy at earnings day close (~4pm on earnings day)

### Exit Rules

1. **Stop Loss: -10%** - Exit if position drops 10% from entry price
   * If stock gaps down and opens below -10%, take the actual gap loss (not -10%)
2. **Time Exit: Day 5** - Exit at market close on trading day 5 if stop loss not triggered
3. **No Profit Cap** - Let winners run; do not cap upside

## Model Rules Summary

| Rule              | Description                                                   |
| ----------------- | ------------------------------------------------------------- |
| Entry Signal      | SMA20 crosses above SMA50 + Barchart Buy + Earnings this week |
| Entry Price (BMO) | Previous day's close                                          |
| Entry Price (AMC) | Earnings day close                                            |
| Stop Loss         | -10% (or actual gap if opens lower)                           |
| Exit Day          | Day 5 close                                                   |
| Position Sizing   | Equal weight per trade                                        |

## Key Metrics (Historical Backtest)

Based on ~177 trades from the universe:

| Metric               | Value   |
| -------------------- | ------- |
| Total Trades         | 177     |
| Average Return/Trade | ~+5-6%  |
| Win Rate             | ~55-60% |
| Best Stop Loss Level | -10%    |

---

## Project Structure

### Data Files

| File                    | Description                                                   |
| ----------------------- | ------------------------------------------------------------- |
| `returns_tracker.csv` | Master data file - all tickers, returns, EPS data, Date Check |
| `hourly_prices.csv`   | Hourly price data for backtesting stop losses                 |

> **Note:** `earnings_universe.csv` is no longer used. All data is consolidated in `returns_tracker.csv`

### Scripts

| File                        | Description                                                         | Location                                                                |
| --------------------------- | ------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `earnings_scraper_v17.py` | Nightly scraper - updates returns_tracker.csv and hourly_prices.csv | `C:\Users\Owner\Desktop\Tyler\OneDrive\Projects\Quantitative Finance` |

### Streamlit App (Modular Structure)

All Streamlit files are in the `streamlit_app/` folder:

| File                      | Purpose                                                           | When to Edit                            |
| ------------------------- | ----------------------------------------------------------------- | --------------------------------------- |
| `streamlit_app.py`      | Main entry point - assembles all tabs                             | Rarely (just imports)                   |
| `config.py`             | CSS styling, page configuration                                   | Change colors, fonts, styling           |
| `data_loader.py`        | Load & filter data from GitHub                                    | Fix data filtering, change data sources |
| `utils.py`              | Scrapers (Finviz, Barchart, yfinance)                             | Fix scraping issues                     |
| `stock_screener.py`     | **Stock Screener tab**- This Week's Earnings + Live Scanner | Change earnings table, scanner logic    |
| `powerbi.py`            | **PowerBI tab**- Embedded dashboard                         | Change PowerBI embed URL                |
| `stop_loss_analysis.py` | **Stop Loss Analysis tab**- Backtest functions + UI         | Change backtest logic, charts           |
| `earnings_analysis.py`  | **Earnings Analysis tab**- Beat/miss analysis               | Change EPS analysis, charts             |

---

## Running the App

### Streamlit App

```bash
cd streamlit_app
streamlit run streamlit_app.py
```

### Nightly Data Update

```bash
python earnings_scraper_v17.py
```

This script:

1. Scans Finviz for stocks meeting SMA criteria with earnings this week
2. Filters through Barchart for buy signals
3. Updates returns_tracker.csv with new tickers and return calculations
4. Exports hourly_prices.csv for backtest visualization
5. Saves files to: `C:\Users\Owner\Desktop\Tyler\OneDrive\Projects\Quantitative Finance`

---

## Data Columns

### returns_tracker.csv

| Column                                                    | Description                                          |
| --------------------------------------------------------- | ---------------------------------------------------- |
| `Ticker`                                                | Stock symbol                                         |
| `Fiscal Quarter`                                        | e.g., "Q3 FY25"                                      |
| `Earnings Date`                                         | Scheduled earnings announcement date                 |
| `Earnings Date (yfinance)`                              | yfinance validation date                             |
| `Date Check`                                            | "OK" or "DATE PASSED" validation                     |
| `Date Added`                                            | When ticker was first added to tracker               |
| `Earnings Timing`                                       | BMO (Before Market Open) or AMC (After Market Close) |
| `Price`                                                 | Entry price (base price)                             |
| `EPS Estimate`                                          | Analyst EPS estimate                                 |
| `Reported EPS`                                          | Actual reported EPS                                  |
| `EPS Surprise (%)`                                      | Percentage beat/miss                                 |
| `1D Return`-`10D Return`                              | Returns at each holding period                       |
| `1W High Return`/`1W Low Return`                      | Best and worst intraday returns                      |
| `Company Name`,`Sector`,`Market Cap`,`Beta`, etc. |                                                      |

### hourly_prices.csv

| Column                                                                    | Description                           |
| ------------------------------------------------------------------------- | ------------------------------------- |
| `Ticker`                                                                | Stock symbol                          |
| `Datetime`                                                              | Timestamp of price observation        |
| `Close`                                                                 | Closing price for that hour           |
| `Base Price`                                                            | Entry price (based on BMO/AMC timing) |
| `Trading Day`                                                           | 1-5 (days after entry)                |
| `Return From Earnings (%)`                                              | Return from base price                |
| `Earnings Date`,`Earnings Timing`,`Company Name`,`Fiscal Quarter` |                                       |

---

## Data Filtering Logic

The app applies consistent filtering across all tabs:

1. **Remove DATE PASSED** - Tickers where `Date Check = 'DATE PASSED'` are excluded
2. **Require 5D Return** - Only tickers with valid 5D Return data are included in analysis

**Exception:** The "This Week's Reported Earnings" table in Stock Screener uses **unfiltered data** to show recent earnings even before 5D returns are available.

---

## Streamlit App Features

### Stock Screener Tab

* **This Week's Reported Earnings** - Shows tickers from returns_tracker with earnings in current week (Sun-Sat)
* **Live Stock Scanner** - Scans Finviz/Barchart for new opportunities
  * Skips tickers with DATE PASSED
  * Skips tickers where earnings passed but weren't in tracker (missed opportunities)

### PowerBI Tab

* Embedded PowerBI dashboard for visual analysis

### Stop Loss Analysis Tab

* Backtest with configurable stop loss levels
* Compare strategy vs buy & hold
* Exit breakdown (Gap Down, Stop Loss, Held to Day 5)
* Stop loss level comparison tool

### Earnings Analysis Tab

* Beat vs Miss performance analysis
* EPS Surprise magnitude correlation
* Raw data explorer

---

## Important Notes

### Gap Down Handling

If a stock gaps down and the first candle opens below the stop loss level:

* You take the  **actual gap loss** , not the stop loss price
* Example: Stop is -10%, stock opens at -15% → You lose 15%, not 10%
* This reflects real trading where you cannot exit at a price that was never traded

### Date Validation

The model validates earnings dates by comparing Finviz dates with yfinance:

* `DATE PASSED` = Finviz shows future date but earnings already happened
* These tickers are filtered out of backtests

### Earnings Week Definition

* Week runs **Sunday to Saturday**
* Example: If today is Tuesday Jan 14, the week is Jan 12 (Sun) - Jan 18 (Sat)

### Data Limitations

* Hourly data from yfinance only available for last ~730 days
* Some tickers may not have complete hourly data

---

## Strategy Rationale

1. **Momentum** - SMA20 > SMA50 indicates positive price momentum
2. **Catalyst** - Earnings announcement provides potential for gap up
3. **Confirmation** - Barchart buy signal adds technical confirmation
4. **Risk Management** - 10% stop loss limits downside while allowing upside
5. **Time Decay** - Exit by Day 5 captures post-earnings momentum before mean reversion

---

## Quick Reference: What to Edit

| If you need to...                   | Edit this file              |
| ----------------------------------- | --------------------------- |
| Change "This Week's Earnings" table | `stock_screener.py`       |
| Change Live Scanner logic           | `stock_screener.py`       |
| Change backtest logic               | `stop_loss_analysis.py`   |
| Change backtest charts              | `stop_loss_analysis.py`   |
| Change earnings analysis            | `earnings_analysis.py`    |
| Change data filtering               | `data_loader.py`          |
| Change scraping functions           | `utils.py`                |
| Change styling/colors               | `config.py`               |
| Change PowerBI embed                | `powerbi.py`              |
| Change nightly data pull            | `earnings_scraper_v17.py` |

---

## License

For personal/educational use only. Not financial advice.
