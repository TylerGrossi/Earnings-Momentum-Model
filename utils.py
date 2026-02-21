import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

# ------------------------------------
# CONSTANTS
# ------------------------------------
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# Forward P/E filter: only include stocks with Forward P/E <= this, or N/A/negative
FORWARD_PE_MAX = 15.0


def is_forward_pe_low(pe_val):
    """
    Return True if the stock passes the low P/E filter: Forward P/E <= FORWARD_PE_MAX, or negative, or N/A.
    Exclude only when we have a numeric Forward P/E and it is > FORWARD_PE_MAX.
    """
    if pd.isna(pe_val) or pe_val is None or pe_val == "" or str(pe_val).strip() == "N/A":
        return True  # N/A → include
    try:
        val = float(pe_val)
        return val <= FORWARD_PE_MAX  # negative or 0 or 1..15 → include; > 15 → exclude
    except (TypeError, ValueError):
        return True  # unparseable → include


def _returns_tracker_pe_column(df, name):
    """Return the actual column name in df that matches name (e.g. 'Forward P/E'), or None. Handles strip."""
    if df is None or df.empty:
        return None
    for c in df.columns:
        if str(c).strip() == name:
            return c
    return None


def filter_returns_by_forward_pe(returns_df):
    """
    Filter to rows where Forward P/E <= 15, or Forward P/E is blank/N/A/negative.
    Uses only the Forward P/E column from returns_tracker.csv (no P/E fallback)
    so counts match Power BI, which only filters on Forward P/E <= 15.
    """
    if returns_df is None or returns_df.empty:
        return returns_df
    pe_col = _returns_tracker_pe_column(returns_df, "Forward P/E")
    if pe_col is None:
        return returns_df

    def passes(row):
        val = row.get(pe_col)
        # Blank/N/A/missing → include (Power BI only filters on Forward P/E when present)
        if pd.isna(val) or val is None or str(val).strip() == "" or str(val).strip() == "N/A":
            return True
        return is_forward_pe_low(val)

    return returns_df[returns_df.apply(passes, axis=1)].copy()

# ------------------------------------
# FINVIZ SCRAPERS
# ------------------------------------
def get_all_tickers():
    """Get all tickers from Finviz screener matching criteria."""
    base_url = "https://finviz.com/screener.ashx?v=111&f=earningsdate_thisweek,ta_sma20_cross50a"
    offset, tickers = 0, []
    while True:
        url = f"{base_url}&r={offset + 1}"
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.text, "html.parser")
        new_tickers = []
        for row in soup.select("table tr"):
            columns = row.find_all("td")
            if len(columns) > 1:
                ticker = columns[1].text.strip()
                if ticker.isupper() and ticker.isalpha() and len(ticker) <= 5:
                    new_tickers.append(ticker)
        if not new_tickers:
            break
        tickers.extend(t for t in new_tickers if t not in tickers)
        offset += 20
    return tickers


def get_finviz_data(ticker):
    """Get stock data from Finviz for a single ticker. Uses the same table and key-value
    parsing as Earnings Momentum Model.get_finviz_stats so Forward P/E (and P/E, Price, etc.)
    come from the same spot."""
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    data = {"Ticker": ticker, "Earnings": "N/A", "Price": "N/A", "P/E": "N/A", "Forward P/E": "N/A", "Beta": "N/A", "Market Cap": "N/A"}
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        soup = BeautifulSoup(r.text, "html.parser")
        tables = soup.find_all("table")
        if len(tables) > 8:
            cells = tables[8].find_all("td")
            # Same key-value extraction as Earnings Momentum Model.get_finviz_stats
            finviz_dict = {cells[i].get_text(strip=True): cells[i + 1].get_text(strip=True) for i in range(0, len(cells), 2)}
            for key in ("Price", "P/E", "Forward P/E", "Beta", "Market Cap", "Earnings"):
                if key in finviz_dict and finviz_dict[key]:
                    data[key] = finviz_dict[key]
    except Exception:
        pass
    return data


# ------------------------------------
# BARCHART SCRAPER
# ------------------------------------
def has_buy_signal(ticker):
    """Check if ticker has a buy signal on Barchart."""
    url = f"https://www.barchart.com/stocks/quotes/{ticker}/opinion"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        sig = soup.find("span", class_="opinion-signal buy")
        return bool(sig and "Buy" in sig.text)
    except:
        return False


# ------------------------------------
# YFINANCE HELPERS
# ------------------------------------
def get_yfinance_earnings_date(ticker):
    """Get earnings date from yfinance."""
    try:
        earnings_df = yf.Ticker(ticker).get_earnings_dates(limit=10)
        if earnings_df is None or earnings_df.empty:
            return None
        today = datetime.today().date()
        best_date = None
        min_diff = 999
        for idx in earnings_df.index:
            try:
                idx_date = idx.date() if hasattr(idx, 'date') else pd.to_datetime(idx).date()
                diff = abs((idx_date - today).days)
                if diff < min_diff and diff <= 60:
                    min_diff = diff
                    best_date = datetime.combine(idx_date, datetime.min.time())
            except:
                continue
        return best_date
    except:
        return None


def get_finviz_earnings_date(ticker):
    """Get earnings date from yfinance info (Finviz backup)."""
    try:
        info = yf.Ticker(ticker).info
        ts = info.get('earningsTimestamp') or info.get('earningsTimestampStart')
        if ts:
            return datetime.fromtimestamp(ts)
    except:
        pass
    return None


# ------------------------------------
# DATE CHECK FUNCTIONS
# ------------------------------------
def check_date_status(earnings_date, yfinance_date):
    """Check if earnings date has passed based on yfinance data."""
    try:
        if earnings_date is None or yfinance_date is None:
            return "OK"
        ed_date = earnings_date.date() if hasattr(earnings_date, 'date') else earnings_date
        yf_date = yfinance_date.date() if hasattr(yfinance_date, 'date') else yfinance_date
        date_diff = abs((ed_date - yf_date).days)
        if date_diff > 14:
            return "OK"
        if yf_date < ed_date:
            return "DATE PASSED"
        return "OK"
    except:
        return "OK"


def get_date_check(ticker):
    """Get full date check info for a ticker."""
    finviz_date = get_finviz_earnings_date(ticker)
    yfinance_date = get_yfinance_earnings_date(ticker)
    status = check_date_status(finviz_date, yfinance_date)
    return {
        "Earnings Date (Finviz)": finviz_date.strftime("%Y-%m-%d") if finviz_date else "N/A",
        "Earnings Date (yfinance)": yfinance_date.strftime("%Y-%m-%d") if yfinance_date else "N/A",
        "Date Check": status
    }


# ------------------------------------
# SORTING HELPERS
# ------------------------------------
def parse_earnings_date(earn_str):
    """Parse earnings date string like 'Jan 15 BMO' to datetime."""
    try:
        parts = (earn_str or "").split()
        if len(parts) >= 2:
            month, day = parts[0], parts[1]
            return datetime(datetime.today().year, datetime.strptime(month, "%b").month, int(day))
    except:
        pass
    return datetime.max


def earnings_sort_key(row):
    """Sort key for earnings - by date, then BMO before AMC."""
    date = parse_earnings_date(row["Earnings"])
    earn_str = (row["Earnings"] or "").upper()
    am_pm_rank = 0 if "BMO" in earn_str else 1 if "AMC" in earn_str else 2
    return (date, am_pm_rank)


# ------------------------------------
# PORTFOLIO METRICS (aligned with Power BI)
# ------------------------------------
CAPITAL_PER_TRADE = 2000


def _buy_date_from_earnings(earnings_date, earnings_timing):
    """
    Compute Buy Date from Earnings Date and Timing (BMO/AMC).
    Matches Power BI: BMO = day before (Mon->Fri, Sun->Fri); AMC = same day (Sat/Sun->Fri).
    Returns a date (no time).
    """
    if earnings_date is None or pd.isna(earnings_date):
        return None
    dt = pd.to_datetime(earnings_date)
    day = dt.date() if hasattr(dt, 'date') else dt
    # Python: Monday=0, Sunday=6
    wk = day.weekday()
    timing = (str(earnings_timing).strip().upper() if pd.notna(earnings_timing) and earnings_timing else "") or "AMC"
    if "BMO" in timing:
        if wk == 0:   # Monday -> Friday
            days_shift = -3
        elif wk == 6:  # Sunday -> Friday
            days_shift = -2
        else:
            days_shift = -1
    else:
        if wk == 5:   # Saturday -> Friday
            days_shift = -1
        elif wk == 6:  # Sunday -> Friday
            days_shift = -2
        else:
            days_shift = 0
    return day + timedelta(days=days_shift)


def compute_portfolio_metrics(returns_df):
    """
    Compute portfolio metrics from returns_tracker-style DataFrame (aligned with Power BI).

    Uses: Earnings Date, Earnings Timing, 3D Return.
    - Total Net Profit = SUM(3D Return * CAPITAL_PER_TRADE)
    - Max Capital High Water Mark = max concurrent trades (after first week) * CAPITAL_PER_TRADE
    - Portfolio Return = Total Net Profit / Max Capital HWM
    - Weekly average return = mean(3D Return) = sum(3D Return) / count(tickers with 3D not blank)
    - Annualized Return = (1 + Portfolio Return)^(365/days) - 1

    Returns dict with: weekly_avg_return, portfolio_return, annualized_return;
    or None if insufficient data.

    weekly_avg_return = mean(3D Return) over all tickers with 3D Return not blank.
    """
    if returns_df is None or returns_df.empty:
        return None
    df = returns_df.copy()
    if '3D Return' not in df.columns or 'Earnings Date' not in df.columns:
        return None

    df['Earnings Date'] = pd.to_datetime(df['Earnings Date'], errors='coerce')
    df['3D Return'] = pd.to_numeric(df['3D Return'], errors='coerce')
    closed = df[df['3D Return'].notna()].copy()
    if closed.empty:
        return None

    # Weekly average return = average of 3D returns (sum / count of tickers with 3D not blank)
    weekly_avg_return = closed['3D Return'].mean()

    timing_col = 'Earnings Timing' if 'Earnings Timing' in closed.columns else None
    closed['_buy_date'] = closed.apply(
        lambda r: _buy_date_from_earnings(r['Earnings Date'], r.get(timing_col) if timing_col else None),
        axis=1
    )
    closed = closed[closed['_buy_date'].notna()].copy()
    closed['_sell_date'] = closed['_buy_date'].apply(lambda d: d + timedelta(days=7))
    if closed.empty:
        return None

    total_net_profit = (closed['3D Return'] * CAPITAL_PER_TRADE).sum()
    global_start = closed['_buy_date'].min()
    cutoff_date = global_start + timedelta(days=7)
    max_sell = closed['_sell_date'].max()
    min_buy = closed['_buy_date'].min()

    # Max concurrent trades: for each day from cutoff to max_sell, count active trades (buy_date > cutoff, buy <= day <= sell)
    day = cutoff_date
    max_concurrent = 0
    while day <= max_sell:
        count = ((closed['_buy_date'] > cutoff_date) &
                 (closed['_buy_date'] <= day) &
                 (closed['_sell_date'] >= day)).sum()
        max_concurrent = max(max_concurrent, count)
        day += timedelta(days=1)
    if max_concurrent <= 0:
        max_capital_hwm = 1.0 * CAPITAL_PER_TRADE  # avoid div by zero
        portfolio_return = total_net_profit / max_capital_hwm
    else:
        max_capital_hwm = max_concurrent * CAPITAL_PER_TRADE
        portfolio_return = total_net_profit / max_capital_hwm

    days_in_period = (max_sell - min_buy).days
    if days_in_period <= 0:
        days_in_period = 1
    num_weeks = days_in_period / 7.0
    annualized_return = (1 + portfolio_return) ** (365 / days_in_period) - 1 if days_in_period > 0 else 0.0

    return {
        'weekly_avg_return': weekly_avg_return,
        'portfolio_return': portfolio_return,
        'annualized_return': annualized_return,
        'total_net_profit': total_net_profit,
        'max_concurrent': max_concurrent,
        'days_in_period': days_in_period,
        'num_weeks': num_weeks,
    }


def compute_portfolio_return_for_return_series(trades_df, return_series, earnings_date_col='Earnings Date', timing_col='Earnings Timing'):
    """
    Compute portfolio return (Total Net Profit / Max Capital HWM) for a given set of trades
    and a return series (e.g. stop-loss strategy returns). Same capital/timing logic as Power BI.

    trades_df: DataFrame with one row per trade, must have earnings_date_col and optionally timing_col.
    return_series: Series of per-trade returns (same index/length as trades_df).

    Returns (portfolio_return_decimal, max_concurrent) or (None, None) if insufficient data.
    """
    if trades_df is None or trades_df.empty or return_series is None or len(return_series) != len(trades_df):
        return None, None
    df = trades_df.copy()
    # Align by index so row order matches
    aligned = return_series.reindex(df.index)
    df['_ret'] = aligned.values
    df = df[df['_ret'].notna()].copy()
    if df.empty:
        return None, None
    if earnings_date_col not in df.columns:
        return None, None
    timing = timing_col if timing_col in df.columns else None
    df['_buy_date'] = df.apply(
        lambda r: _buy_date_from_earnings(r[earnings_date_col], r.get(timing) if timing else None),
        axis=1
    )
    df = df[df['_buy_date'].notna()].copy()
    df['_sell_date'] = df['_buy_date'].apply(lambda d: d + timedelta(days=7))
    if df.empty:
        return None, None
    global_start = df['_buy_date'].min()
    cutoff_date = global_start + timedelta(days=7)
    max_sell = df['_sell_date'].max()
    max_concurrent = 0
    day = cutoff_date
    while day <= max_sell:
        count = ((df['_buy_date'] > cutoff_date) &
                 (df['_buy_date'] <= day) &
                 (df['_sell_date'] >= day)).sum()
        max_concurrent = max(max_concurrent, count)
        day += timedelta(days=1)
    if max_concurrent <= 0:
        max_concurrent = 1
    total_net_profit = (df['_ret'] * CAPITAL_PER_TRADE).sum()
    portfolio_return = total_net_profit / (max_concurrent * CAPITAL_PER_TRADE)
    return portfolio_return, max_concurrent
