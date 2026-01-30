import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo
import yfinance as yf

from utils import get_all_tickers, get_finviz_data, has_buy_signal, get_date_check, earnings_sort_key
from data_loader import get_this_week_earnings

# US Eastern for entry cutoff (4 PM) and "Reported" so behavior matches main model and is consistent across environments
MARKET_TZ = ZoneInfo("America/New_York")


def format_market_cap(value):
    """Format market cap to readable string (e.g., 1.08B, 230.7M)."""
    if pd.isna(value) or value == 'N/A':
        return 'N/A'
    try:
        val = float(value)
        if val >= 1e12:
            return f"{val/1e12:.2f}T"
        elif val >= 1e9:
            return f"{val/1e9:.2f}B"
        elif val >= 1e6:
            return f"{val/1e6:.1f}M"
        else:
            return f"{val:,.0f}"
    except:
        return str(value)


def format_value(value, decimals=2):
    """Format numeric value, return 'N/A' if None/NaN."""
    if pd.isna(value) or value is None or value == '':
        return 'N/A'
    try:
        return f"{float(value):.{decimals}f}"
    except:
        return str(value)


def has_earnings_happened(earnings_date, earnings_timing):
    """
    True if we're past the entry cutoff (so we could not have bought at close).
    Uses current time in US Eastern so GitHub and local match.
    
    Entry cutoff:
    - BMO: 4 PM ET the day *before* earnings (buy at that close)
    - AMC: 4 PM ET the day *of* earnings (buy at that close)
    - No timing: treat as AMC (4 PM on earnings day)
    
    Same cutoff is used for: (1) do not add ticker, (2) Status = "Reported".
    """
    if earnings_date is None:
        return False

    now_et = datetime.now(MARKET_TZ)
    today = now_et.date()
    market_close_hour = 16  # 4 PM ET

    if hasattr(earnings_date, 'date'):
        earn_date = earnings_date.date()
    else:
        earn_date = pd.to_datetime(earnings_date).date() if earnings_date is not None else None
    if earn_date is None:
        return False

    timing = str(earnings_timing).strip().upper() if pd.notna(earnings_timing) and earnings_timing else ""

    if timing == "BMO":
        cutoff_date = earn_date - timedelta(days=1)
        if today > cutoff_date:
            return True
        if today == cutoff_date and now_et.hour >= market_close_hour:
            return True
        return False
    else:
        # AMC or unknown
        if today > earn_date:
            return True
        if today == earn_date and now_et.hour >= market_close_hour:
            return True
        return False


def _entry_date(earnings_date, timing):
    """Entry date (date only): BMO = day before earnings, AMC = day of earnings (4 PM close that day)."""
    if earnings_date is None:
        return None
    ed = earnings_date.date() if hasattr(earnings_date, 'date') else pd.to_datetime(earnings_date).date()
    t = str(timing).strip().upper() if pd.notna(timing) and timing else ""
    if t == "BMO":
        return ed - timedelta(days=1)
    return ed


def _add_return_from_entry(all_rows):
    """
    Add 'Return from entry' to each row: return from entry close (4 PM day before BMO, 4 PM day of AMC)
    until current price when button was pressed. Uses yfinance.
    """
    if not all_rows:
        return all_rows
    tickers = list({r["Ticker"] for r in all_rows})
    entry_dates = []
    for r in all_rows:
        ed = r.get("_ed")
        timing = r.get("_timing", "")
        entry_d = _entry_date(ed, timing)
        entry_dates.append(entry_d)
    start = min(d for d in entry_dates if d is not None) - timedelta(days=5) if entry_dates else datetime.now(MARKET_TZ).date()
    end = datetime.now(MARKET_TZ).date() + timedelta(days=1)
    try:
        px = yf.download(tickers, start=start, end=end, group_by="ticker", auto_adjust=True, progress=False, threads=False)
        if px.empty:
            for r in all_rows:
                r["Return from entry"] = "N/A"
            return all_rows
        single_ticker = len(tickers) == 1
        for i, r in enumerate(all_rows):
            t, entry_d = r["Ticker"], entry_dates[i]
            ret_str = "N/A"
            if entry_d is not None:
                try:
                    if single_ticker:
                        close_series = px["Close"] if "Close" in px.columns else px[tickers[0]]["Close"]
                    else:
                        close_series = px[t]["Close"] if t in px.columns.get_level_values(0) else None
                    if close_series is None or (hasattr(close_series, 'empty') and close_series.empty):
                        r["Return from entry"] = ret_str
                        continue
                    close_series = close_series.dropna()
                    if hasattr(close_series.index, 'tz') and close_series.index.tz is not None:
                        close_series.index = close_series.index.tz_localize(None)
                    entry_series = close_series[close_series.index.date <= entry_d]
                    entry_price = entry_series.iloc[-1] if not entry_series.empty else None
                    current_price = close_series.iloc[-1] if not close_series.empty else None
                    if entry_price is not None and current_price is not None and entry_price > 0:
                        ret = (float(current_price) / float(entry_price)) - 1
                        ret_str = f"{ret:+.2%}"
                except Exception:
                    pass
            r["Return from entry"] = ret_str
    except Exception:
        for r in all_rows:
            r["Return from entry"] = "N/A"
    return all_rows


def render_stock_screener_tab(raw_returns_df):
    """Render the Stock Screener tab."""
    
    st.markdown("### This Week's Earnings")
    st.markdown("**Criteria:** Earnings this week · SMA20 crossed above SMA50 · Barchart Buy Signal")
    
    if st.button("Find Stocks"):
        # Get tickers already reported this week from returns tracker
        this_week_df = get_this_week_earnings(raw_returns_df)
        
        # Get list of tickers already in returns tracker
        tracked_tickers = set()
        if raw_returns_df is not None and not raw_returns_df.empty:
            tracked_tickers = set(raw_returns_df['Ticker'].unique())
        
        # Build the reported earnings rows
        tracked_rows = []
        tracked_tickers_this_week = set()
        if not this_week_df.empty:
            for _, row in this_week_df.iterrows():
                ticker = row.get('Ticker', '')
                tracked_tickers_this_week.add(ticker)
                
                # Get earnings date and timing
                earnings_date = None
                timing = row.get('Earnings Timing', '')
                if pd.notna(row.get('Earnings Date')):
                    earnings_date = pd.to_datetime(row.get('Earnings Date'))
                
                # Format earnings date string
                earnings_str = 'N/A'
                if earnings_date is not None:
                    earnings_str = earnings_date.strftime('%b %d')
                    if pd.notna(timing) and timing:
                        earnings_str += ' ' + str(timing).strip()
                
                # Determine status based on whether earnings have actually happened
                status = "Reported" if has_earnings_happened(earnings_date, timing) else "Upcoming"
                
                tracked_rows.append({
                    "Ticker": ticker,
                    "Earnings": earnings_str,
                    "Price": format_value(row.get('Price')),
                    "P/E": format_value(row.get('P/E')),
                    "Beta": format_value(row.get('Beta')),
                    "Market Cap": format_market_cap(row.get('Market Cap')),
                    "Status": status,
                    "_ed": earnings_date,
                    "_timing": timing,
                })
        
        # Status text for progress
        status_text = st.empty()
        progress = st.progress(0)
        
        # Scan Finviz
        status_text.text("Scanning Finviz...")
        tickers = get_all_tickers()
        
        # Check Barchart
        status_text.text(f"Found {len(tickers)} tickers. Checking Barchart signals...")
        barchart_passed = []
        
        for i, t in enumerate(tickers):
            if has_buy_signal(t):
                barchart_passed.append(t)
            progress.progress((i + 1) / len(tickers) * 0.5)
        
        # Check earnings dates
        status_text.text(f"{len(barchart_passed)} passed Barchart. Checking dates...")
        
        new_rows = []
        skipped = []
        today = datetime.now(MARKET_TZ).date()

        for i, t in enumerate(barchart_passed):
            data = get_finviz_data(t)
            date_info = get_date_check(t)
            
            # Parse the earnings date and timing from Finviz
            earnings_date = None
            earnings_timing = ""
            earnings_str = data.get("Earnings", "")
            if earnings_str and earnings_str != "N/A":
                try:
                    parts = earnings_str.split()
                    if len(parts) >= 2:
                        month_day = f"{parts[0]} {parts[1]}"
                        earnings_date = datetime.strptime(f"{month_day} {today.year}", "%b %d %Y").date()
                        # Check for BMO/AMC in the string
                        if len(parts) >= 3:
                            earnings_timing = parts[2].upper()
                except:
                    pass
            
            # Skip conditions: don't add if past entry cutoff (4 PM day before BMO, 4 PM day of AMC)
            skip_reason = None
            past_cutoff = has_earnings_happened(earnings_date, earnings_timing)

            if date_info["Date Check"] == "DATE PASSED":
                skip_reason = "DATE PASSED"
            elif past_cutoff:
                skip_reason = "Past entry cutoff (can't have bought at close)"
            elif earnings_date and earnings_date < today and t not in tracked_tickers:
                skip_reason = "MISSED (earnings passed, not in tracker)"
            elif t in tracked_tickers_this_week:
                skip_reason = "Already in tracker"

            if skip_reason:
                skipped.append({
                    "Ticker": t,
                    "Earnings": earnings_str,
                    "Reason": skip_reason
                })
            else:
                # Format market cap from Finviz (comes as string like "1.08B")
                market_cap = data.get("Market Cap", "N/A")
                
                # Determine status based on whether earnings have actually happened
                status = "Reported" if has_earnings_happened(earnings_date, earnings_timing) else "Upcoming"
                
                new_rows.append({
                    "Ticker": t,
                    "Earnings": earnings_str,
                    "Price": data.get("Price", "N/A"),
                    "P/E": data.get("P/E", "N/A"),
                    "Beta": data.get("Beta", "N/A"),
                    "Market Cap": market_cap,
                    "Status": status,
                    "_ed": datetime.combine(earnings_date, datetime.min.time()) if earnings_date else None,
                    "_timing": earnings_timing,
                })
            
            progress.progress(0.5 + (i + 1) / len(barchart_passed) * 0.5)
        
        # Clear progress indicators
        progress.empty()
        status_text.empty()
        
        # Combine tracked + new (from Finviz scan)
        all_rows = tracked_rows + new_rows

        # Sort by earnings date
        all_rows = sorted(all_rows, key=earnings_sort_key)

        # Return from entry (4 PM day before BMO, 4 PM day of AMC) until now
        all_rows = _add_return_from_entry(all_rows)

        if not all_rows:
            st.warning("No tickers match all criteria.")
        else:
            reported_count = len([r for r in all_rows if r['Status'] == "Reported"])
            upcoming_count = len([r for r in all_rows if r['Status'] == "Upcoming"])
            display_cols = ["Ticker", "Earnings", "Price", "P/E", "Beta", "Market Cap", "Status", "Return from entry"]
            st.caption(f"{len(all_rows)} tickers found ({reported_count} reported, {upcoming_count} upcoming)")
            st.dataframe(
                pd.DataFrame(all_rows)[display_cols],
                use_container_width=True,
                hide_index=True
            )
        
        if skipped:
            with st.expander(f"{len(skipped)} tickers skipped"):
                st.dataframe(pd.DataFrame(skipped), use_container_width=True, hide_index=True)
    else:
        st.caption("Click Find Stocks to scan.")