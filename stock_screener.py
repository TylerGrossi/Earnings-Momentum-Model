import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from utils import get_all_tickers, get_finviz_data, has_buy_signal, get_date_check, earnings_sort_key
from data_loader import get_this_week_earnings


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
    Determine if earnings have already happened based on date and timing.
    
    Rules:
    - BMO (Before Market Open): Reported after 4pm the day BEFORE earnings
    - AMC (After Market Close): Reported after 4pm ON the earnings day
    - No timing specified: Treat as AMC (conservative - wait until 4pm on earnings day)
    
    Args:
        earnings_date: datetime or date object of the earnings date
        earnings_timing: string like 'BMO', 'AMC', or empty/None
    
    Returns:
        bool: True if earnings have happened, False if upcoming
    """
    if earnings_date is None:
        return False
    
    now = datetime.now()
    today = now.date()
    current_hour = now.hour
    market_close_hour = 16  # 4pm
    
    # Convert to date if datetime
    if hasattr(earnings_date, 'date'):
        earn_date = earnings_date.date()
    else:
        earn_date = earnings_date
    
    # Normalize timing
    timing = str(earnings_timing).strip().upper() if pd.notna(earnings_timing) and earnings_timing else ""
    
    if "BMO" in timing:
        # BMO: Earnings happen before market open on earnings_date
        # So it's "reported" after 4pm the day BEFORE earnings_date
        cutoff_date = earn_date - timedelta(days=1)
        if today > cutoff_date:
            return True
        elif today == cutoff_date and current_hour >= market_close_hour:
            return True
        return False
    else:
        # AMC or unknown: Earnings happen after market close on earnings_date
        # So it's "reported" after 4pm ON the earnings_date
        if today > earn_date:
            return True
        elif today == earn_date and current_hour >= market_close_hour:
            return True
        return False


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
                    "Status": status
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
        today = datetime.today().date()
        
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
            
            # Skip conditions
            skip_reason = None
            
            if date_info["Date Check"] == "DATE PASSED":
                skip_reason = "DATE PASSED"
            elif earnings_date and earnings_date < today and t not in tracked_tickers:
                skip_reason = "MISSED (earnings passed, not in tracker)"
            elif t in tracked_tickers_this_week:
                skip_reason = "Already in tracker"
            
            if skip_reason:
                if skip_reason not in ["Already in tracker"]:
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
                    "Status": status
                })
            
            progress.progress(0.5 + (i + 1) / len(barchart_passed) * 0.5)
        
        # Clear progress indicators
        progress.empty()
        status_text.empty()
        
        # Combine tracked + new (from Finviz scan)
        all_rows = tracked_rows + new_rows
        
        # Sort by earnings date
        all_rows = sorted(all_rows, key=earnings_sort_key)
        
        if not all_rows:
            st.warning("No tickers match all criteria.")
        else:
            reported_count = len([r for r in all_rows if r['Status'] == "Reported"])
            upcoming_count = len([r for r in all_rows if r['Status'] == "Upcoming"])
            
            st.caption(f"{len(all_rows)} tickers found ({reported_count} reported, {upcoming_count} upcoming)")
            st.dataframe(
                pd.DataFrame(all_rows)[["Ticker", "Earnings", "Price", "P/E", "Beta", "Market Cap", "Status"]],
                use_container_width=True, 
                hide_index=True
            )
        
        if skipped:
            with st.expander(f"{len(skipped)} tickers skipped"):
                st.dataframe(pd.DataFrame(skipped), use_container_width=True, hide_index=True)
    else:
        st.caption("Click Find Stocks to scan.")