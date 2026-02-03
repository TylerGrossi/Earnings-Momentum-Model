import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

# Handle zoneinfo import (built-in Python 3.9+, fallback for older versions)
try:
    from zoneinfo import ZoneInfo
    MARKET_TZ = ZoneInfo("America/New_York")
except ImportError:
    # Fallback for older Python versions
    try:
        from backports.zoneinfo import ZoneInfo
        MARKET_TZ = ZoneInfo("America/New_York")
    except ImportError:
        import pytz
        MARKET_TZ = pytz.timezone("America/New_York")

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


@st.cache_data(ttl=300)  # Cache for 5 minutes to avoid excessive API calls
def _get_stock_history(ticker, start_date_str, end_date_str):
    """Helper function to fetch stock history with caching."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date_str, end=end_date_str)
        return hist
    except:
        return None


def calculate_return_to_today(ticker, earnings_date, earnings_timing):
    """
    Calculate return from earnings date to today for a reported ticker.
    Uses yfinance to get historical prices.
    
    Entry price logic:
    - BMO: 4pm ET close on day BEFORE earnings (buy at previous day's close)
    - AMC: 4pm ET close on earnings day (buy at earnings day's close)
    
    Args:
        ticker: Stock ticker symbol
        earnings_date: Date or datetime of earnings
        earnings_timing: 'BMO' or 'AMC' or empty
    
    Returns:
        float: Return percentage, or None if calculation fails
    """
    if earnings_date is None:
        return None
    
    try:
        # Convert earnings_date to date
        if isinstance(earnings_date, str):
            earn_date = pd.to_datetime(earnings_date).date()
        elif hasattr(earnings_date, 'date'):
            earn_date = earnings_date.date()
        elif isinstance(earnings_date, pd.Timestamp):
            earn_date = earnings_date.date()
        else:
            earn_date = earnings_date
        
        # Normalize timing
        timing = str(earnings_timing).strip().upper() if pd.notna(earnings_timing) and earnings_timing else ""
        
        # Determine entry date (4pm ET close price on this date is the entry price)
        # BMO: Entry is 4pm ET on the day BEFORE earnings (buy at previous day's close)
        # AMC: Entry is 4pm ET on the earnings day (buy at earnings day's close)
        if "BMO" in timing:
            entry_date = earn_date - timedelta(days=1)  # Day before earnings
        else:
            entry_date = earn_date  # Earnings day
        
        # Get current date
        today = datetime.now(MARKET_TZ).date()
        
        # Fetch historical data: start a few days before entry_date to account for weekends/holidays
        # Use string format for dates which yfinance handles well
        start_date_str = (entry_date - timedelta(days=10)).strftime('%Y-%m-%d')
        end_date_str = (today + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Get historical data (with caching)
        hist = _get_stock_history(ticker, start_date_str, end_date_str)
        
        if hist is None or hist.empty or len(hist) == 0:
            return None
        
        # Convert index to date for comparison
        hist_dates = []
        for idx in hist.index:
            if isinstance(idx, pd.Timestamp):
                hist_dates.append(idx.date())
            elif hasattr(idx, 'date'):
                hist_dates.append(idx.date())
            else:
                hist_dates.append(pd.to_datetime(idx).date())
        
        # Find entry price: last trading day on or before entry_date
        entry_indices = [i for i, d in enumerate(hist_dates) if d <= entry_date]
        if not entry_indices:
            return None
        
        entry_idx = entry_indices[-1]
        entry_price = float(hist.iloc[entry_idx]['Close'])
        entry_date_actual = hist_dates[entry_idx]
        
        # Get current price: most recent close
        current_price = float(hist['Close'].iloc[-1])
        current_date_actual = hist_dates[-1]
        
        # Only calculate return if we have price data AFTER the entry date
        if current_date_actual <= entry_date_actual:
            return None
        
        # Validate prices
        if pd.isna(entry_price) or pd.isna(current_price) or entry_price == 0:
            return None
        
        # Calculate return percentage
        return_pct = ((current_price / entry_price) - 1) * 100
        return round(return_pct, 2)
        
    except Exception as e:
        # Return None on any error (silent failure for production)
        return None


def has_earnings_happened(earnings_date, earnings_timing):
    """
    Determine if earnings have already happened based on date and timing.
    
    Rules:
    - BMO (Before Market Open): 
        * If earnings are on Monday: Reported after 4pm ON Monday (can't buy on Sunday)
        * Otherwise: Reported after 4pm the day BEFORE earnings
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
    
    # Get current time in Eastern Time (market timezone)
    now_et = datetime.now(MARKET_TZ)
    today = now_et.date()
    current_hour = now_et.hour
    market_close_hour = 16  # 4pm ET
    
    # Convert to date if datetime
    if hasattr(earnings_date, 'date'):
        earn_date = earnings_date.date()
    else:
        earn_date = earnings_date
    
    # Normalize timing
    timing = str(earnings_timing).strip().upper() if pd.notna(earnings_timing) and earnings_timing else ""
    
    if "BMO" in timing:
        # BMO: Earnings happen before market open on earnings_date
        # Special case: If earnings are on Monday, use Monday 4pm as cutoff
        # (can't buy on Sunday since market is closed)
        if earn_date.weekday() == 0:  # Monday is 0
            # Monday BMO: Show "reported" after 4pm on Monday
            if today > earn_date:
                return True
            elif today == earn_date and current_hour >= market_close_hour:
                return True
            return False
        else:
            # Regular BMO: Show "reported" after 4pm the day BEFORE earnings_date
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
        skipped_tracked = []
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
                
                # Skip Monday BMO earnings (can't buy on Sunday)
                timing_upper = str(timing).strip().upper() if pd.notna(timing) and timing else ""
                if earnings_date is not None:
                    earn_date = earnings_date.date() if hasattr(earnings_date, 'date') else earnings_date
                    if "BMO" in timing_upper and earn_date.weekday() == 0:  # Monday is 0
                        skipped_tracked.append({
                            "Ticker": ticker,
                            "Earnings": earnings_str,
                            "Reason": "Monday BMO (no buy opportunity)"
                        })
                        continue
                
                # Determine status based on whether earnings have actually happened
                status = "Reported" if has_earnings_happened(earnings_date, timing) else "Upcoming"
                
                # Calculate return for reported tickers
                return_pct = None
                if status == "Reported" and earnings_date is not None:
                    return_pct = calculate_return_to_today(ticker, earnings_date, timing)
                
                tracked_rows.append({
                    "Ticker": ticker,
                    "Earnings": earnings_str,
                    "Price": format_value(row.get('Price')),
                    "P/E": format_value(row.get('P/E')),
                    "Beta": format_value(row.get('Beta')),
                    "Market Cap": format_market_cap(row.get('Market Cap')),
                    "Status": status,
                    "Return": f"{return_pct:+.2f}%" if return_pct is not None else "N/A"
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
        today = datetime.now(MARKET_TZ).date()  # Use Eastern Time date
        
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
            elif earnings_date and "BMO" in earnings_timing.upper() and earnings_date.weekday() == 0:  # Monday is 0
                skip_reason = "Monday BMO (no buy opportunity)"
            
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
                
                # Calculate return for reported tickers
                return_pct = None
                if status == "Reported" and earnings_date is not None:
                    return_pct = calculate_return_to_today(t, earnings_date, earnings_timing)
                
                new_rows.append({
                    "Ticker": t,
                    "Earnings": earnings_str,
                    "Price": data.get("Price", "N/A"),
                    "P/E": data.get("P/E", "N/A"),
                    "Beta": data.get("Beta", "N/A"),
                    "Market Cap": market_cap,
                    "Status": status,
                    "Return": f"{return_pct:+.2f}%" if return_pct is not None else "N/A"
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
            
            # Create dataframe
            df = pd.DataFrame(all_rows)[["Ticker", "Earnings", "Price", "P/E", "Beta", "Market Cap", "Status", "Return"]]
            
            # Calculate total return
            def parse_return(return_str):
                """Parse return string like '+5.23%' or '-2.15%' to float, or return None for 'N/A'."""
                if return_str == "N/A" or pd.isna(return_str):
                    return None
                try:
                    # Remove % and convert to float
                    return float(str(return_str).replace('%', '').replace('+', ''))
                except:
                    return None
            
            # Sum all valid returns
            returns = [parse_return(r['Return']) for r in all_rows]
            valid_returns = [r for r in returns if r is not None]
            total_return = sum(valid_returns) if valid_returns else None
            
            # Add total row
            if total_return is not None:
                total_row = {
                    "Ticker": "TOTAL",
                    "Earnings": "",
                    "Price": "",
                    "P/E": "",
                    "Beta": "",
                    "Market Cap": "",
                    "Status": f"{len(valid_returns)} reported",
                    "Return": f"{total_return:+.2f}%"
                }
                df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
            
            st.dataframe(
                df,
                use_container_width=True, 
                hide_index=True
            )
        
        # Combine all skipped tickers
        all_skipped = skipped_tracked + skipped
        if all_skipped:
            with st.expander(f"{len(all_skipped)} tickers skipped"):
                st.dataframe(pd.DataFrame(all_skipped), use_container_width=True, hide_index=True)
    else:
        st.caption("Click Find Stocks to scan.")