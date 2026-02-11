import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

# Use Eastern time for "this week" so behavior is consistent on any server (e.g. Streamlit Cloud UTC)
try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except ImportError:
    try:
        from backports.zoneinfo import ZoneInfo
        ET = ZoneInfo("America/New_York")
    except ImportError:
        import pytz
        ET = pytz.timezone("America/New_York")

# ------------------------------------
# DATA LOADING FUNCTIONS
# ------------------------------------

@st.cache_data(ttl=3600)
def load_returns_data_raw():
    """Load raw returns data without filtering."""
    urls = [
        "https://raw.githubusercontent.com/TylerGrossi/Scrapper/main/returns_tracker.csv",
        "https://raw.githubusercontent.com/TylerGrossi/Scrapper/master/returns_tracker.csv",
    ]
    for url in urls:
        try:
            df = pd.read_csv(url)
            if not df.empty and '1D Return' in df.columns:
                df['Earnings Date'] = pd.to_datetime(df['Earnings Date'], errors='coerce')
                return df
        except:
            continue
    try:
        df = pd.read_csv('returns_tracker.csv')
        df['Earnings Date'] = pd.to_datetime(df['Earnings Date'], errors='coerce')
        return df
    except:
        return None


@st.cache_data(ttl=3600)
def load_hourly_prices_raw():
    """Load raw hourly prices data without filtering."""
    urls = [
        "https://raw.githubusercontent.com/TylerGrossi/Scrapper/main/hourly_prices.csv",
        "https://raw.githubusercontent.com/TylerGrossi/Scrapper/master/hourly_prices.csv",
    ]
    for url in urls:
        try:
            df = pd.read_csv(url)
            if not df.empty and 'Trading Day' in df.columns:
                df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df['Earnings Date'] = pd.to_datetime(df['Earnings Date'], errors='coerce')
                return df
        except:
            continue
    try:
        df = pd.read_csv('hourly_prices.csv')
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Earnings Date'] = pd.to_datetime(df['Earnings Date'], errors='coerce')
        return df
    except:
        return None


# ------------------------------------
# FILTERING FUNCTIONS
# ------------------------------------

def apply_consistent_filtering(returns_df, hourly_df):
    """
    Apply consistent filtering across all datasets:
    1. Remove tickers where Date Check = 'DATE PASSED' (from returns_tracker)
    2. Only include tickers with valid 5D Return data
    
    Returns filtered dataframes and filter statistics.
    """
    filter_stats = {
        'original_returns_count': 0,
        'original_hourly_trades': 0,
        'date_passed_tickers': [],
        'date_passed_count': 0,
        'no_5d_return_count': 0,
        'final_count': 0,
        'final_tickers': []
    }
    
    # Get DATE PASSED tickers from returns_tracker itself
    date_passed_tickers = []
    if returns_df is not None and 'Date Check' in returns_df.columns:
        date_passed_tickers = returns_df[returns_df['Date Check'] == 'DATE PASSED']['Ticker'].tolist()
    
    filter_stats['date_passed_tickers'] = date_passed_tickers
    filter_stats['date_passed_count'] = len(date_passed_tickers)
    
    # Filter returns_df
    filtered_returns = None
    if returns_df is not None and not returns_df.empty:
        filter_stats['original_returns_count'] = len(returns_df)
        
        # Step 1: Remove DATE PASSED tickers
        if date_passed_tickers:
            filtered_returns = returns_df[~returns_df['Ticker'].isin(date_passed_tickers)].copy()
        else:
            filtered_returns = returns_df.copy()
        
        # Step 2: Only keep rows with valid 5D Return
        if '5D Return' in filtered_returns.columns:
            before_5d_filter = len(filtered_returns)
            filtered_returns = filtered_returns[filtered_returns['5D Return'].notna()].copy()
            filter_stats['no_5d_return_count'] = before_5d_filter - len(filtered_returns)
        
        filter_stats['final_count'] = len(filtered_returns)
        filter_stats['final_tickers'] = filtered_returns['Ticker'].unique().tolist()
    
    # Filter hourly_df using the same criteria
    filtered_hourly = None
    if hourly_df is not None and not hourly_df.empty:
        filter_stats['original_hourly_trades'] = hourly_df.groupby(['Ticker', 'Earnings Date']).ngroups
        
        # Step 1: Remove DATE PASSED tickers
        if date_passed_tickers:
            filtered_hourly = hourly_df[~hourly_df['Ticker'].isin(date_passed_tickers)].copy()
        else:
            filtered_hourly = hourly_df.copy()
        
        # Step 2: Only keep tickers that have valid 5D Return in returns_df
        if filtered_returns is not None:
            valid_ticker_dates = filtered_returns[['Ticker', 'Earnings Date']].drop_duplicates()
            valid_ticker_dates['key'] = valid_ticker_dates['Ticker'] + '_' + valid_ticker_dates['Earnings Date'].astype(str)
            filtered_hourly['key'] = filtered_hourly['Ticker'] + '_' + filtered_hourly['Earnings Date'].astype(str)
            filtered_hourly = filtered_hourly[filtered_hourly['key'].isin(valid_ticker_dates['key'])].copy()
            filtered_hourly = filtered_hourly.drop(columns=['key'])
    
    return filtered_returns, filtered_hourly, filter_stats


@st.cache_data(ttl=3600)
def load_and_filter_all_data():
    """Load and filter all data with consistent filtering."""
    returns_df = load_returns_data_raw()
    hourly_df = load_hourly_prices_raw()
    
    filtered_returns, filtered_hourly, filter_stats = apply_consistent_filtering(
        returns_df, hourly_df
    )
    
    return {
        'returns': filtered_returns,
        'hourly': filtered_hourly,
        'filter_stats': filter_stats
    }


# ------------------------------------
# THIS WEEK EARNINGS
# ------------------------------------

def get_this_week_earnings(df):
    """Get tickers from returns_tracker that had earnings this week (Sunday–Saturday in Eastern time)."""
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Week boundaries in Eastern so behavior is correct on any server (e.g. Streamlit Cloud UTC)
    now_et = datetime.now(ET)
    today_et = now_et.date()
    days_since_sunday = (today_et.weekday() + 1) % 7  # Sun=0, Mon=1, ..., Sat=6
    week_start_date = today_et - timedelta(days=days_since_sunday)
    week_end_date = week_start_date + timedelta(days=6)
    # Compare date parts so timezone of stored Earnings Date doesn't matter
    ed = pd.to_datetime(df['Earnings Date'], errors='coerce')
    earn_dates = ed.dt.date if hasattr(ed.dt, 'date') else ed.dt.normalize().dt.date
    mask = (earn_dates >= week_start_date) & (earn_dates <= week_end_date)
    return df.loc[mask].copy()
