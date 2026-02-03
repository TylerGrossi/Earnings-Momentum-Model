"""
AI Strategy Assistant - Google Gemini Integration
==================================================
A conversational AI assistant for the Earnings Momentum Strategy.
Uses Google Gemini's free API to answer questions about your strategy data.

API Key Configuration:
- Set GEMINI_API_KEY environment variable, OR
- Add to Streamlit secrets (secrets.toml), OR
- Add to .env file
"""

import streamlit as st
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup

# Try to import sklearn for KNN
try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Try to import Google Generative AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Try to import dotenv for .env file support
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required


# ------------------------------------
# CONSTANTS FOR SCRAPING
# ------------------------------------
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# ------------------------------------
# MODEL CONFIGURATION
# ------------------------------------
# Available models - only ones that work with the API
AVAILABLE_MODELS = {
    "gemini-3-flash-preview": "Gemini 3 Flash",  # Default model (preview)
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite",
}


# ------------------------------------
# API KEY CONFIGURATION (BACKEND)
# ------------------------------------

def get_api_key():
    """
    Get Gemini API key from backend configuration.
    Priority order:
    1. Streamlit secrets (for Streamlit Cloud deployment)
    2. Environment variable
    3. .env file (loaded via dotenv)
    """
    # Option 1: Streamlit secrets (secrets.toml or Streamlit Cloud)
    try:
        if hasattr(st, 'secrets') and 'GEMINI_API_KEY' in st.secrets:
            return st.secrets['GEMINI_API_KEY']
    except:
        pass
    
    # Option 2: Environment variable
    api_key = os.environ.get('GEMINI_API_KEY')
    if api_key:
        return api_key
    
    # Option 3: Check for .env file loading
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        return api_key
    
    return None


# ------------------------------------
# DATA FUNCTIONS
# ------------------------------------

def load_returns_data():
    """Load returns tracker data from GitHub or local."""
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
    return pd.DataFrame()


def load_hourly_prices():
    """Load hourly prices data from GitHub or local."""
    urls = [
        "https://raw.githubusercontent.com/TylerGrossi/Scrapper/main/hourly_prices.csv",
        "https://raw.githubusercontent.com/TylerGrossi/Scrapper/master/hourly_prices.csv",
    ]
    for url in urls:
        try:
            df = pd.read_csv(url)
            if not df.empty and 'Trading Day' in df.columns:
                df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
                df['Earnings Date'] = pd.to_datetime(df['Earnings Date'], errors='coerce')
                return df
        except:
            continue
    return pd.DataFrame()


def filter_data(returns_df):
    """Apply standard filtering: remove DATE PASSED and require 5D Return."""
    if returns_df is None or returns_df.empty:
        return pd.DataFrame()
    
    df = returns_df.copy()
    
    if 'Date Check' in df.columns:
        df = df[df['Date Check'] != 'DATE PASSED']
    
    if '5D Return' in df.columns:
        df = df[df['5D Return'].notna()]
    
    return df


# ------------------------------------
# TOOL FUNCTIONS
# ------------------------------------

def get_strategy_rules() -> str:
    """Return the strategy rules."""
    rules = """
## Earnings Momentum Strategy Rules

### Entry Criteria (ALL must be met):
1. **Earnings This Week** - Stock has earnings scheduled for current week
2. **SMA20 > SMA50** - Golden cross (20-day SMA above 50-day SMA)  
3. **Barchart Buy Signal** - Stock shows "Buy" opinion on Barchart.com

### Entry Timing:
- **BMO (Before Market Open)**: Buy at previous day's close (~4pm day before)
- **AMC (After Market Close)**: Buy at earnings day close (~4pm on earnings day)

### Exit Rules:
1. **Stop Loss: -10%** - Exit if position drops 10% from entry
   - If stock gaps down below -10%, take the actual gap loss
2. **Time Exit: Day 5** - Exit at market close on trading day 5
3. **No Profit Cap** - Let winners run

### Position Sizing:
- Equal weight per trade
"""
    return rules


def get_earnings_this_week() -> str:
    """Get stocks with earnings this week."""
    df = load_returns_data()
    if df.empty:
        return "Could not load data from GitHub."
    
    today = datetime.today()
    days_since_sunday = (today.weekday() + 1) % 7
    week_start = today - timedelta(days=days_since_sunday)
    week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
    week_end = week_start + timedelta(days=6, hours=23, minutes=59, seconds=59)
    
    this_week = df[
        (df['Earnings Date'] >= week_start) & 
        (df['Earnings Date'] <= week_end)
    ].copy()
    
    if this_week.empty:
        return f"No stocks with earnings this week ({week_start.strftime('%b %d')} - {week_end.strftime('%b %d')}) in the tracker."
    
    result = f"## Earnings This Week ({week_start.strftime('%b %d')} - {week_end.strftime('%b %d')})\n\n"
    result += f"**{len(this_week)} stocks found:**\n\n"
    
    for _, row in this_week.iterrows():
        ticker = row.get('Ticker', 'N/A')
        company = row.get('Company Name', 'N/A')
        date = pd.to_datetime(row.get('Earnings Date')).strftime('%b %d') if pd.notna(row.get('Earnings Date')) else 'N/A'
        timing = row.get('Earnings Timing', 'N/A')
        price = row.get('Price', 'N/A')
        date_check = row.get('Date Check', 'N/A')
        
        result += f"- **{ticker}** ({company})\n"
        result += f"  - Earnings: {date} {timing}\n"
        result += f"  - Entry Price: ${price}\n"
        result += f"  - Status: {date_check}\n\n"
    
    return result


def get_stock_details(ticker: str) -> str:
    """Get details for a specific stock."""
    df = load_returns_data()
    if df.empty:
        return "Could not load data from GitHub."
    
    stock_data = df[df['Ticker'].str.upper() == ticker.upper()]
    if stock_data.empty:
        return f"Ticker {ticker.upper()} not found in the database. It may not have met the entry criteria recently."
    
    row = stock_data.sort_values('Earnings Date', ascending=False).iloc[0]
    
    ticker_str = row.get('Ticker', 'N/A')
    company = row.get('Company Name', 'N/A')
    
    result = f"## {ticker_str} - {company}\n\n"
    
    result += "### Company Info\n"
    result += f"- Sector: {row.get('Sector', 'N/A')}\n"
    result += f"- Market Cap: {row.get('Market Cap', 'N/A')}\n"
    result += f"- Beta: {row.get('Beta', 'N/A')}\n"
    result += f"- P/E Ratio: {row.get('P/E', 'N/A')}\n\n"
    
    result += "### Historical Earnings Trade\n"
    earnings_date = pd.to_datetime(row.get('Earnings Date'))
    result += f"- Earnings Date: {earnings_date.strftime('%Y-%m-%d') if pd.notna(earnings_date) else 'N/A'}\n"
    result += f"- Timing: {row.get('Earnings Timing', 'N/A')}\n"
    result += f"- Fiscal Quarter: {row.get('Fiscal Quarter', 'N/A')}\n"
    
    eps_est = row.get('EPS Estimate')
    eps_rep = row.get('Reported EPS')
    eps_surprise = row.get('EPS Surprise (%)')
    
    if pd.notna(eps_est):
        result += f"- EPS Estimate: ${eps_est}\n"
    else:
        result += f"- EPS Estimate: N/A\n"
    
    if pd.notna(eps_rep):
        result += f"- Reported EPS: ${eps_rep}\n"
    else:
        result += f"- Reported EPS: N/A\n"
    
    if pd.notna(eps_surprise):
        result += f"- EPS Surprise: {eps_surprise:+.1f}%\n\n"
    else:
        result += f"- EPS Surprise: N/A\n\n"
    
    result += "### Trade Entry\n"
    result += f"- Entry Price: ${row.get('Price', 'N/A')}\n"
    result += f"- Date Added: {row.get('Date Added', 'N/A')}\n"
    result += f"- Status: {row.get('Date Check', 'N/A')}\n\n"
    
    result += "### Returns (from entry price)\n"
    for day in ['1D', '2D', '3D', '4D', '5D']:
        ret = row.get(f'{day} Return')
        if pd.notna(ret):
            result += f"- {day} Return: {ret*100:+.2f}%\n"
        else:
            result += f"- {day} Return: N/A\n"
    
    return result


def get_strategy_performance() -> str:
    """Get overall strategy performance."""
    df = filter_data(load_returns_data())
    if df.empty:
        return "Could not load data from GitHub."
    
    total_trades = len(df)
    returns_5d = df['5D Return'].dropna() * 100
    
    wins = (returns_5d > 0).sum()
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    
    result = "## Strategy Performance Summary\n\n"
    result += f"**Total Completed Trades:** {total_trades}\n\n"
    
    result += "### 5-Day Return Statistics\n"
    result += f"- **Total Return:** {returns_5d.sum():+.1f}%\n"
    result += f"- **Average Return:** {returns_5d.mean():+.2f}%\n"
    result += f"- **Median Return:** {returns_5d.median():+.2f}%\n"
    result += f"- **Std Deviation:** {returns_5d.std():.2f}%\n"
    result += f"- **Best Trade:** {returns_5d.max():+.1f}%\n"
    result += f"- **Worst Trade:** {returns_5d.min():+.1f}%\n\n"
    
    result += f"### Win Rate\n"
    result += f"- **Wins:** {wins} ({win_rate:.1f}%)\n"
    result += f"- **Losses:** {total_trades - wins} ({100-win_rate:.1f}%)\n\n"
    
    # EPS analysis
    if 'EPS Surprise (%)' in df.columns:
        eps_data = df['EPS Surprise (%)'].dropna()
        if len(eps_data) > 0:
            beat_rate = (eps_data > 0).mean() * 100
            result += f"### Earnings Surprise\n"
            result += f"- **Trades with EPS Data:** {len(eps_data)}\n"
            result += f"- **Beat Rate:** {beat_rate:.1f}%\n"
            result += f"- **Median Surprise:** {eps_data.median():+.1f}%\n\n"
    
    # Sector breakdown
    if 'Sector' in df.columns:
        result += "### Top Sectors by Avg Return\n"
        sector_stats = df.groupby('Sector').agg({
            '5D Return': ['count', 'mean']
        }).round(4)
        sector_stats.columns = ['Count', 'Avg Return']
        sector_stats['Avg Return'] = sector_stats['Avg Return'] * 100
        sector_stats = sector_stats.sort_values('Avg Return', ascending=False).head(5)
        
        for sector, row in sector_stats.iterrows():
            result += f"- **{sector}:** {row['Avg Return']:+.2f}% avg ({int(row['Count'])} trades)\n"
    
    return result


def run_backtest(stop_loss_pct: float = -10, holding_days: int = 5) -> str:
    """Run a backtest with custom parameters."""
    returns_df = filter_data(load_returns_data())
    hourly_df = load_hourly_prices()
    
    if returns_df.empty or hourly_df.empty:
        return "Could not load data for backtest."
    
    stop_loss_pct = max(-50, min(-1, stop_loss_pct))
    holding_days = max(1, min(10, holding_days))
    stop_loss = stop_loss_pct / 100.0
    
    hourly_df = hourly_df.copy()
    hourly_df['Earnings Date'] = pd.to_datetime(hourly_df['Earnings Date']).dt.date
    returns_df = returns_df.copy()
    returns_df['Earnings Date'] = pd.to_datetime(returns_df['Earnings Date']).dt.date
    
    today = datetime.now().date()
    
    valid_trades = returns_df[
        (returns_df['5D Return'].notna()) & 
        (returns_df['Earnings Date'] <= (today - timedelta(days=7)))
    ]
    
    results = []
    exit_reasons = {'held_to_exit': 0, 'stop_loss': 0, 'gap_down': 0}
    
    for _, trade in valid_trades.iterrows():
        ticker = trade['Ticker']
        e_date = trade['Earnings Date']
        normal_return = trade['5D Return']
        
        trade_data = hourly_df[
            (hourly_df['Ticker'] == ticker) & 
            (hourly_df['Earnings Date'] == e_date) &
            (hourly_df['Trading Day'] >= 1)
        ].sort_values('Datetime')
        
        if trade_data.empty:
            continue
        
        exit_day = min(holding_days, trade_data['Trading Day'].max())
        exit_day_data = trade_data[trade_data['Trading Day'] == exit_day]
        if exit_day_data.empty:
            continue
        
        close_return = exit_day_data['Return From Earnings (%)'].iloc[-1] / 100
        
        final_return = close_return
        exit_reason = 'held_to_exit'
        first_candle = True
        
        for _, hour in trade_data.iterrows():
            if int(hour['Trading Day']) > exit_day:
                break
            
            h_ret = hour['Return From Earnings (%)'] / 100
            
            if h_ret <= stop_loss:
                if first_candle and h_ret < stop_loss:
                    final_return = h_ret
                    exit_reason = 'gap_down'
                else:
                    final_return = stop_loss
                    exit_reason = 'stop_loss'
                break
            
            first_candle = False
        
        results.append({
            'ticker': ticker,
            'normal_return': normal_return,
            'strategy_return': final_return,
            'exit_reason': exit_reason
        })
        exit_reasons[exit_reason] += 1
    
    if not results:
        return "No valid trades found for backtest."
    
    results_df = pd.DataFrame(results)
    
    normal_total = results_df['normal_return'].sum() * 100
    strategy_total = results_df['strategy_return'].sum() * 100
    alpha = strategy_total - normal_total
    
    wins = (results_df['strategy_return'] > 0).sum()
    win_rate = wins / len(results_df) * 100
    
    result = f"## Backtest Results\n\n"
    result += f"**Parameters:** {stop_loss_pct}% stop loss, {holding_days}-day holding period\n\n"
    result += f"**Total Trades:** {len(results_df)}\n\n"
    
    result += "### Performance Comparison\n"
    result += f"| Metric | Normal (5D Hold) | With Stop Loss |\n"
    result += f"|--------|------------------|----------------|\n"
    result += f"| Total Return | {normal_total:+.1f}% | {strategy_total:+.1f}% |\n"
    result += f"| Avg Return | {results_df['normal_return'].mean()*100:+.2f}% | {results_df['strategy_return'].mean()*100:+.2f}% |\n"
    result += f"| Win Rate | - | {win_rate:.1f}% |\n\n"
    
    result += f"### Alpha: {alpha:+.1f}%\n\n"
    
    result += "### Exit Breakdown\n"
    result += f"- **Held to Day {holding_days}:** {exit_reasons['held_to_exit']} trades\n"
    result += f"- **Stop Loss Triggered:** {exit_reasons['stop_loss']} trades\n"
    result += f"- **Gap Down Exit:** {exit_reasons['gap_down']} trades\n"
    
    return result


def compare_stop_losses() -> str:
    """Compare different stop loss levels."""
    returns_df = filter_data(load_returns_data())
    hourly_df = load_hourly_prices()
    
    if returns_df.empty or hourly_df.empty:
        return "Could not load data for comparison."
    
    result = "## Stop Loss Level Comparison\n\n"
    result += "Testing stop loss levels from -2% to -20%:\n\n"
    result += "| Stop Loss | Total Return | Alpha | Win Rate |\n"
    result += "|-----------|--------------|-------|----------|\n"
    
    best_sl = None
    best_return = -999
    
    hourly_df_copy = hourly_df.copy()
    hourly_df_copy['Earnings Date'] = pd.to_datetime(hourly_df_copy['Earnings Date']).dt.date
    returns_df_copy = returns_df.copy()
    returns_df_copy['Earnings Date'] = pd.to_datetime(returns_df_copy['Earnings Date']).dt.date
    
    today = datetime.now().date()
    
    valid_trades = returns_df_copy[
        (returns_df_copy['5D Return'].notna()) & 
        (returns_df_copy['Earnings Date'] <= (today - timedelta(days=7)))
    ]
    
    normal_total = valid_trades['5D Return'].sum() * 100
    
    for sl_pct in range(-2, -22, -2):
        sl = sl_pct / 100.0
        strategy_returns = []
        
        for _, trade in valid_trades.iterrows():
            ticker = trade['Ticker']
            e_date = trade['Earnings Date']
            
            trade_data = hourly_df_copy[
                (hourly_df_copy['Ticker'] == ticker) & 
                (hourly_df_copy['Earnings Date'] == e_date) &
                (hourly_df_copy['Trading Day'] >= 1)
            ].sort_values('Datetime')
            
            if trade_data.empty:
                continue
            
            exit_day = min(5, trade_data['Trading Day'].max())
            exit_day_data = trade_data[trade_data['Trading Day'] == exit_day]
            if exit_day_data.empty:
                continue
            
            close_return = exit_day_data['Return From Earnings (%)'].iloc[-1] / 100
            final_return = close_return
            first_candle = True
            
            for _, hour in trade_data.iterrows():
                if int(hour['Trading Day']) > exit_day:
                    break
                h_ret = hour['Return From Earnings (%)'] / 100
                if h_ret <= sl:
                    final_return = h_ret if (first_candle and h_ret < sl) else sl
                    break
                first_candle = False
            
            strategy_returns.append(final_return)
        
        if strategy_returns:
            total_ret = sum(strategy_returns) * 100
            alpha = total_ret - normal_total
            win_rate = sum(1 for r in strategy_returns if r > 0) / len(strategy_returns) * 100
            
            result += f"| {sl_pct}% | {total_ret:+.1f}% | {alpha:+.1f}% | {win_rate:.1f}% |\n"
            
            if total_ret > best_return:
                best_return = total_ret
                best_sl = sl_pct
    
    result += f"\n**Recommendation:** The **{best_sl}%** stop loss provides the highest total return ({best_return:+.1f}%).\n"
    
    return result


def get_beat_miss_analysis() -> str:
    """Analyze beat vs miss performance."""
    df = filter_data(load_returns_data())
    if df.empty:
        return "Could not load data."
    
    if 'EPS Surprise (%)' not in df.columns:
        return "No EPS surprise data available in the dataset."
    
    df = df[df['EPS Surprise (%)'].notna()].copy()
    df['Return Pct'] = df['5D Return'] * 100
    
    beats = df[df['EPS Surprise (%)'] > 0]
    misses = df[df['EPS Surprise (%)'] < 0]
    
    result = "## Beat vs Miss Analysis\n\n"
    result += f"**Total Trades with EPS Data:** {len(df)}\n\n"
    
    result += "### Earnings Beats\n"
    if len(beats) > 0:
        result += f"- **Count:** {len(beats)} trades\n"
        result += f"- **Total Return:** {beats['Return Pct'].sum():+.1f}%\n"
        result += f"- **Average Return:** {beats['Return Pct'].mean():+.2f}%\n"
        result += f"- **Win Rate:** {(beats['Return Pct'] > 0).mean()*100:.1f}%\n"
        result += f"- **Avg EPS Surprise:** {beats['EPS Surprise (%)'].mean():+.1f}%\n\n"
    else:
        result += "No beats in dataset.\n\n"
    
    result += "### Earnings Misses\n"
    if len(misses) > 0:
        result += f"- **Count:** {len(misses)} trades\n"
        result += f"- **Total Return:** {misses['Return Pct'].sum():+.1f}%\n"
        result += f"- **Average Return:** {misses['Return Pct'].mean():+.2f}%\n"
        result += f"- **Win Rate:** {(misses['Return Pct'] > 0).mean()*100:.1f}%\n"
        result += f"- **Avg EPS Surprise:** {misses['EPS Surprise (%)'].mean():+.1f}%\n\n"
    else:
        result += "No misses in dataset.\n\n"
    
    if len(beats) > 0 and len(misses) > 0:
        spread = beats['Return Pct'].mean() - misses['Return Pct'].mean()
        result += f"### Key Insight\n"
        result += f"Beats outperform misses by **{spread:+.2f}%** on average.\n"
    
    return result


def list_all_tickers() -> str:
    """List all tickers in the database."""
    df = load_returns_data()
    if df.empty:
        return "Could not load data."
    
    tickers = df['Ticker'].unique()
    
    result = f"## All Tickers in Database\n\n"
    result += f"**Total:** {len(tickers)} unique tickers\n\n"
    
    sorted_tickers = sorted(tickers)
    result += ", ".join(sorted_tickers)
    
    return result


def find_similar_tickers(ticker: str, n_neighbors: int = 10) -> str:
    """
    Find similar tickers based on financial metrics using KNN.
    Compares stocks based on P/E ratio, Beta, Market Cap, and Sector.
    
    Args:
        ticker: Stock ticker to find similar stocks for
        n_neighbors: Number of similar stocks to return (default: 10)
    
    Returns:
        Formatted string with similar tickers and their stats
    """
    if not SKLEARN_AVAILABLE:
        return "Error: scikit-learn package not installed. Run: `pip install scikit-learn`"
    
    df = load_returns_data()
    if df.empty:
        return "Could not load data."
    
    # Filter to only stocks with 5D Return data (completed trades)
    if '5D Return' in df.columns:
        df = df[df['5D Return'].notna()].copy()
    
    if df.empty:
        return "No stocks with 5D Return data found in the database."
    
    # Filter to most recent entry per ticker
    df = df.sort_values('Earnings Date', ascending=False)
    df = df.drop_duplicates(subset=['Ticker'], keep='first')
    
    # Find the target ticker
    target_row = df[df['Ticker'].str.upper() == ticker.upper()]
    if target_row.empty:
        return f"Ticker {ticker.upper()} not found in the database."
    
    target_row = target_row.iloc[0]
    
    # Prepare features for KNN
    # Features: P/E, Beta, Market Cap (numeric), Sector (encoded)
    feature_cols = ['P/E', 'Beta', 'Market Cap']
    
    # Parse Market Cap to numeric
    def parse_market_cap(val):
        if pd.isna(val) or val == 'N/A':
            return np.nan
        val = str(val).upper().replace(',', '').replace('$', '')
        multiplier = 1
        if 'T' in val:
            multiplier = 1_000_000_000_000
            val = val.replace('T', '')
        elif 'B' in val:
            multiplier = 1_000_000_000
            val = val.replace('B', '')
        elif 'M' in val:
            multiplier = 1_000_000
            val = val.replace('M', '')
        elif 'K' in val:
            multiplier = 1_000
            val = val.replace('K', '')
        try:
            return float(val) * multiplier
        except:
            return np.nan
    
    # Create feature dataframe
    feature_df = df.copy()
    feature_df['P/E Numeric'] = pd.to_numeric(feature_df['P/E'], errors='coerce')
    feature_df['Beta Numeric'] = pd.to_numeric(feature_df['Beta'], errors='coerce')
    feature_df['Market Cap Numeric'] = feature_df['Market Cap'].apply(parse_market_cap)
    
    # Encode Sector as numeric (simple approach)
    if 'Sector' in feature_df.columns:
        sectors = feature_df['Sector'].dropna().unique()
        sector_map = {sector: idx for idx, sector in enumerate(sectors)}
        feature_df['Sector Numeric'] = feature_df['Sector'].map(sector_map)
    else:
        feature_df['Sector Numeric'] = 0
    
    # Select features for KNN
    features = ['P/E Numeric', 'Beta Numeric', 'Market Cap Numeric', 'Sector Numeric']
    
    # Remove rows with missing values in any feature
    valid_mask = feature_df[features].notna().all(axis=1)
    feature_df_valid = feature_df[valid_mask].copy()
    
    if len(feature_df_valid) < n_neighbors + 1:
        return f"Not enough stocks with complete data. Found {len(feature_df_valid)} stocks with all metrics."
    
    # Check if target ticker has all features
    target_valid_mask = feature_df['Ticker'].str.upper() == ticker.upper()
    target_valid = valid_mask[target_valid_mask]
    if not target_valid.any():
        missing_features = []
        target_idx = feature_df[feature_df['Ticker'].str.upper() == ticker.upper()].index[0]
        target_row_check = feature_df.loc[target_idx]
        for feat in features:
            if pd.isna(target_row_check.get(feat)):
                missing_features.append(feat.replace(' Numeric', ''))
        return f"Ticker {ticker.upper()} is missing required metrics: {', '.join(missing_features)}. Cannot find similar stocks."
    
    # Prepare feature matrix
    X = feature_df_valid[features].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find target index in valid dataframe
    target_idx = feature_df_valid[feature_df_valid['Ticker'].str.upper() == ticker.upper()].index[0]
    target_pos = feature_df_valid.index.get_loc(target_idx)
    
    # Fit KNN
    nbrs = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(X_scaled)), algorithm='auto')
    nbrs.fit(X_scaled)
    
    # Find neighbors (including itself, so we'll exclude it)
    distances, indices = nbrs.kneighbors([X_scaled[target_pos]])
    
    # Get similar tickers (exclude the target itself)
    similar_indices = indices[0][1:]  # Skip first (itself)
    similar_distances = distances[0][1:]
    
    # Build result with target stock info
    result = f"## Similar Stocks to {ticker.upper()}\n\n"
    result += f"**Target Stock:** {target_row.get('Ticker', 'N/A')} - {target_row.get('Company Name', 'N/A')}\n"
    result += f"- P/E: {target_row.get('P/E', 'N/A')}\n"
    result += f"- Beta: {target_row.get('Beta', 'N/A')}\n"
    result += f"- Market Cap: {target_row.get('Market Cap', 'N/A')}\n"
    result += f"- Sector: {target_row.get('Sector', 'N/A')}\n"
    
    # Add target's 5D return if available
    if '5D Return' in target_row and pd.notna(target_row['5D Return']):
        result += f"- 5D Return: {target_row['5D Return']*100:+.2f}%\n"
    
    result += "\n"
    result += f"**Top {len(similar_indices)} Most Similar Stocks:**\n\n"
    
    # Create table header
    result += "| Rank | Ticker | Company | Similarity | P/E | Beta | Market Cap | Sector | 5D Return |\n"
    result += "|------|--------|---------|------------|-----|------|------------|--------|-----------|\n"
    
    # Add rows for similar stocks
    for i, (idx, dist) in enumerate(zip(similar_indices, similar_distances), 1):
        similar_row = feature_df_valid.iloc[idx]
        similarity_score = (1 - dist) * 100  # Convert distance to similarity percentage
        
        ticker_val = similar_row['Ticker']
        company = similar_row.get('Company Name', 'N/A')
        pe = similar_row.get('P/E', 'N/A')
        beta = similar_row.get('Beta', 'N/A')
        mcap = similar_row.get('Market Cap', 'N/A')
        sector = similar_row.get('Sector', 'N/A')
        
        # Format 5D return
        if '5D Return' in similar_row and pd.notna(similar_row['5D Return']):
            return_str = f"{similar_row['5D Return']*100:+.2f}%"
        else:
            return_str = "N/A"
        
        result += f"| {i} | {ticker_val} | {company} | {similarity_score:.1f}% | {pe} | {beta} | {mcap} | {sector} | {return_str} |\n"
    
    result += "\n_Note: Similarity is calculated based on P/E ratio, Beta, Market Cap, and Sector using K-Nearest Neighbors algorithm. Returns shown are 5-day returns from earnings date._"
    
    return result


def scan_live_signals() -> str:
    """
    Run live scanner to find stocks currently signaling:
    1. Scan Finviz for stocks with earnings this week + SMA20 > SMA50
    2. Check Barchart for buy signals
    3. Return list of stocks meeting all criteria
    """
    result = "## Live Stock Scanner Results\n\n"
    result += f"Scan Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    
    # Step 1: Get tickers from Finviz
    result += "### Step 1: Finviz Screener\n"
    result += "Criteria: Earnings this week + SMA20 crossed above SMA50\n\n"
    
    try:
        base_url = "https://finviz.com/screener.ashx?v=111&f=earningsdate_thisweek,ta_sma20_cross50a"
        offset = 0
        finviz_tickers = []
        
        while True:
            url = f"{base_url}&r={offset + 1}"
            response = requests.get(url, headers=HEADERS, timeout=15)
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
            
            finviz_tickers.extend(t for t in new_tickers if t not in finviz_tickers)
            offset += 20
            
            # Limit to prevent too many requests
            if offset >= 100:
                break
        
        if not finviz_tickers:
            result += "No stocks found matching Finviz criteria.\n\n"
            return result
        
        result += f"Found {len(finviz_tickers)} stocks: {', '.join(finviz_tickers)}\n\n"
        
    except Exception as e:
        result += f"Error scanning Finviz: {str(e)}\n\n"
        return result
    
    # Step 2: Check Barchart for buy signals
    result += "### Step 2: Barchart Buy Signal Filter\n\n"
    
    qualified_tickers = []
    
    for ticker in finviz_tickers:
        try:
            url = f"https://www.barchart.com/stocks/quotes/{ticker}/opinion"
            r = requests.get(url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")
            sig = soup.find("span", class_="opinion-signal buy")
            
            if sig and "Buy" in sig.text:
                qualified_tickers.append(ticker)
                result += f"- {ticker}: BUY signal confirmed\n"
            else:
                result += f"- {ticker}: No buy signal (filtered out)\n"
        except Exception as e:
            result += f"- {ticker}: Error checking ({str(e)})\n"
    
    result += "\n"
    
    # Step 3: Get details for qualified tickers
    if qualified_tickers:
        result += "### Qualified Stocks (All Criteria Met)\n\n"
        result += f"Total: {len(qualified_tickers)} stocks\n\n"
        
        for ticker in qualified_tickers:
            try:
                # Get Finviz data
                url = f"https://finviz.com/quote.ashx?t={ticker}"
                r = requests.get(url, headers=HEADERS, timeout=12)
                soup = BeautifulSoup(r.text, "html.parser")
                
                data = {"Earnings": "N/A", "Price": "N/A", "Market Cap": "N/A"}
                
                for t in soup.select("table.snapshot-table2"):
                    tds = t.find_all("td")
                    for i in range(0, len(tds) - 1, 2):
                        k, v = tds[i].get_text(strip=True), tds[i + 1].get_text(strip=True)
                        if k in data:
                            data[k] = v
                
                result += f"**{ticker}**\n"
                result += f"- Earnings: {data['Earnings']}\n"
                result += f"- Price: ${data['Price']}\n"
                result += f"- Market Cap: {data['Market Cap']}\n\n"
                
            except:
                result += f"**{ticker}** - Could not fetch details\n\n"
    else:
        result += "### No Stocks Currently Meet All Criteria\n\n"
        result += "No stocks passed both the Finviz screen AND have a Barchart buy signal.\n"
    
    # Also check what's already in the tracker for this week
    result += "---\n\n"
    result += "### Already in Tracker (This Week)\n\n"
    
    df = load_returns_data()
    if not df.empty:
        today = datetime.today()
        days_since_sunday = (today.weekday() + 1) % 7
        week_start = today - timedelta(days=days_since_sunday)
        week_end = week_start + timedelta(days=6)
        
        this_week = df[
            (df['Earnings Date'] >= week_start) & 
            (df['Earnings Date'] <= week_end)
        ]
        
        if not this_week.empty:
            for _, row in this_week.iterrows():
                ticker = row.get('Ticker', 'N/A')
                earnings_date = pd.to_datetime(row.get('Earnings Date'))
                timing = row.get('Earnings Timing', 'N/A')
                result += f"- {ticker}: {earnings_date.strftime('%b %d')} {timing}\n"
        else:
            result += "No stocks with earnings this week in tracker yet.\n"
    
    return result


def get_risk_metrics() -> str:
    """Calculate risk metrics for the strategy."""
    df = filter_data(load_returns_data())
    if df.empty:
        return "Could not load data for risk analysis."
    
    returns_5d = df['5D Return'].dropna() * 100  # Convert to percentage
    
    result = "## Strategy Risk Metrics\n\n"
    
    # Basic Stats
    total_trades = len(returns_5d)
    result += f"Total Trades Analyzed: {total_trades}\n\n"
    
    # Drawdown Analysis
    result += "### Drawdown Analysis\n"
    worst_trade = returns_5d.min()
    worst_idx = df['5D Return'].idxmin()
    worst_ticker = df.loc[worst_idx, 'Ticker'] if pd.notna(worst_idx) else 'N/A'
    best_trade = returns_5d.max()
    best_idx = df['5D Return'].idxmax()
    best_ticker = df.loc[best_idx, 'Ticker'] if pd.notna(best_idx) else 'N/A'
    
    result += f"- Worst Single Trade: {worst_trade:.2f}% ({worst_ticker})\n"
    result += f"- Best Single Trade: {best_trade:.2f}% ({best_ticker})\n"
    
    # Count significant losses
    losses_over_10 = (returns_5d < -10).sum()
    losses_over_20 = (returns_5d < -20).sum()
    losses_over_30 = (returns_5d < -30).sum()
    
    result += f"- Trades with > 10% loss: {losses_over_10} ({losses_over_10/total_trades*100:.1f}%)\n"
    result += f"- Trades with > 20% loss: {losses_over_20} ({losses_over_20/total_trades*100:.1f}%)\n"
    result += f"- Trades with > 30% loss: {losses_over_30} ({losses_over_30/total_trades*100:.1f}%)\n\n"
    
    # Volatility Metrics
    result += "### Volatility Metrics\n"
    std_dev = returns_5d.std()
    result += f"- Standard Deviation: {std_dev:.2f}%\n"
    result += f"- Return Range: {worst_trade:.2f}% to {best_trade:.2f}%\n"
    
    # Percentiles
    p5 = returns_5d.quantile(0.05)
    p25 = returns_5d.quantile(0.25)
    p75 = returns_5d.quantile(0.75)
    p95 = returns_5d.quantile(0.95)
    
    result += f"- 5th Percentile (worst 5%): {p5:.2f}%\n"
    result += f"- 25th Percentile: {p25:.2f}%\n"
    result += f"- 75th Percentile: {p75:.2f}%\n"
    result += f"- 95th Percentile (best 5%): {p95:.2f}%\n\n"
    
    # Risk-Adjusted Returns
    result += "### Risk-Adjusted Metrics\n"
    avg_return = returns_5d.mean()
    
    # Sharpe-like ratio (simplified, assuming 0 risk-free rate)
    sharpe_like = avg_return / std_dev if std_dev > 0 else 0
    result += f"- Average Return: {avg_return:.2f}%\n"
    result += f"- Return/Risk Ratio: {sharpe_like:.2f}\n"
    
    # Win/Loss Analysis
    wins = returns_5d[returns_5d > 0]
    losses = returns_5d[returns_5d < 0]
    
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    
    result += f"- Average Win: {avg_win:.2f}%\n"
    result += f"- Average Loss: {avg_loss:.2f}%\n"
    
    # Profit Factor
    total_wins = wins.sum()
    total_losses = abs(losses.sum())
    profit_factor = total_wins / total_losses if total_losses > 0 else 0
    result += f"- Profit Factor: {profit_factor:.2f} (total wins / total losses)\n\n"
    
    # Consecutive Analysis
    result += "### Streak Analysis\n"
    
    # Calculate consecutive wins/losses
    signs = (returns_5d > 0).astype(int)
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    current_wins = 0
    current_losses = 0
    
    for val in signs:
        if val == 1:
            current_wins += 1
            current_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, current_losses)
    
    result += f"- Max Consecutive Wins: {max_consecutive_wins}\n"
    result += f"- Max Consecutive Losses: {max_consecutive_losses}\n\n"
    
    # Worst 5 trades
    result += "### Worst 5 Trades\n"
    worst_5 = df.nsmallest(5, '5D Return')[['Ticker', 'Earnings Date', '5D Return', 'Sector']].copy()
    worst_5['5D Return'] = worst_5['5D Return'] * 100
    
    for _, row in worst_5.iterrows():
        ed = pd.to_datetime(row['Earnings Date']).strftime('%Y-%m-%d') if pd.notna(row['Earnings Date']) else 'N/A'
        result += f"- {row['Ticker']}: {row['5D Return']:.2f}% ({ed}, {row.get('Sector', 'N/A')})\n"
    
    result += "\n### Risk Summary\n"
    result += f"This strategy has a {(returns_5d > 0).mean()*100:.1f}% win rate with an average return of {avg_return:.2f}% per trade. "
    result += f"However, {losses_over_10} trades ({losses_over_10/total_trades*100:.1f}%) resulted in losses greater than 10%. "
    result += f"The worst single trade lost {worst_trade:.2f}%. Consider using stop losses to limit downside risk."
    
    return result


def analyze_by_filter(filter_type: str, filter_value: str) -> str:
    """
    Analyze strategy performance filtered by various criteria.
    
    filter_type: 'market_cap', 'sector', 'beta', 'timing'
    filter_value: the value to filter by (e.g., '100B', 'Technology', '1.5', 'BMO')
    """
    df = filter_data(load_returns_data())
    if df.empty:
        return "Could not load data."
    
    result = ""
    filtered_df = None
    filter_description = ""
    
    filter_type = filter_type.lower().strip()
    filter_value = filter_value.strip()
    
    # Market Cap Filter
    if filter_type in ['market_cap', 'marketcap', 'cap', 'market']:
        if 'Market Cap' not in df.columns:
            return "Market Cap data not available."
        
        # Parse market cap values to numeric
        def parse_market_cap(val):
            if pd.isna(val) or val == 'N/A':
                return None
            val = str(val).upper().replace(',', '').replace('$', '')
            multiplier = 1
            if 'T' in val:
                multiplier = 1_000_000_000_000
                val = val.replace('T', '')
            elif 'B' in val:
                multiplier = 1_000_000_000
                val = val.replace('B', '')
            elif 'M' in val:
                multiplier = 1_000_000
                val = val.replace('M', '')
            try:
                return float(val) * multiplier
            except:
                return None
        
        df['Market Cap Numeric'] = df['Market Cap'].apply(parse_market_cap)
        
        # Parse filter value
        filter_value_upper = filter_value.upper()
        threshold = None
        comparison = 'above'
        
        if 'ABOVE' in filter_value_upper or 'OVER' in filter_value_upper or '>' in filter_value_upper:
            comparison = 'above'
            # Extract number
            for part in filter_value_upper.replace('>', ' ').replace('ABOVE', ' ').replace('OVER', ' ').split():
                parsed = parse_market_cap(part)
                if parsed:
                    threshold = parsed
                    break
        elif 'BELOW' in filter_value_upper or 'UNDER' in filter_value_upper or '<' in filter_value_upper:
            comparison = 'below'
            for part in filter_value_upper.replace('<', ' ').replace('BELOW', ' ').replace('UNDER', ' ').split():
                parsed = parse_market_cap(part)
                if parsed:
                    threshold = parsed
                    break
        else:
            # Just a number, assume "above"
            threshold = parse_market_cap(filter_value)
        
        if threshold is None:
            return f"Could not parse market cap filter: {filter_value}. Try '100B', 'above 50B', 'below 10B'"
        
        # Format threshold for display
        if threshold >= 1_000_000_000_000:
            threshold_str = f"${threshold/1_000_000_000_000:.1f}T"
        elif threshold >= 1_000_000_000:
            threshold_str = f"${threshold/1_000_000_000:.0f}B"
        else:
            threshold_str = f"${threshold/1_000_000:.0f}M"
        
        if comparison == 'above':
            filtered_df = df[df['Market Cap Numeric'] >= threshold].copy()
            filter_description = f"Market Cap {threshold_str} or above"
        else:
            filtered_df = df[df['Market Cap Numeric'] < threshold].copy()
            filter_description = f"Market Cap below {threshold_str}"
    
    # Sector Filter
    elif filter_type == 'sector':
        if 'Sector' not in df.columns:
            return "Sector data not available."
        
        # Find matching sector (case insensitive partial match)
        sectors = df['Sector'].dropna().unique()
        matched_sector = None
        
        for s in sectors:
            if filter_value.lower() in s.lower():
                matched_sector = s
                break
        
        if not matched_sector:
            return f"Sector '{filter_value}' not found. Available sectors: {', '.join(sectors)}"
        
        filtered_df = df[df['Sector'] == matched_sector].copy()
        filter_description = f"Sector: {matched_sector}"
    
    # Beta Filter
    elif filter_type == 'beta':
        if 'Beta' not in df.columns:
            return "Beta data not available."
        
        df['Beta Numeric'] = pd.to_numeric(df['Beta'], errors='coerce')
        
        filter_value_upper = filter_value.upper()
        threshold = None
        comparison = 'above'
        
        if 'ABOVE' in filter_value_upper or 'OVER' in filter_value_upper or '>' in filter_value_upper:
            comparison = 'above'
            for part in filter_value_upper.replace('>', ' ').replace('ABOVE', ' ').replace('OVER', ' ').split():
                try:
                    threshold = float(part)
                    break
                except:
                    pass
        elif 'BELOW' in filter_value_upper or 'UNDER' in filter_value_upper or '<' in filter_value_upper:
            comparison = 'below'
            for part in filter_value_upper.replace('<', ' ').replace('BELOW', ' ').replace('UNDER', ' ').split():
                try:
                    threshold = float(part)
                    break
                except:
                    pass
        else:
            try:
                threshold = float(filter_value)
            except:
                pass
        
        if threshold is None:
            return f"Could not parse beta filter: {filter_value}. Try '1.5', 'above 1', 'below 0.5'"
        
        if comparison == 'above':
            filtered_df = df[df['Beta Numeric'] >= threshold].copy()
            filter_description = f"Beta {threshold} or above"
        else:
            filtered_df = df[df['Beta Numeric'] < threshold].copy()
            filter_description = f"Beta below {threshold}"
    
    # Timing Filter (BMO/AMC)
    elif filter_type in ['timing', 'time', 'earnings_timing']:
        if 'Earnings Timing' not in df.columns:
            return "Earnings timing data not available."
        
        filter_value_upper = filter_value.upper()
        if 'BMO' in filter_value_upper or 'BEFORE' in filter_value_upper or 'MORNING' in filter_value_upper:
            filtered_df = df[df['Earnings Timing'].str.upper().str.contains('BMO', na=False)].copy()
            filter_description = "Earnings Timing: BMO (Before Market Open)"
        elif 'AMC' in filter_value_upper or 'AFTER' in filter_value_upper or 'CLOSE' in filter_value_upper:
            filtered_df = df[df['Earnings Timing'].str.upper().str.contains('AMC', na=False)].copy()
            filter_description = "Earnings Timing: AMC (After Market Close)"
        else:
            return f"Could not parse timing filter: {filter_value}. Try 'BMO' or 'AMC'"
    
    else:
        return f"Unknown filter type: {filter_type}. Available filters: market_cap, sector, beta, timing"
    
    # Calculate stats for filtered data
    if filtered_df is None or filtered_df.empty:
        return f"No trades found matching filter: {filter_description}"
    
    total_trades = len(filtered_df)
    returns_5d = filtered_df['5D Return'].dropna() * 100
    
    if len(returns_5d) == 0:
        return f"No trades with return data found for filter: {filter_description}"
    
    # Build result
    result = f"## Performance Analysis: {filter_description}\n\n"
    result += f"Total Trades: {total_trades}\n\n"
    
    result += "### Return Statistics\n"
    result += f"- Total Return: {returns_5d.sum():.2f}%\n"
    result += f"- Average Return: {returns_5d.mean():.2f}%\n"
    result += f"- Median Return: {returns_5d.median():.2f}%\n"
    result += f"- Std Deviation: {returns_5d.std():.2f}%\n"
    result += f"- Best Trade: {returns_5d.max():.2f}%\n"
    result += f"- Worst Trade: {returns_5d.min():.2f}%\n\n"
    
    result += "### Win/Loss\n"
    wins = (returns_5d > 0).sum()
    win_rate = wins / len(returns_5d) * 100
    result += f"- Win Rate: {win_rate:.1f}% ({wins}/{len(returns_5d)})\n"
    
    avg_win = returns_5d[returns_5d > 0].mean() if (returns_5d > 0).any() else 0
    avg_loss = returns_5d[returns_5d < 0].mean() if (returns_5d < 0).any() else 0
    result += f"- Average Win: {avg_win:.2f}%\n"
    result += f"- Average Loss: {avg_loss:.2f}%\n\n"
    
    # Compare to overall
    all_returns = filter_data(load_returns_data())['5D Return'].dropna() * 100
    overall_avg = all_returns.mean()
    overall_win_rate = (all_returns > 0).mean() * 100
    
    result += "### Comparison to Overall Strategy\n"
    result += f"- This filter avg return: {returns_5d.mean():.2f}% vs Overall: {overall_avg:.2f}%\n"
    result += f"- This filter win rate: {win_rate:.1f}% vs Overall: {overall_win_rate:.1f}%\n"
    
    diff = returns_5d.mean() - overall_avg
    if diff > 0:
        result += f"- Outperforms overall by {diff:.2f}% per trade\n"
    else:
        result += f"- Underperforms overall by {abs(diff):.2f}% per trade\n"
    
    # List some example trades
    result += "\n### Sample Trades\n"
    sample = filtered_df.head(5)[['Ticker', 'Earnings Date', '5D Return', 'Market Cap']].copy()
    sample['5D Return'] = sample['5D Return'] * 100
    
    for _, row in sample.iterrows():
        ed = pd.to_datetime(row['Earnings Date']).strftime('%Y-%m-%d') if pd.notna(row['Earnings Date']) else 'N/A'
        result += f"- {row['Ticker']}: {row['5D Return']:.2f}% ({ed}, {row.get('Market Cap', 'N/A')})\n"
    
    return result


# ------------------------------------
# GEMINI INTEGRATION
# ------------------------------------

SYSTEM_PROMPT = """You are an AI assistant for the Earnings Momentum Strategy, a quantitative trading system. You help users understand their strategy performance, analyze data, and answer questions.

IMPORTANT DATA CONTEXT:
- The tracker stores HISTORICAL trades that already happened
- "Earnings Date" in the tracker is when the stock HAD earnings (past tense)
- Returns (1D, 2D, 3D, 4D, 5D) are the actual returns AFTER that earnings date
- You can also run a LIVE scanner to find stocks currently signaling

You have access to the following tools:

1. **get_strategy_rules** - Returns the entry/exit rules of the strategy
2. **scan_live_signals** - LIVE SCANNER: Scans Finviz and Barchart RIGHT NOW to find stocks currently meeting entry criteria
3. **get_earnings_this_week** - Lists stocks that HAD earnings this week from the tracker (historical)
4. **get_stock_details(ticker)** - Gets full details for a specific stock's HISTORICAL trade
5. **get_strategy_performance** - Returns overall historical performance metrics
6. **get_risk_metrics** - Returns detailed risk analysis: drawdowns, volatility, worst trades, profit factor, streaks
7. **analyze_by_filter(filter_type, filter_value)** - Analyze performance filtered by: market_cap, sector, beta, timing
8. **run_backtest(stop_loss_pct, holding_days)** - Runs a backtest with custom parameters
9. **compare_stop_losses** - Compares stop loss levels from -2% to -20%
10. **get_beat_miss_analysis** - Analyzes historical performance by earnings beat vs miss
11. **list_all_tickers** - Lists all tickers in the database
12. **find_similar_tickers(ticker, n_neighbors)** - Find similar stocks based on financial metrics (P/E, Beta, Market Cap, Sector) using KNN. Example: find_similar_tickers(AAPL, 5)

When a user asks a question:
1. Determine which tool(s) would help answer their question
2. Call the appropriate tool by responding with: TOOL_CALL: tool_name(arguments)
3. I will execute the tool and give you the results
4. Then provide a helpful, conversational response based on the data

IMPORTANT TOOL SELECTION:
- "What stocks are signaling?" or "What should I buy?" -> Use scan_live_signals()
- "What are the risks?" or "Risk factors" -> Use get_risk_metrics()
- Questions about market cap, sector, beta, or timing filters -> Use analyze_by_filter()
- "Tell me about [ticker]" -> Use get_stock_details(ticker)
- "Find similar stocks to [ticker]" or "What stocks are similar to [ticker]?" -> Use find_similar_tickers(ticker)

Examples:
- "What stocks are signaling this week?" -> TOOL_CALL: scan_live_signals()
- "What are the risk factors?" -> TOOL_CALL: get_risk_metrics()
- "Returns for stocks with market cap above 100 billion" -> TOOL_CALL: analyze_by_filter(market_cap, above 100B)
- "How do large cap stocks perform?" -> TOOL_CALL: analyze_by_filter(market_cap, above 50B)
- "Performance for technology sector" -> TOOL_CALL: analyze_by_filter(sector, Technology)
- "How do high beta stocks perform?" -> TOOL_CALL: analyze_by_filter(beta, above 1.5)
- "BMO vs AMC performance" -> TOOL_CALL: analyze_by_filter(timing, BMO)
- "Tell me about KSS" -> TOOL_CALL: get_stock_details(KSS)
- "What's the win rate?" -> TOOL_CALL: get_strategy_performance()
- "Test with 15% stop loss" -> TOOL_CALL: run_backtest(-15, 5)
- "Find similar stocks to AAPL" -> TOOL_CALL: find_similar_tickers(AAPL, 5)
- "What stocks are similar to TSLA?" -> TOOL_CALL: find_similar_tickers(TSLA, 5)

RESPONSE FORMATTING RULES:
- Be conversational and explain the data in plain English
- For historical data, use past tense
- For live scanner results, explain these are current opportunities
- Format numbers cleanly with proper spacing
- DO NOT use asterisks (*) for emphasis in your response - use plain text instead
- When listing stock details, format clearly: "GE: Price $325.12, Market Cap $342.94B, Earnings Jan 22 BMO"
- Keep responses concise and readable
"""


def execute_tool_call(tool_call: str) -> str:
    """Parse and execute a tool call from Gemini."""
    try:
        tool_call = tool_call.strip()
        
        if '(' not in tool_call:
            return f"Invalid tool call format: {tool_call}"
        
        tool_name = tool_call.split('(')[0].strip()
        args_str = tool_call.split('(')[1].rstrip(')')
        
        if tool_name == 'get_strategy_rules':
            return get_strategy_rules()
        elif tool_name == 'get_earnings_this_week':
            return get_earnings_this_week()
        elif tool_name == 'scan_live_signals':
            return scan_live_signals()
        elif tool_name == 'get_risk_metrics':
            return get_risk_metrics()
        elif tool_name == 'analyze_by_filter':
            # Parse two arguments: filter_type, filter_value
            # Handle various formats: (market_cap, above 100B) or ("market_cap", "above 100B")
            args_str_clean = args_str.strip()
            
            # Split on first comma only
            if ',' in args_str_clean:
                parts = args_str_clean.split(',', 1)
                filter_type = parts[0].strip().strip('"').strip("'").lower()
                filter_value = parts[1].strip().strip('"').strip("'")
                return analyze_by_filter(filter_type, filter_value)
            else:
                return "analyze_by_filter requires two arguments: filter_type and filter_value. Example: analyze_by_filter(market_cap, above 100B)"
        elif tool_name == 'get_stock_details':
            ticker = args_str.strip().strip('"').strip("'")
            return get_stock_details(ticker)
        elif tool_name == 'get_strategy_performance':
            return get_strategy_performance()
        elif tool_name == 'run_backtest':
            args = [a.strip() for a in args_str.split(',')]
            stop_loss = float(args[0]) if args[0] else -10
            holding_days = int(args[1]) if len(args) > 1 and args[1] else 5
            return run_backtest(stop_loss, holding_days)
        elif tool_name == 'compare_stop_losses':
            return compare_stop_losses()
        elif tool_name == 'get_beat_miss_analysis':
            return get_beat_miss_analysis()
        elif tool_name == 'list_all_tickers':
            return list_all_tickers()
        elif tool_name == 'find_similar_tickers':
            # Parse arguments: ticker, optional n_neighbors
            args_str_clean = args_str.strip()
            if ',' in args_str_clean:
                parts = args_str_clean.split(',', 1)
                ticker = parts[0].strip().strip('"').strip("'")
                try:
                    n_neighbors = int(parts[1].strip())
                except:
                    n_neighbors = 5
            else:
                ticker = args_str_clean.strip().strip('"').strip("'")
                n_neighbors = 5
            return find_similar_tickers(ticker, n_neighbors)
        else:
            return f"Unknown tool: {tool_name}"
    except Exception as e:
        return f"Error executing tool: {str(e)}"


def chat_with_gemini(user_message: str, chat_history: list, api_key: str, model_name: str) -> tuple[str, str]:
    """
    Send a message to Gemini and get a response.
    Returns tuple of (response_text, model_used)
    """
    if not GEMINI_AVAILABLE:
        return ("Error: google-generativeai package not installed. Run: `pip install google-generativeai`", "none")
    
    genai.configure(api_key=api_key)
    
    # Build conversation with system prompt
    messages = [SYSTEM_PROMPT]
    
    for msg in chat_history:
        messages.append(f"{msg['role'].upper()}: {msg['content']}")
    
    messages.append(f"USER: {user_message}")
    
    full_prompt = "\n\n".join(messages)
    
    try:
        model = genai.GenerativeModel(model_name)
        
        # First call - let Gemini decide if it needs a tool
        response = model.generate_content(full_prompt)
        response_text = response.text
        
        # Check if Gemini wants to call a tool
        if 'TOOL_CALL:' in response_text:
            # Extract tool call
            tool_line = [line for line in response_text.split('\n') if 'TOOL_CALL:' in line][0]
            tool_call = tool_line.split('TOOL_CALL:')[1].strip()
            
            # Execute the tool
            tool_result = execute_tool_call(tool_call)
            
            # Send tool result back to Gemini for final response
            follow_up = f"{full_prompt}\n\nASSISTANT: {response_text}\n\nTOOL_RESULT:\n{tool_result}\n\nNow provide a helpful response to the user based on this data. Be conversational and explain the key insights."
            
            final_response = model.generate_content(follow_up)
            return (final_response.text, model_name)
        else:
            return (response_text, model_name)
            
    except Exception as e:
        error_str = str(e).lower()
        
        if ('resource exhausted' in error_str or 
            'quota' in error_str or 
            '429' in error_str or 
            'rate limit' in error_str):
            return (f"Rate limit reached for {model_name}. Try selecting a different model from the dropdown.", model_name)
        elif ('404' in error_str or 
              'not found' in error_str or 
              'not supported' in error_str):
            return (f"Model '{model_name}' is not available via API. Try a different model.", model_name)
        else:
            return (f"Error: {str(e)}", model_name)


# ------------------------------------
# STREAMLIT TAB
# ------------------------------------

@st.fragment
def chat_fragment(api_key: str):
    """Fragment for the chat interface - only this part reruns on interaction."""
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize selected model
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = list(AVAILABLE_MODELS.keys())[0]
    
    # Model selector at top
    col1, col2 = st.columns([5, 2])
    with col2:
        selected = st.selectbox(
            "Model",
            options=list(AVAILABLE_MODELS.keys()),
            format_func=lambda x: AVAILABLE_MODELS[x],
            index=list(AVAILABLE_MODELS.keys()).index(st.session_state.selected_model),
            label_visibility="collapsed",
            key="model_selector"
        )
        st.session_state.selected_model = selected
    
    # Input row with clear button
    input_col, clear_col = st.columns([6, 1])
    
    with clear_col:
        clear_clicked = st.button("Clear", use_container_width=True, key="clear_chat")
    
    with input_col:
        prompt = st.chat_input("Ask about your strategy...")
    
    # Handle clear - wipe history and rerun to refresh the UI
    if clear_clicked:
        st.session_state.chat_history = []
        st.rerun()
    
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])
    
    # Process new message
    if prompt:
        # Display and save user message
        with st.chat_message('user'):
            st.markdown(prompt)
        st.session_state.chat_history.append({'role': 'user', 'content': prompt})
        
        # Get AI response
        response, model_used = chat_with_gemini(
            prompt, 
            st.session_state.chat_history[:-1],
            api_key,
            st.session_state.selected_model
        )
        
        # Display and save response
        with st.chat_message('assistant'):
            st.markdown(response)
        st.session_state.chat_history.append({'role': 'assistant', 'content': response})


def render_ai_assistant_tab():
    """Render the AI Assistant tab."""
    
    # Get API key from backend
    api_key = get_api_key()
    
    # Check dependencies
    if not GEMINI_AVAILABLE:
        st.error("Google Generative AI package not installed.")
        st.code("pip install google-generativeai", language="bash")
        return
    
    if not api_key:
        st.error("Gemini API key not configured.")
        st.markdown("""
        **For the administrator:** Configure the API key using one of these methods:
        
        **Option 1: Environment Variable**
        ```bash
        export GEMINI_API_KEY="your-api-key-here"
        ```
        
        **Option 2: Streamlit Secrets** (for Streamlit Cloud)
        
        Create `.streamlit/secrets.toml`:
        ```toml
        GEMINI_API_KEY = "your-api-key-here"
        ```
        
        **Option 3: .env File**
        ```
        GEMINI_API_KEY=your-api-key-here
        ```
        
        Get a free API key at [Google AI Studio](https://aistudio.google.com/apikey)
        """)
        return
    
    # Initialize
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "gemini-3-flash-preview"  # Default to Gemini 3 Flash
    
    # Header
    st.markdown("**Ask me anything about your Earnings Momentum Strategy**")
    
    # Empty state hint
    if len(st.session_state.chat_history) == 0:
        st.markdown("_Try: \"What stocks are signaling?\" or \"Show me the strategy performance\" or \"What are the risk metrics?\"_")
    
    # Render the chat fragment (fast - no full page rerun)
    chat_fragment(api_key)