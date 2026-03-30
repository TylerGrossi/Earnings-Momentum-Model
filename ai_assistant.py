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

# yfinance for reliable current/latest prices in Returns table
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# Win probability model for Returns table
try:
    from win_probability_predictor import train_win_probability_model, predict_batch
    WIN_PROB_AVAILABLE = True
except ImportError:
    WIN_PROB_AVAILABLE = False

# Same filtering and stop-loss backtest as Earnings Analysis / Stop Loss tabs
try:
    from data_loader import apply_consistent_filtering
except ImportError:
    apply_consistent_filtering = None

try:
    from stop_loss_analysis import run_comparative_analysis
except ImportError:
    run_comparative_analysis = None

from utils import (
    compute_portfolio_return_for_return_series,
    _buy_date_from_earnings,
    CAPITAL_PER_TRADE,
)

# Try to import dotenv for .env file support
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required

# Market timezone for "reported" cutoff (match stock_screener.py)
try:
    from zoneinfo import ZoneInfo
    MARKET_TZ = ZoneInfo("America/New_York")
except ImportError:
    try:
        from backports.zoneinfo import ZoneInfo
        MARKET_TZ = ZoneInfo("America/New_York")
    except ImportError:
        try:
            import pytz
            MARKET_TZ = pytz.timezone("America/New_York")
        except ImportError:
            MARKET_TZ = None  # fallback: use UTC in has_earnings_happened


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

def _earnings_date_only(earnings_date):
    """Return earnings_date as a date (no time)."""
    if earnings_date is None:
        return None
    return earnings_date.date() if hasattr(earnings_date, "date") else earnings_date


def has_earnings_happened(earnings_date, earnings_timing):
    """
    True if earnings have already occurred (used to filter "reported" vs "upcoming").
    Matches stock_screener.py: BMO = reported after 4pm ET day before (or 4pm Monday for Monday BMO);
    AMC = reported after 4pm ET on earnings day.
    """
    if earnings_date is None or MARKET_TZ is None:
        return False
    now_et = datetime.now(MARKET_TZ)
    today = now_et.date()
    current_hour = now_et.hour
    market_close_hour = 16
    earn_date = _earnings_date_only(earnings_date)
    timing = str(earnings_timing).strip().upper() if pd.notna(earnings_timing) and earnings_timing else ""

    if "BMO" in timing:
        if earn_date.weekday() == 0:  # Monday BMO: reported after 4pm Monday
            if today > earn_date:
                return True
            if today == earn_date and current_hour >= market_close_hour:
                return True
            return False
        cutoff_date = earn_date - timedelta(days=1)
        if today > cutoff_date:
            return True
        if today == cutoff_date and current_hour >= market_close_hour:
            return True
        return False
    else:
        if today > earn_date:
            return True
        if today == earn_date and current_hour >= market_close_hour:
            return True
        return False


def load_returns_data():
    """Load returns tracker data from local file first, then GitHub."""
    # Try local returns_tracker.csv first (script dir or cwd)
    for base in [os.path.dirname(os.path.abspath(__file__)), os.getcwd()]:
        local_path = os.path.join(base, "returns_tracker.csv")
        if os.path.isfile(local_path):
            try:
                df = pd.read_csv(local_path)
                if not df.empty:
                    if '1D Return' not in df.columns and 'Price' in df.columns:
                        pass  # still valid for returns-this-week
                    df['Earnings Date'] = pd.to_datetime(df['Earnings Date'], errors='coerce')
                    return df
            except Exception:
                pass
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
        except Exception:
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
    """
    Match earnings_analysis / stop_loss / data_loader: exclude tickers with any DATE PASSED row,
    require valid 3D Return, Forward P/E <= 15 (or N/A per utils.filter_returns_by_forward_pe).
    """
    if returns_df is None or returns_df.empty:
        return pd.DataFrame()
    if apply_consistent_filtering is not None:
        filtered_returns, _, _ = apply_consistent_filtering(returns_df.copy(), None)
        if filtered_returns is None or filtered_returns.empty:
            return pd.DataFrame()
        return filtered_returns
    df = returns_df.copy()
    if 'Date Check' in df.columns:
        df = df[df['Date Check'] != 'DATE PASSED']
    if '3D Return' in df.columns:
        df = df[df['3D Return'].notna()].copy()
    return df


# ------------------------------------
# TOOL FUNCTIONS
# ------------------------------------

def get_strategy_rules() -> str:
    """Return the strategy rules (aligned with stock_screener and quant dashboard)."""
    rules = """
## Earnings Momentum Strategy Rules

### Entry Criteria (ALL must be met):
1. **Earnings This Week** - Stock has earnings scheduled for the current week
2. **SMA20 > SMA50** - 20-day SMA above 50-day SMA
3. **Barchart Buy Signal** - Barchart opinion shows "Buy"
4. **Low Forward P/E** - Forward P/E at most 15 when a numeric Forward P/E is present (matches tracker / screener)

### Entry Timing:
- **BMO (Before Market Open)**: Track from the previous session's close (before the earnings morning)
- **AMC (After Market Close)**: Track from the close on the earnings date

### Measured Holding Period (Dashboard):
- **3 trading days (3D Return)** is the primary post-earnings return metric in Earnings Analysis and Stop Loss optimization.

### Stop Loss (Backtest / Optimization Tab):
- Stop levels from **-1% through -20%** on a fixed grid (every 1% up to -10%, then -12, -14, -16, -18, -20).
- Intraday path uses **hourly** data through day 3; if price hits the stop, exit at the stop unless the first observation is a **gap down** through the stop (then the actual gap return is used).
- The app recommends a stop only if it **beats the normal 3D hold** on **portfolio return** (same capital / overlap logic as Power BI); otherwise it recommends **no stop loss**.

### Position Sizing:
- Equal capital per trade in the portfolio return model; analysis tabs use the filtered tracker universe (valid 3D, Forward P/E rule, DATE PASSED tickers excluded).
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


def _best_available_return(row, entry_f):
    """
    Return (current_price, return_pct) using best available: Return to Today, then 1D..3D.
    entry_f is the entry price (float). Returns (current_str, ret_str) for table cells.
    """
    if entry_f is None:
        return "—", "—"
    # Prefer Return to Today
    for col in ["Return to Today", "1D Return", "2D Return", "3D Return"]:
        if col not in row or pd.isna(row.get(col)):
            continue
        try:
            r = float(row[col])
        except (TypeError, ValueError):
            continue
        current_f = entry_f * (1 + r)
        pct = r * 100
        ret_str = f"{pct:+.2f}%" if pct >= 0 else f"{pct:.2f}%"
        return f"${current_f:.2f}", ret_str
    return "—", "—"


def _fetch_current_prices_yfinance(tickers):
    """
    Fetch current or latest closing price for each ticker via yfinance.
    Returns dict mapping ticker (uppercase) -> price (float). Uses latest available close.
    """
    if not tickers or not YFINANCE_AVAILABLE:
        return {}
    result = {}
    for t in tickers:
        sym = str(t).strip().upper()
        if not sym:
            continue
        try:
            obj = yf.Ticker(sym)
            # Prefer last close from history (most reliable); fallback to fast_info
            hist = obj.history(period="3d", interval="1d", auto_adjust=False)
            if hist is not None and not hist.empty and "Close" in hist.columns:
                last_close = float(hist["Close"].iloc[-1])
                if last_close > 0:
                    result[sym] = last_close
                    continue
            info = getattr(obj, "fast_info", None)
            if info is not None:
                try:
                    p = getattr(info, "last_price", None) or getattr(info, "previous_close", None)
                    if p is not None and float(p) > 0:
                        result[sym] = float(p)
                except Exception:
                    pass
        except Exception:
            pass
    return result


def _fetch_current_prices_gemini(tickers, api_key, model_name=None):
    """
    Ask Gemini for current or latest stock prices for the given tickers.
    Returns dict mapping ticker -> price (float), or {} on failure.
    """
    if not tickers or not api_key or not GEMINI_AVAILABLE:
        return {}
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name or list(AVAILABLE_MODELS.keys())[0])
        ticker_list = ", ".join(sorted(set(str(t).upper() for t in tickers)))
        today_str = datetime.now().strftime("%Y-%m-%d")
        prompt = (
            f"Today's date is {today_str}. For each US stock ticker below, give the most recent US market "
            f"closing price in US dollars (regular close, not adjusted for splits). "
            f"Tickers: {ticker_list}. "
            f"Return ONLY a valid JSON object: keys = ticker symbols, values = price numbers. "
            f"No explanation, no markdown. Example: {{\"AAPL\": 225.50, \"AMD\": 198.20}}"
        )
        response = model.generate_content(prompt)
        text = (response.text or "").strip()
        # Strip markdown code blocks if present
        if "```" in text:
            for start in ("```json", "```"):
                if start in text:
                    text = text.split(start, 1)[-1]
            text = text.split("```")[0].strip()
        data = json.loads(text)
        result = {}
        ticker_set = {str(t).upper() for t in tickers}
        for k, v in data.items():
            key = str(k).strip().upper()
            if key not in ticker_set:
                continue
            try:
                val = float(v)
                if val > 0:
                    result[key] = val
            except (TypeError, ValueError):
                pass
        return result
    except Exception:
        return {}


def get_returns_this_week() -> str:
    """
    Get returns for tickers that have already reported earnings this week (reported only).
    Uses same "reported" logic as stock_screener.py (has_earnings_happened). Fills current
    prices and returns from Gemini when API is available; otherwise tracker data or dash.
    """
    df = load_returns_data()
    if df.empty:
        return "Could not load returns tracker data."

    today = datetime.today()
    days_since_sunday = (today.weekday() + 1) % 7
    week_start = today - timedelta(days=days_since_sunday)
    week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
    week_end = week_start + timedelta(days=6, hours=23, minutes=59, seconds=59)

    # This week by date
    this_week = df[
        (df["Earnings Date"] >= week_start) & (df["Earnings Date"] <= week_end)
    ].copy()

    # Only stocks that have already reported (earnings have happened) — same as stock_screener "Reported"
    reported = []
    for _, row in this_week.iterrows():
        ed = row.get("Earnings Date")
        timing = row.get("Earnings Timing", "")
        if has_earnings_happened(ed, timing):
            reported.append(row)
    if not reported:
        return (
            f"No stocks in the tracker have **reported** earnings so far this week "
            f"({week_start.strftime('%b %d')} - {week_end.strftime('%b %d')}). "
            "Only tickers whose earnings have already occurred are included."
        )

    reported_df = pd.DataFrame(reported)
    count = len(reported_df)
    week_start_str = week_start.strftime("%B %d")
    week_end_str = week_end.strftime("%B %d")

    ticker_list = reported_df["Ticker"].tolist()
    # Prefer yfinance for accurate current/latest prices; fall back to Gemini then tracker
    yf_prices = _fetch_current_prices_yfinance(ticker_list) if YFINANCE_AVAILABLE else {}
    gemini_prices = _fetch_current_prices_gemini(ticker_list, get_api_key()) if get_api_key() else {}
    current_prices = yf_prices if yf_prices else gemini_prices
    prices_from_yf = bool(yf_prices)

    # Win probabilities: use returns_tracker value when present (from Earnings Momentum Model for new stocks), else predict via win_probability_predictor (same features as training: P/E or Forward P/E, Beta, Market Cap, Sector, Earnings Timing)
    win_probs_display = []
    if WIN_PROB_AVAILABLE:
        try:
            training_df = filter_data(load_returns_data())
            if training_df is not None and not training_df.empty and len(training_df) >= 10:
                ml_model, ml_encoders, _ = train_win_probability_model(training_df)
                # Use Forward P/E when present (matches win_probability_predictor training and Earnings Momentum Model)
                stock_data_list = [
                    {
                        "P/E": row.get("Forward P/E") if pd.notna(row.get("Forward P/E")) else row.get("P/E", "N/A"),
                        "Beta": row.get("Beta", "N/A"),
                        "Market Cap": row.get("Market Cap", "N/A"),
                        "Sector": row.get("Sector", "Unknown"),
                        "Earnings Timing": row.get("Earnings Timing", "Unknown"),
                    }
                    for _, row in reported_df.iterrows()
                ]
                win_probs = predict_batch(ml_model, ml_encoders, stock_data_list)
                # Prefer Win Probability from returns_tracker.csv when present (written by Earnings Momentum Model for new stocks)
                for i, (_, row) in enumerate(reported_df.iterrows()):
                    wp = row.get("Win Probability")
                    if wp is not None and not (pd.isna(wp) or wp == ""):
                        try:
                            p = float(wp)
                            win_probs_display.append(f"{p:.1%}" if 0 <= p <= 1 else "N/A")
                        except (TypeError, ValueError):
                            win_probs_display.append(f"{win_probs[i]:.1%}" if i < len(win_probs) and 0 <= win_probs[i] <= 1 else "N/A")
                    else:
                        win_probs_display.append(f"{win_probs[i]:.1%}" if i < len(win_probs) and 0 <= win_probs[i] <= 1 else "N/A")
            else:
                # Fallback: use tracker Win Probability when available
                for _, row in reported_df.iterrows():
                    wp = row.get("Win Probability")
                    if wp is not None and not (pd.isna(wp) or wp == ""):
                        try:
                            p = float(wp)
                            win_probs_display.append(f"{p:.1%}" if 0 <= p <= 1 else "N/A")
                        except (TypeError, ValueError):
                            win_probs_display.append("N/A")
                    else:
                        win_probs_display.append("N/A")
        except Exception:
            win_probs_display = ["N/A"] * len(reported_df)
    else:
        # No ML: show tracker Win Probability when present
        for _, row in reported_df.iterrows():
            wp = row.get("Win Probability")
            if wp is not None and not (pd.isna(wp) or wp == ""):
                try:
                    p = float(wp)
                    win_probs_display.append(f"{p:.1%}" if 0 <= p <= 1 else "N/A")
                except (TypeError, ValueError):
                    win_probs_display.append("N/A")
            else:
                win_probs_display.append("N/A")

    # Lead with table; short summary
    result = (
        f"**{count}** stocks have reported earnings so far this week "
        f"({week_start_str} – {week_end_str}). "
    )
    if prices_from_yf:
        result += "Current prices from market data (yfinance).\n\n"
    elif current_prices:
        result += "Current prices from Gemini when available; otherwise tracker.\n\n"
    else:
        result += "Returns use tracker data when available (Return to Today or 1D–3D).\n\n"
    result += "## Returns Summary\n\n"
    result += (
        "The **Return** column is price movement from the tracking start to current/latest price. "
        "Tracking start: BMO = previous day close; AMC = earnings day close (per strategy). "
    )
    if prices_from_yf:
        result += "Current/latest prices from yfinance (last close).\n\n"
    elif current_prices:
        result += "Current prices from Gemini API.\n\n"
    else:
        result += "Source: tracker when available.\n\n"
    result += "| Ticker | Reporting Time | Tracking Start (Price) | Current/Latest Price | Return (%) | Win Probability |\n"
    result += "|--------|----------------|------------------------|----------------------|------------|------------------|\n"

    returns_pct = []
    for i, (_, row) in enumerate(reported_df.iterrows()):
        ticker = row.get("Ticker", "N/A")
        company = row.get("Company Name", "N/A")
        short_name = (company.split(",")[0].strip() if isinstance(company, str) else str(company)) or ticker
        ed = pd.to_datetime(row.get("Earnings Date"))
        date_str = ed.strftime("%b %d") if pd.notna(ed) else "N/A"
        timing = row.get("Earnings Timing", "N/A")
        reporting = f"{timing} ({date_str})"
        entry_price = row.get("Price")
        entry_f = None
        if not (pd.isna(entry_price) or entry_price is None):
            entry_f = float(entry_price)
        # Tracking start label: BMO = previous day close, AMC = earnings day close (per strategy)
        timing_upper = str(timing).upper() if pd.notna(timing) else ""
        if pd.notna(ed) and ed is not None and "BMO" in timing_upper:
            track_date = ed - timedelta(days=1)
            track_date_str = track_date.strftime("%b %d")
        else:
            track_date_str = date_str
        entry_str = f"{track_date_str} Close (${entry_f:.2f})" if entry_f is not None else "N/A"
        # Prefer yfinance (then Gemini) current price; sanity-check Gemini only
        ticker_key = str(ticker).upper() if ticker else ""
        use_price = False
        pct = None
        if ticker_key in current_prices and entry_f is not None and entry_f > 0:
            current_f = current_prices[ticker_key]
            if current_f > 0.01:
                pct = (current_f - entry_f) / entry_f * 100
                if prices_from_yf or (-90 <= pct <= 400):  # trust yf; sanity-check Gemini
                    use_price = True
        if use_price:
            current_f = current_prices[ticker_key]
            pct = (current_f - entry_f) / entry_f * 100
            current_str = f"${current_f:.2f}"
            ret_str = f"{pct:+.2f}%" if pct >= 0 else f"{pct:.2f}%"
            returns_pct.append(pct)
        else:
            current_str, ret_str = _best_available_return(row, entry_f)
            if ret_str and ret_str != "—":
                try:
                    returns_pct.append(float(ret_str.replace("%", "").replace("+", "")))
                except ValueError:
                    pass
        win_prob_str = win_probs_display[i] if i < len(win_probs_display) else "N/A"
        result += f"| {ticker} ({short_name}) | {reporting} | {entry_str} | {current_str} | {ret_str} | {win_prob_str} |\n"

    if returns_pct:
        total_return = sum(returns_pct)
        avg_return = total_return / len(returns_pct)
        result += "\n**Total return:** " + f"{total_return:+.2f}%"
        result += " | **Average return:** " + f"{avg_return:+.2f}%\n\n"
    else:
        result += "\n"

    result += (
        "For a specific ticker's full post-earnings breakdown (1D–3D), ask for that ticker by name."
    )
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
    fwd_pe = row.get('Forward P/E')
    if pd.notna(fwd_pe) and fwd_pe != '' and str(fwd_pe).strip().upper() != 'N/A':
        result += f"- Forward P/E: {fwd_pe}\n"
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
    for day in ['1D', '2D', '3D']:
        ret = row.get(f'{day} Return')
        if pd.notna(ret):
            result += f"- {day} Return: {ret*100:+.2f}%\n"
        else:
            result += f"- {day} Return: N/A\n"
    
    return result


def get_strategy_performance() -> str:
    """
    Strategy performance echoing the Power BI dashboard cards:
    Max Concurrent, # Trades, Win%, Avg 3D Return, Portfolio Return,
    Annualized Return — all using the correct 3D sell-date math.
    """
    df = filter_data(load_returns_data())
    if df.empty:
        return "Could not load data from GitHub."

    m = _compute_dashboard_metrics(df)

    total_trades = len(df)
    returns_3d = df["3D Return"].dropna()
    wins = (returns_3d > 0).sum()
    win_pct = wins / len(returns_3d) * 100 if len(returns_3d) > 0 else 0
    avg_3d = returns_3d.mean() * 100

    result = "## Strategy Performance Summary\n\n"

    result += "### Key Metrics\n\n"
    result += "| Metric | Value |\n"
    result += "|--------|-------|\n"
    if m:
        result += f"| Max Concurrent Trades | {m['max_concurrent']} |\n"
    result += f"| Number of Trades | {total_trades} |\n"
    result += f"| Win % | {win_pct:.2f}% |\n"
    result += f"| Average 3D Return | {avg_3d:.2f}% |\n"
    if m:
        result += f"| Portfolio Return | {m['portfolio_return']*100:.2f}% |\n"
        result += f"| Annualized Return | {m['annualized_return']*100:.2f}% |\n"

    result += "\n### Per-Trade Stats\n"
    pct = returns_3d * 100
    result += f"- Median 3D Return: {pct.median():+.2f}%\n"
    result += f"- Best Trade: {pct.max():+.1f}%\n"
    result += f"- Worst Trade: {pct.min():+.1f}%\n"
    result += f"- Std Deviation: {pct.std():.2f}%\n\n"

    if "EPS Surprise (%)" in df.columns:
        eps_data = df["EPS Surprise (%)"].dropna()
        if len(eps_data) > 0:
            beat_rate = (eps_data > 0).mean() * 100
            result += "### Earnings Surprise\n"
            result += f"- Trades with EPS Data: {len(eps_data)}\n"
            result += f"- Beat Rate: {beat_rate:.1f}%\n"
            result += f"- Median Surprise: {eps_data.median():+.1f}%\n\n"

    if "Sector" in df.columns:
        result += "### Top Sectors by Avg Return\n"
        sector_stats = (
            df.groupby("Sector")["3D Return"]
            .agg(["count", "mean"])
            .rename(columns={"count": "Count", "mean": "Avg Return"})
        )
        sector_stats["Avg Return"] = sector_stats["Avg Return"] * 100
        sector_stats = sector_stats.sort_values("Avg Return", ascending=False).head(5)
        for sector, row in sector_stats.iterrows():
            result += f"- {sector}: {row['Avg Return']:+.2f}% avg ({int(row['Count'])} trades)\n"

    return result


def run_backtest(stop_loss_pct: float = -10, holding_days: int = 3) -> str:
    """Backtest one stop level vs normal 3D hold; horizon capped at 3 days to match Stop Loss tab."""
    returns_df = filter_data(load_returns_data())
    hourly_df = load_hourly_prices()
    
    if returns_df.empty or hourly_df.empty:
        return "Could not load data for backtest."
    
    stop_loss_pct = max(-50, min(-1, stop_loss_pct))
    holding_days = max(1, min(10, holding_days))
    holding_days = min(holding_days, 3)  # Align primary horizon with stop_loss_analysis / hourly grid
    stop_loss = stop_loss_pct / 100.0
    
    hourly_df = hourly_df.copy()
    hourly_df['Earnings Date'] = pd.to_datetime(hourly_df['Earnings Date']).dt.date
    returns_df = returns_df.copy()
    returns_df['Earnings Date'] = pd.to_datetime(returns_df['Earnings Date']).dt.date
    
    today = datetime.now().date()
    
    valid_trades = returns_df[
        (returns_df['3D Return'].notna()) & 
        (returns_df['Earnings Date'] <= (today - timedelta(days=7)))
    ]
    
    results = []
    exit_reasons = {'held_to_exit': 0, 'stop_loss': 0, 'gap_down': 0}
    
    for _, trade in valid_trades.iterrows():
        ticker = trade['Ticker']
        e_date = trade['Earnings Date']
        normal_return = trade['3D Return']
        
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
    result += f"| Metric | Normal (3D Hold) | With Stop Loss |\n"
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
    """Match Stop Loss tab: SL grid, 3-day path, matrix on avg return; best pick by portfolio return."""
    returns_df = filter_data(load_returns_data())
    hourly_df = load_hourly_prices()

    if returns_df.empty or hourly_df.empty:
        return "Could not load data for comparison."

    if run_comparative_analysis is None:
        return "Stop loss backtest helper is unavailable (stop_loss_analysis import failed)."

    master_results = run_comparative_analysis(hourly_df, returns_df, max_days=3)
    if master_results.empty:
        return (
            "No valid trades for stop-loss comparison. Need hourly_prices coverage and "
            "earnings dates at least 7 days in the past."
        )

    rd = returns_df.copy()
    rd["Earnings Date"] = pd.to_datetime(rd["Earnings Date"], errors="coerce")
    rd["_ed"] = rd["Earnings Date"].dt.date
    trades_meta = master_results[["Ticker", "Date"]].copy()
    trades_meta = trades_meta.merge(
        rd[["Ticker", "_ed", "Earnings Timing"]].drop_duplicates(subset=["Ticker", "_ed"]),
        left_on=["Ticker", "Date"],
        right_on=["Ticker", "_ed"],
        how="left",
    )
    trades_meta["Earnings Date"] = pd.to_datetime(trades_meta["Date"])

    base_col = "Normal Model (3D)"
    sl_cols = [c for c in master_results.columns if c.startswith("SL ")]

    port_ret_normal, _ = compute_portfolio_return_for_return_series(
        trades_meta,
        master_results[base_col],
        earnings_date_col="Earnings Date",
        timing_col="Earnings Timing",
    )
    port_ret_normal = port_ret_normal if port_ret_normal is not None else 0.0

    portfolio_returns = {}
    for col in sl_cols:
        pr, _ = compute_portfolio_return_for_return_series(
            trades_meta,
            master_results[col],
            earnings_date_col="Earnings Date",
            timing_col="Earnings Timing",
        )
        portfolio_returns[col] = pr if pr is not None else 0.0

    if not portfolio_returns:
        return (
            "## Stop Loss Level Comparison\n\n"
            "Backtest ran but produced no stop-loss columns.\n"
        )

    best_sl_col = max(portfolio_returns, key=portfolio_returns.get)
    best_port = portfolio_returns[best_sl_col]
    use_stop_loss = bool(best_sl_col and best_port > port_ret_normal)

    rd_3d = returns_df["3D Return"].dropna()
    avg_normal_3d = (
        (rd_3d.mean() * 100) if len(rd_3d) > 0 else float(master_results[base_col].mean() * 100)
    )

    result = "## Stop Loss Level Comparison\n\n"
    result += (
        "Aligned with the **Stop Loss Optimization** tab: **3 trading days**, hourly stops, gap-down rule, "
        "SL levels -1%…-10% then -12/-14/-16/-18/-20%. "
        f"Backtest: **{len(master_results)}** trades with hourly data; filtered universe **{len(returns_df)}** rows.\n\n"
    )

    result += "### Matrix (matches Performance Comparison Matrix)\n\n"
    result += "| Strategy | Avg Return (%) | Alpha vs Normal 3D Avg (%) | Win Rate (%) |\n"
    result += "|----------|----------------|-----------------------------|-------------|\n"

    sl_cols_sorted = sorted(
        sl_cols, key=lambda c: int(c.replace("SL ", "").replace("%", "")), reverse=True
    )
    for col in sl_cols_sorted:
        avg_ret = master_results[col].mean() * 100
        alpha_avg = avg_ret - avg_normal_3d
        wr = (master_results[col] > 0).mean() * 100
        result += f"| {col} | {avg_ret:+.2f}% | {alpha_avg:+.2f}% | {wr:.1f}% |\n"

    sub_normal_mean = master_results[base_col].mean() * 100
    result += (
        f"\n_Benchmark **Normal 3D Average** (full filtered universe, {len(returns_df)} trades): "
        f"{avg_normal_3d:+.2f}%. "
        f"Backtest subsample mean for normal 3D ({len(master_results)} trades): {sub_normal_mean:+.2f}%._\n"
    )

    result += "\n### Portfolio-based recommendation (same as tab)\n\n"
    pn, pb = port_ret_normal * 100, best_port * 100
    if use_stop_loss and best_sl_col:
        result += (
            f"**{best_sl_col}** maximizes portfolio return (**{pb:.2f}%** vs normal **{pn:.2f}%** "
            "under the model’s capital / overlap rules).\n"
        )
    else:
        result += (
            f"**No stop loss** — no tested stop beat the normal 3D hold on portfolio return "
            f"(normal **{pn:.2f}%**; best stop portfolio **{pb:.2f}%**).\n"
        )

    return result


def get_beat_miss_analysis() -> str:
    """Analyze beat vs miss performance (mean 3D return — same idea as Earnings Analysis tab)."""
    df = filter_data(load_returns_data())
    if df.empty:
        return "Could not load data."
    
    if 'EPS Surprise (%)' not in df.columns:
        return "No EPS surprise data available in the dataset."
    
    df = df[df['EPS Surprise (%)'].notna()].copy()
    df['Return Pct'] = df['3D Return'] * 100
    
    beats = df[df['EPS Surprise (%)'] > 0]
    misses = df[df['EPS Surprise (%)'] < 0]
    
    result = "## Beat vs Miss Analysis\n\n"
    result += (
        "_Filtered universe matches the dashboard (3D Return, Forward P/E rule, DATE PASSED). "
        "Beat = positive EPS surprise; metrics use **3D** returns._\n\n"
    )
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


def _pe_display_for_row(row) -> str:
    """Prefer Forward P/E for display and modeling (matches win probability / screener)."""
    fp = row.get("Forward P/E") if hasattr(row, "get") else None
    if fp is not None and not (pd.isna(fp) or str(fp).strip() == "" or str(fp).strip().upper() == "N/A"):
        return fp
    return row.get("P/E", "N/A") if hasattr(row, "get") else "N/A"


def find_similar_tickers(ticker: str, n_neighbors: int = 10) -> str:
    """
    Find similar tickers based on financial metrics using KNN.
    Compares stocks based on Forward P/E (fallback P/E), Beta, Market Cap, and Sector.
    
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
    
    # Filter to most recent entry per ticker (keep all stocks, even without 3D return)
    df_all = df.sort_values('Earnings Date', ascending=False)
    df_all = df_all.drop_duplicates(subset=['Ticker'], keep='first')
    
    # Find the target ticker (can be any stock, even without 3D return)
    target_row = df_all[df_all['Ticker'].str.upper() == ticker.upper()]
    if target_row.empty:
        return f"Ticker {ticker.upper()} not found in the database."
    
    target_row = target_row.iloc[0]
    
    # For KNN comparison, use all stocks with required features
    # But we'll filter results later to only show stocks with 3D returns
    df = df_all.copy()
    
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
    if "Forward P/E" in feature_df.columns:
        feature_df["P/E Numeric"] = pd.to_numeric(feature_df["Forward P/E"], errors="coerce")
        if "P/E" in feature_df.columns:
            feature_df["P/E Numeric"] = feature_df["P/E Numeric"].fillna(
                pd.to_numeric(feature_df["P/E"], errors="coerce")
            )
    else:
        feature_df["P/E Numeric"] = pd.to_numeric(feature_df["P/E"], errors="coerce")
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
    # Use a larger number to account for filtering out stocks without 3D returns
    max_neighbors = min(n_neighbors * 3, len(X_scaled))  # Get more candidates to filter
    nbrs_large = NearestNeighbors(n_neighbors=min(max_neighbors + 1, len(X_scaled)), algorithm='auto')
    nbrs_large.fit(X_scaled)
    distances, indices = nbrs_large.kneighbors([X_scaled[target_pos]])
    
    # Get similar tickers (exclude the target itself)
    candidate_indices = indices[0][1:]  # Skip first (itself)
    candidate_distances = distances[0][1:]
    
    # Filter to only show stocks with 3D returns in the results
    similar_indices = []
    similar_distances = []
    
    for idx, dist in zip(candidate_indices, candidate_distances):
        similar_row = feature_df_valid.iloc[idx]
        # Only include if it has 3D return data
        if '3D Return' in similar_row and pd.notna(similar_row['3D Return']):
            similar_indices.append(idx)
            similar_distances.append(dist)
            # Stop once we have enough results
            if len(similar_indices) >= n_neighbors:
                break
    
    if len(similar_indices) == 0:
        return f"Found similar stocks to {ticker.upper()}, but none of them have 3D return data available."
    
    # Calculate max distance for normalization (use the largest distance found)
    max_dist = max(similar_distances) if similar_distances else 1.0
    
    # Build result header
    result = f"## Similar Stocks to {ticker.upper()}\n\n"
    
    # Create table header
    result += "| Rank | Ticker | Company | Similarity | Fwd P/E | Beta | Market Cap | Sector | 3D Return |\n"
    result += "|------|--------|---------|------------|---------|------|------------|--------|-----------|\n"
    
    # Add target ticker as first row (reference row)
    target_ticker_val = target_row.get('Ticker', 'N/A')
    target_company = target_row.get('Company Name', 'N/A')
    target_pe = _pe_display_for_row(target_row)
    target_beta = target_row.get('Beta', 'N/A')
    target_mcap = target_row.get('Market Cap', 'N/A')
    target_sector = target_row.get('Sector', 'N/A')
    
    # Format target's 3D return
    if '3D Return' in target_row and pd.notna(target_row['3D Return']):
        target_return_str = f"{target_row['3D Return']*100:+.2f}%"
    else:
        target_return_str = "N/A"
    
    # Add target as reference row (Rank 0 or "Reference")
    result += f"| **Reference** | **{target_ticker_val}** | **{target_company}** | **100.0%** | {target_pe} | {target_beta} | {target_mcap} | {target_sector} | {target_return_str} |\n"
    
    # Add rows for similar stocks
    for i, (idx, dist) in enumerate(zip(similar_indices, similar_distances), 1):
        similar_row = feature_df_valid.iloc[idx]
        
        # Normalize similarity score: convert distance to similarity percentage
        # Use inverse distance normalized by max distance, then scale to 0-100%
        # Formula: similarity = (1 - (dist / max_dist)) * 100
        # This ensures scores are always positive and higher = more similar
        if max_dist > 0:
            similarity_score = (1 - (dist / max_dist)) * 100
        else:
            similarity_score = 100.0
        
        ticker_val = similar_row['Ticker']
        company = similar_row.get('Company Name', 'N/A')
        pe = _pe_display_for_row(similar_row)
        beta = similar_row.get('Beta', 'N/A')
        mcap = similar_row.get('Market Cap', 'N/A')
        sector = similar_row.get('Sector', 'N/A')
        
        # Format 3D return
        if '3D Return' in similar_row and pd.notna(similar_row['3D Return']):
            return_str = f"{similar_row['3D Return']*100:+.2f}%"
        else:
            return_str = "N/A"
        
        result += f"| {i} | {ticker_val} | {company} | {similarity_score:.1f}% | {pe} | {beta} | {mcap} | {sector} | {return_str} |\n"
    
    result += "\n**Similarity Score Explanation:**\n"
    result += "- The similarity score measures how similar each stock is to the reference stock (100%) based on financial metrics.\n"
    result += "- Scores range from 0% to 100%, where 100% = identical to reference stock, and lower scores = less similar.\n"
    result += "- Similarity uses K-Nearest Neighbors on Forward P/E (or P/E if Forward is missing), Beta, Market Cap, and Sector.\n"
    result += "- 3D Return is the three-day post-earnings return from the tracker.\n"
    
    return result


def scan_live_signals() -> str:
    """
    Live Finviz screen (earnings this week + SMA20 > SMA50) plus Barchart Buy.
    Full strategy also requires Forward P/E ≤ 15 on the tracker; Finviz URL does not apply that filter.
    """
    result = "## Live Stock Scanner Results\n\n"
    result += f"Scan Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    
    # Step 1: Get tickers from Finviz
    result += "### Step 1: Finviz Screener\n"
    result += "Criteria: Earnings this week · SMA20 above SMA50 (add Barchart Buy + Forward P/E ≤ 15 in the full strategy)\n\n"
    
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


def _sell_date_3d(buy_date):
    """
    Compute Sell Date (3D) from Buy Date — matches Power BI exactly.
    3 trading days: Mon/Tue buy -> +3 cal days, Wed/Thu/Fri -> +5 (crosses weekend).
    """
    from datetime import date as _date_type
    if buy_date is None:
        return None
    d = buy_date if isinstance(buy_date, _date_type) else buy_date.date()
    wk = d.weekday()  # Mon=0 .. Sun=6
    if wk <= 1:       # Mon/Tue
        days_add = 3
    elif wk <= 4:     # Wed/Thu/Fri
        days_add = 5
    elif wk == 5:     # Sat (rare)
        days_add = 4
    else:             # Sun (rare)
        days_add = 3
    return d + timedelta(days=days_add)


def _dax_weeknum_2(d):
    """Replicate DAX WEEKNUM(date, 2) — week starts on Monday, week 1 contains Jan 1."""
    jan1 = d.replace(month=1, day=1)
    day_of_year_0 = (d - jan1).days
    jan1_wd = jan1.weekday()  # Mon=0
    return (day_of_year_0 + jan1_wd) // 7 + 1


def _week_id_3d(sell_date):
    """YEAR(sell) * 100 + WEEKNUM(sell, 2) — matches Power BI Week ID (3D)."""
    if sell_date is None:
        return None
    return sell_date.year * 100 + _dax_weeknum_2(sell_date)


def _compute_dashboard_metrics(df):
    """
    One-stop function that replicates every headline number on the Power BI
    dashboard using the correct 3D sell date for all calculations.

    Returns dict with portfolio stats AND risk metrics, or None.
    """
    if df is None or df.empty:
        return None
    if "3D Return" not in df.columns or "Earnings Date" not in df.columns:
        return None

    df = df.copy()
    df["Earnings Date"] = pd.to_datetime(df["Earnings Date"], errors="coerce")
    df["3D Return"] = pd.to_numeric(df["3D Return"], errors="coerce")
    closed = df[df["3D Return"].notna()].copy()
    if closed.empty:
        return None

    timing_col = "Earnings Timing" if "Earnings Timing" in closed.columns else None

    closed["_buy_date"] = closed.apply(
        lambda r: _buy_date_from_earnings(
            r["Earnings Date"], r.get(timing_col) if timing_col else None
        ),
        axis=1,
    )
    closed = closed[closed["_buy_date"].notna()].copy()
    closed["_sell_date_3d"] = closed["_buy_date"].apply(_sell_date_3d)
    closed = closed[closed["_sell_date_3d"].notna()].copy()
    if closed.empty:
        return None

    closed["_trade_profit"] = closed["3D Return"] * CAPITAL_PER_TRADE
    closed["_week_id"] = closed["_sell_date_3d"].apply(_week_id_3d)

    # ------------------------------------------------------------------
    # Max Capital High Water Mark (3D)
    # ------------------------------------------------------------------
    global_start = closed["_buy_date"].min()
    max_sell = closed["_sell_date_3d"].max()
    max_concurrent = 0
    day = global_start
    while day <= max_sell:
        count = (
            (closed["_buy_date"] <= day) & (closed["_sell_date_3d"] >= day)
        ).sum()
        max_concurrent = max(max_concurrent, count)
        day += timedelta(days=1)
    if max_concurrent <= 0:
        max_concurrent = 1
    fund_size = max_concurrent * CAPITAL_PER_TRADE

    # ------------------------------------------------------------------
    # Portfolio Return & Annualized Return (3D)
    # ------------------------------------------------------------------
    total_net_profit = closed["_trade_profit"].sum()
    portfolio_return = total_net_profit / fund_size  # decimal

    start_date = closed["_buy_date"].min()
    end_date = closed["_sell_date_3d"].max()
    days_in_period = (end_date - start_date).days
    if days_in_period > 0:
        annualized_return = (1 + portfolio_return) ** (365 / days_in_period) - 1
    else:
        annualized_return = 0.0

    # ------------------------------------------------------------------
    # Weekly risk metrics — ALL grouped by Week ID (3D) per DAX
    # ------------------------------------------------------------------
    risk_free_rate = 0.04
    weekly_rf = risk_free_rate / 52.0

    weekly_profit = closed.groupby("_week_id")["_trade_profit"].sum()
    weekly_returns = weekly_profit / fund_size  # DIVIDE(SUM(Trade Profit), FundSize)

    n_weeks = len(weekly_returns)
    has_risk = n_weeks >= 2

    if has_risk:
        avg_wr = weekly_returns.mean()           # AVERAGEX(WeeklyData, [WeeklyReturn])
        sigma_p = weekly_returns.std(ddof=0)     # STDEVX.P

        # Sharpe: (AvgWeeklyReturn - WeeklyRiskFree) / WeeklySigma * SQRT(52)
        if sigma_p > 0:
            sharpe = (avg_wr - weekly_rf) / sigma_p * np.sqrt(52)
        else:
            sharpe = 0.0

        # Volatility: WeeklySigma * SQRT(52)
        ann_vol = sigma_p * np.sqrt(52)

        # VaR 95%: MeanReturn - 1.645 * WeeklySigma
        z_score = 1.645
        var_weekly = avg_wr - (z_score * sigma_p) if sigma_p > 0 else 0.0
        ann_var = var_weekly * np.sqrt(52)
    else:
        avg_wr = sigma_p = sharpe = ann_vol = 0.0
        var_weekly = ann_var = 0.0

    return {
        # portfolio cards
        "max_concurrent": max_concurrent,
        "n_trades": len(closed),
        "portfolio_return": portfolio_return,
        "annualized_return": annualized_return,
        "days_in_period": days_in_period,
        "fund_size": fund_size,
        # risk panel
        "has_risk": has_risk,
        "n_weeks": n_weeks,
        "sharpe": sharpe,
        "weekly_volatility": sigma_p,
        "ann_volatility": ann_vol,
        "var_weekly": var_weekly,
        "var_annualized": ann_var,
        "avg_weekly_return": avg_wr,
    }


def get_risk_metrics() -> str:
    """Weekly and Annualized Risk Metrics only: Sharpe, Volatility, VaR 95%."""
    df = filter_data(load_returns_data())
    if df.empty:
        return "Could not load data for risk analysis."

    m = _compute_dashboard_metrics(df)
    if m is None or not m["has_risk"]:
        return "Not enough weekly data for risk metrics (need 2+ weeks of trades)."

    result = "## Risk Metrics (3D)\n\n"
    result += "| Metric | Weekly | Annualized |\n"
    result += "|--------|--------|------------|\n"
    result += f"| Sharpe Ratio | — | {m['sharpe']:.2f} |\n"
    result += (
        f"| Volatility | {m['weekly_volatility']*100:.2f}% | "
        f"{m['ann_volatility']*100:.2f}% |\n"
    )
    result += (
        f"| VaR 95% | {m['var_weekly']*100:.2f}% | "
        f"{m['var_annualized']*100:.2f}% |\n"
    )

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
    returns_3d = filtered_df['3D Return'].dropna() * 100
    
    if len(returns_3d) == 0:
        return f"No trades with return data found for filter: {filter_description}"
    
    # Build result
    result = f"## Performance Analysis: {filter_description}\n\n"
    result += f"Total Trades: {total_trades}\n\n"
    
    result += "### Return Statistics\n"
    result += f"- Total Return: {returns_3d.sum():.2f}%\n"
    result += f"- Average Return: {returns_3d.mean():.2f}%\n"
    result += f"- Median Return: {returns_3d.median():.2f}%\n"
    result += f"- Std Deviation: {returns_3d.std():.2f}%\n"
    result += f"- Best Trade: {returns_3d.max():.2f}%\n"
    result += f"- Worst Trade: {returns_3d.min():.2f}%\n\n"
    
    result += "### Win/Loss\n"
    wins = (returns_3d > 0).sum()
    win_rate = wins / len(returns_3d) * 100
    result += f"- Win Rate: {win_rate:.1f}% ({wins}/{len(returns_3d)})\n"
    
    avg_win = returns_3d[returns_3d > 0].mean() if (returns_3d > 0).any() else 0
    avg_loss = returns_3d[returns_3d < 0].mean() if (returns_3d < 0).any() else 0
    result += f"- Average Win: {avg_win:.2f}%\n"
    result += f"- Average Loss: {avg_loss:.2f}%\n\n"
    
    # Compare to overall
    all_returns = filter_data(load_returns_data())['3D Return'].dropna() * 100
    overall_avg = all_returns.mean()
    overall_win_rate = (all_returns > 0).mean() * 100
    
    result += "### Comparison to Overall Strategy\n"
    result += f"- This filter avg return: {returns_3d.mean():.2f}% vs Overall: {overall_avg:.2f}%\n"
    result += f"- This filter win rate: {win_rate:.1f}% vs Overall: {overall_win_rate:.1f}%\n"
    
    diff = returns_3d.mean() - overall_avg
    if diff > 0:
        result += f"- Outperforms overall by {diff:.2f}% per trade\n"
    else:
        result += f"- Underperforms overall by {abs(diff):.2f}% per trade\n"
    
    # List some example trades
    result += "\n### Sample Trades\n"
    sample = filtered_df.head(5)[['Ticker', 'Earnings Date', '3D Return', 'Market Cap']].copy()
    sample['3D Return'] = sample['3D Return'] * 100
    
    for _, row in sample.iterrows():
        ed = pd.to_datetime(row['Earnings Date']).strftime('%Y-%m-%d') if pd.notna(row['Earnings Date']) else 'N/A'
        result += f"- {row['Ticker']}: {row['3D Return']:.2f}% ({ed}, {row.get('Market Cap', 'N/A')})\n"
    
    return result


# ------------------------------------
# GEMINI INTEGRATION
# ------------------------------------

SYSTEM_PROMPT = """You are an AI assistant for the Earnings Momentum Strategy, a quantitative trading system. You help users understand their strategy performance, analyze data, and answer questions.

IMPORTANT DATA CONTEXT:
- The tracker stores HISTORICAL trades that already happened
- "Earnings Date" is when the stock had earnings (past tense)
- The headline outcome in the app is **3D Return** (three trading days after the strategy entry). 1D/2D are also in the tracker when present
- Full strategy entry: earnings this week, SMA20 > SMA50, Barchart Buy, and **Forward P/E ≤ 15** when a numeric Forward P/E exists (see get_strategy_rules)
- **get_strategy_performance**, **get_risk_metrics**, **analyze_by_filter**, **get_beat_miss_analysis**, **compare_stop_losses**, and **run_backtest** use the SAME filtered universe as the Earnings Analysis and Stop Loss tabs: tickers with any DATE PASSED row removed, valid 3D Return, Forward P/E rule
- **get_returns_this_week** / **get_earnings_this_week** use the raw tracker slice for the week (not re-filtered by Forward P/E) and may use yfinance (then Gemini) for live prices when configured
- You can run **scan_live_signals** for a live Finviz + Barchart pass (Finviz does not encode the Forward P/E cap; mention that gap when relevant)

TOOLS:

1. **get_strategy_rules** - Entry rules, 3D focus, stop-loss grid, portfolio-based stop recommendation (matches app copy)
2. **scan_live_signals** - Live Finviz earnings+SMA screen, then Barchart Buy filter
3. **get_earnings_this_week** - Tracker rows with earnings this calendar week
4. **get_stock_details(ticker)** - Latest tracker row for a ticker (includes Forward P/E when present)
5. **get_returns_this_week** - Reported-this-week only; includes Returns Summary table (Ticker, Reporting Time, Tracking Start, Current/Latest Price, Return %, Win Probability). Preserve the table in your reply
6. **get_strategy_performance** - Win rate, 3D return stats, EPS surprise summary, sectors (filtered universe)
7. **get_risk_metrics** - Weekly and Annualized Risk Metrics ONLY: Sharpe Ratio, Volatility, VaR 95% (matching Power BI). Does NOT include performance stats — use get_strategy_performance for those
8. **analyze_by_filter(filter_type, filter_value)** - market_cap, sector, beta, timing (BMO/AMC); uses filtered universe
9. **run_backtest(stop_loss_pct, holding_days)** - One stop level vs normal 3D hold; horizon is capped at **3 days** to match hourly backtest. Default second argument **3** if omitted
10. **compare_stop_losses** - Same as Stop Loss tab: SL grid (-1..-10, then -12..-20), 3-day path, matrix of avg return / alpha vs full-universe normal 3D average, **portfolio-based** best stop or "no stop loss"
11. **get_beat_miss_analysis** - Beat vs miss on **3D** returns (aligned with Earnings Analysis)
12. **list_all_tickers** - Unique tickers in raw tracker load
13. **find_similar_tickers(ticker, n_neighbors)** - KNN on Forward P/E (fallback P/E), Beta, Market Cap, Sector; table includes **Fwd P/E** and **3D Return**. Example: find_similar_tickers(AAPL, 5)

When a user asks a question:
1. Pick tool(s) that match the new methodology (3D, Forward P/E universe, portfolio stop logic)
2. Respond with: TOOL_CALL: tool_name(arguments)
3. Use tool results in a clear, conversational answer

TOOL SELECTION:
- Optimal stop / compare stops / which SL -> compare_stop_losses()
- Single SL what-if -> run_backtest(-10, 3) style (stop % negative, days usually 3)
- Beat vs miss / EPS surprise vs returns -> get_beat_miss_analysis()
- Dashboard-style performance / win rate -> get_strategy_performance()
- Sharpe / volatility / VaR -> get_risk_metrics()
- Rules recap -> get_strategy_rules()
- Similar names / peers -> find_similar_tickers()

Examples:
- "What stop loss does the model recommend?" -> TOOL_CALL: compare_stop_losses()
- "Backtest a 7% stop" -> TOOL_CALL: run_backtest(-7, 3)
- "How do beats vs misses do on 3-day returns?" -> TOOL_CALL: get_beat_miss_analysis()
- "What's the win rate on the filtered universe?" -> TOOL_CALL: get_strategy_performance()
- "Returns for stocks that reported this week" -> TOOL_CALL: get_returns_this_week() and include the tool’s table
- "BMO vs AMC on 3D returns" -> TOOL_CALL: analyze_by_filter(timing, BMO) and mention AMC needs a second call or comparison

RESPONSE FORMATTING RULES:
- Be conversational; past tense for history
- Do not use asterisks for emphasis
- When listing stocks: include Forward P/E when relevant (e.g. GE: Forward P/E 12, Beta 1.1, …)

TABLE RULES:
- If the user wants a table of similar stocks with returns, use the find_similar_tickers output as-is (Fwd P/E, 3D Return)
- For generic return tables, include identifiers and key fundamentals plus 3D Return
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
        elif tool_name == 'get_returns_this_week':
            return get_returns_this_week()
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
            holding_days = int(args[1]) if len(args) > 1 and args[1] else 3
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


def _build_prompt(user_message: str, chat_history: list) -> str:
    """Assemble the full prompt from system prompt + history + new message."""
    parts = [SYSTEM_PROMPT]
    for msg in chat_history:
        parts.append(f"{msg['role'].upper()}: {msg['content']}")
    parts.append(f"USER: {user_message}")
    return "\n\n".join(parts)


def _stream_gemini(full_prompt: str, api_key: str, model_name: str):
    """Yield text chunks from Gemini (streaming). Falls back to single chunk."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    try:
        for chunk in model.generate_content(full_prompt, stream=True):
            text = chunk.text
            if text:
                yield text
    except Exception as first_err:
        if "stream" in str(first_err).lower():
            resp = model.generate_content(full_prompt)
            yield resp.text
        else:
            raise


def _friendly_error(e: Exception, model_name: str) -> str:
    err = str(e).lower()
    if any(k in err for k in ("resource exhausted", "quota", "429", "rate limit")):
        return f"Rate limit hit for {model_name} -- try a different model."
    if any(k in err for k in ("404", "not found", "not supported")):
        return f"Model *{model_name}* is not available. Pick another one from the dropdown."
    return f"Something went wrong: {e}"


# ------------------------------------
# SUGGESTION CHIPS
# ------------------------------------
SUGGESTIONS = [
    ("What stop loss does the model recommend?", "Stop Loss"),
    ("Show strategy performance", "Performance"),
    ("How do earnings beats vs misses look?", "Beats vs Misses"),
    ("What are the risk metrics?", "Risk"),
    ("What stocks are signaling this week?", "Live Signals"),
]

# ------------------------------------
# CSS
# ------------------------------------
_CHAT_CSS = """
<style>
/* welcome card */
.chat-welcome {
    text-align: center;
    padding: 2.5rem 1rem 1rem;
}
.chat-welcome h2 {
    font-size: 1.5rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 0.25rem;
}
.chat-welcome p {
    color: #94a3b8;
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
}

/* tool status pill */
.tool-pill {
    display: inline-block;
    background: #1e293b;
    color: #94a3b8;
    border: 1px solid #334155;
    border-radius: 999px;
    padding: 0.3rem 0.85rem;
    font-size: 0.8rem;
    margin-bottom: 0.5rem;
}
.tool-pill .dot {
    display: inline-block;
    width: 6px; height: 6px;
    background: #22c55e;
    border-radius: 50%;
    margin-right: 0.4rem;
    animation: pulse 1.2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}
</style>
"""

# ------------------------------------
# STREAMLIT TAB
# ------------------------------------

@st.fragment
def chat_fragment(api_key: str):
    """Main chat fragment -- reruns only this part on interaction."""

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = list(AVAILABLE_MODELS.keys())[0]
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None

    st.markdown(_CHAT_CSS, unsafe_allow_html=True)

    # ---- top bar: model selector + clear (compact) ----
    bar_spacer, bar_model, bar_clear = st.columns([5, 2, 1])
    with bar_model:
        st.selectbox(
            "model",
            options=list(AVAILABLE_MODELS.keys()),
            format_func=lambda k: AVAILABLE_MODELS[k],
            index=list(AVAILABLE_MODELS.keys()).index(st.session_state.selected_model),
            label_visibility="collapsed",
            key="model_sel",
        )
        st.session_state.selected_model = st.session_state.model_sel
    with bar_clear:
        if st.button("Clear", use_container_width=True, key="clear_btn"):
            st.session_state.chat_history = []
            st.session_state.pending_prompt = None
            st.rerun(scope="fragment")

    # ---- welcome screen (only when history is empty) ----
    if not st.session_state.chat_history and not st.session_state.pending_prompt:
        st.markdown(
            '<div class="chat-welcome">'
            "<h2>Earnings Momentum Assistant</h2>"
            "<p>Ask me anything about your strategy -- returns, stop losses, "
            "risk, earnings analysis, or live signals.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        cols = st.columns(len(SUGGESTIONS))
        for i, (full_q, label) in enumerate(SUGGESTIONS):
            if cols[i].button(label, key=f"chip_{i}", use_container_width=True):
                st.session_state.pending_prompt = full_q
                st.rerun(scope="fragment")

    # ---- replay history ----
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ---- chat input (outside columns so it pins to bottom) ----
    typed = st.chat_input("Ask about your strategy...", key="chat_box")

    # ---- resolve prompt (typed text OR chip click) ----
    prompt = typed or st.session_state.pending_prompt
    if st.session_state.pending_prompt and not typed:
        st.session_state.pending_prompt = None
    if not prompt:
        return

    # ---- user bubble ----
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ---- assistant response (streaming) ----
    full_prompt = _build_prompt(prompt, st.session_state.chat_history[:-1])
    model_name = st.session_state.selected_model

    with st.chat_message("assistant"):
        try:
            collected = ""
            placeholder = st.empty()
            for chunk in _stream_gemini(full_prompt, api_key, model_name):
                collected += chunk
                visible = collected.split("TOOL_CALL:")[0].strip()
                if visible:
                    placeholder.markdown(visible + " ...")
                else:
                    placeholder.markdown("Thinking ...")

            if "TOOL_CALL:" in collected:
                tool_lines = [l for l in collected.split("\n") if "TOOL_CALL:" in l]
                tool_call = tool_lines[0].split("TOOL_CALL:")[1].strip()
                tool_name_display = (
                    tool_call.split("(")[0].strip().replace("_", " ").title()
                )

                placeholder.markdown("")
                st.markdown(
                    '<span class="tool-pill">'
                    '<span class="dot"></span>'
                    f"Running {tool_name_display}..."
                    "</span>",
                    unsafe_allow_html=True,
                )
                tool_result = execute_tool_call(tool_call)

                follow_up = (
                    f"{full_prompt}\n\nASSISTANT: {collected}\n\n"
                    f"TOOL_RESULT:\n{tool_result}\n\n"
                    "Now provide a helpful, conversational response to the user "
                    "based on this data. Explain key insights clearly."
                )

                final_text = ""
                placeholder2 = st.empty()
                for chunk in _stream_gemini(follow_up, api_key, model_name):
                    final_text += chunk
                    placeholder2.markdown(final_text + " ...")
                placeholder2.markdown(final_text)
                collected = final_text
            else:
                placeholder.markdown(collected)

            if "TOOL_CALL:" in collected:
                collected = (
                    collected.split("TOOL_CALL:")[0].strip()
                    or "I ran into an issue processing that request. Please try again."
                )

            st.session_state.chat_history.append(
                {"role": "assistant", "content": collected}
            )

        except Exception as exc:
            err_msg = _friendly_error(exc, model_name)
            st.error(err_msg)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": err_msg}
            )


def render_ai_assistant_tab():
    """Render the AI Assistant tab."""

    api_key = get_api_key()

    if not GEMINI_AVAILABLE:
        st.error("Google Generative AI package not installed.")
        st.code("pip install google-generativeai", language="bash")
        return

    if not api_key:
        st.error("Gemini API key not configured.")
        st.markdown(
            "Set `GEMINI_API_KEY` as an environment variable, in "
            "`.streamlit/secrets.toml`, or in a `.env` file. "
            "Get a free key at [Google AI Studio](https://aistudio.google.com/apikey)."
        )
        return

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "gemini-3-flash-preview"

    chat_fragment(api_key)
