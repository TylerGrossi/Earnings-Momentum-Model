import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

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

from utils import get_all_tickers, get_finviz_data, has_buy_signal, get_date_check, earnings_sort_key, parse_earnings_date
from data_loader import get_this_week_earnings
from win_probability_predictor import train_win_probability_model, predict_batch


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


def _earnings_date_only(earnings_date):
    """Return earnings_date as a date (no time)."""
    if earnings_date is None:
        return None
    return earnings_date.date() if hasattr(earnings_date, 'date') else earnings_date


def is_past_entry_deadline(earnings_date, earnings_timing):
    """
    True if we are past the time when the user could have entered the trade.
    Entry: BMO = 4pm day before earnings; AMC = 4pm day of earnings.
    If we're past that, a ticker appearing now cannot be added (too late to act).
    """
    if earnings_date is None:
        return False
    now_et = datetime.now(MARKET_TZ)
    today = now_et.date()
    hour = now_et.hour
    market_close_hour = 16  # 4pm ET
    earn_date = _earnings_date_only(earnings_date)
    timing = str(earnings_timing).strip().upper() if pd.notna(earnings_timing) and earnings_timing else ""

    if "BMO" in timing:
        # Entry was 4pm day before earnings
        entry_date = earn_date - timedelta(days=1)
        if today > entry_date:
            return True
        if today == entry_date and hour >= market_close_hour:
            return True
        return False
    else:
        # AMC or unknown: entry is 4pm on earnings day
        if today > earn_date:
            return True
        if today == earn_date and hour >= market_close_hour:
            return True
        return False


def has_earnings_happened(earnings_date, earnings_timing):
    """
    Determine if earnings have already happened based on date and timing.
    Used only for tickers already in the tracker (to show Reported vs Upcoming).

    Rules:
    - BMO (Before Market Open): Entry was 4pm day before. Earnings before open on earnings_date.
        * If earnings are on Monday: Show "reported" after 4pm ON Monday (can't buy on Sunday)
        * Otherwise: Show "reported" after 4pm the day BEFORE earnings
    - AMC (After Market Close): Entry is 4pm day of. Show "reported" after 4pm ON the earnings day.
    - No timing specified: Treat as AMC.
    """
    if earnings_date is None:
        return False

    now_et = datetime.now(MARKET_TZ)
    today = now_et.date()
    current_hour = now_et.hour
    market_close_hour = 16  # 4pm ET
    earn_date = _earnings_date_only(earnings_date)
    timing = str(earnings_timing).strip().upper() if pd.notna(earnings_timing) and earnings_timing else ""

    if "BMO" in timing:
        if earn_date.weekday() == 0:  # Monday BMO: reported after 4pm Monday
            if today > earn_date:
                return True
            if today == earn_date and current_hour >= market_close_hour:
                return True
            return False
        else:
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


def render_stock_screener_tab(raw_returns_df):
    """Render the Stock Screener tab."""
    
    st.markdown("### This Week's Earnings")
    st.markdown("**Criteria:** Earnings this week · SMA20 crossed above SMA50 · Barchart Buy Signal")
    
    if st.button("Find Stocks"):
        # Single progress bar and status for entire run (Finviz, Barchart, ML training)
        status_text = st.empty()
        progress = st.progress(0)

        # 1) Train ML model (show in same bar)
        status_text.text("Training win probability model...")
        progress.progress(0.02)
        ml_model = None
        ml_encoders = None
        ml_metrics = None
        ml_available = False
        if raw_returns_df is not None and not raw_returns_df.empty:
            training_df = raw_returns_df[raw_returns_df['5D Return'].notna()].copy()
            if len(training_df) >= 10:
                try:
                    ml_model, ml_encoders, ml_metrics = train_win_probability_model(training_df)
                    ml_available = True
                except Exception as e:
                    st.warning(f"Could not train ML model: {str(e)}")
                    ml_available = False
        progress.progress(0.08)

        # Get tickers already reported this week from returns tracker
        this_week_df = get_this_week_earnings(raw_returns_df)
        tracked_tickers = set()
        if raw_returns_df is not None and not raw_returns_df.empty:
            tracked_tickers = set(raw_returns_df['Ticker'].unique())
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
                
                # Get Sector (default to Unknown if not available)
                sector = row.get('Sector', 'Unknown')
                if pd.isna(sector) or sector == '':
                    sector = 'Unknown'
                
                tracked_rows.append({
                    "Ticker": ticker,
                    "Earnings": earnings_str,
                    "Price": format_value(row.get('Price')),
                    "P/E": format_value(row.get('P/E')),
                    "Beta": format_value(row.get('Beta')),
                    "Market Cap": format_market_cap(row.get('Market Cap')),
                    "Sector": sector,
                    "Earnings Timing": timing if pd.notna(timing) else 'Unknown',
                    "Status": status
                })
        
        # Scan Finviz (same bar)
        status_text.text("Scanning Finviz...")
        progress.progress(0.1)
        tickers = get_all_tickers()
        
        # Check Barchart (same bar)
        status_text.text(f"Found {len(tickers)} tickers. Checking Barchart signals...")
        barchart_passed = []
        n_t = len(tickers)
        for i, t in enumerate(tickers):
            if has_buy_signal(t):
                barchart_passed.append(t)
            progress.progress(0.1 + 0.4 * (i + 1) / n_t if n_t else 0.5)
        
        # Check earnings dates (same bar)
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
            
            # Skip conditions for Finviz tickers (not in tracker)
            # Reported section = only from returns_tracker. We never add a new ticker as "Reported".
            # Entry: BMO = 4pm day before earnings; AMC = 4pm day of. Past that = too late to add.
            skip_reason = None

            if date_info["Date Check"] == "DATE PASSED":
                skip_reason = "DATE PASSED"
            elif t in tracked_tickers_this_week:
                skip_reason = "Already in tracker"
            elif earnings_date and "BMO" in earnings_timing.upper() and earnings_date.weekday() == 0:
                skip_reason = "Monday BMO (no buy opportunity)"
            elif earnings_date and is_past_entry_deadline(earnings_date, earnings_timing):
                # Past entry deadline: BMO needed to be in by 4pm day before; AMC by 4pm day of
                skip_reason = "MISSED (past entry deadline)"
            elif earnings_date and earnings_date < today and t not in tracked_tickers:
                skip_reason = "MISSED (earnings passed, not in tracker)"

            if skip_reason:
                if skip_reason not in ["Already in tracker"]:
                    skipped.append({
                        "Ticker": t,
                        "Earnings": earnings_str,
                        "Reason": skip_reason
                    })
            else:
                # Only add tickers the user can still act on. Status is always "Upcoming" for
                # Finviz-origin rows (Reported only comes from returns_tracker).
                market_cap = data.get("Market Cap", "N/A")
                sector = 'Unknown'
                try:
                    import yfinance as yf
                    ticker_info = yf.Ticker(t).info
                    sector = ticker_info.get('sector', 'Unknown')
                    if not sector or sector == '':
                        sector = 'Unknown'
                except Exception:
                    sector = 'Unknown'

                new_rows.append({
                    "Ticker": t,
                    "Earnings": earnings_str,
                    "Price": data.get("Price", "N/A"),
                    "P/E": data.get("P/E", "N/A"),
                    "Beta": data.get("Beta", "N/A"),
                    "Market Cap": market_cap,
                    "Sector": sector,
                    "Earnings Timing": earnings_timing if earnings_timing else 'Unknown',
                    "Status": "Upcoming",
                })
            
            progress.progress(0.5 + 0.5 * (i + 1) / len(barchart_passed) if barchart_passed else 1.0)
        
        progress.progress(1.0)
        progress.empty()
        status_text.empty()

        # Combine tracked + new (from Finviz scan)
        all_rows = tracked_rows + new_rows
        
        # Sort by earnings date
        all_rows = sorted(all_rows, key=earnings_sort_key)
        
        if not all_rows:
            st.warning("No tickers match all criteria.")
            for key in ("screener_df", "screener_ml_available", "screener_ml_metrics", "screener_skipped"):
                if key in st.session_state:
                    del st.session_state[key]
        else:
            # Predict win probabilities if ML model is available
            if ml_available and ml_model is not None and ml_encoders is not None:
                try:
                    # Prepare stock data for prediction
                    stock_data_list = []
                    for row in all_rows:
                        stock_data_list.append({
                            'P/E': row.get('P/E', 'N/A'),
                            'Beta': row.get('Beta', 'N/A'),
                            'Market Cap': row.get('Market Cap', 'N/A'),
                            'Sector': row.get('Sector', 'Unknown'),
                            'Earnings Timing': row.get('Earnings Timing', 'Unknown')
                        })
                    
                    # Predict probabilities
                    win_probs = predict_batch(ml_model, ml_encoders, stock_data_list)
                    
                    # Add win probabilities to rows
                    for i, row in enumerate(all_rows):
                        if i < len(win_probs):
                            row['Win Probability'] = f"{win_probs[i]:.1%}"
                        else:
                            row['Win Probability'] = "N/A"
                except Exception as e:
                    st.warning(f"Could not predict win probabilities: {str(e)}")
                    for row in all_rows:
                        row['Win Probability'] = "N/A"
            else:
                # No ML model available
                for row in all_rows:
                    row['Win Probability'] = "N/A"
            
            reported_count = len([r for r in all_rows if r['Status'] == "Reported"])
            upcoming_count = len([r for r in all_rows if r['Status'] == "Upcoming"])
            
            # Build dataframe and save to session state so filters work on rerun
            display_cols = ["Ticker", "Earnings", "Price", "P/E", "Beta", "Market Cap", "Win Probability", "Status"]
            df = pd.DataFrame(all_rows)[display_cols].copy()
            df["Earnings Date"] = df["Earnings"].apply(parse_earnings_date)
            df["_win_prob_num"] = df["Win Probability"].apply(
                lambda x: float(str(x).replace("%", "")) / 100 if x != "N/A" and "%" in str(x) else 0.0
            )
            st.session_state.screener_df = df
            st.session_state.screener_ml_available = ml_available
            st.session_state.screener_ml_metrics = ml_metrics
            st.session_state.screener_skipped = skipped_tracked + skipped

        # Show table (no filters)
        if "screener_df" in st.session_state:
            _render_screener_results()

    else:
        st.caption("Click Find Stocks to scan.")


def _render_screener_results():
    """Render screener table only (no filters)."""
    if "screener_df" not in st.session_state:
        return
    df = st.session_state.screener_df.copy()

    # Sort by earnings date (earliest first); Earnings Date used for sort only, not displayed
    filtered = df.sort_values("Earnings Date", ascending=True)

    show_cols = ["Ticker", "Earnings", "Price", "P/E", "Beta", "Market Cap", "Win Probability", "Status"]
    filtered_display = filtered[[c for c in show_cols if c in filtered.columns]].copy()

    st.dataframe(filtered_display, use_container_width=True, hide_index=True, height=525)

    # ML Model Info
    ml_available = st.session_state.get("screener_ml_available", False)
    ml_metrics = st.session_state.get("screener_ml_metrics")
    if ml_available and ml_metrics is not None:
        st.markdown("---")
        with st.expander("ML Model Performance (Click to view)"):
            st.write("**Model Performance:**")
            st.write(f"- Accuracy: {ml_metrics['accuracy']:.1%}")
            st.write(f"- Precision: {ml_metrics['precision']:.1%}")
            st.write(f"- Recall: {ml_metrics['recall']:.1%}")
            st.write(f"- ROC-AUC: {ml_metrics['roc_auc']:.3f}")
            st.write(f"- Cross-validation accuracy: {ml_metrics['cv_accuracy_mean']:.1%} (±{ml_metrics['cv_accuracy_std']:.1%})")
            st.write(f"- Training samples: {ml_metrics['train_size']} | Test: {ml_metrics['test_size']} | Historical win rate: {ml_metrics['win_rate_train']:.1%}")
            imp = ml_metrics.get('feature_importance')
            if imp:
                st.write("**Feature importance (what drives predictions):**")
                for name, weight in sorted(imp.items(), key=lambda x: -x[1])[:10]:
                    st.write(f"- {name}: {weight:.3f}")

    all_skipped = st.session_state.get("screener_skipped", [])
    if all_skipped:
        with st.expander(f"{len(all_skipped)} tickers skipped"):
            st.dataframe(pd.DataFrame(all_skipped), use_container_width=True, hide_index=True)