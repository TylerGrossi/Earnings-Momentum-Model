import io
import logging
import streamlit as st
import pandas as pd
import time
from contextlib import redirect_stderr
from datetime import datetime, timedelta

try:
    import yfinance as yf
except ImportError:
    yf = None

# Eastern time (EST/EDT) for market hours — America/New_York is the correct US Eastern zone
try:
    from zoneinfo import ZoneInfo
    MARKET_TZ = ZoneInfo("America/New_York")
except ImportError:
    try:
        from backports.zoneinfo import ZoneInfo
        MARKET_TZ = ZoneInfo("America/New_York")
    except ImportError:
        import pytz
        MARKET_TZ = pytz.timezone("America/New_York")

from utils import get_all_tickers, get_finviz_data, has_buy_signal, earnings_sort_key, parse_earnings_date
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
    market_close_hour = 16  # 4pm EST
    earn_date = _earnings_date_only(earnings_date)
    timing = str(earnings_timing).strip().upper() if pd.notna(earnings_timing) and earnings_timing else ""

    if "BMO" in timing:
        # Entry was 4pm EST day before earnings
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
    market_close_hour = 16  # 4pm EST
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


def _entry_date_for_return(earnings_date, earnings_timing):
    """
    Date whose 4pm EST close we use as entry price.
    BMO: day before earnings; AMC: day of earnings.
    If that date is weekend, use previous Friday.
    """
    if earnings_date is None:
        return None
    earn_date = _earnings_date_only(earnings_date)
    timing = str(earnings_timing).strip().upper() if pd.notna(earnings_timing) and earnings_timing else ""
    if "BMO" in timing:
        entry_date = earn_date - timedelta(days=1)
    else:
        entry_date = earn_date
    # If weekend, use previous Friday
    while entry_date.weekday() >= 5:
        entry_date -= timedelta(days=1)
    return entry_date


def get_reported_return(ticker, earnings_date, earnings_timing):
    """
    Return (return_pct, entry_price, current_price) for reported earnings.
    return_pct = (current - entry) / entry as decimal (e.g. 0.05 for 5%).
    Entry = close on entry_date (4pm EST = daily close; BMO = day before earnings, AMC = day of).
    Uses yfinance daily data. Returns None on error or if yfinance unavailable.
    """
    if yf is None:
        return None
    entry_date = _entry_date_for_return(earnings_date, earnings_timing)
    if entry_date is None:
        return None
    now_est = datetime.now(MARKET_TZ)
    today = now_est.date()
    if entry_date > today:
        return None
    try:
        # Request extra history so we have entry_date even if yfinance trims or delays
        start = entry_date - timedelta(days=60)
        end = today + timedelta(days=1)
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")
        # Suppress yfinance "Failed to get ticker" / "No timezone found" (stderr + logging)
        yf_log = logging.getLogger("yfinance")
        old_level = yf_log.level
        yf_log.setLevel(logging.CRITICAL)
        if getattr(yf, "config", None) is not None and hasattr(yf.config, "debug"):
            try:
                old_logging = getattr(yf.config.debug, "logging", True)
                yf.config.debug.logging = False
            except Exception:
                old_logging = True
        else:
            old_logging = True
        try:
            hist = None
            for attempt in range(3):  # up to 3 attempts (429 / empty on Cloud)
                with redirect_stderr(io.StringIO()):
                    hist = yf.Ticker(ticker).history(
                        interval="1d",
                        start=start_str,
                        end=end_str,
                        auto_adjust=True,
                    )
                if hist is not None and not hist.empty and "Close" in hist.columns:
                    break
                if attempt < 2:
                    time.sleep(2.0)  # back off before retry (helps 429 on Streamlit Cloud)
        finally:
            yf_log.setLevel(old_level)
            if getattr(yf, "config", None) is not None and hasattr(yf.config, "debug"):
                try:
                    yf.config.debug.logging = old_logging
                except Exception:
                    pass
        if hist is None or hist.empty or "Close" not in hist.columns:
            return None
        # Use date part of index (if tz-aware, convert to Eastern so date = US trading day)
        idx = hist.index
        try:
            if hasattr(idx, "tz") and idx.tz is not None:
                idx = idx.tz_convert(MARKET_TZ)
        except Exception:
            pass
        hist_dates = pd.Series(idx).dt.date
        mask = hist_dates <= entry_date
        entry_bars = hist.loc[mask]
        if entry_bars.empty:
            return None
        entry_price = float(entry_bars["Close"].iloc[-1])
        current_price = float(hist["Close"].iloc[-1])
        if entry_price <= 0:
            return None
        ret_pct = (current_price - entry_price) / entry_price
        return (ret_pct, entry_price, current_price)
    except Exception:
        return None


def render_stock_screener_tab(raw_returns_df):
    """Render the Stock Screener tab."""
    
    st.markdown("### This Week's Earnings")
    st.markdown("**Criteria:** Earnings this week · SMA20 crossed above SMA50 · Barchart Buy Signal")
    
    if st.button("Find Stocks"):
        # Single progress bar and status for entire run
        status_text = st.empty()
        progress = st.progress(0)

        # 1) Find stocks: Scan Finviz, then Barchart
        status_text.text("Scanning Finviz...")
        progress.progress(0.05)
        tickers = get_all_tickers()

        status_text.text(f"Found {len(tickers)} tickers. Checking Barchart signals...")
        barchart_passed = []
        n_t = len(tickers)
        for i, t in enumerate(tickers):
            if has_buy_signal(t):
                barchart_passed.append(t)
            progress.progress(0.05 + 0.35 * (i + 1) / n_t if n_t else 0.4)

        # 2) Load tracker & fetch returns for reported tickers (need tracked_tickers_this_week before date check)
        this_week_df = get_this_week_earnings(raw_returns_df)
        tracked_tickers = set()
        if raw_returns_df is not None and not raw_returns_df.empty:
            tracked_tickers = set(raw_returns_df['Ticker'].unique())
        tracked_rows = []
        tracked_tickers_this_week = set()
        skipped_tracked = []
        if not this_week_df.empty:
            status_text.text("Fetching returns for reported tickers...")
            n_week = len(this_week_df)
            for i, (_, row) in enumerate(this_week_df.iterrows()):
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
                
                # Reported return from yfinance (entry = 4pm close on entry date, current = latest)
                return_pct = "N/A"
                open_price = "N/A"
                current_price = "N/A"
                if status == "Reported" and yf is not None:
                    time.sleep(1.0)  # throttle to avoid Yahoo 429 (Streamlit Cloud)
                    result = get_reported_return(ticker, earnings_date, timing)
                    if result is not None:
                        ret, entry_px, curr_px = result
                        return_pct = f"{ret:.1%}"
                        open_price = f"{entry_px:.2f}"
                        current_price = f"{curr_px:.2f}"
                progress.progress(0.4 + 0.05 * (i + 1) / n_week)
                
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
                    "Status": status,
                    "Return": return_pct,
                    "Open": open_price,
                    "Current": current_price,
                })
        progress.progress(0.45)

        # 3) Check earnings dates for Finviz/Barchart tickers
        status_text.text(f"{len(barchart_passed)} passed Barchart. Checking dates...")
        
        new_rows = []
        skipped = []
        today = datetime.now(MARKET_TZ).date()  # Use EST date
        
        for i, t in enumerate(barchart_passed):
            # Skip yfinance calls for tickers already in tracker (avoids 429 and duplicate errors)
            if t in tracked_tickers_this_week:
                progress.progress(0.45 + 0.45 * (i + 1) / len(barchart_passed) if barchart_passed else 0.9)
                continue
            data = get_finviz_data(t)
            # Parse the earnings date and timing from Finviz (no yfinance earnings API — avoids 429 / "No earnings dates found")
            earnings_date = None
            earnings_timing = ""
            earnings_str = data.get("Earnings", "")
            if earnings_str and earnings_str != "N/A":
                try:
                    parts = earnings_str.split()
                    if len(parts) >= 2:
                        month_day = f"{parts[0]} {parts[1]}"
                        earnings_date = datetime.strptime(f"{month_day} {today.year}", "%b %d %Y").date()
                        if len(parts) >= 3:
                            earnings_timing = parts[2].upper()
                except Exception:
                    pass
            # Date check from Finviz only (earnings already passed = DATE PASSED)
            date_passed = earnings_date is not None and earnings_date < today

            # Skip conditions for Finviz tickers (not in tracker)
            skip_reason = None
            if date_passed:
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
                # Finviz-origin rows. Sector left as Unknown to avoid extra yfinance calls (429).
                market_cap = data.get("Market Cap", "N/A")
                new_rows.append({
                    "Ticker": t,
                    "Earnings": earnings_str,
                    "Price": data.get("Price", "N/A"),
                    "P/E": data.get("P/E", "N/A"),
                    "Beta": data.get("Beta", "N/A"),
                    "Market Cap": market_cap,
                    "Sector": "Unknown",
                    "Earnings Timing": earnings_timing if earnings_timing else 'Unknown',
                    "Status": "Upcoming",
                    "Return": "N/A",
                    "Open": "N/A",
                    "Current": "N/A",
                })
            
            progress.progress(0.45 + 0.45 * (i + 1) / len(barchart_passed) if barchart_passed else 0.9)
        
        # 4) Train win probability model (uses historical returns, not current run)
        status_text.text("Training win probability model...")
        progress.progress(0.92)
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
                    st.warning(f"Could not train Machine Learning model: {str(e)}")
                    ml_available = False
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
            display_cols = ["Ticker", "Earnings", "Win Probability", "Price", "P/E", "Beta", "Market Cap", "Open", "Current", "Return", "Status"]
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


def _earnings_timing_rank(earn_str):
    """Return sort rank so BMO (0) comes before AMC (1) before unknown (2)."""
    s = (earn_str or "").upper()
    return 0 if "BMO" in s else 1 if "AMC" in s else 2


def _parse_return_pct(s):
    """Parse Return column value to decimal (e.g. '5.2%' -> 0.052). Returns None for N/A."""
    if pd.isna(s) or s == "N/A" or not s or "%" not in str(s):
        return None
    try:
        return float(str(s).replace("%", "").replace(",", "")) / 100
    except (ValueError, TypeError):
        return None


def _render_screener_results():
    """Render screener table only (no filters)."""
    if "screener_df" not in st.session_state:
        return
    df = st.session_state.screener_df.copy()

    # Sort by earnings date (earliest first), then BMO before AMC
    df["_timing_rank"] = df["Earnings"].apply(_earnings_timing_rank)
    filtered = df.sort_values(["Earnings Date", "_timing_rank"], ascending=[True, True])

    # Total return for reported tickers this week (sum of individual returns)
    reported = filtered[filtered["Status"] == "Reported"]
    return_vals = reported["Return"].apply(_parse_return_pct) if "Return" in reported.columns else pd.Series(dtype=float)
    return_vals = return_vals.dropna()
    if len(return_vals) > 0:
        total_return = return_vals.sum()
        n_reported = len(return_vals)
        st.markdown(f"**Total return (reported this week):** {total_return:.1%} ({n_reported} ticker{'s' if n_reported != 1 else ''})")

    show_cols = ["Ticker", "Earnings", "Win Probability", "Price", "P/E", "Beta", "Market Cap", "Open", "Current", "Return", "Status"]
    filtered_display = filtered[[c for c in show_cols if c in filtered.columns]].copy()

    st.dataframe(filtered_display, width="stretch", hide_index=True, height=525)

    # ML Model Info
    ml_available = st.session_state.get("screener_ml_available", False)
    ml_metrics = st.session_state.get("screener_ml_metrics")
    if ml_available and ml_metrics is not None:
        st.markdown("---")
        with st.expander("Machine Learning Model Performance (Click to view)"):
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
            st.dataframe(pd.DataFrame(all_skipped), width="stretch", hide_index=True)