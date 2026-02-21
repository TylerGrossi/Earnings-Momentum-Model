import streamlit as st

# Import configuration and styling
from config import setup_page, apply_styling

# Import data loader
from data_loader import load_and_filter_all_data, load_returns_data_raw, _FILTER_CACHE_VERSION

# Import tab renderers
from stock_screener import render_stock_screener_tab
from powerbi import render_powerbi_tab
from stop_loss_analysis import render_stop_loss_tab
from earnings_analysis import render_earnings_analysis_tab
from ai_assistant import render_ai_assistant_tab

# ------------------------------------
# PAGE SETUP
# ------------------------------------
setup_page()
apply_styling()

# ------------------------------------
# MAIN APP - Title first so it shows even if data fails
# ------------------------------------
st.title("Earnings Momentum Strategy")

# ------------------------------------
# LOAD DATA
# ------------------------------------
# Load filtered data for analysis tabs
try:
    all_data = load_and_filter_all_data(_cache_version=_FILTER_CACHE_VERSION)
    returns_df = all_data['returns']
    hourly_df = all_data['hourly']
    filter_stats = all_data['filter_stats']
except Exception as e:
    st.error(f"Error loading filtered data: {e}")
    returns_df = None
    hourly_df = None
    filter_stats = {'final_count': 0, 'final_tickers': []}

# Load raw data for stock screener (includes stocks without 5D return)
try:
    raw_returns_df = load_returns_data_raw()
except Exception as e:
    st.error(f"Error loading raw data: {e}")
    raw_returns_df = None

# ------------------------------------
# TABS
# ------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Stock Screener", 
    "PowerBI", 
    "Stop Loss Analysis", 
    "Earnings Analysis",
    "AI Assistant"
])

# Render each tab
with tab1:
    render_stock_screener_tab(raw_returns_df)

with tab2:
    render_powerbi_tab()

with tab3:
    render_stop_loss_tab(returns_df, hourly_df, filter_stats)

with tab4:
    render_earnings_analysis_tab(returns_df, filter_stats)

with tab5:
    render_ai_assistant_tab()

# FOOTER
st.markdown("---")
st.caption("Earnings Momentum Strategy")