import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import plotly.graph_objects as go

from utils import compute_portfolio_return_for_return_series

# Stop loss levels: 1% to 10% every 1%, then 12, 14, 16, 18, 20%
SL_LEVELS_PCT = [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -12, -14, -16, -18, -20]


@st.cache_data(ttl=3600)
def run_comparative_analysis(hourly_df, returns_df, max_days=3):
    """
    Automated backtest over SL_LEVELS_PCT. Cached so changing the dropdown doesn't re-run the backtest.
    Filters: Valid 3D return and Date Passed (>7 days ago).
    """
    if hourly_df is None or hourly_df.empty or returns_df is None or returns_df.empty:
        return pd.DataFrame()

    hourly_df = hourly_df.copy()
    hourly_df['Earnings Date'] = pd.to_datetime(hourly_df['Earnings Date']).dt.date
    returns_df = returns_df.copy()
    returns_df['Earnings Date'] = pd.to_datetime(returns_df['Earnings Date']).dt.date
    
    today = datetime.now().date()
    sl_levels = [x / 100.0 for x in SL_LEVELS_PCT]
    
    # Completed trades only (have 3D return)
    valid_trades = returns_df[
        (returns_df['3D Return'].notna()) & 
        (returns_df['Earnings Date'] <= (today - timedelta(days=7)))
    ]
    
    analysis_results = []
    for _, trade in valid_trades.iterrows():
        ticker, e_date, normal_3d = trade['Ticker'], trade['Earnings Date'], trade['3D Return']
        
        trade_data = hourly_df[
            (hourly_df['Ticker'] == ticker) & 
            (hourly_df['Earnings Date'] == e_date) &
            (hourly_df['Trading Day'] >= 1)
        ].sort_values('Datetime')
        
        if trade_data.empty: continue
            
        exit_day = min(max_days, trade_data['Trading Day'].max())
        exit_day_data = trade_data[trade_data['Trading Day'] == exit_day]
        if exit_day_data.empty: continue
        
        close_ret = exit_day_data['Return From Earnings (%)'].iloc[-1] / 100
        row = {'Ticker': ticker, 'Date': e_date, 'Normal Model (3D)': normal_3d}
        
        for sl in sl_levels:
            label = f"SL {int(sl*100)}%"  # e.g. SL -10%
            final_ret, first_candle = close_ret, True
            for _, hour in trade_data.iterrows():
                if int(hour['Trading Day']) > exit_day: break
                h_ret = hour['Return From Earnings (%)'] / 100
                if h_ret <= sl:
                    # Gap down handling per strategy rules
                    final_ret = h_ret if (first_candle and h_ret < sl) else sl
                    break
                first_candle = False
            row[label] = final_ret
        analysis_results.append(row)
        
    return pd.DataFrame(analysis_results)


def _get_sl_result_and_trigger(hourly_df, ticker, e_date, sl_pct, max_days=3):
    """
    For one trade and one stop-loss level: return (final_return, trigger_label).
    Uses same logic as run_comparative_analysis: first-candle gap down = take actual loss, not SL.
    """
    if hourly_df is None or hourly_df.empty:
        return None, "—"
    h = hourly_df.copy()
    h["Earnings Date"] = pd.to_datetime(h["Earnings Date"]).dt.date
    e_date_only = e_date.date() if hasattr(e_date, "date") else e_date
    trade_data = h[
        (h["Ticker"] == ticker) &
        (h["Earnings Date"] == e_date_only) &
        (h["Trading Day"] >= 1)
    ].sort_values("Datetime")
    if trade_data.empty:
        return None, "—"
    exit_day = min(max_days, int(trade_data["Trading Day"].max()))
    exit_day_data = trade_data[trade_data["Trading Day"] == exit_day]
    if exit_day_data.empty:
        return None, "—"
    close_ret = exit_day_data["Return From Earnings (%)"].iloc[-1] / 100
    sl = sl_pct / 100.0
    first_candle = True
    for _, hour in trade_data.iterrows():
        if int(hour["Trading Day"]) > exit_day:
            break
        h_ret = hour["Return From Earnings (%)"] / 100
        if h_ret <= sl:
            # Gap down: before open we take actual loss instead of stop loss
            final_ret = h_ret if (first_candle and h_ret < sl) else sl
            dt = hour.get("Datetime")
            if pd.notna(dt):
                try:
                    t = pd.to_datetime(dt)
                    day_num = int(hour["Trading Day"])
                    time_str = t.strftime("%H:%M") if hasattr(t, "strftime") else str(dt)
                    if first_candle and h_ret < sl:
                        trigger_label = f"Gap down (Day {day_num} {time_str})"
                    else:
                        trigger_label = f"Day {day_num} {time_str}"
                except Exception:
                    trigger_label = "Gap down (pre-open)" if (first_candle and h_ret < sl) else "Triggered"
            else:
                trigger_label = "Gap down (pre-open)" if (first_candle and h_ret < sl) else "Triggered"
            return final_ret, trigger_label
        first_candle = False
    return close_ret, "No trigger (3D hold)"


def render_stop_loss_tab(returns_df, hourly_df, filter_stats):
    st.subheader("Stop Loss Optimization")
    # returns_df is already filtered by data_loader (Date Check, 3D Return, Forward P/E <= 15 from returns_tracker.csv)

    if (hourly_df is None or hourly_df.empty) and os.path.exists("hourly_prices.csv"):
        hourly_df = pd.read_csv("hourly_prices.csv")
    
    if hourly_df is not None and not hourly_df.empty:
        master_results = run_comparative_analysis(hourly_df, returns_df)
        if master_results.empty: return

        # Build trades frame for portfolio return (same rows as master_results, with Earnings Date + Timing)
        rd = returns_df.copy()
        rd['Earnings Date'] = pd.to_datetime(rd['Earnings Date'], errors='coerce')
        rd['_ed'] = rd['Earnings Date'].dt.date
        trades_meta = master_results[['Ticker', 'Date']].copy()
        trades_meta = trades_meta.merge(
            rd[['Ticker', '_ed', 'Earnings Timing']].drop_duplicates(subset=['Ticker', '_ed']),
            left_on=['Ticker', 'Date'], right_on=['Ticker', '_ed'], how='left'
        )
        trades_meta['Earnings Date'] = pd.to_datetime(trades_meta['Date'])

        # Portfolio return for each strategy (Total Net Profit / Max Capital HWM) — quant model
        base_col = 'Normal Model (3D)'
        sl_cols = [c for c in master_results.columns if "SL" in c]
        port_ret_normal, _ = compute_portfolio_return_for_return_series(
            trades_meta, master_results[base_col], earnings_date_col='Earnings Date', timing_col='Earnings Timing'
        )
        portfolio_returns = {}
        for col in sl_cols:
            pr, _ = compute_portfolio_return_for_return_series(
                trades_meta, master_results[col], earnings_date_col='Earnings Date', timing_col='Earnings Timing'
            )
            portfolio_returns[col] = pr if pr is not None else 0.0
        best_sl_col = max(portfolio_returns, key=portfolio_returns.get)
        best_portfolio_return = portfolio_returns[best_sl_col]
        port_ret_normal = port_ret_normal if port_ret_normal is not None else 0.0

        # If no stop loss beats the normal model, recommend no stop loss
        use_stop_loss = best_portfolio_return > port_ret_normal
        recommended_label = best_sl_col.replace("SL ", "") if use_stop_loss else "No stop loss"
        best_strategy_col = best_sl_col if use_stop_loss else base_col
        # Normal 3D Average = mean(3D Return) over full universe (152) to match Power BI "Average 3D Return"
        rd_3d = returns_df['3D Return'].dropna() if returns_df is not None and '3D Return' in returns_df.columns else pd.Series(dtype=float)
        avg_normal_3d = (rd_3d.mean() * 100) if len(rd_3d) > 0 else (master_results[base_col].mean() * 100)
        # Optimized = same as normal when no stop loss; otherwise backtest average with best SL
        avg_optimized_3d = avg_normal_3d if not use_stop_loss else (master_results[best_strategy_col].mean() * 100)
        alpha_delta_avg = avg_optimized_3d - avg_normal_3d if use_stop_loss else 0.0

        # 1. Summary Metrics (3D Average = mean return per trade)
        # Total Trades = full universe (match Earnings Analysis); backtest runs on subset with hourly data and earnings >7 days ago
        total_universe = len(returns_df) if returns_df is not None else 0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Trades", total_universe)
        c2.metric("Normal 3D Average", f"{avg_normal_3d:+.2f}%")
        c3.metric("Recommendation", recommended_label)
        c4.metric("Optimized 3D Average", f"{avg_optimized_3d:+.2f}%",
                  delta=f"{alpha_delta_avg:+.2f}% Alpha" if use_stop_loss else "No improvement")
        st.markdown("---")

        # 2. Performance Comparison Matrix — Avg Return second, no Total Return column
        st.markdown("### Performance Comparison Matrix")
        # Alpha vs Normal = (avg return with stop loss) − (Normal 3D Average from full universe)
        matrix_df = pd.DataFrame([
            {
                "SL_Value": int(col.replace("SL ", "").replace("%", "")),
                "Strategy": col,
                "Avg Return (%)": round(master_results[col].mean() * 100, 2),
                "Alpha vs Normal (%)": round(master_results[col].mean() * 100 - avg_normal_3d, 2),
                "Win Rate (%)": round((master_results[col] > 0).mean() * 100, 1),
            } for col in sl_cols
        ]).sort_values("SL_Value", ascending=False)
        matrix_display = matrix_df.drop(columns=["SL_Value"])[
            ["Strategy", "Avg Return (%)", "Alpha vs Normal (%)", "Win Rate (%)"]
        ]
        st.dataframe(
            matrix_display,
            width="stretch",
            hide_index=True,
            column_config={
                "Avg Return (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                "Alpha vs Normal (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                "Win Rate (%)": st.column_config.NumberColumn(format="%.1f%%"),
            }
        )

        # Per-trade table: choose stop loss type, show 3D return, return with that SL, and hour SL triggered (gap down = take loss before open)
        st.markdown("### Per-trade returns by stop loss")
        sl_options = ["No stop loss"] + [f"SL {x}%" for x in SL_LEVELS_PCT]
        default_ix = 0
        if use_stop_loss and best_sl_col in sl_options:
            default_ix = sl_options.index(best_sl_col)
        selected_sl_label = st.selectbox(
            "Stop loss type",
            options=sl_options,
            index=default_ix,
            help="Return and trigger hour use this level. If price gaps down before open, the actual loss is taken instead of the stop loss.",
        )
        if selected_sl_label == "No stop loss":
            selected_col = base_col
            per_trade = master_results[["Ticker", "Date", base_col]].copy()
            per_trade["3D Return (%)"] = (per_trade[base_col] * 100).round(2)
            per_trade["Return with selected SL (%)"] = (per_trade[base_col] * 100).round(2)
            per_trade["SL trigger hour"] = "—"
        else:
            selected_col = selected_sl_label
            sl_pct = int(selected_sl_label.replace("SL ", "").replace("%", ""))
            per_trade = master_results[["Ticker", "Date", base_col]].copy()
            per_trade["3D Return (%)"] = (per_trade[base_col] * 100).round(2)
            per_trade["Return with selected SL (%)"] = (master_results[selected_col] * 100).round(2)
            h_df = hourly_df.copy()
            h_df["Earnings Date"] = pd.to_datetime(h_df["Earnings Date"]).dt.date
            triggers = []
            for _, r in master_results.iterrows():
                _, trigger_label = _get_sl_result_and_trigger(h_df, r["Ticker"], r["Date"], sl_pct)
                triggers.append(trigger_label)
            per_trade["SL trigger hour"] = triggers
        per_trade = per_trade.drop(columns=[base_col])

        # Saved vs 3D = SL level − 3D return (when triggered at SL). Lost vs 3D = gap return − 3D return (when gap down).
        trigger_hr = per_trade["SL trigger hour"].astype(str)
        triggered_at_sl = trigger_hr.str.match(r"^Day \d", na=False)
        gap_down = trigger_hr.str.contains("Gap down", na=False)
        no_trigger = trigger_hr.str.contains("No trigger", na=False)
        three_d = per_trade["3D Return (%)"]
        ret_sl = per_trade["Return with selected SL (%)"]
        if selected_sl_label == "No stop loss":
            per_trade["Saved vs 3D (%)"] = np.nan
            per_trade["Lost vs 3D (%)"] = np.nan
        else:
            sl_pct = int(selected_sl_label.replace("SL ", "").replace("%", ""))
            per_trade["Saved vs 3D (%)"] = np.where(triggered_at_sl, (sl_pct - three_d).round(2), np.nan)
            per_trade["Lost vs 3D (%)"] = np.where(gap_down, (ret_sl - three_d).round(2), np.nan)

        # Ensure all return columns display with 2 decimal places
        for col in ["3D Return (%)", "Return with selected SL (%)", "Saved vs 3D (%)", "Lost vs 3D (%)"]:
            if col in per_trade.columns:
                per_trade[col] = pd.to_numeric(per_trade[col], errors="coerce").round(2)
        per_trade = per_trade.sort_values("Date", ascending=False)

        # Cards = averages of Saved and Lost (over trades where they apply); third card = avg return (all trades) to match matrix
        avg_return_all = ret_sl.mean()  # Same as matrix "Avg Return (%)" for this strategy
        if selected_sl_label == "No stop loss":
            total_saved_str = "—"
            total_lost_gap_str = "—"
            card3_label = "Avg return (all trades)"
            card3_value = f"{avg_return_all:+.2f}% ({len(per_trade)} trades)"
            card3_help = "Average 3D return across all trades (no stop loss). Matches matrix."
        else:
            saved_vals = per_trade.loc[triggered_at_sl, "Saved vs 3D (%)"]
            lost_vals = per_trade.loc[gap_down, "Lost vs 3D (%)"]
            avg_saved = saved_vals.mean() if triggered_at_sl.any() else np.nan
            avg_lost = lost_vals.mean() if gap_down.any() else np.nan
            n_saved = triggered_at_sl.sum()
            n_lost = gap_down.sum()
            total_saved_str = f"{avg_saved:+.2f}% ({n_saved})" if pd.notna(avg_saved) else "—"
            total_lost_gap_str = f"{avg_lost:+.2f}% ({n_lost})" if pd.notna(avg_lost) else "—"
            other_count = no_trigger.sum()
            other_avg = three_d[no_trigger].mean() if other_count else np.nan
            card3_label = "Avg return (all trades)"
            card3_value = f"{avg_return_all:+.2f}% ({len(per_trade)} trades)"
            card3_help = (
                f"Average return with this SL across all trades — matches the matrix row for {selected_sl_label}. "
                f"Of these, {other_count} held to 3D (avg {other_avg:+.2f}%)."
            ) if other_count and pd.notna(other_avg) else (
                f"Average return with this SL across all trades — matches the matrix row for {selected_sl_label}."
                + (f" Of these, {other_count} held to 3D." if other_count else "")
            )
        # Formula + explanation for metric tooltips (question marks)
        help_avg_saved = (
            "Formula: mean(SL_level − 3D_return) over trades that triggered at the stop.\n\n"
            "Meaning: For each trade where the stop was hit (not a gap down), we take (stop level − 3D return). "
            "A positive value means the 3D return was worse than the stop—you 'saved' by exiting at the stop instead of holding to day 3."
        )
        help_avg_lost_gap = (
            "Formula: mean(return_at_gap − 3D_return) over trades that had a gap down.\n\n"
            "Meaning: When price gaps down before the open, you get the gap return instead of the stop. "
            "This metric is the average difference between that gap return and the 3D return. Negative means you lost more by gapping down than you would have by holding to 3D."
        )
        help_avg_return_all = (
            "Formula: mean(Return with selected SL) over all trades.\n\n"
            "Meaning: Simple average per-trade return for the selected strategy (with or without stop loss). "
            "Matches the 'Avg Return (%)' in the Performance Comparison Matrix for this strategy."
        )
        card1, card2, card3 = st.columns(3)
        with card1:
            st.metric("Avg saved from stop loss", total_saved_str, help=help_avg_saved)
        with card2:
            st.metric("Avg lost on gap down", total_lost_gap_str, help=help_avg_lost_gap)
        with card3:
            st.metric(card3_label, card3_value, help=help_avg_return_all)

        col_config = {
            "Date": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "3D Return (%)": st.column_config.NumberColumn(format="%+.2f%%"),
            "Return with selected SL (%)": st.column_config.NumberColumn(format="%+.2f%%"),
        }
        if "Saved vs 3D (%)" in per_trade.columns:
            col_config["Saved vs 3D (%)"] = st.column_config.NumberColumn(format="%+.2f%%")
        if "Lost vs 3D (%)" in per_trade.columns:
            col_config["Lost vs 3D (%)"] = st.column_config.NumberColumn(format="%+.2f%%")
        st.dataframe(per_trade, width="stretch", hide_index=True, column_config=col_config)

        # 3. Chart and Analysis Row
        st.markdown("### Strategy Equity Curves & Insights")
        col_chart, col_analysis = st.columns([2.5, 1])

        with col_chart:
            top_3_sl = sorted(portfolio_returns, key=portfolio_returns.get, reverse=True)[:3]
            chart_cols = [base_col] + top_3_sl
            res_cum = master_results.sort_values('Date').copy()
            fig = go.Figure()
            for col in chart_cols:
                is_base = col == base_col
                fig.add_trace(go.Scatter(
                    x=res_cum['Date'], y=res_cum[col].cumsum() * 100, name=col,
                    line=dict(width=3 if is_base else 2, dash='dash' if is_base else 'solid', color='white' if is_base else None)
                ))
            fig.update_layout(template="plotly_dark", hovermode="x unified", height=400, margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig, width="stretch")

        with col_analysis:
            st.markdown("**Backtest Insights**")
            if use_stop_loss:
                best_avg_alpha = (master_results[best_sl_col].mean() - master_results[base_col].mean()) * 100
                st.write(f"**Primary Target:** {best_sl_col}")
                st.write(f"**Alpha per Trade:** {best_avg_alpha:+.2f}%")
                total_wins_normal = (master_results[base_col] > 0).sum()
                total_wins_best = (master_results[best_sl_col] > 0).sum()
                win_delta = total_wins_best - total_wins_normal
                st.write(f"**Win Rate Delta:** {win_delta:+} trades")
                st.info(f"The {best_sl_col} strategy provides the highest portfolio return over {len(master_results)} trades while maintaining the optimal balance between 'breathing room' and protection.")
            else:
                st.write("**Recommendation:** No stop loss")
                st.write("None of the stop-loss levels improved portfolio return over the normal (no stop loss) model.")
                st.info("Sticking with the normal 3D hold is optimal for this backtest. Re-run as more data accumulates to see if a stop loss becomes beneficial.")

    else:
        st.error("Missing hourly_prices.csv in repository.")