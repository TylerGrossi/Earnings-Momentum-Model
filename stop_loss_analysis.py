import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import plotly.graph_objects as go

def run_comparative_analysis(hourly_df, returns_df, max_days=5):
    """
    Automated backtest from -2% to -20% in 2% steps.
    Filters: Valid 5D return and Date Passed (>7 days ago).
    """
    if hourly_df is None or hourly_df.empty or returns_df is None or returns_df.empty:
        return pd.DataFrame()

    hourly_df = hourly_df.copy()
    hourly_df['Earnings Date'] = pd.to_datetime(hourly_df['Earnings Date']).dt.date
    returns_df = returns_df.copy()
    returns_df['Earnings Date'] = pd.to_datetime(returns_df['Earnings Date']).dt.date
    
    today = datetime.now().date()
    sl_levels = [x / 100.0 for x in range(-2, -22, -2)]
    
    # Target 174 tickers (completed trades only)
    valid_trades = returns_df[
        (returns_df['5D Return'].notna()) & 
        (returns_df['Earnings Date'] <= (today - timedelta(days=7)))
    ]
    
    analysis_results = []
    for _, trade in valid_trades.iterrows():
        ticker, e_date, normal_5d = trade['Ticker'], trade['Earnings Date'], trade['5D Return']
        
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
        row = {'Ticker': ticker, 'Date': e_date, 'Normal Model (5D)': normal_5d}
        
        for sl in sl_levels:
            label = f"SL {int(sl*100)}%"
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

def render_stop_loss_tab(returns_df, hourly_df, filter_stats):
    st.subheader("Stop Loss Optimization")
    
    if (hourly_df is None or hourly_df.empty) and os.path.exists("hourly_prices.csv"):
        hourly_df = pd.read_csv("hourly_prices.csv")
    
    if hourly_df is not None and not hourly_df.empty:
        master_results = run_comparative_analysis(hourly_df, returns_df)
        if master_results.empty: return

        # Calculation logic
        base_col = 'Normal Model (5D)'
        sl_cols = [c for c in master_results.columns if "SL" in c]
        totals = {col: master_results[col].sum() for col in sl_cols}
        best_sl_col = max(totals, key=totals.get)
        best_sl_total = totals[best_sl_col]
        
        # 1. Summary Metrics
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Trades", len(master_results))
        c2.metric("Normal 5D Return", f"{master_results[base_col].sum()*100:+.1f}%")
        c3.metric("Best Stop Level", best_sl_col.replace("SL ", ""))
        c4.metric("Optimized Return", f"{best_sl_total*100:+.1f}%", 
                  delta=f"{(best_sl_total - master_results[base_col].sum())*100:+.1f}% Alpha")
        st.markdown("---")

        # 2. Performance Comparison Matrix (Sortable Table)
        st.markdown("### Performance Comparison Matrix")
        matrix_df = pd.DataFrame([
            {
                "SL_Value": int(col.replace("SL ", "").replace("%", "")),
                "Strategy": col,
                "Total Return (%)": round(master_results[col].sum() * 100, 2),
                "Alpha vs Normal (%)": round((master_results[col].sum() - master_results[base_col].sum()) * 100, 2),
                "Win Rate (%)": round((master_results[col] > 0).mean() * 100, 1),
                "Avg Return (%)": round(master_results[col].mean() * 100, 2)
            } for col in sl_cols
        ]).sort_values("SL_Value", ascending=False) # Sorted -2% to -20%

        st.dataframe(
            matrix_df.drop(columns=["SL_Value"]), 
            width="stretch", 
            hide_index=True,
            column_config={
                "Total Return (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                "Alpha vs Normal (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                "Win Rate (%)": st.column_config.NumberColumn(format="%.1f%%"),
                "Avg Return (%)": st.column_config.NumberColumn(format="%+.2f%%"),
            }
        )

        # 3. Chart and Analysis Row
        st.markdown("### Strategy Equity Curves & Insights")
        col_chart, col_analysis = st.columns([2.5, 1])

        with col_chart:
            top_3_sl = sorted(totals, key=totals.get, reverse=True)[:3]
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
            # Calculate Alpha per trade
            best_avg_alpha = (master_results[best_sl_col].mean() - master_results[base_col].mean()) * 100
            
            st.write(f"**Primary Target:** {best_sl_col}")
            st.write(f"**Alpha per Trade:** {best_avg_alpha:+.2f}%")
            
            # Improvement calculation
            total_wins_normal = (master_results[base_col] > 0).sum()
            total_wins_best = (master_results[best_sl_col] > 0).sum()
            win_delta = total_wins_best - total_wins_normal
            
            st.write(f"**Win Rate Delta:** {win_delta:+} trades")
            
            # Risk/Reward Note
            st.info(f"The {best_sl_col} strategy provides the highest total return over {len(master_results)} trades while maintaining the optimal balance between 'breathing room' and protection.")

    else:
        st.error("Missing hourly_prices.csv in repository.")