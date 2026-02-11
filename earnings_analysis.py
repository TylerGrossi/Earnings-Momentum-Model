import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def render_earnings_analysis_tab(returns_df, filter_stats):
    """Render the Earnings Analysis tab."""
    
    st.subheader("Earnings Surprise Analysis")
    st.markdown("Analyze how earnings beats/misses and surprise magnitude affect stock returns")
    
    # Use the filtered returns_df
    analysis_df = returns_df.copy() if returns_df is not None else None
    
    if analysis_df is None or analysis_df.empty:
        st.warning("No returns data available. Please ensure returns_tracker.csv is accessible.")
        
        uploaded_file = st.file_uploader("Upload returns_tracker.csv:", type=['csv'], key='earnings_upload')
        if uploaded_file:
            analysis_df = pd.read_csv(uploaded_file)
            analysis_df['Earnings Date'] = pd.to_datetime(analysis_df['Earnings Date'], errors='coerce')
            st.success(f"Loaded {len(analysis_df)} rows")
    
    if analysis_df is not None and not analysis_df.empty:
        # Convert EPS Surprise to numeric if needed
        if 'EPS Surprise (%)' in analysis_df.columns:
            analysis_df['EPS Surprise (%)'] = pd.to_numeric(analysis_df['EPS Surprise (%)'], errors='coerce')
        
        # Use 5D Return
        return_col = '5D Return'
        
        # Convert returns to percentage for display
        analysis_df[return_col] = pd.to_numeric(analysis_df[return_col], errors='coerce') * 100
        
        # Count trades with and without EPS Surprise data
        total_trades = len(analysis_df)
        valid_surprise = analysis_df['EPS Surprise (%)'].notna().sum() if 'EPS Surprise (%)' in analysis_df.columns else 0
        missing_surprise = total_trades - valid_surprise
        
        # Calculate beat rate and median surprise
        if 'EPS Surprise (%)' in analysis_df.columns:
            surprise_data = analysis_df['EPS Surprise (%)'].dropna()
            beat_rate = (surprise_data > 0).mean() * 100 if len(surprise_data) > 0 else 0
            median_surprise = surprise_data.median() if len(surprise_data) > 0 else 0
        else:
            beat_rate = 0
            median_surprise = 0
        
        st.markdown("---")
        
        # Quick Stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", total_trades)
        with col2:
            st.metric("With Surprise Data", valid_surprise)
        with col3:
            st.metric("Beat Rate", f"{beat_rate:.1f}%")
        with col4:
            st.metric("Median EPS Surprise", f"{median_surprise:+.1f}%" if pd.notna(median_surprise) else "N/A")
        
        st.markdown("---")
        
        # Create analysis tabs
        analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs([
            "Beat vs Miss", "Surprise Magnitude", "Raw Data"
        ])
        
        with analysis_tab1:
            _render_beat_vs_miss(analysis_df, return_col, valid_surprise, total_trades)
        
        with analysis_tab2:
            _render_surprise_magnitude(analysis_df, return_col, valid_surprise, total_trades)
        
        with analysis_tab3:
            _render_raw_data(analysis_df, return_col)


def _render_beat_vs_miss(analysis_df, return_col, valid_surprise, total_trades):
    """Render Beat vs Miss analysis."""
    st.markdown("#### Returns by Surprise Buckets")
    
    if 'EPS Surprise (%)' not in analysis_df.columns or valid_surprise == 0:
        st.warning("No 'EPS Surprise (%)' column found in data.")
        return
    
    # Filter for valid data
    scatter_df = analysis_df[
        analysis_df['EPS Surprise (%)'].notna() & 
        analysis_df[return_col].notna() &
        (analysis_df['EPS Surprise (%)'] >= -100) &
        (analysis_df['EPS Surprise (%)'] <= 100)
    ].copy()
    
    if len(scatter_df) < 5:
        st.warning("Not enough data for analysis.")
        return
    
    # Define bucket ranges for drilldown
    bucket_ranges = {
        '< -10%': (-100, -10),
        '-10% to -5%': (-10, -5),
        '-5% to 0%': (-5, 0),
        '0% to 5%': (0, 5),
        '5% to 10%': (5, 10),
        '10% to 20%': (10, 20),
        '> 20%': (20, 100)
    }
    
    # Create surprise buckets
    def bucket_surprise(x):
        if pd.isna(x):
            return None
        elif x < -10:
            return '< -10%'
        elif x < -5:
            return '-10% to -5%'
        elif x < 0:
            return '-5% to 0%'
        elif x < 5:
            return '0% to 5%'
        elif x < 10:
            return '5% to 10%'
        elif x < 20:
            return '10% to 20%'
        else:
            return '> 20%'
    
    scatter_df['Surprise Bucket'] = scatter_df['EPS Surprise (%)'].apply(bucket_surprise)
    
    # Calculate IQR stats per bucket
    def calc_iqr_mean(group):
        q1 = group.quantile(0.25)
        q3 = group.quantile(0.75)
        iqr_data = group[(group >= q1) & (group <= q3)]
        return iqr_data.mean() if len(iqr_data) > 0 else group.mean()
    
    # Calculate 2σ average (excluding outliers beyond 2 std devs)
    def calc_2sigma_mean(group):
        mean = group.mean()
        std = group.std()
        if pd.isna(std) or std == 0:
            return mean
        filtered = group[(group >= mean - 2 * std) & (group <= mean + 2 * std)]
        return filtered.mean() if len(filtered) > 0 else mean
    
    # Session state for drilldown
    if 'drilldown_bucket' not in st.session_state:
        st.session_state.drilldown_bucket = None
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        # Dropdowns side by side
        dropdown_col1, dropdown_col2 = st.columns(2)
        
        with dropdown_col1:
            metric_type = st.selectbox("Display metric:", ["Median", "Average", "Average (excl. 2σ outliers)", "IQR Mean (25th-75th)", "Total"], index=0)
        
        with dropdown_col2:
            selected_bucket = st.selectbox("Drill down into:", ["All Buckets"] + list(bucket_ranges.keys()), index=0)
        
        if selected_bucket != "All Buckets":
            # Drilldown view - 1% increments
            low, high = bucket_ranges[selected_bucket]
            drilldown_df = scatter_df[
                (scatter_df['EPS Surprise (%)'] >= low) & 
                (scatter_df['EPS Surprise (%)'] < high)
            ].copy()
            
            # Create 1% buckets
            def one_pct_bucket(x):
                floor_val = int(np.floor(x))
                return f"{floor_val}% to {floor_val+1}%"
            
            drilldown_df['Detail Bucket'] = drilldown_df['EPS Surprise (%)'].apply(one_pct_bucket)
            
            if len(drilldown_df) > 0:
                detail_stats = drilldown_df.groupby('Detail Bucket').agg({
                    return_col: ['sum', 'mean', 'median', 'count'],
                }).round(2)
                detail_stats.columns = ['Total Return', 'Avg Return', 'Median', 'Count']
                detail_stats['IQR Mean'] = drilldown_df.groupby('Detail Bucket')[return_col].apply(calc_iqr_mean).round(2)
                detail_stats['Avg (2σ)'] = drilldown_df.groupby('Detail Bucket')[return_col].apply(calc_2sigma_mean).round(2)
                detail_stats['Win Rate'] = drilldown_df.groupby('Detail Bucket')[return_col].apply(
                    lambda x: (x > 0).mean() * 100
                ).round(1)
                detail_stats = detail_stats.reset_index()
                
                # Sort by the numeric value
                detail_stats['sort_val'] = detail_stats['Detail Bucket'].apply(
                    lambda x: int(x.split('%')[0])
                )
                detail_stats = detail_stats.sort_values('sort_val').drop(columns=['sort_val'])
                
                display_stats = detail_stats
                x_col = 'Detail Bucket'
                chart_title_prefix = f"Drilldown: {selected_bucket}"
            else:
                st.warning("No data in this bucket.")
                return
        else:
            # Main view - original buckets
            bucket_stats = scatter_df.groupby('Surprise Bucket').agg({
                return_col: ['sum', 'mean', 'median', 'count'],
            }).round(2)
            bucket_stats.columns = ['Total Return', 'Avg Return', 'Median', 'Count']
            bucket_stats['IQR Mean'] = scatter_df.groupby('Surprise Bucket')[return_col].apply(calc_iqr_mean).round(2)
            bucket_stats['Avg (2σ)'] = scatter_df.groupby('Surprise Bucket')[return_col].apply(calc_2sigma_mean).round(2)
            bucket_stats['Win Rate'] = scatter_df.groupby('Surprise Bucket')[return_col].apply(
                lambda x: (x > 0).mean() * 100
            ).round(1)
            bucket_stats = bucket_stats.reset_index()
            
            bucket_order = ['< -10%', '-10% to -5%', '-5% to 0%', '0% to 5%', '5% to 10%', '10% to 20%', '> 20%']
            bucket_stats['sort_order'] = bucket_stats['Surprise Bucket'].apply(
                lambda x: bucket_order.index(x) if x in bucket_order else 99
            )
            bucket_stats = bucket_stats.sort_values('sort_order').drop(columns=['sort_order'])
            
            display_stats = bucket_stats
            x_col = 'Surprise Bucket'
            chart_title_prefix = ""
        
        # Choose which column to display based on dropdown
        if metric_type == "Average":
            display_col = 'Avg Return'
            metric_label = "Average"
        elif metric_type == "Median":
            display_col = 'Median'
            metric_label = "Median"
        elif metric_type == "Total":
            display_col = 'Total Return'
            metric_label = "Total"
        elif metric_type == "Average (excl. 2σ outliers)":
            display_col = 'Avg (2σ)'
            metric_label = "Avg (2σ)"
        else:
            display_col = 'IQR Mean'
            metric_label = "IQR Mean"
        
        chart_title = f"{chart_title_prefix} {metric_label} Return by EPS Surprise Bucket".strip()
        y_label = f"{metric_label} {return_col} (%)"
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=display_stats[x_col],
            y=display_stats[display_col],
            marker_color='#3b82f6',
            text=display_stats[display_col].apply(lambda x: f"{x:+.2f}%"),
            textposition='outside'
        ))
        
        y_min = display_stats[display_col].min()
        y_max = display_stats[display_col].max()
        y_padding = (y_max - y_min) * 0.2 if y_max != y_min else 5
        
        fig.update_layout(
            title=chart_title,
            xaxis_title="EPS Surprise %",
            yaxis_title=y_label,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8',
            height=400,
            yaxis=dict(range=[min(y_min - y_padding, -2), max(y_max + y_padding, 2)]),
            margin=dict(t=50, b=50)
        )
        fig.add_hline(y=0, line_dash="dash", line_color="#475569")
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        st.dataframe(
            display_stats, 
            width="stretch", 
            hide_index=True,
            column_config={
                "Total Return": st.column_config.NumberColumn(format="%.1f%%"),
                "Avg Return": st.column_config.NumberColumn(format="%.2f%%"),
                "Median": st.column_config.NumberColumn(format="%.2f%%"),
                "IQR Mean": st.column_config.NumberColumn(format="%.2f%%"),
                "Avg (2σ)": st.column_config.NumberColumn(format="%.2f%%"),
                "Win Rate": st.column_config.NumberColumn(format="%.1f%%"),
            }
        )
    
    # Simple Beat vs Miss summary
    st.markdown("---")
    st.markdown("#### Simple Beat vs Miss (Any Amount)")
    
    # Use the same filtered data as the table above (scatter_df) for consistency
    known_df = scatter_df.copy()
    known_df['Beat/Miss'] = known_df['EPS Surprise (%)'].apply(
        lambda x: 'Beat' if x > 0 else 'Miss'
    )
    
    simple_stats = known_df.groupby('Beat/Miss').agg({
        return_col: ['sum', 'mean', 'median', 'count'],
        'EPS Surprise (%)': 'mean'
    }).round(2)
    simple_stats.columns = ['Total Return', 'Avg Return', 'Median Return', 'Count', 'Avg Surprise %']
    simple_stats['Win Rate'] = known_df.groupby('Beat/Miss')[return_col].apply(
        lambda x: (x > 0).mean() * 100
    ).round(1)
    simple_stats = simple_stats.reset_index()
    
    col1, col2, col3 = st.columns(3)
    
    beat_row = simple_stats[simple_stats['Beat/Miss'] == 'Beat']
    miss_row = simple_stats[simple_stats['Beat/Miss'] == 'Miss']
    
    with col1:
        if len(beat_row) > 0:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value metric-green">{beat_row['Total Return'].values[0]:+.1f}%</div>
                <div class="metric-label">Beat Total Return ({int(beat_row['Count'].values[0])} trades)</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if len(miss_row) > 0:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value metric-red">{miss_row['Total Return'].values[0]:+.1f}%</div>
                <div class="metric-label">Miss Total Return ({int(miss_row['Count'].values[0])} trades)</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if len(beat_row) > 0 and len(miss_row) > 0:
            spread = beat_row['Avg Return'].values[0] - miss_row['Avg Return'].values[0]
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value metric-blue">{spread:+.2f}%</div>
                <div class="metric-label">Beat - Miss Avg Spread</div>
            </div>
            """, unsafe_allow_html=True)


def _render_surprise_magnitude(analysis_df, return_col, valid_surprise, total_trades):
    """Render Surprise Magnitude analysis - scatter plot with regression."""
    st.markdown("#### Surprise Magnitude vs Returns")
    
    if 'EPS Surprise (%)' not in analysis_df.columns:
        st.warning("No EPS Surprise data available for magnitude analysis.")
        return
    
    scatter_df = analysis_df[
        analysis_df['EPS Surprise (%)'].notna() & 
        analysis_df[return_col].notna() &
        (analysis_df['EPS Surprise (%)'] >= -100) &
        (analysis_df['EPS Surprise (%)'] <= 100)
    ].copy()
    
    outliers = analysis_df[
        analysis_df['EPS Surprise (%)'].notna() & 
        ((analysis_df['EPS Surprise (%)'] < -100) | (analysis_df['EPS Surprise (%)'] > 100))
    ]
    if len(outliers) > 0:
        st.caption(f"Note: {len(outliers)} outliers with EPS Surprise outside -100% to 100% range excluded from chart")
    
    if len(scatter_df) > 5:
        col1, col2 = st.columns([2, 1])
        
        x = scatter_df['EPS Surprise (%)'].values
        y = scatter_df[return_col].values
        n = len(x)
        x_mean, y_mean = np.mean(x), np.mean(y)
        slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
        intercept = y_mean - slope * x_mean
        
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        se = np.sqrt(ss_res / (n - 2)) if n > 2 else 0
        se_slope = se / np.sqrt(np.sum((x - x_mean) ** 2)) if np.sum((x - x_mean) ** 2) > 0 else 0
        t_stat = slope / se_slope if se_slope > 0 else 0
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=scatter_df['EPS Surprise (%)'],
                y=scatter_df[return_col],
                mode='markers',
                marker=dict(size=10, opacity=0.7, color='#3b82f6'),
                text=scatter_df['Ticker'],
                hovertemplate='<b>%{text}</b><br>EPS Surprise: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>',
                name='Trades'
            ))
            
            x_line = np.array([-100, 100])
            y_line = slope * x_line + intercept
            fig.add_trace(go.Scatter(
                x=x_line,
                y=y_line,
                mode='lines',
                line=dict(color='#f59e0b', width=2, dash='solid'),
                name=f'Trend (R²={r_squared:.3f})'
            ))
            
            fig.update_layout(
                title="EPS Surprise % vs Stock Return",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#94a3b8',
                height=500,
                xaxis=dict(range=[-100, 100], title='EPS Surprise (%)'),
                yaxis=dict(title=f'{return_col} (%)'),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig.add_hline(y=0, line_dash="dash", line_color="#475569")
            fig.add_vline(x=0, line_dash="dash", line_color="#475569")
            st.plotly_chart(fig, width="stretch")
        
        with col2:
            correlation = scatter_df['EPS Surprise (%)'].corr(scatter_df[return_col])
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value metric-blue">{correlation:.3f}</div>
                <div class="metric-label">Correlation</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("")
            
            st.markdown(f"""
            **Regression Stats:**
            - R²: **{r_squared:.3f}**
            - Slope: **{slope:.4f}**
            - T-stat: **{t_stat:.2f}**
            """)
            
            if abs(t_stat) > 1.96:
                st.success("Statistically significant (|t| > 1.96)")
            elif abs(t_stat) > 1.65:
                st.info("Marginally significant")
            else:
                st.warning("Not significant")
    else:
        st.warning("Not enough data for magnitude analysis.")


def _render_raw_data(analysis_df, return_col):
    """Render Raw Data explorer."""
    st.markdown("#### Raw Data Explorer")
    
    all_cols = list(analysis_df.columns)
    default_cols = ['Ticker', 'Company Name', 'Earnings Date', 'EPS Estimate', 
                   'Reported EPS', 'EPS Surprise (%)', return_col]
    default_cols = [c for c in default_cols if c in all_cols]
    
    selected_cols = st.multiselect(
        "Select columns to display:",
        options=all_cols,
        default=default_cols[:10]
    )
    
    if selected_cols:
        display_data = analysis_df[selected_cols].copy()
        
        col1, col2 = st.columns(2)
        
        with col1:
            sort_col = st.selectbox("Sort by:", selected_cols, index=0)
        
        with col2:
            sort_order = st.radio("Order:", ["Descending", "Ascending"], horizontal=True)
        
        display_data = display_data.sort_values(
            sort_col, 
            ascending=(sort_order == "Ascending")
        )
        
        st.dataframe(
            display_data,
            width="stretch",
            hide_index=True,
            height=500
        )
        
        st.caption(f"Showing {len(display_data)} rows")
        
        csv = display_data.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data",
            data=csv,
            file_name="earnings_analysis_data.csv",
            mime="text/csv"
        )
    
    st.markdown("---")
    st.markdown("#### Summary Statistics")
    
    numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
    key_cols = ['EPS Surprise (%)', return_col, 'EPS Estimate', 'Reported EPS', 'P/E', 'Beta']
    key_cols = [c for c in key_cols if c in numeric_cols]
    
    if key_cols:
        summary = analysis_df[key_cols].describe().T
        st.dataframe(summary.round(2), width="stretch")