import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path

# Import modules
from data_loader import load_assets, load_dividend_data
from strategies import run_strategy_1, run_strategy_2, run_strategy_3, run_strategy_4
from analytics import (calculate_correlation_data, calculate_rolling_correlation, 
                      calculate_rolling_beta, calculate_performance_metrics)
from visualization import (create_correlation_bar_chart, create_rolling_correlation_plot,
                         create_price_comparison_plot, create_rolling_beta_plot,
                         create_correlation_volatility_plot, create_asset_volatility_comparison_plot)
from strategy_viz import (create_asset_pnl_chart, create_qqq_pnl_chart, 
                         create_total_pnl_chart, create_correlation_entry_exit_chart,
                         create_volatility_entry_exit_chart, create_dividend_chart,
                         create_daily_pnl_chart, create_metrics_dataframe)
from ui_components import *

# Set page configuration - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Hedging Strategy App",
    page_icon="📈",
    layout="wide"
)

# Load custom CSS
load_css()

# Define directory paths
local_dir = r"D:\Benson\aUpWork\Douglas Backtester Algo\Backtester Algorithm\Hedging strategy\Yahoo Finance Data"
github_dir = "Hedging strategy/Yahoo Finance Data"

# Choose which directory to use (True for local, False for GitHub)
use_local = False
data_dir = local_dir if use_local else github_dir

# Set the current working directory to the data directory
if os.path.exists(data_dir):
    os.chdir(data_dir)
else:
    st.error(f"Directory not found: {data_dir}")
    st.stop()

# Add a header
main_header()

# Add introductory paragraph explaining the app
st.markdown("""
<div class="explanation-text" style="font-size: 1.1rem; margin-bottom: 25px;">
    This Multi-Leg Hedging Strategy Analysis Tool helps investors design and backtest market-neutral investment strategies 
    using ETF pairs. The app analyzes correlation patterns between various ETFs (like YMAX, JEPQ, QYLD) and QQQ, 
    allowing users to implement long-short hedging strategies that can potentially reduce market risk while generating income. 
    Users can explore asset correlations, test different entry/exit criteria based on correlation levels or market volatility indicators, 
    and visualize historical performance including dividend payments. The tool aims to help construct more resilient portfolios 
    that can perform well in various market conditions.
</div>
""", unsafe_allow_html=True)

# Load the data
data_result = load_assets()

if data_result is not None:
    data, individual_dfs = data_result
    
    # Load dividend data
    dividend_result = load_dividend_data()
    dividend_data = None
    dividend_dfs = {}
    
    if dividend_result is not None:
        dividend_data, dividend_dfs = dividend_result
    
    # Set Date as index for correlation calculation but keep a copy with Date as column
    data_with_date = data.copy()
    data.set_index('Date', inplace=True)
    
    # Data Preview Section
    section_header("Data Preview")
    description_text("This is a preview of the data (complete data) that we are using to run the backtests below.")
    
    # Add "Select the asset to preview" text
    select_text("Select the asset to preview:")
    
    # Get list of assets (columns excluding Date)
    assets = data.columns.tolist()
    
    # Add VIX and VVIX if they're available in the individual dataframes but not in the merged data
    if 'VIX' in individual_dfs and 'VIX' not in assets:
        assets.append('VIX')
    if 'VVIX' in individual_dfs and 'VVIX' not in assets:
        assets.append('VVIX')
    
    # Create a single radio button group for all assets
    available_options = assets.copy()
    available_options.append("All")  # Add option to show all assets
    
    selected_preview_option = st.radio(
        "Select asset to preview:", 
        available_options,
        index=available_options.index("All"),  # Changed default to "All"
        horizontal=True,
        label_visibility="collapsed"
    )
    
    # Show the data for the selected asset
    if selected_preview_option:
        if selected_preview_option == "All":
            st.subheader("Price data for all assets")
            st.dataframe(data.reset_index(), height=300, use_container_width=True, hide_index=False)
        else:
            st.subheader(f"Price data for {selected_preview_option}")
            
            # Get the data for the selected asset
            if selected_preview_option in individual_dfs:
                preview_df = individual_dfs[selected_preview_option]
                # Show the full dataset
                st.dataframe(preview_df, height=300, use_container_width=True, hide_index=False)
            else:
                # Fallback to showing the column from the merged dataframe
                preview_df = data_with_date[['Date', selected_preview_option]].dropna()
                st.dataframe(preview_df, height=300, use_container_width=True, hide_index=False)
    
    # Calculate QQQ Correlation Section
    section_header("Correlation with QQQ")
    
    # Check if QQQ is in the available assets
    if "QQQ" not in assets:
        st.warning("QQQ data is not available for correlation analysis.")
    else:
        # Add explanation text
        explanation_text("The correlation analysis below shows how each asset's daily returns correlate with QQQ's daily returns over their overlapping time periods. Higher values (closer to 100%) indicate stronger positive correlation, meaning the asset tends to move in the same direction as QQQ. The date ranges show the periods over which the correlation is calculated.")
        
        # Calculate daily returns for each asset
        returns_dict = {}
        for asset in assets:
            returns_dict[asset] = data[asset].pct_change().dropna()
        
        # QQQ returns
        qqq_returns = returns_dict["QQQ"]
        
        # Calculate correlation for each asset with QQQ
        correlation_df, date_ranges = calculate_correlation_data(assets, data, qqq_returns)
        
        # Display correlation data in two columns
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Create the correlation bar chart - now sorted by correlation value
            fig = create_correlation_bar_chart(correlation_df)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Create a nicer looking table for display
            display_df = correlation_df.copy()
            
            # Exclude VIX and VVIX
            display_df = display_df[~display_df.index.isin(['VIX', 'VVIX'])]
            
            # Sort by correlation value from lowest to highest
            display_df = display_df.sort_values('Correlation')
            
            # Format the correlation as percentage with 2 decimal places
            display_df['Correlation'] = display_df['Correlation'].map(lambda x: f"{x*100:.2f}%")
            
            st.subheader("Correlation Table")
            st.dataframe(display_df, height=400, use_container_width=True, hide_index=False)
        
        # Rolling Correlation and Asset vs QQQ Comparison Section
        section_header("Advanced Analysis")
        explanation_text("The analysis below provides deeper insights into the relationships between each asset and QQQ over time. The rolling correlation shows how the correlation changes over different periods, while the price comparison helps visualize how the assets move in relation to QQQ.")
        
        # Create two rows with two columns each
        row1_col1, row1_col2 = st.columns(2)
        
        # ---- First Column: Rolling Correlation Plot ----
        with row1_col1:
            subsection_start()
            st.subheader("Rolling Correlation with QQQ")
            
            # Create a container div for both inputs
            input_container_start()
            
            # Create two columns for inputs with adjusted widths
            input_col1, input_col2 = st.columns([7, 3])  # 70% for assets, 30% for window size
            
            # Asset selector in first column
            with input_col1:
                input_item_start()
                select_text("Select asset to analyze:")
                asset_for_rolling = st.radio(
                    "Select asset to analyze:",
                    [a for a in assets if a != "QQQ" and a != "VIX" and a != "VVIX"],
                    index=0,
                    horizontal=True,
                    label_visibility="collapsed"
                )
                input_item_end()
            
            # Window size input in second column
            with input_col2:
                input_item_start()
                window_size = window_size_input(key="rolling_corr_window")
                input_item_end()
            
            # Close the container div
            input_container_end()
            
            # Add custom CSS to reduce spacing between radio options and reduce vertical spacing
            st.markdown("""
            <style>
            div.row-widget.stRadio > div {
                flex-direction: row;
                align-items: center;
                gap: 5px; /* Reduce gap between radio options */
            }
            div.row-widget.stRadio > div label {
                padding: 3px 5px; /* Reduce padding around labels */
                min-width: 50px; /* Ensure minimum width */
                font-size: 0.9em; /* Slightly smaller font */
            }
            /* Reduce vertical spacing between inputs and plot */
            div.block-container {
                padding-top: 1rem;
                padding-bottom: 1rem;
            }
            div.stPlotlyChart {
                margin-top: -15px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Calculate rolling correlation
            if asset_for_rolling and "QQQ" in assets:
                # Get returns for selected asset and QQQ
                asset_returns = returns_dict[asset_for_rolling]
                
                    # Calculate rolling correlation
                rolling_corr = calculate_rolling_correlation(asset_returns, qqq_returns, window_size)
                
                if rolling_corr is not None:
                    # Calculate overall correlation for annotation
                    aligned_returns = pd.concat([asset_returns, qqq_returns], axis=1).dropna()
                    aligned_returns.columns = [asset_for_rolling, 'QQQ']
                    overall_corr = aligned_returns[asset_for_rolling].corr(aligned_returns['QQQ'])
                    
                    # Create the rolling correlation plot
                    fig = create_rolling_correlation_plot(rolling_corr, window_size, asset_for_rolling, overall_corr)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Not enough data points for {asset_for_rolling} to calculate a {window_size}-day rolling correlation.")
            subsection_end()
                
        # ---- Second Column: Asset vs QQQ Price Comparison ----
        with row1_col2:
            subsection_start()
            st.subheader("Asset vs QQQ Price Comparison")
            
            # Create a container div for the input
            input_container_start()
            input_item_start()
            
            # Asset selector using radio buttons (matching screenshot style)
            select_text("Select asset to compare with QQQ:")
            asset_for_comparison = st.radio(
                "Select asset to compare with QQQ:",
                [a for a in assets if a != "QQQ" and a != "VIX" and a != "VVIX"],
                index=0,
                key="price_comparison_asset",
                horizontal=True,
                label_visibility="collapsed"
            )
            
            input_item_end()
            input_container_end()
            
            # Create a price comparison plot with dual Y axes
            if asset_for_comparison and "QQQ" in assets:
                # Get data for selected asset and QQQ
                asset_data = data[asset_for_comparison].dropna()
                qqq_data = data["QQQ"].dropna()
                
                # Create a dataframe with aligned dates
                merged_data = pd.concat([asset_data, qqq_data], axis=1).dropna()
                merged_data.columns = [asset_for_comparison, 'QQQ']
                
                if not merged_data.empty:
                    # Create the price comparison plot
                    fig = create_price_comparison_plot(merged_data, asset_for_comparison)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"No overlapping data found between {asset_for_comparison} and QQQ.")
            subsection_end()
        
        # Create a second row for the new plots
        row2_col1, row2_col2 = st.columns(2)
        
        # ---- Third Column: New Volatility Correlation Plot ----
        with row2_col1:
            subsection_start()
            st.subheader("Rolling Correlation of Volatilities with the Assets")
            
            # Create a container div for both inputs
            input_container_start()
            
            # Create columns for inputs with more balanced widths
            input_col1, input_col2, input_col3 = st.columns([5, 2.5, 2.5])  # More balanced column widths
            
            # Asset selector in first column
            with input_col1:
                input_item_start()
                select_text("Select asset to analyze:")
                
                # Get the list of available assets (QQQ and all other non-volatility assets)
                available_assets = ["QQQ"] + [a for a in assets if a != "QQQ" and a != "VIX" and a != "VVIX"]
                
                # Create a single radio selection group with a more compact layout
                vol_corr_asset = st.radio(
                    "Select asset to analyze:",
                    available_assets,
                    index=0,
                    horizontal=True,
                    key="vol_corr_asset_selector",
                    label_visibility="collapsed"
                )
                
                input_item_end()
            
            # Window size input in second column
            with input_col2:
                input_item_start()
                vol_corr_window = window_size_input(key="vol_corr_window")
                input_item_end()
            
            # Volatility selector in third column
            with input_col3:
                input_item_start()
                select_text("Correlate with:")
                # Replace checkboxes with radio button for mutual exclusivity
                volatility_source = st.radio(
                    "Select volatility index:",
                    options=["VIX", "VVIX"],
                    index=0,
                    key="volatility_source",
                    label_visibility="collapsed"
                )
                # Set flags based on radio selection
                use_vix_for_corr = (volatility_source == "VIX")
                use_vvix_for_corr = (volatility_source == "VVIX")
                input_item_end()
            
            # Close the container div
            input_container_end()
            
            # Add custom CSS to reduce spacing between radio options and reduce vertical spacing
            st.markdown("""
            <style>
            div.row-widget.stRadio > div {
                flex-direction: row;
                align-items: center;
                gap: 5px; /* Reduce gap between radio options */
            }
            div.row-widget.stRadio > div label {
                padding: 3px 5px; /* Reduce padding around labels */
                min-width: 50px; /* Ensure minimum width */
                font-size: 0.9em; /* Slightly smaller font */
            }
            /* Reduce vertical spacing between inputs and plot */
            div.block-container {
                padding-top: 1rem;
                padding-bottom: 1rem;
            }
            div.stPlotlyChart {
                margin-top: -15px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Process and display the plot
            if vol_corr_asset:
                # Check if we have the required data
                if vol_corr_asset in individual_dfs and ((use_vix_for_corr and "VIX" in individual_dfs) or (use_vvix_for_corr and "VVIX" in individual_dfs)):
                    # Get price data for selected asset
                    asset_df = individual_dfs[vol_corr_asset].copy()
                    
                    # Get price data for selected volatility index
                    vol_index = "VIX" if use_vix_for_corr else "VVIX"
                    vol_df = individual_dfs[vol_index].copy()
                    
                    # Ensure Date is correctly formatted
                    asset_df['Date'] = pd.to_datetime(asset_df['Date'])
                    vol_df['Date'] = pd.to_datetime(vol_df['Date'])
                    
                    # Set Date as index for both dataframes
                    asset_df.set_index('Date', inplace=True)
                    vol_df.set_index('Date', inplace=True)
                    
                    # Rename columns to avoid collision
                    asset_df = asset_df[[vol_corr_asset]].rename(columns={vol_corr_asset: f"{vol_corr_asset}_price"})
                    vol_df = vol_df[[vol_index]].rename(columns={vol_index: f"{vol_index}_price"})
                    
                    # Merge dataframes with inner join (only keep dates that exist in both)
                    merged_df = asset_df.join(vol_df, how='inner').dropna()
                    
                    # Check if we have enough data points
                    if len(merged_df) > vol_corr_window:
                        # Calculate returns for both series
                        returns_df = merged_df.pct_change().dropna()
                        
                        # Calculate rolling correlation
                        rolling_corr = returns_df[f"{vol_corr_asset}_price"].rolling(vol_corr_window).corr(returns_df[f"{vol_index}_price"])
                        
                        # Create a dataframe for plotting
                        plot_df = pd.DataFrame({
                            'Date': rolling_corr.index,
                            'Correlation': rolling_corr.values
                        }).dropna()
                        
                        if not plot_df.empty:
                            # Create the plot
                            import plotly.graph_objects as go
                            import numpy as np
                            
                            # Calculate statistics for the correlation data
                            corr_values = plot_df['Correlation'].values
                            overall_corr = np.mean(corr_values).round(4)
                            mean_corr = np.mean(corr_values).round(4)
                            min_corr = np.min(corr_values).round(4)
                            max_corr = np.max(corr_values).round(4)
                            q25 = np.percentile(corr_values, 25).round(4)
                            q50 = np.percentile(corr_values, 50).round(4) # median
                            q75 = np.percentile(corr_values, 75).round(4)
                            
                            # Determine dynamic y-axis range with some padding
                            y_min = min(min_corr, -0.1) - 0.05  # Add padding below min
                            y_max = max(max_corr, 0.1) + 0.05   # Add padding above max
                            
                            # For very negative correlations, ensure we show some positive space
                            if y_max < 0.1:
                                y_max = 0.1
                            
                            # For very positive correlations, ensure we show some negative space
                            if y_min > -0.1:
                                y_min = -0.1
                            
                            # Ensure y-axis ticks at regular intervals
                            y_tick_values = np.arange(np.floor(y_min*10)/10, np.ceil(y_max*10)/10 + 0.1, 0.1)
                            y_tick_values = [round(val, 1) for val in y_tick_values]
                            
                            # Create figure with dynamic height based on content
                            fig = go.Figure()
                            
                            # Add color-coded background regions
                            # Strong negative correlation region (-1.0 to -0.7)
                            if y_min < -0.7:
                                fig.add_shape(
                                    type="rect",
                                    xref="paper", yref="y",
                                    x0=0, x1=1,
                                    y0=-1.0, y1=-0.7,
                                    fillcolor="rgba(255, 200, 200, 0.3)",
                                    line_width=0,
                                )
                            
                            # Moderate negative correlation region (-0.7 to -0.3)
                            if y_min < -0.3 and y_max > -0.7:
                                fig.add_shape(
                                    type="rect",
                                    xref="paper", yref="y",
                                    x0=0, x1=1,
                                    y0=-0.7, y1=-0.3,
                                    fillcolor="rgba(255, 230, 230, 0.3)",
                                    line_width=0,
                                )
                            
                            # Weak correlation region (-0.3 to 0.3)
                            if y_min < 0.3 and y_max > -0.3:
                                fig.add_shape(
                                    type="rect",
                                    xref="paper", yref="y",
                                    x0=0, x1=1,
                                    y0=-0.3, y1=0.3,
                                    fillcolor="rgba(240, 240, 240, 0.3)",
                                    line_width=0,
                                )
                            
                            # Moderate positive correlation region (0.3 to 0.7)
                            if y_min < 0.7 and y_max > 0.3:
                                fig.add_shape(
                                    type="rect",
                                    xref="paper", yref="y",
                                    x0=0, x1=1,
                                    y0=0.3, y1=0.7,
                                    fillcolor="rgba(200, 255, 200, 0.3)",
                                    line_width=0,
                                )
                            
                            # Strong positive correlation region (0.7 to 1.0)
                            if y_max > 0.7:
                                fig.add_shape(
                                    type="rect",
                                    xref="paper", yref="y",
                                    x0=0, x1=1,
                                    y0=0.7, y1=1.0,
                                    fillcolor="rgba(150, 255, 150, 0.3)",
                                    line_width=0,
                                )
                            
                            # Add the correlation line
                            fig.add_trace(
                                go.Scatter(
                                    x=plot_df['Date'],
                                    y=plot_df['Correlation'],
                                    mode='lines',
                                    name=f'{vol_corr_asset}-{vol_index} Correlation',
                                    line=dict(color='#0066CC', width=2),
                                    hovertemplate='<b>Date:</b> %{x|%d-%b-%Y}<br>' +
                                                 f'<b>Correlation of {vol_index} and {vol_corr_asset}:</b> %{{y:.2f}}<extra></extra>'
                                )
                            )
                            
                            # Add horizontal reference line at 0
                            fig.add_shape(
                                type="line", 
                                line=dict(dash="solid", width=1.5, color="gray"),
                                y0=0, y1=0, 
                                x0=plot_df['Date'].min(), 
                                x1=plot_df['Date'].max()
                            )
                            
                            # Add overall correlation box
                            corr_text = f"Overall Correlation: {overall_corr:.2%}"
                            fig.add_annotation(
                                xref="paper", yref="paper",
                                x=0.02, y=0.05,
                                text=corr_text,
                                showarrow=False,
                                font=dict(size=12),
                                bgcolor="rgba(255, 255, 255, 0.8)",
                                bordercolor="gray",
                                borderwidth=1,
                                borderpad=4,
                                align="left"
                            )
                            
                            # Add statistics box
                            stats_text = f"<b>Statistics:</b><br>" + \
                                        f"Mean: {mean_corr:.2f}<br>" + \
                                        f"Q25: {q25:.2f}<br>" + \
                                        f"Median: {q50:.2f}<br>" + \
                                        f"Q75: {q75:.2f}<br>" + \
                                        f"Min: {min_corr:.2f}<br>" + \
                                        f"Max: {max_corr:.2f}"
                            
                            fig.add_annotation(
                                xref="paper", yref="paper",
                                x=0.98, y=0.95,
                                text=stats_text,
                                showarrow=False,
                                font=dict(size=11),
                                bgcolor="rgba(255, 255, 255, 0.8)",
                                bordercolor="gray",
                                borderwidth=1,
                                borderpad=4,
                                align="left",
                                xanchor="right",
                                yanchor="top"
                            )
                            
                            # Create legend for correlation regions
                            legend_traces = [
                                go.Scatter(
                                    x=[None], y=[None],
                                    mode='markers',
                                    marker=dict(size=10, color='rgba(150, 255, 150, 0.3)'),
                                    name='Strong Positive',
                                    showlegend=True
                                ),
                                go.Scatter(
                                    x=[None], y=[None],
                                    mode='markers',
                                    marker=dict(size=10, color='rgba(255, 200, 200, 0.3)'),
                                    name='Strong Negative',
                                    showlegend=True
                                ),
                                go.Scatter(
                                    x=[None], y=[None],
                                    mode='markers',
                                    marker=dict(size=10, color='rgba(240, 240, 240, 0.3)'),
                                    name='Weak Correlation',
                                    showlegend=True
                                ),
                                go.Scatter(
                                    x=[None], y=[None],
                                    mode='lines',
                                    line=dict(color='#0066CC', width=2),
                                    name=f'{vol_corr_window}-Day Rolling Correlation',
                                    showlegend=True
                                )
                            ]
                            
                            for trace in legend_traces:
                                fig.add_trace(trace)
                            
                            # Update layout with improved settings
                            fig.update_layout(
                                title=f"{vol_corr_window}-Day Rolling Correlation of {vol_corr_asset} with {vol_index}",
                                xaxis=dict(
                                    title="Date",
                                    showgrid=True,
                                    gridcolor='rgba(200, 200, 200, 0.2)',
                                    zeroline=False
                                ),
                                yaxis=dict(
                                    title="Correlation Coefficient",
                                    range=[y_min, y_max],
                                    tickvals=y_tick_values,
                                    tickformat='.1f',
                                    showgrid=True,
                                    gridcolor='rgba(200, 200, 200, 0.2)',
                                    zeroline=False
                                ),
                                height=480,
                                margin=dict(l=50, r=50, t=60, b=50),
                                hovermode="x unified",
                                template="plotly_white",
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="center",
                                    x=0.5
                                ),
                                plot_bgcolor='white'
                            )
                            
                            # Display the plot
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("No valid correlation data after calculation. Try a smaller window size.")
                    else:
                        st.warning(f"Not enough overlapping data between {vol_corr_asset} and {vol_index} to calculate {vol_corr_window}-day rolling correlation. Found {len(merged_df)} overlapping dates, need at least {vol_corr_window+1}.")
                else:
                    missing_data = []
                    if vol_corr_asset not in individual_dfs:
                        missing_data.append(vol_corr_asset)
                    if use_vix_for_corr and "VIX" not in individual_dfs:
                        missing_data.append("VIX")
                    if use_vvix_for_corr and "VVIX" not in individual_dfs:
                        missing_data.append("VVIX")
                    
                    st.warning(f"Missing required data for: {', '.join(missing_data)}. Please make sure all required datasets are loaded.")
            subsection_end()
                
        # ---- Fourth Column: Move Rolling Beta Plot here (previously "Coming Soon") ----
        with row2_col2:
            subsection_start()
            st.subheader("Rolling Beta with QQQ")
            
            # Create a container div for both inputs
            input_container_start()
            
            # Create two columns for inputs with adjusted widths
            input_col1, input_col2 = st.columns([7, 3])  # 70% for assets, 30% for window size
            
            # Asset selector in first column
            with input_col1:
                input_item_start()
                select_text("Select asset to analyze:")
                asset_for_beta = st.radio(
                    "Select asset to analyze:",
                    [a for a in assets if a != "QQQ" and a != "VIX" and a != "VVIX"],
                    index=0,
                    horizontal=True,
                    key="beta_asset_selector",
                    label_visibility="collapsed"
                )
                input_item_end()
            
            # Window size input in second column
            with input_col2:
                input_item_start()
                select_text("Select window size:")
                beta_window_size = st.number_input(
                    "Rolling window size for beta (trading days):",
                    min_value=1,
                    max_value=150,
                    value=30,
                    step=1,
                    key="beta_window_size",
                    label_visibility="collapsed"
                )
                input_item_end()
            
            # Close the container div
            input_container_end()
            
            # Calculate rolling beta
            if asset_for_beta and "QQQ" in assets:
                # Get returns for selected asset and QQQ
                asset_returns = returns_dict[asset_for_beta]
                
                # Calculate rolling beta
                rolling_beta = calculate_rolling_beta(asset_returns, qqq_returns, beta_window_size)
                
                # Process results
                if rolling_beta is not None:
                    # Calculate overall beta for annotation
                    aligned_returns = pd.concat([asset_returns, qqq_returns], axis=1).dropna()
                    aligned_returns.columns = [asset_for_beta, 'QQQ']
                    overall_beta = aligned_returns[asset_for_beta].cov(aligned_returns['QQQ']) / aligned_returns['QQQ'].var()
                    
                    # Create the rolling beta plot
                    fig = create_rolling_beta_plot(rolling_beta, beta_window_size, asset_for_beta, overall_beta)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Not enough data points for {asset_for_beta} to calculate a {beta_window_size}-day rolling beta.")
            subsection_end()
        
        # Create a third row for the new plots
        row3_col1, row3_col2 = st.columns(2)
        
        # ---- Fifth Column: Correlation with Volatility Plot ----
        with row3_col1:
            subsection_start()
            st.subheader("Correlation with Volatility Index")
            
            # Create a container div for the inputs
            input_container_start()
            
            # Create columns for inputs
            input_col1, input_col2, input_col3 = st.columns([5, 2.5, 2.5])
            
            # Asset selector in first column
            with input_col1:
                input_item_start()
                select_text("Select asset to analyze:")
                
                # Get the list of available assets (QQQ and all other non-volatility assets)
                available_assets = ["QQQ"] + [a for a in assets if a != "QQQ" and a != "VIX" and a != "VVIX"]
                
                # Create a single radio selection group
                corr_vol_asset = st.radio(
                    "Select asset to analyze:",
                    available_assets,
                    index=0,
                    horizontal=True,
                    key="corr_vol_asset_selector",
                    label_visibility="collapsed"
                )
                input_item_end()
            
            # Window size input in second column
            with input_col2:
                input_item_start()
                corr_vol_window = window_size_input(key="corr_vol_window")
                input_item_end()
            
            # Volatility selector in third column
            with input_col3:
                input_item_start()
                select_text("Correlate with:")
                # Radio button for volatility selection
                volatility_index = st.radio(
                    "Select volatility index:",
                    options=["VIX", "VVIX"],
                    index=0,
                    key="corr_vol_source",
                    label_visibility="collapsed"
                )
                input_item_end()
            
            # Close the container div
            input_container_end()
            
            # Process and display the plot
            if corr_vol_asset and volatility_index in ["VIX", "VVIX"]:
                # Check if we have the required data
                if corr_vol_asset in returns_dict and volatility_index in individual_dfs:
                    # Get returns for selected asset
                    asset_returns = returns_dict[corr_vol_asset]
                    
                    # Get volatility index data from individual_dfs
                    vol_df = individual_dfs[volatility_index]
                    
                    # Convert Date column to datetime if necessary
                    if 'Date' in vol_df.columns:
                        vol_df['Date'] = pd.to_datetime(vol_df['Date'])
                        vol_df.set_index('Date', inplace=True)
                    
                    # Get the actual volatility data and calculate returns
                    vol_data = vol_df[volatility_index]
                    vol_returns = vol_data.pct_change().dropna()
                    
                    # Calculate rolling correlation between asset and volatility index
                    rolling_corr = calculate_rolling_correlation(asset_returns, vol_returns, corr_vol_window)
                    
                    if rolling_corr is not None:
                        # Create the correlation-volatility plot
                        fig = create_correlation_volatility_plot(
                            rolling_corr, 
                            vol_data, 
                            volatility_index, 
                            corr_vol_window, 
                            corr_vol_asset, 
                            overall_corr=None
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"Not enough data points for {corr_vol_asset} to calculate a {corr_vol_window}-day rolling correlation.")
                else:
                    st.warning(f"Missing required data for either {corr_vol_asset} or {volatility_index}.")
            subsection_end()
        
        # ---- Sixth Column: Asset vs Volatility Price Comparison ----
        with row3_col2:
            subsection_start()
            st.subheader("Asset vs Volatility Index Comparison")
            
            # Create a container div for the inputs
            input_container_start()
            
            # Create columns for inputs
            input_col1, input_col2 = st.columns([7, 3])
            
            # Asset selector in first column
            with input_col1:
                input_item_start()
                select_text("Select asset to compare:")
                # Get the list of available assets (excluding volatility indices)
                avail_assets = [a for a in assets if a != "VIX" and a != "VVIX"]
                
                # Create a single radio selection group
                asset_vol_compare = st.radio(
                    "Select asset to compare:",
                    avail_assets,
                    index=0,
                    horizontal=True,
                    key="asset_vol_compare_selector",
                    label_visibility="collapsed"
                )
                input_item_end()
            
            # Volatility selector in second column
            with input_col2:
                input_item_start()
                select_text("Compare with:")
                # Radio button for volatility selection
                vol_compare_index = st.radio(
                    "Select volatility index:",
                    options=["VIX", "VVIX"],
                    index=0,
                    key="vol_compare_source",
                    label_visibility="collapsed"
                )
                input_item_end()
            
            # Close the container div
            input_container_end()
            
            # Process and display the plot
            if asset_vol_compare and vol_compare_index in ["VIX", "VVIX"]:
                # Check if we have the required data
                if asset_vol_compare in data.columns and vol_compare_index in individual_dfs:
                    # Get data for selected asset and volatility index
                    asset_data = data[asset_vol_compare]
                    
                    # Get volatility index data from individual_dfs
                    vol_df = individual_dfs[vol_compare_index]
                    
                    # Convert Date column to datetime if necessary
                    if 'Date' in vol_df.columns:
                        vol_df['Date'] = pd.to_datetime(vol_df['Date'])
                        vol_df.set_index('Date', inplace=True)
                    
                    # Get the actual volatility data
                    vol_data = vol_df[vol_compare_index]
                    
                    # Create a dataframe with aligned dates
                    merged_data = pd.concat([asset_data, vol_data], axis=1).dropna()
                    
                    if not merged_data.empty:
                        # Create the price comparison plot
                        fig = create_asset_volatility_comparison_plot(
                            merged_data, 
                            asset_vol_compare, 
                            vol_compare_index
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"No overlapping data found between {asset_vol_compare} and {vol_compare_index}.")
                else:
                    st.warning(f"Missing required data for either {asset_vol_compare} or {vol_compare_index}.")
            subsection_end()
        
        # Add a separator between sections
        horizontal_rule()
                
        # ---- Backtesting Section ----
        section_header("Backtesting the Hedging Strategy")
        
        explanation_text("""
            This section implements a backtesting framework for a long-short hedging strategy aimed at reducing market exposure while maintaining potential for returns. 
            The strategy involves going long on selected assets (YMAX, JEPQ, QYLD, YMAG, or QQQI) while simultaneously shorting QQQ to hedge against 
            broader market movements. By adjusting position sizes based on correlation and other factors, the strategy attempts to create a 
            market-neutral or reduced-beta portfolio that can potentially perform well in various market conditions.
        """)
        
        # Start backtest section with custom styling
        backtest_section_start()
        
        # Create a three-column layout for the backtest parameters (removing frequency)
        bt_col1, bt_col2, bt_col3 = st.columns(3)
        
        # Column 1: Strategy Selection
        with bt_col1:
            backtest_column_start()
            st.subheader("Select Strategy")
            
            strategy_options = ["Strategy 1", "Strategy 2", "Strategy 3", "Strategy 4"]
            strategy = st.radio(
                "Select strategy to backtest:",
                options=strategy_options,
                index=0,
                label_visibility="collapsed"
            )
            backtest_column_end()
        
        # Column 2: Asset Selection
        with bt_col2:
            backtest_column_start()
            st.subheader("Select Asset")
            
            # Get the available assets (excluding QQQ which is always used for hedging)
            backtest_assets = [a for a in assets if a != "QQQ" and a != "VIX" and a != "VVIX"]
            
            # Session state to track which asset is selected
            if 'selected_asset_index' not in st.session_state:
                st.session_state.selected_asset_index = 0  # Default to first asset
                
            # Function to handle checkbox changes
            def handle_asset_selection(index):
                st.session_state.selected_asset_index = index
            
            # Create 3 columns for the checkboxes
            asset_col1, asset_col2, asset_col3 = st.columns(3)
            
            # Arrange assets with 4 assets per row
            # Column 1 gets indices 0, 3, 6, 9, etc.
            # Column 2 gets indices 1, 4, 7, 10, etc.
            # Column 3 gets indices 2, 5, 8, 11, etc.
            
            with asset_col1:
                for i in range(0, len(backtest_assets), 3):
                    if i < len(backtest_assets):
                        asset = backtest_assets[i]
                        is_selected = i == st.session_state.selected_asset_index
                        if st.checkbox(asset, value=is_selected, key=f"asset_{i}", 
                                     on_change=handle_asset_selection, args=(i,)):
                            if i != st.session_state.selected_asset_index:
                                st.session_state.selected_asset_index = i
                                st.experimental_rerun()
            
            with asset_col2:
                for i in range(1, len(backtest_assets), 3):
                    if i < len(backtest_assets):
                        asset = backtest_assets[i]
                        is_selected = i == st.session_state.selected_asset_index
                        if st.checkbox(asset, value=is_selected, key=f"asset_{i}", 
                                     on_change=handle_asset_selection, args=(i,)):
                            if i != st.session_state.selected_asset_index:
                                st.session_state.selected_asset_index = i
                                st.experimental_rerun()
            
            with asset_col3:
                for i in range(2, len(backtest_assets), 3):
                    if i < len(backtest_assets):
                        asset = backtest_assets[i]
                        is_selected = i == st.session_state.selected_asset_index
                        if st.checkbox(asset, value=is_selected, key=f"asset_{i}", 
                                     on_change=handle_asset_selection, args=(i,)):
                            if i != st.session_state.selected_asset_index:
                                st.session_state.selected_asset_index = i
                                st.experimental_rerun()
            
            # Set the selected asset based on which checkbox is checked
            selected_asset = backtest_assets[st.session_state.selected_asset_index]
            
            backtest_column_end()
        
        # Column 3: Date Range Settings
        with bt_col3:
            backtest_column_start()
            st.subheader("Date Range")
            
            # Checkbox to use full available data range
            use_full_range = st.checkbox("Use full available data range for each asset", value=True)
            
            if use_full_range:
                full_range_message()
            else:
                # Get asset data to determine available date range
                if selected_asset in data.columns and "QQQ" in data.columns:
                    # Find common date range between selected asset and QQQ
                    asset_data = data[selected_asset].dropna()
                    qqq_data = data["QQQ"].dropna()
                    
                    # Get the date range with overlapping data
                    both_available = pd.concat([asset_data, qqq_data], axis=1).dropna()
                    if not both_available.empty:
                        min_date = both_available.index.min().date()
                        max_date = both_available.index.max().date()
                        
                        # Wrap date inputs in container
                        date_input_container_start()
                        
                        # Date inputs with default values
                        select_text("Start Date")
                        start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date, label_visibility="collapsed")
                        
                        select_text("End Date")
                        end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date, label_visibility="collapsed")
                        
                        date_input_container_end()
                    else:
                        st.warning(f"No overlapping data between {selected_asset} and QQQ")
                else:
                    st.warning("Select a valid asset to see available date range")
            backtest_column_end()
        
        # Run strategy when selected
        if strategy:
            # Set up container for the backtest results
            results_container = st.container()
            
            with results_container:
                if strategy == "Strategy 1":
                    st.subheader(f"Backtesting {strategy} - Correlation-Based Pairs Trading with {selected_asset}")
                    
                    # Strategy description
                    explanation_text(
                        "<b>Strategy Description:</b> This strategy implements a correlation-based pairs trading approach between the selected asset and QQQ. "
                        "It enters a long asset/short QQQ position when correlation drops to a specific level, then exits when correlation rises above a threshold. "
                        "The strategy aims to profit from temporary correlation divergences while maintaining market neutrality."
                    )
                    
                    # Add title for parameters section
                    st.markdown("### Parameters for Strategy 1")
                    
                    # Add checkbox for including dividends
                    include_dividends = st.checkbox("Factor in dividends in the backtest", value=False, key="s1_include_dividends")
                    
                    # Set up parameters for Strategy 1
                    param_col1, param_col2, param_col3 = st.columns(3)
                    
                    with param_col1:
                        window = st.number_input("Rolling Window (days)", min_value=1, value=30, step=1, key="window_input")
                        entry_corr = st.number_input("Entry Correlation", min_value=-1.0, max_value=1.0, value=0.85, step=0.01, format="%.2f", key="entry_corr_input")
                    
                    with param_col2:
                        exit_corr = st.number_input("Exit Correlation", 
                                                min_value=0.0, 
                                                max_value=2.0, 
                                                value=0.95, 
                                                step=0.01, 
                                                format="%.2f", 
                                                key="exit_corr_input")
                        asset_amount = st.number_input("Long Amount ($)", min_value=1, value=28000, step=1000, key="asset_amount_input")
                    
                    with param_col3:
                        qqq_amount = st.number_input("Short Amount ($)", min_value=1, value=10000, step=1000, key="qqq_amount_input")
                    
                    # Add horizontal line
                    st.markdown("---")
                    
                    # Place Run Backtest button separately
                    run_backtest = st.button("Run Backtest")
                    
                    if selected_asset in data.columns and "QQQ" in data.columns:
                        # Get asset data
                        asset_data = data[selected_asset].dropna()
                        qqq_data = data["QQQ"].dropna()
                        
                        # Load dividend data if we're including dividends
                        asset_dividends = None
                        qqq_dividends = None
                        
                        if include_dividends:
                                # Get asset dividend data if available
                                if selected_asset in dividend_dfs:
                                    asset_dividends = dividend_dfs[selected_asset]
                                else:
                                    st.warning(f"No dividend data found for {selected_asset}")
                                
                                # Get QQQ dividend data if available
                                if "QQQ" in dividend_dfs:
                                    qqq_dividends = dividend_dfs["QQQ"]
                                else:
                                    st.warning("No dividend data found for QQQ")
                        
                        # Determine date range
                        if use_full_range:
                            start_date = None
                            end_date = None
                            date_range_text = "Using full available date range"
                        else:
                            start_date = pd.Timestamp(start_date)
                            end_date = pd.Timestamp(end_date)
                            date_range_text = f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                        
                        explanation_text(date_range_text)
                        
                        # Run the backtest when button is clicked
                        if run_backtest:
                            with st.spinner(f"Running {strategy} backtest..."):
                                # Run the strategy
                                strategy_data, portfolio_results, metrics = run_strategy_1(
                                    asset_data, qqq_data, start_date, end_date,
                                    window, entry_corr, exit_corr, asset_amount, qqq_amount,
                                    asset_name=selected_asset,
                                    include_dividends=include_dividends,
                                    asset_dividends=asset_dividends,
                                    qqq_dividends=qqq_dividends
                                )
                                
                                # Create a 2x2 grid of plots
                                st.subheader("Backtest Results")
                                
                                # Create a 2-column, 3-row layout for plots with metrics table at the bottom right
                                row1_col1, row1_col2 = st.columns(2)  # Asset & QQQ PnL
                                row2_col1, row2_col2 = st.columns(2)  # Total PnL & Rolling Correlation/Entry/Exit
                                row3_col1, row3_col2 = st.columns(2)  # Dividend Bar Chart & Metrics Table
                                
                                with row1_col1:
                                    # Asset Cumulative PnL Plot
                                    fig1 = create_asset_pnl_chart(strategy_data, selected_asset)
                                    st.plotly_chart(fig1, use_container_width=True)
                                
                                with row1_col2:
                                    # QQQ Cumulative PnL Plot
                                    fig2 = create_qqq_pnl_chart(strategy_data)
                                    st.plotly_chart(fig2, use_container_width=True)
                                
                                with row2_col1:
                                    # Total Cumulative PnL Plot
                                    fig3 = create_total_pnl_chart(strategy_data)
                                    st.plotly_chart(fig3, use_container_width=True)
                                
                                with row2_col2:
                                    # Rolling Correlation with Entry/Exit Points
                                    fig4 = create_correlation_entry_exit_chart(
                                        strategy_data, entry_corr, exit_corr, window
                                    )
                                    st.plotly_chart(fig4, use_container_width=True)
                                
                                with row3_col1:
                                    # Only show dividend bar chart if dividends are included
                                    if include_dividends:
                                        # Create dividend bar chart
                                        fig_div = create_dividend_chart(strategy_data)
                                        if fig_div:
                                            st.plotly_chart(fig_div, use_container_width=True)
                                        else:
                                            st.info("No dividend payments received during trading periods.")
                                    else:
                                        st.info("Enable 'Factor in dividends' to see dividend payment data.")
                                
                                with row3_col2:
                                    # Calculate dividend metrics
                                    total_div_amount = strategy_data.loc[strategy_data['in_position'], 'dividend_payment'].sum() if include_dividends else 0.0
                                    dividend_days = sum((strategy_data['in_position']) & (strategy_data['dividend_payment'] > 0)) if include_dividends else 0
                                    
                                    # Calculate annualized percentage return
                                    days_in_market = metrics.get("Number of days in market", 0)
                                    final_pnl = strategy_data['Total_Cum_PnL'].iloc[-1] if not strategy_data.empty else 0.0
                                    
                                    if days_in_market > 0:
                                        annualized_percentage = ((final_pnl + total_div_amount) * 250 / days_in_market) / 100
                                    else:
                                        annualized_percentage = 0.0
                                    
                                    # Create metrics table
                                    metrics_df = create_metrics_dataframe(
                                        metrics, days_in_market, total_div_amount, dividend_days, final_pnl, annualized_percentage
                                    )
                                    
                                    st.markdown("### Strategy Metrics")
                                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                                
                                # Add data table
                                st.subheader("Strategy Data")
                                
                                # Create a download button for the data
                                csv = portfolio_results.to_csv(index=True)
                                st.download_button(
                                    label="Download Data as CSV",
                                    data=csv,
                                    file_name="strategy_data.csv",
                                    mime="text/csv",
                                )
                                
                                # Display the dataframe
                                st.dataframe(
                                    portfolio_results, 
                                    use_container_width=True,
                                    hide_index=False
                                )
                
                elif strategy == "Strategy 2":
                    st.subheader(f"Backtesting {strategy} - VIX/VVIX-Based Pairs Trading with {selected_asset}")
                    
                    # Strategy description
                    explanation_text(
                        "<b>Strategy Description:</b> This strategy implements a VIX/VVIX-based pairs trading approach between the selected asset and QQQ. "
                        "It enters a long asset/short QQQ position when VIX and/or VVIX is within specified ranges AND correlation is above a threshold level. "
                        "The strategy aims to capture opportunities during specific market volatility conditions while maintaining correlation safety checks."
                    )
                    
                    # Add title for parameters section
                    st.markdown("### Parameters for Strategy 2")
                    
                    # Add checkbox for including dividends
                    include_dividends = st.checkbox("Factor in dividends in the backtest", value=False, key="s2_include_dividends")
                    
                    # Set up checkboxes for using VIX/VVIX
                    use_vix_vvix_col1, use_vix_vvix_col2 = st.columns(2)
                    
                    with use_vix_vvix_col1:
                        use_vix = st.checkbox("Use VIX for trading decisions:", value=True, key="use_vix_checkbox")
                    
                    with use_vix_vvix_col2:
                        use_vvix = st.checkbox("Use VVIX for trading decisions:", value=False, key="use_vvix_checkbox")
                    
                    # VIX parameters section
                    if use_vix:
                        st.subheader("VIX Parameters")
                        vix_col1, vix_col2 = st.columns(2)
                        
                        with vix_col1:
                            vix_lower = st.number_input("VIX Lower Bound:", 
                                                   min_value=0.0, 
                                                   value=20.0, 
                                                   step=1.0, 
                                                   format="%.1f", 
                                                   key="vix_lower_input",
                                                   help="Enter position when VIX is above this level")
                        
                        with vix_col2:
                            vix_upper = st.number_input("VIX Upper Bound:", 
                                                   min_value=5.0, 
                                                   value=30.0, 
                                                   step=1.0, 
                                                   format="%.1f", 
                                                   key="vix_upper_input",
                                                   help="Enter position when VIX is below this level")
                    else:
                        # Define default values if not using VIX
                        vix_lower = 20.0
                        vix_upper = 30.0
                    
                    # VVIX parameters section
                    if use_vvix:
                        st.subheader("VVIX Parameters")
                        vvix_col1, vvix_col2 = st.columns(2)
                        
                        with vvix_col1:
                            vvix_lower = st.number_input("VVIX Lower Bound:", 
                                                    min_value=0.0, 
                                                    value=100.0, 
                                                    step=1.0, 
                                                    format="%.1f", 
                                                    key="vvix_lower_input",
                                                    help="Enter position when VVIX is above this level")
                        
                        with vvix_col2:
                            vvix_upper = st.number_input("VVIX Upper Bound:", 
                                                    min_value=50.0, 
                                                    value=150.0, 
                                                    step=1.0, 
                                                    format="%.1f", 
                                                    key="vvix_upper_input",
                                                    help="Enter position when VVIX is below this level")
                    else:
                        # Define default values if not using VVIX
                        vvix_lower = 100.0
                        vvix_upper = 150.0
                    
                    # Correlation and investment parameters
                    st.subheader("Correlation & Investment Parameters")
                    param_col1, param_col2, param_col3 = st.columns(3)
                    
                    with param_col1:
                        window = st.number_input("Rolling Window (days)", 
                                            min_value=1, 
                                            value=30, 
                                            step=1, 
                                            key="s2_window_input")
                        
                    with param_col2:
                        corr_above = st.number_input("Correlation Above", 
                                                min_value=0.0, 
                                                max_value=1.0, 
                                                value=0.9, 
                                                step=0.01, 
                                                format="%.2f", 
                                                key="corr_above_input")
                        
                    with param_col3:
                        asset_amount = st.number_input("Long Amount ($)", 
                                                  min_value=1, 
                                                  value=10000, 
                                                  step=1000, 
                                                  key="s2_asset_amount_input")
                        qqq_amount = st.number_input("Short Amount ($)", 
                                                min_value=1, 
                                                value=10000, 
                                                step=1000, 
                                                key="s2_qqq_amount_input")
                    
                    # Add horizontal line
                    st.markdown("---")
                    
                    # Place Run Backtest button separately
                    run_backtest = st.button("Run Backtest", key="s2_run_backtest")
                    
                    # Check for required data
                    required_data_missing = False
                    error_message = ""
                    
                    if use_vix and "VIX" not in data.columns:
                        required_data_missing = True
                        error_message += "VIX data is required but not available. "
                    
                    if use_vvix and "VVIX" not in data.columns:
                        required_data_missing = True
                        error_message += "VVIX data is required but not available. "
                    
                    if required_data_missing:
                        st.error(f"{error_message}Please ensure the required data is loaded.")
                    elif selected_asset in data.columns and "QQQ" in data.columns:
                        # Get asset data
                        asset_data = data[selected_asset].dropna()
                        qqq_data = data["QQQ"].dropna()
                        vix_data = data["VIX"].dropna() if "VIX" in data.columns else None
                        vvix_data = data["VVIX"].dropna() if "VVIX" in data.columns else None
                        
                        # Load dividend data if we're including dividends
                        asset_dividends = None
                        qqq_dividends = None
                        
                        if include_dividends:
                                # Get asset dividend data if available
                                if selected_asset in dividend_dfs:
                                    asset_dividends = dividend_dfs[selected_asset]
                                else:
                                    st.warning(f"No dividend data found for {selected_asset}")
                                
                                # Get QQQ dividend data if available
                                if "QQQ" in dividend_dfs:
                                    qqq_dividends = dividend_dfs["QQQ"]
                                else:
                                    st.warning("No dividend data found for QQQ")
                        
                        # Determine date range
                        if use_full_range:
                            start_date = None
                            end_date = None
                            date_range_text = "Using full available date range"
                        else:
                            start_date = pd.Timestamp(start_date)
                            end_date = pd.Timestamp(end_date)
                            date_range_text = f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                        
                        explanation_text(date_range_text)
                        
                        # Run the backtest when button is clicked
                        if run_backtest:
                            with st.spinner(f"Running {strategy} backtest..."):
                                # Run the strategy
                                strategy_data, portfolio_results, metrics = run_strategy_2(
                                    asset_data, qqq_data, vix_data, vvix_data, start_date, end_date,
                                    window, vix_lower, vix_upper, vvix_lower, vvix_upper, corr_above, 
                                    use_vix, use_vvix, asset_amount, qqq_amount,
                                    asset_name=selected_asset,
                                    include_dividends=include_dividends,
                                    asset_dividends=asset_dividends,
                                    qqq_dividends=qqq_dividends
                                )
                                
                                # Create a 2x2 grid of plots
                                st.subheader("Backtest Results")
                                
                                # Create a 2-column, 3-row layout for plots with metrics table at the bottom right
                                row1_col1, row1_col2 = st.columns(2)  # Asset & QQQ PnL
                                row2_col1, row2_col2 = st.columns(2)  # Total PnL & Volatility Entry/Exit
                                row3_col1, row3_col2 = st.columns(2)  # Dividend Bar Chart or Daily PnL & Metrics Table
                                
                                with row1_col1:
                                    # Asset Cumulative PnL Plot
                                    fig1 = create_asset_pnl_chart(strategy_data, selected_asset)
                                    st.plotly_chart(fig1, use_container_width=True)
                                
                                with row1_col2:
                                    # QQQ Cumulative PnL Plot
                                    fig2 = create_qqq_pnl_chart(strategy_data)
                                    st.plotly_chart(fig2, use_container_width=True)
                                
                                with row2_col1:
                                    # Total Cumulative PnL Plot
                                    fig3 = create_total_pnl_chart(strategy_data)
                                    st.plotly_chart(fig3, use_container_width=True)
                                
                                with row2_col2:
                                    # Volatility Plot with Entry/Exit Points
                                    fig4 = create_volatility_entry_exit_chart(
                                        strategy_data, use_vix, use_vvix,
                                        vix_lower, vix_upper,
                                        vvix_lower, vvix_upper
                                    )
                                    st.plotly_chart(fig4, use_container_width=True)
                                
                                with row3_col1:
                                    # Show dividend chart if dividends are included, otherwise show daily PnL
                                    if include_dividends:
                                        fig_div = create_dividend_chart(strategy_data)
                                        if fig_div:
                                            st.plotly_chart(fig_div, use_container_width=True)
                                        else:
                                            st.info("No dividend payments received during trading periods.")
                                    else:
                                        # Show daily PnL chart
                                        fig_daily = create_daily_pnl_chart(strategy_data)
                                        st.plotly_chart(fig_daily, use_container_width=True)
                                
                                with row3_col2:
                                    # Calculate dividend metrics directly from the data
                                    dividend_amount = strategy_data.loc[strategy_data['in_position'], 'dividend_payment'].sum() if include_dividends else 0
                                    dividend_days = sum((strategy_data['in_position']) & (strategy_data['dividend_payment'] > 0)) if include_dividends else 0
                                        
                                    # Calculate annualized percentage return
                                    days_in_market = metrics.get("Number of days in market", 0)
                                    final_pnl = strategy_data['Total_Cum_PnL'].iloc[-1] if not strategy_data.empty else 0.0
                                    
                                    if days_in_market > 0:
                                        annualized_percentage = ((final_pnl + dividend_amount) * 250 / days_in_market) / 100
                                    else:
                                        annualized_percentage = 0.0
                                        
                                    # Create metrics table
                                    metrics_df = create_metrics_dataframe(
                                        metrics, days_in_market, dividend_amount, dividend_days, final_pnl, annualized_percentage
                                    )
                                    
                                    st.markdown("### Strategy Metrics")
                                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                                
                                # Add data table
                                st.subheader("Strategy Data")
                                
                                # Create a download button for the data
                                csv = portfolio_results.to_csv(index=True)
                                st.download_button(
                                    label="Download Data as CSV",
                                    data=csv,
                                    file_name="strategy_data.csv",
                                    mime="text/csv",
                                )
                                
                                # Display the dataframe
                                st.dataframe(
                                    portfolio_results, 
                                    use_container_width=True,
                                    hide_index=False
                                )
                
                elif strategy == "Strategy 3":
                    st.subheader(f"Backtesting {strategy} - Correlation-Bound Pairs Trading with {selected_asset}")
                    
                    # Strategy description
                    explanation_text(
                        "<b>Strategy Description:</b> This strategy implements a correlation-bound pairs trading approach between the selected asset and QQQ. "
                        "It enters a long asset/short QQQ position when correlation is within a specified range (lower and upper bounds, inclusive), "
                        "then exits when correlation moves outside that range. The strategy aims to capture opportunities in specific correlation regions."
                    )
                    
                    # Add title for parameters section
                    st.markdown("### Parameters for Strategy 3")
                    
                    # Add checkbox for including dividends
                    include_dividends = st.checkbox("Factor in dividends in the backtest", value=False, key="s3_include_dividends")
                    
                    # Set up parameters for Strategy 3
                    param_col1, param_col2, param_col3 = st.columns(3)
                    
                    with param_col1:
                        window = st.number_input("Rolling Window (days)", 
                                               min_value=1, 
                                               value=30, 
                                               step=1, 
                                               key="s3_window_input")
                        corr_lower = st.number_input("Lower Correlation Bound", 
                                                   min_value=0.0, 
                                                   max_value=1.0, 
                                                   value=0.80, 
                                                   step=0.01, 
                                                   format="%.2f", 
                                                   key="corr_lower_input")
                    
                    with param_col2:
                        corr_upper = st.number_input("Upper Correlation Bound", 
                                                   min_value=0.0, 
                                                   max_value=2.0, 
                                                   value=0.90, 
                                                   step=0.01, 
                                                   format="%.2f", 
                                                   key="corr_upper_input")
                        asset_amount = st.number_input("Long Amount ($)", 
                                                    min_value=1, 
                                                    value=10000, 
                                                    step=1000, 
                                                    key="s3_asset_amount_input")
                    
                    with param_col3:
                        qqq_amount = st.number_input("Short Amount ($)", 
                                                  min_value=1, 
                                                  value=10000, 
                                                  step=1000, 
                                                  key="s3_qqq_amount_input")
                    
                    # Add horizontal line
                    st.markdown("---")
                    
                    # Place Run Backtest button separately
                    run_backtest = st.button("Run Backtest", key="s3_run_backtest")
                    
                    if selected_asset in data.columns and "QQQ" in data.columns:
                        # Get asset data
                        asset_data = data[selected_asset].dropna()
                        qqq_data = data["QQQ"].dropna()
                        
                        # Load dividend data if we're including dividends
                        asset_dividends = None
                        qqq_dividends = None
                        
                        if include_dividends:
                                # Get asset dividend data if available
                                if selected_asset in dividend_dfs:
                                    asset_dividends = dividend_dfs[selected_asset]
                                else:
                                    st.warning(f"No dividend data found for {selected_asset}")
                                
                                # Get QQQ dividend data if available
                                if "QQQ" in dividend_dfs:
                                    qqq_dividends = dividend_dfs["QQQ"]
                                else:
                                    st.warning("No dividend data found for QQQ")
                        
                        # Determine date range
                        if use_full_range:
                            start_date = None
                            end_date = None
                            date_range_text = "Using full available date range"
                        else:
                            start_date = pd.Timestamp(start_date)
                            end_date = pd.Timestamp(end_date)
                            date_range_text = f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                        
                        explanation_text(date_range_text)
                        
                        # Run the backtest when button is clicked
                        if run_backtest:
                            with st.spinner(f"Running {strategy} backtest..."):
                                # Run the strategy
                                strategy_data, portfolio_results, metrics = run_strategy_3(
                                    asset_data, qqq_data, start_date, end_date,
                                    window, corr_lower, corr_upper, asset_amount, qqq_amount,
                                    asset_name=selected_asset,
                                    include_dividends=include_dividends,
                                    asset_dividends=asset_dividends,
                                    qqq_dividends=qqq_dividends
                                )
                                
                                # Create a 2x2 grid of plots
                                st.subheader("Backtest Results")
                                
                                # Create a 2-column, 3-row layout for plots with metrics table at the bottom right
                                row1_col1, row1_col2 = st.columns(2)  # Asset & QQQ PnL
                                row2_col1, row2_col2 = st.columns(2)  # Total PnL & Rolling Correlation/Entry/Exit
                                row3_col1, row3_col2 = st.columns(2)  # Dividend Bar Chart & Metrics Table
                                
                                with row1_col1:
                                    # Asset Cumulative PnL Plot
                                    fig1 = create_asset_pnl_chart(strategy_data, selected_asset)
                                    st.plotly_chart(fig1, use_container_width=True)
                                
                                with row1_col2:
                                    # QQQ Cumulative PnL Plot
                                    fig2 = create_qqq_pnl_chart(strategy_data)
                                    st.plotly_chart(fig2, use_container_width=True)
                                
                                with row2_col1:
                                    # Total Cumulative PnL Plot
                                    fig3 = create_total_pnl_chart(strategy_data)
                                    st.plotly_chart(fig3, use_container_width=True)
                                
                                with row2_col2:
                                    # Create a custom correlation entry/exit visualization for Strategy 3
                                    # that shows both upper and lower bounds
                                    fig4 = create_correlation_entry_exit_chart(
                                        strategy_data, corr_lower, corr_upper, window
                                    )
                                    st.plotly_chart(fig4, use_container_width=True)
                                
                                with row3_col1:
                                    # Only show dividend bar chart if dividends are included
                                    if include_dividends:
                                        # Create dividend bar chart
                                        fig_div = create_dividend_chart(strategy_data)
                                        if fig_div:
                                            st.plotly_chart(fig_div, use_container_width=True)
                                        else:
                                            st.info("No dividend payments received during trading periods.")
                                    else:
                                        # Show daily PnL chart if dividends are not included
                                        fig_daily = create_daily_pnl_chart(strategy_data)
                                        st.plotly_chart(fig_daily, use_container_width=True)
                                
                                with row3_col2:
                                    # Calculate dividend metrics
                                    total_div_amount = strategy_data.loc[strategy_data['in_position'], 'dividend_payment'].sum() if include_dividends else 0.0
                                    dividend_days = sum((strategy_data['in_position']) & (strategy_data['dividend_payment'] > 0)) if include_dividends else 0
                                    
                                    # Calculate annualized percentage return
                                    days_in_market = metrics.get("Number of days in market", 0)
                                    final_pnl = strategy_data['Total_Cum_PnL'].iloc[-1] if not strategy_data.empty else 0.0
                                    
                                    if days_in_market > 0:
                                        annualized_percentage = ((final_pnl + total_div_amount) * 250 / days_in_market) / 100
                                    else:
                                        annualized_percentage = 0.0
                                    
                                    # Create metrics table
                                    metrics_df = create_metrics_dataframe(
                                        metrics, days_in_market, total_div_amount, dividend_days, final_pnl, annualized_percentage
                                    )
                                    
                                    st.markdown("### Strategy Metrics")
                                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                                
                                # Add data table
                                st.subheader("Strategy Data")
                                
                                # Create a download button for the data
                                csv = portfolio_results.to_csv(index=True)
                                st.download_button(
                                    label="Download Data as CSV",
                                    data=csv,
                                    file_name="strategy_data.csv",
                                    mime="text/csv",
                                )
                                
                                # Display the dataframe
                                st.dataframe(
                                    portfolio_results, 
                                    use_container_width=True,
                                    hide_index=False
                                )
                    
                elif strategy == "Strategy 4":
                    st.subheader(f"Backtesting {strategy} - Advanced Volatility & Correlation Strategy with {selected_asset}")
                    
                    # Strategy description
                    explanation_text(
                        "<b>Strategy Description:</b> This advanced strategy combines volatility indicators (VIX/VVIX) with multiple correlation signals. "
                        "It extends Strategy 2 by adding correlation analysis between VIX/VVIX and both the selected asset and QQQ. "
                        "The strategy enters positions when all enabled conditions are met simultaneously, allowing for more refined "
                        "market timing based on both volatility levels and correlation patterns."
                    )
                    
                    # Add title for parameters section
                    st.markdown("### Parameters for Strategy 4")
                    
                    # Add checkbox for including dividends
                    include_dividends = st.checkbox("Factor in dividends in the backtest:", value=False, key="s4_include_dividends")
                    
                    # Set up checkboxes for using VIX/VVIX for volatility conditions
                    st.subheader("Volatility Indicators")
                    use_vix_vvix_col1, use_vix_vvix_col2 = st.columns(2)
                    
                    with use_vix_vvix_col1:
                        use_vix = st.checkbox("Use VIX for trading decisions:", value=True, key="s4_use_vix_checkbox")
                    
                    with use_vix_vvix_col2:
                        use_vvix = st.checkbox("Use VVIX for trading decisions:", value=False, key="s4_use_vvix_checkbox")
                    
                    # VIX parameters section
                    if use_vix:
                        st.subheader("VIX Range Parameters")
                        vix_col1, vix_col2 = st.columns(2)
                        
                        with vix_col1:
                            vix_lower = st.number_input("VIX Lower Bound:", 
                                                   min_value=0.0, 
                                                   value=20.0, 
                                                   step=1.0, 
                                                   format="%.1f", 
                                                   key="s4_vix_lower_input",
                                                   help="Enter position when VIX is above this level")
                        
                        with vix_col2:
                            vix_upper = st.number_input("VIX Upper Bound:", 
                                                   min_value=5.0, 
                                                   value=30.0, 
                                                   step=1.0, 
                                                   format="%.1f", 
                                                   key="s4_vix_upper_input",
                                                   help="Enter position when VIX is below this level")
                    else:
                        # Define default values if not using VIX
                        vix_lower = 20.0
                        vix_upper = 30.0
                    
                    # VVIX parameters section
                    if use_vvix:
                        st.subheader("VVIX Range Parameters")
                        vvix_col1, vvix_col2 = st.columns(2)
                        
                        with vvix_col1:
                            vvix_lower = st.number_input("VVIX Lower Bound:", 
                                                    min_value=0.0, 
                                                    value=100.0, 
                                                    step=1.0, 
                                                    format="%.1f", 
                                                    key="s4_vvix_lower_input",
                                                    help="Enter position when VVIX is above this level")
                        
                        with vvix_col2:
                            vvix_upper = st.number_input("VVIX Upper Bound:", 
                                                    min_value=50.0, 
                                                    value=150.0, 
                                                    step=1.0, 
                                                    format="%.1f", 
                                                    key="s4_vvix_upper_input",
                                                    help="Enter position when VVIX is below this level")
                    else:
                        # Define default values if not using VVIX
                        vvix_lower = 100.0
                        vvix_upper = 150.0
                    
                    # Add additional correlation parameters for Strategy 4
                    st.subheader("Advanced Correlation Parameters")
                    
                    # VIX/VVIX vs Asset Correlation
                    corr_col1, corr_col2 = st.columns(2)
                    
                    with corr_col1:
                        use_vix_asset_corr = st.checkbox(f"Include VIX vs {selected_asset} Correlation:", value=False, key="use_vix_asset_corr")
                        
                        if use_vix_asset_corr:
                            vix_asset_window = st.number_input(
                                f"VIX-{selected_asset} Rolling Window (days):", 
                                min_value=1, 
                                value=30, 
                                step=1, 
                                key="vix_asset_window"
                            )
                            
                            vix_asset_corr_threshold = st.number_input(
                                f"VIX-{selected_asset} Correlation less than or equal to:", 
                                min_value=-1.0, 
                                max_value=1.0, 
                                value=-0.70, 
                                step=0.01, 
                                format="%.2f", 
                                key="vix_asset_corr_threshold"
                            )
                        else:
                            vix_asset_window = 30
                            vix_asset_corr_threshold = -0.70
                    
                    with corr_col2:
                        use_vvix_asset_corr = st.checkbox(f"Include VVIX vs {selected_asset} Correlation:", value=False, key="use_vvix_asset_corr")
                        
                        if use_vvix_asset_corr:
                            vvix_asset_window = st.number_input(
                                f"VVIX-{selected_asset} Rolling Window (days):", 
                                min_value=1, 
                                value=30, 
                                step=1, 
                                key="vvix_asset_window"
                            )
                            
                            vvix_asset_corr_threshold = st.number_input(
                                f"VVIX-{selected_asset} Correlation less than or equal to:", 
                                min_value=-1.0, 
                                max_value=1.0, 
                                value=-0.70, 
                                step=0.01, 
                                format="%.2f", 
                                key="vvix_asset_corr_threshold"
                            )
                        else:
                            vvix_asset_window = 30
                            vvix_asset_corr_threshold = -0.70
                    
                    # VIX/VVIX vs QQQ Correlation
                    corr_col3, corr_col4 = st.columns(2)
                    
                    with corr_col3:
                        use_vix_qqq_corr = st.checkbox("Include VIX vs QQQ Correlation:", value=False, key="use_vix_qqq_corr")
                        
                        if use_vix_qqq_corr:
                            vix_qqq_window = st.number_input(
                                "VIX-QQQ Rolling Window (days):", 
                                min_value=1, 
                                value=30, 
                                step=1, 
                                key="vix_qqq_window"
                            )
                            
                            vix_qqq_corr_threshold = st.number_input(
                                "VIX-QQQ Correlation less than or equal to:", 
                                min_value=-1.0, 
                                max_value=1.0, 
                                value=-0.70, 
                                step=0.01, 
                                format="%.2f", 
                                key="vix_qqq_corr_threshold"
                            )
                        else:
                            vix_qqq_window = 30
                            vix_qqq_corr_threshold = -0.70
                    
                    with corr_col4:
                        use_vvix_qqq_corr = st.checkbox("Include VVIX vs QQQ Correlation:", value=False, key="use_vvix_qqq_corr")
                        
                        if use_vvix_qqq_corr:
                            vvix_qqq_window = st.number_input(
                                "VVIX-QQQ Rolling Window (days):", 
                                min_value=1, 
                                value=30, 
                                step=1, 
                                key="vvix_qqq_window"
                            )
                            
                            vvix_qqq_corr_threshold = st.number_input(
                                "VVIX-QQQ Correlation less than or equal to:", 
                                min_value=-1.0, 
                                max_value=1.0, 
                                value=-0.70, 
                                step=0.01, 
                                format="%.2f", 
                                key="vvix_qqq_corr_threshold"
                            )
                        else:
                            vvix_qqq_window = 30
                            vvix_qqq_corr_threshold = -0.70
                    
                    # New section for VIX/VVIX value change constraint
                    st.subheader("Volatility Value Change Constraint")
                    vol_change_col1, vol_change_col2 = st.columns(2)
                    
                    with vol_change_col1:
                        use_vol_change_constraint = st.checkbox("Stay in market if volatility index value is decreasing:", value=False, key="use_vol_change_constraint")
                    
                    with vol_change_col2:
                        if use_vol_change_constraint:
                            lookback_days = st.number_input(
                                "Compare with value from N days ago:", 
                                min_value=1, 
                                value=1, 
                                step=1, 
                                key="lookback_days",
                                help="Stay in market if today's volatility index value <= value from N days ago"
                            )
                        else:
                            lookback_days = 1
                    
                    # Add explanation for new constraint
                    if use_vol_change_constraint:
                        st.markdown(
                            """
                            <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                            <b>Volatility Value Change Constraint:</b> When this option is enabled, the strategy will stay in market (maintain position) 
                            if the selected volatility index (VIX or VVIX or both) value today is less than or equal to its value from the specified number of days ago. 
                            This allows the strategy to remain in position when volatility is decreasing or stable.
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    
                    # Asset-QQQ correlation parameters (from Strategy 2)
                    st.subheader("Asset-QQQ Correlation Parameters")
                    param_col1, param_col2, param_col3 = st.columns(3)
                    
                    with param_col1:
                        window = st.number_input(f"{selected_asset}-QQQ Rolling Window (days):", 
                                            min_value=1, 
                                            value=30, 
                                            step=1, 
                                            key="s4_window_input")
                        
                    with param_col2:
                        corr_above = st.number_input(f"{selected_asset}-QQQ Correlation greater than or equal to:", 
                                                min_value=0.0, 
                                                max_value=1.0, 
                                                value=0.90, 
                                                step=0.01, 
                                                format="%.2f", 
                                                key="s4_corr_above_input")
                        
                    with param_col3:
                        asset_amount = st.number_input(f"Long {selected_asset} Amount ($):", 
                                                  min_value=1, 
                                                  value=10000, 
                                                  step=1000, 
                                                  key="s4_asset_amount_input")
                        qqq_amount = st.number_input("Short QQQ Amount ($):", 
                                                min_value=1, 
                                                value=10000, 
                                                step=1000, 
                                                key="s4_qqq_amount_input")
                    
                    # Add horizontal line
                    st.markdown("---")
                    
                    # Place Run Backtest button separately
                    run_backtest = st.button("Run Backtest", key="s4_run_backtest")
                    
                    # Check for required data
                    required_data_missing = False
                    error_message = ""
                    
                    if use_vix and "VIX" not in data.columns:
                        required_data_missing = True
                        error_message += "VIX data is required but not available. "
                    
                    if use_vvix and "VVIX" not in data.columns:
                        required_data_missing = True
                        error_message += "VVIX data is required but not available. "
                    
                    if required_data_missing:
                        st.error(f"{error_message}Please ensure the required data is loaded.")
                    elif selected_asset in data.columns and "QQQ" in data.columns:
                        # Get asset data
                        asset_data = data[selected_asset].dropna()
                        qqq_data = data["QQQ"].dropna()
                        vix_data = data["VIX"].dropna() if "VIX" in data.columns else None
                        vvix_data = data["VVIX"].dropna() if "VVIX" in data.columns else None
                        
                        # Load dividend data if we're including dividends
                        asset_dividends = None
                        qqq_dividends = None
                        
                        if include_dividends:
                                # Get asset dividend data if available
                                if selected_asset in dividend_dfs:
                                    asset_dividends = dividend_dfs[selected_asset]
                                else:
                                    st.warning(f"No dividend data found for {selected_asset}")
                                
                                # Get QQQ dividend data if available
                                if "QQQ" in dividend_dfs:
                                    qqq_dividends = dividend_dfs["QQQ"]
                                else:
                                    st.warning("No dividend data found for QQQ")
                        
                        # Determine date range
                        if use_full_range:
                            start_date = None
                            end_date = None
                            date_range_text = "Using full available date range"
                        else:
                            start_date = pd.Timestamp(start_date)
                            end_date = pd.Timestamp(end_date)
                            date_range_text = f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                        
                        explanation_text(date_range_text)
                        
                        # Run the backtest when button is clicked
                        if run_backtest:
                            with st.spinner(f"Running {strategy} backtest..."):
                                # Run the strategy
                                strategy_data, portfolio_results, metrics = run_strategy_4(
                                    asset_data, qqq_data, vix_data, vvix_data, start_date, end_date,
                                    window, vix_lower, vix_upper, vvix_lower, vvix_upper, corr_above, 
                                    use_vix, use_vvix, asset_amount, qqq_amount,
                                    # New parameters for Strategy 4
                                    use_vix_asset_corr=use_vix_asset_corr,
                                    vix_asset_window=vix_asset_window,
                                    vix_asset_corr_threshold=vix_asset_corr_threshold,
                                    use_vvix_asset_corr=use_vvix_asset_corr,
                                    vvix_asset_window=vvix_asset_window,
                                    vvix_asset_corr_threshold=vvix_asset_corr_threshold,
                                    use_vix_qqq_corr=use_vix_qqq_corr,
                                    vix_qqq_window=vix_qqq_window,
                                    vix_qqq_corr_threshold=vix_qqq_corr_threshold,
                                    use_vvix_qqq_corr=use_vvix_qqq_corr,
                                    vvix_qqq_window=vvix_qqq_window,
                                    vvix_qqq_corr_threshold=vvix_qqq_corr_threshold,
                                    # New volatility value change constraint
                                    use_vol_change_constraint=use_vol_change_constraint,
                                    lookback_days=lookback_days,
                                    asset_name=selected_asset,
                                    include_dividends=include_dividends,
                                    asset_dividends=asset_dividends,
                                    qqq_dividends=qqq_dividends
                                )
                                
                                # Create a 2x2 grid of plots
                                st.subheader("Backtest Results")
                                
                                # Create a 2-column, 3-row layout for plots with metrics table at the bottom right
                                row1_col1, row1_col2 = st.columns(2)  # Asset & QQQ PnL
                                row2_col1, row2_col2 = st.columns(2)  # Total PnL & Volatility Entry/Exit
                                row3_col1, row3_col2 = st.columns(2)  # Dividend Bar Chart or Daily PnL & Metrics Table
                                
                                with row1_col1:
                                    # Asset Cumulative PnL Plot
                                    fig1 = create_asset_pnl_chart(strategy_data, selected_asset)
                                    st.plotly_chart(fig1, use_container_width=True)
                                
                                with row1_col2:
                                    # QQQ Cumulative PnL Plot
                                    fig2 = create_qqq_pnl_chart(strategy_data)
                                    st.plotly_chart(fig2, use_container_width=True)
                                
                                with row2_col1:
                                    # Total Cumulative PnL Plot
                                    fig3 = create_total_pnl_chart(strategy_data)
                                    st.plotly_chart(fig3, use_container_width=True)
                                
                                with row2_col2:
                                    # Volatility Plot with Entry/Exit Points
                                    fig4 = create_volatility_entry_exit_chart(
                                        strategy_data, use_vix, use_vvix,
                                        vix_lower, vix_upper,
                                        vvix_lower, vvix_upper
                                    )
                                    st.plotly_chart(fig4, use_container_width=True)
                                
                                with row3_col1:
                                    # Show dividend chart if dividends are included, otherwise show daily PnL
                                    if include_dividends:
                                        fig_div = create_dividend_chart(strategy_data)
                                        if fig_div:
                                            st.plotly_chart(fig_div, use_container_width=True)
                                        else:
                                            st.info("No dividend payments received during trading periods.")
                                    else:
                                        # Show daily PnL chart
                                        fig_daily = create_daily_pnl_chart(strategy_data)
                                        st.plotly_chart(fig_daily, use_container_width=True)
                                
                                with row3_col2:
                                    # Calculate dividend metrics directly from the data
                                    dividend_amount = strategy_data.loc[strategy_data['in_position'], 'dividend_payment'].sum() if include_dividends else 0
                                    dividend_days = sum((strategy_data['in_position']) & (strategy_data['dividend_payment'] > 0)) if include_dividends else 0
                                        
                                    # Calculate annualized percentage return
                                    days_in_market = metrics.get("Number of days in market", 0)
                                    final_pnl = strategy_data['Total_Cum_PnL'].iloc[-1] if not strategy_data.empty else 0.0
                                    
                                    if days_in_market > 0:
                                        annualized_percentage = ((final_pnl + dividend_amount) * 250 / days_in_market) / 100
                                    else:
                                        annualized_percentage = 0.0
                                        
                                    # Create metrics table
                                    metrics_df = create_metrics_dataframe(
                                        metrics, days_in_market, dividend_amount, dividend_days, final_pnl, annualized_percentage
                                    )
                                    
                                    st.markdown("### Strategy Metrics")
                                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                                
                                # Add data table
                                st.subheader("Strategy Data")
                                
                                # Create a download button for the data
                                csv = portfolio_results.to_csv(index=True)
                                st.download_button(
                                    label="Download Data as CSV",
                                    data=csv,
                                    file_name="strategy_data.csv",
                                    mime="text/csv",
                                )
                                
                                # Display the dataframe
                                st.dataframe(
                                    portfolio_results, 
                                    use_container_width=True,
                                    hide_index=False
                                )
        
        # End backtest section
        backtest_section_end()
else:
    st.error("Failed to load asset data. Please check if the directory and data files exist.")
