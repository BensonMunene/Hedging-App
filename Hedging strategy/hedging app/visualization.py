import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_correlation_bar_chart(correlation_df, custom_order=None):
    """
    Create a bar chart visualization of asset correlations with QQQ
    
    Parameters:
    -----------
    correlation_df: pd.DataFrame
        DataFrame containing correlation data
    custom_order: list or None
        Custom ordering of assets for the chart (not used anymore, kept for compatibility)
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    # Sort by correlation value from smallest to largest (exclude VIX and VVIX)
    corr_df_filtered = correlation_df[~correlation_df.index.isin(['VIX', 'VVIX'])]
    corr_df_sorted = corr_df_filtered.sort_values('Correlation')
    
    # Format for display
    formatted_values = [f"{x*100:.2f}%" for x in corr_df_sorted['Correlation']]
    
    # Create custom hover text
    hover_text = []
    for idx, val in enumerate(corr_df_sorted['Correlation']):
        asset = corr_df_sorted.index[idx]
        hover_text.append(
            f"Asset: {asset}<br>Correlation: {val*100:.2f}%"
        )
    
    # Create a bar chart to visualize correlations
    fig = go.Figure()
    
    # Add bars with customized appearance - use a color gradient from light to dark blue based on correlation
    num_assets = len(corr_df_sorted)
    color_scale = [
        f'rgba({30 + int(160 * (1 - i/num_assets))}, ' + 
        f'{120 + int(110 * (1 - i/num_assets))}, ' + 
        f'{190 + int(52 * (1 - i/num_assets))}, 0.8)'
        for i in range(num_assets)
    ]
    
    fig.add_trace(go.Bar(
        x=corr_df_sorted.index,
        y=corr_df_sorted['Correlation'] * 100,
        text=formatted_values,
        textposition='outside',
        hovertext=hover_text,
        hoverinfo='text',
        marker=dict(
            color=color_scale,
            line=dict(width=0)
        ),
        width=0.6  # Make bars slightly narrower
    ))
    
    # Set up the layout
    fig.update_layout(
        title='Correlation of Assets with QQQ',
        title_font=dict(size=20),
        xaxis_title='Asset',
        yaxis_title='Correlation Coefficient (%)',
        yaxis=dict(
            range=[0, 110],  # Give a little space above 100%
            tickmode='linear',
            tick0=0,
            dtick=20,  # Ticks at 0, 20, 40, 60, 80, 100
            tickformat='.0f',  # No decimal places for ticks
            gridcolor='rgba(200, 200, 200, 0.2)'
        ),
        xaxis=dict(
            tickangle=0,  # Horizontal labels
        ),
        plot_bgcolor='white',
        showlegend=False,
        height=450,
        margin=dict(l=40, r=20, t=50, b=40)
    )
    
    # Add horizontal gridlines
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.2)')
    
    return fig

def create_rolling_correlation_plot(rolling_corr, window_size, asset_name, overall_corr=None):
    """
    Create a dual-axis plot showing a rolling correlation between an asset and QQQ
    
    Parameters:
    -----------
    rolling_corr: pd.Series
        Rolling correlation time series
    window_size: int
        Window size used for the rolling correlation
    asset_name: str
        Name of the asset being analyzed
    overall_corr: float or None
        Overall correlation to display on the chart
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    # Create the rolling correlation plot
    fig = go.Figure()
    
    # Color mapping based on the original notebook
    colors = {
        'YMAX': 'teal',
        'YMAG': 'green',
        'QYLD': 'red',
        'QQQI': 'purple',
        'JEPQ': 'orange',
        'QQQ': 'blue',
        'PBP': 'brown',
        'GPIQ': 'olive',
        'FEPI': 'darkblue',
        'IQQQ': 'coral',
        'FTQI': 'magenta',
        'QYLG': 'darkgreen',
        'QDTY': 'darkred',
        'VIX': '#9467bd',  # purple
        'VVIX': '#ff7f0e'  # orange
    }
    
    # Default color for assets not in the colors dictionary
    default_color = 'blue'
    
    # Add the rolling correlation line
    fig.add_trace(go.Scatter(
        x=rolling_corr.index,
        y=rolling_corr.values,
        mode='lines',
        name=f'{window_size}-Day Rolling Correlation',
        line=dict(color=colors.get(asset_name, default_color), width=3),
        hovertemplate='Date: %{x}<br>Correlation: %{y:.2f}<extra></extra>'
    ))
    
    # Add shaded areas for correlation zones
    fig.add_trace(go.Scatter(
        x=rolling_corr.index,
        y=[0.7] * len(rolling_corr),
        fill=None,
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add a trace to create strong positive correlation shaded area (0.7 to 1.0)
    fig.add_trace(go.Scatter(
        x=rolling_corr.index,
        y=[1.0] * len(rolling_corr),
        fill='tonexty',  # Fill to next y value
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        fillcolor='rgba(0, 128, 0, 0.2)',
        name='Strong Positive',
        hoverinfo='skip'
    ))
    
    # Add trace for weak correlation (-0.3 to 0.3)
    fig.add_trace(go.Scatter(
        x=rolling_corr.index,
        y=[-0.3] * len(rolling_corr),
        fill=None,
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=rolling_corr.index,
        y=[0.3] * len(rolling_corr),
        fill='tonexty',  # Fill to next y value
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        fillcolor='rgba(128, 128, 128, 0.2)',
        name='Weak Correlation',
        hoverinfo='skip'
    ))
    
    # Add reference lines
    fig.add_shape(
        type="line",
        x0=rolling_corr.index[0],
        y0=0,
        x1=rolling_corr.index[-1],
        y1=0,
        line=dict(color="black", width=1, dash="dash"),
    )
    
    fig.add_shape(
        type="line",
        x0=rolling_corr.index[0],
        y0=1,
        x1=rolling_corr.index[-1],
        y1=1,
        line=dict(color="green", width=1, dash="dot"),
    )
    
    # Add annotation for overall correlation if provided
    if overall_corr is not None:
        fig.add_annotation(
            x=0.02,
            y=0.05,
            xref="paper",
            yref="paper",
            text=f"Overall Correlation: {overall_corr:.2%}",
            showarrow=False,
            font=dict(size=12),
            bgcolor="white",
            bordercolor="gray",
            borderwidth=1,
            borderpad=4,
            opacity=0.8,
        )
    
    # Set layout
    fig.update_layout(
        title=f'Rolling {window_size}-Day Correlation: {asset_name} vs QQQ',
        xaxis_title='Date',
        yaxis_title='Correlation Coefficient',
        yaxis=dict(
            range=[-0.3, 1.1],
            tickmode='linear',
            tick0=-0.3,
            dtick=0.1,
            gridcolor='rgba(200, 200, 200, 0.2)'
        ),
        plot_bgcolor='white',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        height=450
    )
    
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.2)')
    
    return fig

def create_price_comparison_plot(merged_data, asset_name):
    """
    Create a dual-axis plot comparing an asset's price with QQQ
    
    Parameters:
    -----------
    merged_data: pd.DataFrame
        DataFrame containing price data for the asset and QQQ
    asset_name: str
        Name of the asset being compared
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    # Define color scheme
    colors = {
        'YMAX': 'teal',
        'YMAG': 'green',
        'QYLD': 'red',
        'QQQI': 'purple',
        'JEPQ': 'orange',
        'QQQ': 'blue',
        'PBP': 'brown',
        'GPIQ': 'olive',
        'FEPI': 'darkblue',
        'IQQQ': 'coral',
        'FTQI': 'magenta',
        'QYLG': 'darkgreen',
        'QDTY': 'darkred',
        'VIX': '#9467bd',
        'VVIX': '#ff7f0e'
    }
    
    # Default color for assets not in the colors dictionary
    default_color = 'teal'
    
    # Create a subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add trace for the selected asset on first y-axis
    fig.add_trace(
        go.Scatter(
            x=merged_data.index,
            y=merged_data[asset_name],
            name=asset_name,
            line=dict(color=colors.get(asset_name, default_color), width=3),
            hovertemplate='Date: %{x}<br>' + asset_name + ': $%{y:.2f}<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Add trace for QQQ on second y-axis
    fig.add_trace(
        go.Scatter(
            x=merged_data.index,
            y=merged_data['QQQ'],
            name='QQQ',
            line=dict(color=colors['QQQ'], width=2),
            hovertemplate='Date: %{x}<br>QQQ: $%{y:.2f}<extra></extra>'
        ),
        secondary_y=True
    )
    
    # Set titles and axis labels
    fig.update_layout(
        title=f'{asset_name} vs QQQ Price Comparison',
        hovermode='x unified',
        plot_bgcolor='white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        height=450
    )
    
    # Set y-axes titles with matching colors
    fig.update_yaxes(
        title_text=f"{asset_name} Price ($)",
        titlefont=dict(color=colors.get(asset_name, default_color)),
        tickfont=dict(color=colors.get(asset_name, default_color)),
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(200, 200, 200, 0.2)',
        secondary_y=False
    )
    
    fig.update_layout(
        yaxis2=dict(
            title_text="QQQ Price ($)",
            titlefont=dict(color=colors['QQQ']),
            tickfont=dict(color=colors['QQQ']),
            overlaying="y",
            side="right"
        )
    )
    
    # Add gridlines
    fig.update_xaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='rgba(200, 200, 200, 0.2)'
    )
    
    # Calculate and display correlation
    corr = merged_data[asset_name].corr(merged_data['QQQ'])
    
    # Add annotation for correlation
    fig.add_annotation(
        x=0.02,
        y=0.05,
        xref="paper",
        yref="paper",
        text=f"Correlation: {corr:.2%}",
        showarrow=False,
        font=dict(size=12),
        bgcolor="white",
        bordercolor="gray",
        borderwidth=1,
        borderpad=4,
        opacity=0.8,
    )
    
    return fig

def create_rolling_beta_plot(rolling_beta, window_size, asset_name, overall_beta=None):
    """
    Create a plot of rolling beta with appropriate annotations
    
    Parameters:
    -----------
    rolling_beta: pd.Series
        Rolling beta time series
    window_size: int
        Window size used for the rolling beta
    asset_name: str
        Name of the asset being analyzed
    overall_beta: float or None
        Overall beta to display on the chart
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    # Create the rolling beta plot
    fig = go.Figure()
    
    # Add the rolling beta line
    fig.add_trace(go.Scatter(
        x=rolling_beta.index,
        y=rolling_beta.values,
        mode='lines',
        name=f'{window_size}-Day Rolling Beta',
        line=dict(color='#2ca02c', width=3),
        hovertemplate='Date: %{x}<br>Beta: %{y:.2f}<extra></extra>'
    ))
    
    # Add reference lines for beta=0 and beta=1
    fig.add_shape(
        type="line",
        x0=rolling_beta.index[0],
        y0=1,
        x1=rolling_beta.index[-1],
        y1=1,
        line=dict(color="#ff5555", width=1.5, dash="dash"),
    )
    
    fig.add_shape(
        type="line",
        x0=rolling_beta.index[0],
        y0=0,
        x1=rolling_beta.index[-1],
        y1=0,
        line=dict(color="white", width=1, dash="solid"),
    )
    
    # Add shaded area from beta line to zero
    fig.add_trace(go.Scatter(
        x=rolling_beta.index,
        y=rolling_beta.values,
        fill='tozeroy',
        fillcolor='rgba(44, 160, 44, 0.2)',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add annotations for reference lines
    fig.add_annotation(
        x=rolling_beta.index[0],
        y=1.02,
        text="β = 1",
        showarrow=False,
        font=dict(color="#ff5555", size=12, family="Arial"),
    )
    
    fig.add_annotation(
        x=rolling_beta.index[0],
        y=0.02,
        text="β = 0",
        showarrow=False,
        font=dict(color="white", size=12, family="Arial"),
    )
    
    # Add annotation for overall beta if provided
    if overall_beta is not None:
        fig.add_annotation(
            x=0.02,
            y=0.05,
            xref="paper",
            yref="paper",
            text=f"Overall Beta: {overall_beta:.2f}",
            showarrow=False,
            font=dict(size=12),
            bgcolor="white",
            bordercolor="gray",
            borderwidth=1,
            borderpad=4,
            opacity=0.8,
        )
    
    # Add note about beta interpretation
    fig.add_annotation(
        x=0.5,
        y=-0.25,
        xref="paper",
        yref="paper",
        text="Note: Beta measures the volatility of an asset relative to QQQ. β=1 means equal volatility, β<1 means lower volatility.",
        showarrow=False,
        font=dict(size=10, color="gray", style="italic"),
    )
    
    # Add horizontal grid lines at specific beta values
    for beta_val in [0.25, 0.5, 0.75]:
        fig.add_shape(
            type="line",
            x0=rolling_beta.index[0],
            y0=beta_val,
            x1=rolling_beta.index[-1],
            y1=beta_val,
            line=dict(color="rgba(150, 150, 150, 0.3)", width=1, dash="dot"),
        )
    
    # Set layout
    fig.update_layout(
        title=f'Rolling {window_size}-Day Beta: {asset_name} vs QQQ',
        xaxis_title='Date',
        yaxis_title='Beta Coefficient',
        plot_bgcolor='white',
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.1)",
            borderwidth=1
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        height=450
    )
    
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.2)')
    
    return fig

def create_correlation_volatility_plot(rolling_corr, vol_data, vol_index, window_size, asset_name, overall_corr=None):
    """
    Create a dual-axis plot showing rolling correlation and volatility index values
    
    Parameters:
    -----------
    rolling_corr: pd.Series
        Rolling correlation time series
    vol_data: pd.Series
        Volatility index (VIX or VVIX) data
    vol_index: str
        Name of the volatility index ('VIX' or 'VVIX')
    window_size: int
        Window size used for the rolling correlation
    asset_name: str
        Name of the asset being analyzed
    overall_corr: float or None
        Overall correlation to display on the chart
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    # Create a subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Color mapping based on the original notebook
    colors = {
        'YMAX': 'teal',
        'YMAG': 'green',
        'QYLD': 'red',
        'QQQI': 'purple',
        'JEPQ': 'orange',
        'QQQ': 'blue',
        'PBP': 'brown',
        'GPIQ': 'olive',
        'FEPI': 'darkblue',
        'IQQQ': 'coral',
        'FTQI': 'magenta',
        'QYLG': 'darkgreen',
        'QDTY': 'darkred',
        'VIX': '#9467bd',  # purple
        'VVIX': '#ff7f0e'  # orange
    }
    
    # Default color for assets not in the colors dictionary
    default_color = 'blue'
    
    # Select color for volatility index
    vol_color = colors.get(vol_index, '#ff7f0e')  # Default to orange for unknown volatility index
    
    # Align volatility data with correlation data date range
    # Filter volatility data to only include the dates in rolling correlation
    corr_start_date = rolling_corr.index.min()
    corr_end_date = rolling_corr.index.max()
    
    # Filter vol_data to match the correlation date range
    aligned_vol_data = vol_data[vol_data.index >= corr_start_date]
    aligned_vol_data = aligned_vol_data[aligned_vol_data.index <= corr_end_date]
    
    # Determine dynamic y-axis range based on correlation values
    min_corr = rolling_corr.min()
    max_corr = rolling_corr.max()
    
    # Add some padding to the range
    y_min = min(min_corr - 0.1, -0.3) # Ensure we include at least -0.3
    y_max = max(max_corr + 0.1, 0.3)  # Ensure we include at least 0.3
    
    # Round to nearest 0.1 to make axis labels nice
    y_min = np.floor(y_min * 10) / 10
    y_max = np.ceil(y_max * 10) / 10
    
    # Add the rolling correlation line on the primary y-axis
    fig.add_trace(
        go.Scatter(
            x=rolling_corr.index,
            y=rolling_corr.values,
            mode='lines',
            name=f'{window_size}-Day Rolling Correlation',
            line=dict(color=colors.get(asset_name, default_color), width=3),
            hovertemplate='Date: %{x}<br>Correlation: %{y:.2f}<extra></extra>'
        )
    )
    
    # Add the volatility index line on the secondary y-axis (using aligned data)
    fig.add_trace(
        go.Scatter(
            x=aligned_vol_data.index,
            y=aligned_vol_data.values,
            mode='lines',
            name=vol_index,
            line=dict(color=vol_color, width=2),
            hovertemplate='Date: %{x}<br>' + vol_index + ': %{y:.2f}<extra></extra>',
            yaxis="y2"
        )
    )
    
    # Create shaded areas for correlation zones
    # Strong negative correlation (-1.0 to -0.7)
    if y_min <= -0.7:
        fig.add_trace(
            go.Scatter(
                x=rolling_corr.index,
                y=[-1.0] * len(rolling_corr),
                fill=None,
                mode='lines',
                line=dict(color='rgba(0,0,0,0)'),
                showlegend=False,
                hoverinfo='skip'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=rolling_corr.index,
                y=[-0.7] * len(rolling_corr),
                fill='tonexty',  # Fill to next y value
                mode='lines',
                line=dict(color='rgba(0,0,0,0)'),
                fillcolor='rgba(0, 128, 0, 0.2)',
                name='Strong Negative',
                hoverinfo='skip'
            )
        )
    
    # Weak correlation (-0.3 to 0.3)
    fig.add_trace(
        go.Scatter(
            x=rolling_corr.index,
            y=[-0.3] * len(rolling_corr),
            fill=None,
            mode='lines',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False,
            hoverinfo='skip'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=rolling_corr.index,
            y=[0.3] * len(rolling_corr),
            fill='tonexty',  # Fill to next y value
            mode='lines',
            line=dict(color='rgba(0,0,0,0)'),
            fillcolor='rgba(128, 128, 128, 0.2)',
            name='Weak Correlation',
            hoverinfo='skip'
        )
    )
    
    # Strong positive correlation (0.7 to 1.0)
    if y_max >= 0.7:
        fig.add_trace(
            go.Scatter(
                x=rolling_corr.index,
                y=[0.7] * len(rolling_corr),
                fill=None,
                mode='lines',
                line=dict(color='rgba(0,0,0,0)'),
                showlegend=False,
                hoverinfo='skip'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=rolling_corr.index,
                y=[1.0] * len(rolling_corr),
                fill='tonexty',  # Fill to next y value
                mode='lines',
                line=dict(color='rgba(0,0,0,0)'),
                fillcolor='rgba(0, 128, 0, 0.2)',
                name='Strong Positive',
                hoverinfo='skip'
            )
        )
    
    # Add reference lines for correlation (primary y-axis)
    fig.add_shape(
        type="line",
        x0=rolling_corr.index[0],
        y0=0,
        x1=rolling_corr.index[-1],
        y1=0,
        line=dict(color="black", width=1, dash="dash"),
        yref="y"
    )
    
    # Add reference lines at -0.7, -0.3, 0.3, and 0.7 if they're in range
    if y_min <= -0.7 <= y_max:
        fig.add_shape(
            type="line",
            x0=rolling_corr.index[0],
            y0=-0.7,
            x1=rolling_corr.index[-1],
            y1=-0.7,
            line=dict(color="green", width=1, dash="dot"),
            yref="y"
        )
    
    if y_min <= -0.3 <= y_max:
        fig.add_shape(
            type="line",
            x0=rolling_corr.index[0],
            y0=-0.3,
            x1=rolling_corr.index[-1],
            y1=-0.3,
            line=dict(color="gray", width=1, dash="dot"),
            yref="y"
        )
    
    if y_min <= 0.3 <= y_max:
        fig.add_shape(
            type="line",
            x0=rolling_corr.index[0],
            y0=0.3,
            x1=rolling_corr.index[-1],
            y1=0.3,
            line=dict(color="gray", width=1, dash="dot"),
            yref="y"
        )
    
    if y_min <= 0.7 <= y_max:
        fig.add_shape(
            type="line",
            x0=rolling_corr.index[0],
            y0=0.7,
            x1=rolling_corr.index[-1],
            y1=0.7,
            line=dict(color="green", width=1, dash="dot"),
            yref="y"
        )
    
    # Add annotation for overall correlation if provided
    if overall_corr is not None:
        fig.add_annotation(
            x=0.02,
            y=0.05,
            xref="paper",
            yref="paper",
            text=f"Overall Correlation: {overall_corr:.2%}",
            showarrow=False,
            font=dict(size=12),
            bgcolor="white",
            bordercolor="gray",
            borderwidth=1,
            borderpad=4,
            opacity=0.8,
        )
    
    # Calculate and display correlation between asset returns and volatility
    if len(rolling_corr) > 0 and len(aligned_vol_data) > 0:
        # Align the data
        aligned_data = pd.DataFrame({
            'corr': rolling_corr,
            'vol': aligned_vol_data
        }).dropna()
        
        if not aligned_data.empty:
            vol_corr = aligned_data['corr'].corr(aligned_data['vol'])
            
            # Add annotation for volatility correlation
            fig.add_annotation(
                x=0.98,
                y=0.05,
                xref="paper",
                yref="paper",
                text=f"Corr-{vol_index} Correlation: {vol_corr:.2f}",
                showarrow=False,
                font=dict(size=12),
                bgcolor="white",
                bordercolor="gray",
                borderwidth=1,
                borderpad=4,
                opacity=0.8,
                align="right",
                xanchor="right"
            )
    
    # Set layout with appropriate titles and colors
    fig.update_layout(
        title=f'Rolling {window_size}-Day Correlation: {asset_name} vs {vol_index}',
        xaxis_title='Date',
        hovermode='x unified',
        plot_bgcolor='white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        height=450,
        # Configure the secondary y-axis (volatility)
        yaxis2=dict(
            title_text=f"{vol_index} Value",
            titlefont=dict(color=vol_color),
            tickfont=dict(color=vol_color),
            overlaying="y",
            side="right"
        )
    )
    
    # Set primary y-axis (correlation) with dynamic range
    fig.update_yaxes(
        title_text="Correlation Coefficient",
        range=[y_min, y_max],
        tickmode='linear',
        tick0=-1.0,
        dtick=0.1,
        gridcolor='rgba(200, 200, 200, 0.2)',
        titlefont=dict(color=colors.get(asset_name, default_color)),
        tickfont=dict(color=colors.get(asset_name, default_color))
    )
    
    return fig

def create_asset_volatility_comparison_plot(merged_data, asset_name, vol_index):
    """
    Create a dual-axis plot comparing an asset's price with a volatility index
    
    Parameters:
    -----------
    merged_data: pd.DataFrame
        DataFrame containing price data for the asset and volatility index
    asset_name: str
        Name of the asset being compared
    vol_index: str
        Name of the volatility index ('VIX' or 'VVIX')
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    # Define color scheme
    colors = {
        'YMAX': 'teal',
        'YMAG': 'green',
        'QYLD': 'red',
        'QQQI': 'purple',
        'JEPQ': 'orange',
        'QQQ': 'blue',
        'PBP': 'brown',
        'GPIQ': 'olive',
        'FEPI': 'darkblue',
        'IQQQ': 'coral',
        'FTQI': 'magenta',
        'QYLG': 'darkgreen',
        'QDTY': 'darkred',
        'VIX': '#9467bd',  # purple
        'VVIX': '#ff7f0e'  # orange
    }
    
    # Default color for assets not in the colors dictionary
    default_color = 'teal'
    
    # Select color for volatility index
    vol_color = colors.get(vol_index, '#ff7f0e')  # Default to orange for unknown volatility index
    
    # Create a subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add trace for the selected asset on first y-axis
    fig.add_trace(
        go.Scatter(
            x=merged_data.index,
            y=merged_data[asset_name],
            name=asset_name,
            line=dict(color=colors.get(asset_name, default_color), width=3),
            hovertemplate='Date: %{x}<br>' + asset_name + ': $%{y:.2f}<extra></extra>'
        )
    )
    
    # Add trace for volatility index on second y-axis
    fig.add_trace(
        go.Scatter(
            x=merged_data.index,
            y=merged_data[vol_index],
            name=vol_index,
            line=dict(color=vol_color, width=2),
            hovertemplate='Date: %{x}<br>' + vol_index + ': %{y:.2f}<extra></extra>',
            yaxis="y2"
        )
    )
    
    # Set titles and axis labels
    fig.update_layout(
        title=f'Asset vs Volatility Index Comparison: {asset_name} and {vol_index}',
        hovermode='x unified',
        plot_bgcolor='white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        height=450,
        # Configure the secondary y-axis (volatility)
        yaxis2=dict(
            title_text=f"{vol_index} Value",
            titlefont=dict(color=vol_color),
            tickfont=dict(color=vol_color),
            overlaying="y",
            side="right"
        )
    )
    
    # Set y-axes titles with matching colors
    fig.update_yaxes(
        title_text=f"{asset_name} Price ($)",
        titlefont=dict(color=colors.get(asset_name, default_color)),
        tickfont=dict(color=colors.get(asset_name, default_color)),
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(200, 200, 200, 0.2)'
    )
    
    # Add gridlines
    fig.update_xaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='rgba(200, 200, 200, 0.2)'
    )
    
    # Calculate and display correlation
    corr = merged_data[asset_name].corr(merged_data[vol_index])
    
    # Add annotation for correlation
    fig.add_annotation(
        x=0.02,
        y=0.05,
        xref="paper",
        yref="paper",
        text=f"Correlation: {corr:.2f}",
        showarrow=False,
        font=dict(size=12),
        bgcolor="white",
        bordercolor="gray",
        borderwidth=1,
        borderpad=4,
        opacity=0.8,
    )
    
    return fig 