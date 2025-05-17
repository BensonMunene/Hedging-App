import plotly.graph_objects as go
import pandas as pd
import numpy as np

def create_asset_pnl_chart(strategy_data, asset_name):
    """
    Create a chart of asset cumulative PnL
    
    Parameters:
    -----------
    strategy_data: pd.DataFrame
        DataFrame with strategy results
    asset_name: str
        Name of the asset
    
    Returns:
    --------
    go.Figure
        Plotly figure with the chart
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=strategy_data.index, 
        y=strategy_data[f'{asset_name}_Cum_PnL'],
        mode='lines',
        line=dict(color='green', width=2),
        name=f'{asset_name} Cumulative PnL'
    ))
    
    # Add horizontal line at y=0
    fig.add_shape(
        type="line",
        x0=strategy_data.index[0],
        x1=strategy_data.index[-1],
        y0=0,
        y1=0,
        line=dict(color="black", dash="dash")
    )
    
    fig.update_layout(
        title=f'{asset_name} Cumulative PnL',
        xaxis_title='Date',
        yaxis_title='Profit/Loss ($)',
        plot_bgcolor='white',
        height=300,
        margin=dict(l=40, r=40, t=40, b=30)
    )
    
    fig.update_yaxes(gridcolor='rgba(200, 200, 200, 0.2)')
    
    return fig

def create_qqq_pnl_chart(strategy_data):
    """
    Create a chart of QQQ cumulative PnL
    
    Parameters:
    -----------
    strategy_data: pd.DataFrame
        DataFrame with strategy results
    
    Returns:
    --------
    go.Figure
        Plotly figure with the chart
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=strategy_data.index, 
        y=strategy_data['QQQ_Cum_PnL'],
        mode='lines',
        line=dict(color='red', width=2),
        name='QQQ Cumulative PnL'
    ))
    
    # Add horizontal line at y=0
    fig.add_shape(
        type="line",
        x0=strategy_data.index[0],
        x1=strategy_data.index[-1],
        y0=0,
        y1=0,
        line=dict(color="black", dash="dash")
    )
    
    fig.update_layout(
        title='QQQ Cumulative PnL',
        xaxis_title='Date',
        yaxis_title='Profit/Loss ($)',
        plot_bgcolor='white',
        height=300,
        margin=dict(l=40, r=40, t=40, b=30)
    )
    
    fig.update_yaxes(gridcolor='rgba(200, 200, 200, 0.2)')
    
    return fig

def create_total_pnl_chart(strategy_data):
    """
    Create a chart of total cumulative PnL
    
    Parameters:
    -----------
    strategy_data: pd.DataFrame
        DataFrame with strategy results
    
    Returns:
    --------
    go.Figure
        Plotly figure with the chart
    """
    fig = go.Figure()
    
    # Add total PnL line
    fig.add_trace(go.Scatter(
        x=strategy_data.index, 
        y=strategy_data['Total_Cum_PnL'],
        mode='lines',
        line=dict(color='blue', width=2),
        name='Total Cumulative PnL'
    ))
    
    # Add horizontal line at y=0
    fig.add_shape(
        type="line",
        x0=strategy_data.index[0],
        x1=strategy_data.index[-1],
        y0=0,
        y1=0,
        line=dict(color="black", dash="dash")
    )
    
    fig.update_layout(
        title='Total Cumulative PnL',
        xaxis_title='Date',
        yaxis_title='Profit/Loss ($)',
        plot_bgcolor='white',
        height=300,
        margin=dict(l=40, r=40, t=40, b=30)
    )
    
    fig.update_yaxes(gridcolor='rgba(200, 200, 200, 0.2)')
    
    return fig

def create_correlation_entry_exit_chart(strategy_data, entry_threshold=None, exit_threshold=None, window=30):
    """
    Create a chart showing rolling correlation with entry/exit points
    
    Parameters:
    -----------
    strategy_data: pd.DataFrame
        DataFrame with strategy results
    entry_threshold: float or None
        Entry correlation threshold (for Strategy 1), or Lower bound (for Strategy 3)
    exit_threshold: float or None
        Exit correlation threshold (for Strategy 1), or Upper bound (for Strategy 3)
    window: int
        Rolling window size
    
    Returns:
    --------
    go.Figure
        Plotly figure with the chart
    """
    fig = go.Figure()
    
    # Add correlation line
    fig.add_trace(go.Scatter(
        x=strategy_data.index, 
        y=strategy_data['rolling_corr'],
        mode='lines',
        name=f'{window}d Rolling Correlation',
        line=dict(color='blue', width=2)
    ))
    
    # Determine if we're visualizing Strategy 1 or Strategy 3 based on the thresholds
    # Strategy 1: entry_corr is lower than exit_corr (enter when correlation drops below, exit when rises above)
    # Strategy 3: both thresholds define a range (enter when within range, exit when outside range)
    is_strategy_3 = (entry_threshold is not None and exit_threshold is not None and 
                    entry_threshold <= exit_threshold)
    
    # Add threshold lines with appropriate styling
    if is_strategy_3:
        # Strategy 3: Add lower bound
        fig.add_shape(
            type="line",
            x0=strategy_data.index[0],
            x1=strategy_data.index[-1],
            y0=entry_threshold,
            y1=entry_threshold,
            line=dict(color="green", dash="dash", width=1.5)
        )
        
        # Strategy 3: Add upper bound
        fig.add_shape(
            type="line",
            x0=strategy_data.index[0],
            x1=strategy_data.index[-1],
            y0=exit_threshold,
            y1=exit_threshold,
            line=dict(color="red", dash="dash", width=1.5)
        )
        
        # Add shaded area between lower and upper bounds
        fig.add_trace(go.Scatter(
            x=strategy_data.index,
            y=[entry_threshold] * len(strategy_data),
            fill=None,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=strategy_data.index,
            y=[exit_threshold] * len(strategy_data),
            fill='tonexty',
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(0, 128, 0, 0.15)',
            name='Trading Zone',
            hoverinfo='skip'
        ))
        
    else:
        # Strategy 1: Add entry correlation threshold
        if entry_threshold is not None:
            fig.add_shape(
                type="line",
                x0=strategy_data.index[0],
                x1=strategy_data.index[-1],
                y0=entry_threshold,
                y1=entry_threshold,
                line=dict(color="green", dash="dash")
            )
        
        # Strategy 1: Add exit correlation threshold
        if exit_threshold is not None:
            fig.add_shape(
                type="line",
                x0=strategy_data.index[0],
                x1=strategy_data.index[-1],
                y0=exit_threshold,
                y1=exit_threshold,
                line=dict(color="red", dash="dash")
            )
    
    # Mark entry and exit points
    entries = strategy_data[strategy_data['entry_exit'] == 'Entry']
    exits = strategy_data[strategy_data['entry_exit'] == 'Exit']
    
    if not entries.empty:
        fig.add_trace(go.Scatter(
            x=entries.index,
            y=entries['rolling_corr'],
            mode='markers',
            marker=dict(color='green', size=10, symbol='triangle-up'),
            name='Entry Points'
        ))
    
    if not exits.empty:
        fig.add_trace(go.Scatter(
            x=exits.index,
            y=exits['rolling_corr'],
            mode='markers',
            marker=dict(color='red', size=10, symbol='triangle-down'),
            name='Exit Points'
        ))
    
    # Add labels for the thresholds
    if is_strategy_3:
        # Strategy 3: Label the lower and upper bounds
        fig.add_annotation(
            x=strategy_data.index[0],
            y=entry_threshold,
            text=f"Lower: {entry_threshold:.2f}",
            xanchor="left",
            yanchor="bottom",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="green",
            borderwidth=1,
            borderpad=3,
            font=dict(color="green")
        )
        
        fig.add_annotation(
            x=strategy_data.index[0],
            y=exit_threshold,
            text=f"Upper: {exit_threshold:.2f}",
            xanchor="left",
            yanchor="top",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="red",
            borderwidth=1,
            borderpad=3,
            font=dict(color="red")
        )
        
        title_text = 'Correlation Trading Zone & Entry/Exit Points'
        
    else:
        # Strategy 1: Use original labeling
        title_text = 'Rolling Correlation & Entry/Exit Points'
    
    fig.update_layout(
        title=title_text,
        xaxis_title='Date',
        yaxis_title='Correlation',
        plot_bgcolor='white',
        yaxis=dict(
            range=[0, 1],
            gridcolor='rgba(200, 200, 200, 0.2)'
        ),
        height=300,
        margin=dict(l=40, r=40, t=40, b=30)
    )
    
    return fig

def create_volatility_entry_exit_chart(strategy_data, use_vix=True, use_vvix=False, 
                                      vix_lower=None, vix_upper=None, 
                                      vvix_lower=None, vvix_upper=None):
    """
    Create a chart showing volatility with entry/exit points for Strategy 2
    
    Parameters:
    -----------
    strategy_data: pd.DataFrame
        DataFrame with strategy results
    use_vix: bool
        Whether VIX data is being used
    use_vvix: bool
        Whether VVIX data is being used
    vix_lower: float or None
        Lower VIX threshold
    vix_upper: float or None
        Upper VIX threshold
    vvix_lower: float or None
        Lower VVIX threshold
    vvix_upper: float or None
        Upper VVIX threshold
        
    Returns:
    --------
    go.Figure
        Plotly figure with the chart
    """
    # Create volatility plot
    fig = go.Figure()
    
    # Add VIX line if used
    if use_vix and 'VIX' in strategy_data.columns:
        fig.add_trace(go.Scatter(
            x=strategy_data.index, 
            y=strategy_data['VIX'],
            mode='lines',
            name='VIX',
            line=dict(color='purple', width=2)
        ))
        
        # Add upper and lower VIX bounds if provided
        if vix_lower is not None:
            fig.add_shape(
                type="line",
                x0=strategy_data.index[0],
                x1=strategy_data.index[-1],
                y0=vix_lower,
                y1=vix_lower,
                line=dict(color="green", dash="dash")
            )
        
        if vix_upper is not None:
            fig.add_shape(
                type="line",
                x0=strategy_data.index[0],
                x1=strategy_data.index[-1],
                y0=vix_upper,
                y1=vix_upper,
                line=dict(color="red", dash="dash")
            )
    
    # Add VVIX line if used
    if use_vvix and 'VVIX' in strategy_data.columns:
        # Create a secondary y-axis for VVIX
        fig.add_trace(go.Scatter(
            x=strategy_data.index, 
            y=strategy_data['VVIX'],
            mode='lines',
            name='VVIX',
            line=dict(color='orange', width=2),
            yaxis="y2"
        ))
        
        # Add VVIX bounds on secondary y-axis if provided
        if vvix_lower is not None:
            fig.add_shape(
                type="line",
                x0=strategy_data.index[0],
                x1=strategy_data.index[-1],
                y0=vvix_lower,
                y1=vvix_lower,
                line=dict(color="green", dash="dot"),
                yref="y2"
            )
        
        if vvix_upper is not None:
            fig.add_shape(
                type="line",
                x0=strategy_data.index[0],
                x1=strategy_data.index[-1],
                y0=vvix_upper,
                y1=vvix_upper,
                line=dict(color="red", dash="dot"),
                yref="y2"
            )
    
    # Mark entry and exit points
    entries = strategy_data[strategy_data['entry_exit'] == 'Entry']
    exits = strategy_data[strategy_data['entry_exit'] == 'Exit']
    
    if not entries.empty and use_vix and 'VIX' in strategy_data.columns:
        fig.add_trace(go.Scatter(
            x=entries.index,
            y=entries['VIX'],
            mode='markers',
            marker=dict(color='green', size=10, symbol='triangle-up'),
            name='Entry Points'
        ))
    
    if not exits.empty and use_vix and 'VIX' in strategy_data.columns:
        fig.add_trace(go.Scatter(
            x=exits.index,
            y=exits['VIX'],
            mode='markers',
            marker=dict(color='red', size=10, symbol='triangle-down'),
            name='Exit Points'
        ))
    
    # Set up layout with secondary y-axis if needed
    layout_settings = {
        'title': 'Volatility with Entry/Exit Points',
        'xaxis_title': 'Date',
        'plot_bgcolor': 'white',
        'height': 300,
        'margin': dict(l=40, r=40, t=40, b=30)
    }
    
    if use_vix and not use_vvix:
        layout_settings['yaxis_title'] = 'VIX Value'
    elif use_vvix and not use_vix:
        layout_settings['yaxis_title'] = 'VVIX Value'
    elif use_vix and use_vvix:
        layout_settings['yaxis_title'] = 'VIX Value'
        layout_settings['yaxis2'] = {
            'title': 'VVIX Value',
            'overlaying': 'y',
            'side': 'right'
        }
    
    fig.update_layout(**layout_settings)
    fig.update_yaxes(gridcolor='rgba(200, 200, 200, 0.2)')
    
    return fig

def create_dividend_chart(strategy_data):
    """
    Create a bar chart showing dividend payments
    
    Parameters:
    -----------
    strategy_data: pd.DataFrame
        DataFrame with strategy results
    
    Returns:
    --------
    go.Figure or None
        Plotly figure with the chart, or None if no dividend payments
    """
    # Get dividend payment dates and amounts
    dividend_payments = strategy_data[
        (strategy_data['in_position']) & 
        (strategy_data['dividend_payment'] > 0)
    ].copy()
    
    if dividend_payments.empty:
        return None
    
    # Convert dates to string format for x-axis
    dividend_payments['date_str'] = dividend_payments.index.strftime('%b %d<br>%Y')
    
    # Create a bar chart for each dividend payment
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dividend_payments['date_str'],
        y=dividend_payments['dividend_payment'],
        marker_color='lightblue',
        text=[f"${x:.2f}" for x in dividend_payments['dividend_payment']],
        textposition='outside',
        hovertemplate='<b>Date</b>: %{customdata}<br><b>Dividend Paid</b>: $%{y:.2f}<extra></extra>',
        customdata=dividend_payments.index.strftime('%b %d, %Y')
    ))
    
    fig.update_layout(
        title='Dividend Payments by Date',
        xaxis_title='Payment Date',
        yaxis_title='Amount ($)',
        plot_bgcolor='white',
        height=300,
        margin=dict(l=40, r=40, t=40, b=50),
        showlegend=False,
        xaxis=dict(
            tickangle=45,
            tickfont=dict(size=10)
        ),
        # Ensure sufficient space for the text above bars
        yaxis=dict(
            range=[0, max(dividend_payments['dividend_payment']) * 1.15],
            gridcolor='rgba(200, 200, 200, 0.2)'
        )
    )
    
    # Add horizontal gridlines
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.2)')
    
    return fig

def create_daily_pnl_chart(strategy_data):
    """
    Create a chart showing daily PnL
    
    Parameters:
    -----------
    strategy_data: pd.DataFrame
        DataFrame with strategy results
    
    Returns:
    --------
    go.Figure
        Plotly figure with the chart
    """
    fig = go.Figure()
    
    # Add daily PnL line
    fig.add_trace(go.Scatter(
        x=strategy_data.index,
        y=strategy_data['total_pnl_daily'],
        mode='lines',
        line=dict(color='purple', width=2),
        name='Daily PnL'
    ))
    
    # Add horizontal line at y=0
    fig.add_shape(
        type="line",
        x0=strategy_data.index[0],
        x1=strategy_data.index[-1],
        y0=0,
        y1=0,
        line=dict(color="black", dash="dash")
    )
    
    fig.update_layout(
        title='Daily Total PnL',
        xaxis_title='Date',
        yaxis_title='Profit/Loss ($)',
        plot_bgcolor='white',
        height=300,
        margin=dict(l=40, r=40, t=40, b=30)
    )
    
    fig.update_yaxes(gridcolor='rgba(200, 200, 200, 0.2)')
    
    return fig

def create_metrics_dataframe(metrics, days_in_market, dividend_amount, dividend_days, final_pnl, annualized_percentage):
    """
    Create a DataFrame with strategy metrics for display
    
    Parameters:
    -----------
    metrics: dict
        Dictionary of strategy metrics
    days_in_market: int
        Number of days in the market
    dividend_amount: float
        Total dividends received
    dividend_days: int
        Number of days with dividend payments
    final_pnl: float
        Final cumulative PnL
    annualized_percentage: float
        Annualized return as percentage
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with metrics for display
    """
    metrics_df = pd.DataFrame({
        'Metric': ['Number of days in market', 
                  'Amount of Dividends paid', 
                  'Days with dividend payments',
                  'Final cumulative PnL',
                  'Annualized Return (%)'],
        'Value': [days_in_market,
                 f"${dividend_amount:.2f}",
                 dividend_days,
                 f"${final_pnl:.2f}",
                 f"{annualized_percentage:.2f}%"]
    })
    
    return metrics_df 