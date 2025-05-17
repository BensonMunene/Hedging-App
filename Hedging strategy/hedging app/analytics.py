import pandas as pd
import numpy as np

def calculate_correlation_data(assets, data, qqq_returns):
    """
    Calculate correlation between each asset's returns and QQQ returns
    
    Parameters:
    -----------
    assets: list
        List of asset names
    data: pd.DataFrame
        DataFrame containing price data for all assets
    qqq_returns: pd.Series
        Return series for QQQ
        
    Returns:
    --------
    tuple
        (correlation_df, date_ranges)
    """
    correlations = {}
    date_ranges = {}
    
    # Calculate returns for each asset
    returns_dict = {}
    for asset in assets:
        returns_dict[asset] = data[asset].pct_change().dropna()
    
    for asset_name in assets:
        if asset_name != "QQQ" and asset_name != "VIX" and asset_name != "VVIX":  # Skip QQQ, VIX, and VVIX
            # Align the asset returns with QQQ returns
            asset_returns = returns_dict[asset_name]
            aligned_returns = pd.concat([asset_returns, qqq_returns], axis=1).dropna()
            
            if not aligned_returns.empty:
                # Calculate correlation if there are overlapping dates
                correlations[asset_name] = aligned_returns.iloc[:, 0].corr(aligned_returns["QQQ"])
                # Store the start and end dates of the aligned data
                date_ranges[asset_name] = {
                    'start': aligned_returns.index.min().strftime('%Y-%m-%d'),
                    'end': aligned_returns.index.max().strftime('%Y-%m-%d')
                }
            else:
                correlations[asset_name] = np.nan
                date_ranges[asset_name] = {'start': 'N/A', 'end': 'N/A'}
    
    # Add QQQ's self-correlation
    correlations['QQQ'] = 1.0
    date_ranges['QQQ'] = {
        'start': qqq_returns.index.min().strftime('%Y-%m-%d'),
        'end': qqq_returns.index.max().strftime('%Y-%m-%d')
    }
    
    # Convert to Series for visualization
    correlations_series = pd.Series(correlations)
    
    # Create a DataFrame with correlation values and date ranges
    correlation_df = pd.DataFrame({
        'Correlation': correlations_series,
        'Start Date': [date_ranges[asset]['start'] for asset in correlations_series.index],
        'End Date': [date_ranges[asset]['end'] for asset in correlations_series.index]
    })
    
    return correlation_df, date_ranges

def calculate_rolling_correlation(asset_returns, qqq_returns, window=30):
    """
    Calculate rolling correlation between asset returns and QQQ returns
    
    Parameters:
    -----------
    asset_returns: pd.Series
        Return series for the asset
    qqq_returns: pd.Series
        Return series for QQQ
    window: int
        Rolling window size in days
        
    Returns:
    --------
    pd.Series
        Rolling correlation series
    """
    # Align data
    aligned_returns = pd.concat([asset_returns, qqq_returns], axis=1).dropna()
    
    if len(aligned_returns) <= window:
        return None
    
    # Rename columns for clarity
    aligned_returns.columns = ['Asset', 'QQQ']
    
    # Calculate rolling correlation
    rolling_corr = aligned_returns['Asset'].rolling(window=window).corr(aligned_returns['QQQ'])
    
    return rolling_corr

def calculate_rolling_beta(asset_returns, qqq_returns, window=30):
    """
    Calculate rolling beta between asset returns and QQQ returns
    
    Parameters:
    -----------
    asset_returns: pd.Series
        Return series for the asset
    qqq_returns: pd.Series
        Return series for QQQ
    window: int
        Rolling window size in days
        
    Returns:
    --------
    pd.Series
        Rolling beta series
    """
    # Align data
    aligned_returns = pd.concat([asset_returns, qqq_returns], axis=1).dropna()
    
    if len(aligned_returns) <= window:
        return None
    
    # Rename columns for clarity
    aligned_returns.columns = ['Asset', 'QQQ']
    
    # Calculate rolling beta using covariance and variance
    rolling_beta = pd.Series(index=aligned_returns.index)
    
    for i in range(window, len(aligned_returns)):
        window_data = aligned_returns.iloc[i-window:i]
        cov = window_data['Asset'].cov(window_data['QQQ'])
        var = window_data['QQQ'].var()
        
        # Avoid division by zero
        if var != 0:
            beta = cov / var
        else:
            beta = np.nan
            
        rolling_beta.iloc[i-1] = beta
    
    return rolling_beta

def get_active_trading_days(strategy_data):
    """
    Extract days in market, trading periods, and dividend information from strategy data
    
    Parameters:
    -----------
    strategy_data: pd.DataFrame
        DataFrame containing strategy backtest results
        
    Returns:
    --------
    dict
        Dictionary with trading metrics
    """
    if strategy_data is None or strategy_data.empty:
        return {
            "days_in_market": 0,
            "trades": 0,
            "dividend_days": 0,
            "dividend_amount": 0.0
        }
    
    days_in_market = strategy_data['in_position'].sum()
    
    # Find entries and exits
    entries = strategy_data[strategy_data['entry_exit'] == 'Entry']
    exits = strategy_data[strategy_data['entry_exit'] == 'Exit']
    
    # Count total trades
    total_trades = len(entries)
    
    # Calculate dividend information
    if 'dividend_payment' in strategy_data.columns:
        dividend_payments = strategy_data.loc[strategy_data['in_position'], 'dividend_payment']
        total_dividend_amount = dividend_payments.sum()
        dividend_days = (dividend_payments > 0).sum()
    else:
        total_dividend_amount = 0.0
        dividend_days = 0
    
    return {
        "days_in_market": int(days_in_market),
        "trades": total_trades,
        "dividend_days": int(dividend_days),
        "dividend_amount": float(total_dividend_amount)
    }

def calculate_performance_metrics(strategy_data, include_dividends=False):
    """
    Calculate performance metrics for a strategy
    
    Parameters:
    -----------
    strategy_data: pd.DataFrame
        DataFrame containing strategy backtest results
    include_dividends: bool
        Whether to include dividends in the calculations
        
    Returns:
    --------
    dict
        Dictionary with performance metrics
    """
    if strategy_data is None or strategy_data.empty:
        return {
            "final_pnl": "$0.00",
            "win_rate": "0.00%",
            "annualized_return": "0.00%",
            "max_drawdown": "0.00%"
        }
    
    # Get essential metrics
    trading_days = get_active_trading_days(strategy_data)
    days_in_market = trading_days["days_in_market"]
    total_dividend_amount = trading_days["dividend_amount"]
    
    # Calculate final PnL
    if 'Total_Cum_PnL' in strategy_data.columns:
        final_pnl = strategy_data['Total_Cum_PnL'].iloc[-1]
    else:
        final_pnl = 0.0
    
    # Calculate win rate using entry/exit pairs
    entries = strategy_data[strategy_data['entry_exit'] == 'Entry']
    exits = strategy_data[strategy_data['entry_exit'] == 'Exit']
    
    if not entries.empty and not exits.empty:
        # For each entry-exit pair, calculate if the trade was profitable
        trade_results = []
        for i in range(min(len(entries), len(exits))):
            entry_date = entries.index[i]
            exit_date = exits.index[i]
            
            # Find all dates between entry and exit
            trade_dates = strategy_data.loc[entry_date:exit_date].index
            
            # Sum daily PnL for this trade period
            trade_pnl = strategy_data.loc[trade_dates, 'total_pnl_daily'].sum()
            trade_results.append(trade_pnl > 0)
        
        winning_trades = sum(trade_results)
        win_rate = winning_trades / len(trade_results) if trade_results else 0
    else:
        win_rate = 0
    
    # Calculate annualized return
    if days_in_market > 0:
        total_return = final_pnl
        if include_dividends:
            total_return += total_dividend_amount
        
        # Assume 250 trading days per year
        annualized_return = (total_return * 250 / days_in_market) / 100
    else:
        annualized_return = 0.0
    
    # Calculate maximum drawdown
    if 'Total_Cum_PnL' in strategy_data.columns:
        cum_pnl = strategy_data['Total_Cum_PnL']
        running_max = cum_pnl.cummax()
        drawdown = (cum_pnl - running_max) / 100  # As percentage
        max_drawdown = drawdown.min() if not drawdown.empty else 0.0
    else:
        max_drawdown = 0.0
    
    return {
        "final_pnl": f"${final_pnl:.2f}",
        "win_rate": f"{win_rate:.2%}",
        "annualized_return": f"{annualized_return:.2%}",
        "max_drawdown": f"{max_drawdown:.2%}"
    } 