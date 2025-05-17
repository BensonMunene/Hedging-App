import pandas as pd
import numpy as np

# Function to implement Strategy 1
def run_strategy_1(asset_data, qqq_data, start_date=None, end_date=None, window=30, entry_corr=0.85, exit_corr=0.95, 
                 asset_amount=28000, qqq_amount=10000, asset_name="Asset", 
                 include_dividends=False, asset_dividends=None, qqq_dividends=None):
    """
    Implements a correlation-based pairs trading strategy between a selected asset and QQQ.
    - Enters long asset/short QQQ position when correlation drops to entry_corr level
    - Exits when correlation rises above exit_corr level
    
    Parameters:
    -----------
    asset_data: pd.Series
        Price data for the selected asset
    qqq_data: pd.Series
        Price data for QQQ
    start_date, end_date: datetime or None
        Date range for the backtest (if None, uses all available data)
    window: int
        Rolling window in days for correlation calculation
    entry_corr: float
        Enter position when correlation reaches this level
    exit_corr: float
        Exit position when correlation reaches this level
    asset_amount: float
        Dollar amount to invest in the asset
    qqq_amount: float
        Dollar amount to short in QQQ
    asset_name: str
        Name of the selected asset for column headers
    include_dividends: bool
        Whether to include dividend payments in the calculations
    asset_dividends: pd.Series or None
        Dividend data for the selected asset
    qqq_dividends: pd.Series or None
        Dividend data for QQQ
    """
    # Merge the data
    strategy_data = pd.DataFrame({
        asset_name: asset_data,
        'QQQ': qqq_data
    })
    
    # Add dividend data if available and requested
    if include_dividends:
        # Add asset dividends
        if asset_dividends is not None:
            dividend_col_name = f'{asset_name}_Dividends'
            strategy_data = strategy_data.merge(
                asset_dividends[['Date', dividend_col_name]], 
                left_index=True, right_on='Date', 
                how='left'
            )
            strategy_data.set_index('Date', inplace=True)
            strategy_data[dividend_col_name].fillna(0, inplace=True)
        else:
            strategy_data[f'{asset_name}_Dividends'] = 0
            
        # Add QQQ dividends
        if qqq_dividends is not None:
            strategy_data = strategy_data.merge(
                qqq_dividends[['Date', 'QQQ_Dividends']], 
                left_index=True, right_on='Date', 
                how='left'
            )
            strategy_data.set_index('Date', inplace=True)
            strategy_data['QQQ_Dividends'].fillna(0, inplace=True)
        else:
            strategy_data['QQQ_Dividends'] = 0
    else:
        # Still create the columns but fill with zeros
        strategy_data[f'{asset_name}_Dividends'] = 0
        strategy_data['QQQ_Dividends'] = 0
    
    # Filter by date range if provided
    if start_date is not None and end_date is not None:
        strategy_data = strategy_data.loc[start_date:end_date]
    
    # Calculate returns
    strategy_data[f'{asset_name}_Rets'] = strategy_data[asset_name].pct_change()
    strategy_data['QQQ_Rets'] = strategy_data['QQQ'].pct_change()
    strategy_data = strategy_data.dropna()
    
    # Calculate rolling correlation
    strategy_data['rolling_corr'] = (
        strategy_data[f'{asset_name}_Rets']
          .rolling(window)
          .corr(strategy_data['QQQ_Rets'])
    )
    
    # Round the rolling correlation to 2 decimal places to avoid precision issues
    strategy_data['rolling_corr'] = strategy_data['rolling_corr'].round(2)
    
    # Initialize columns for tracking positions and PnL
    strategy_data['signal'] = None  # New column to track entry/exit signals
    strategy_data['in_position'] = False
    strategy_data[f'{asset_name}_shares'] = 0
    strategy_data['QQQ_shares'] = 0
    strategy_data[f'{asset_name}_value'] = 0.0
    strategy_data['QQQ_value'] = 0.0
    strategy_data[f'{asset_name}_pnl_daily'] = 0.0
    strategy_data['QQQ_pnl_daily'] = 0.0
    strategy_data['total_pnl_daily'] = 0.0  # New column for total daily PnL
    strategy_data['entry_exit'] = None  # To track entry and exit points
    strategy_data['dividend_payment'] = 0.0  # Track dividend payments
    
    # Entry and exit logic
    in_position = False
    pending_entry = False
    pending_exit = False
    entry_price_asset = None
    entry_price_qqq = None
    shares_asset = 0
    shares_qqq = 0
    entry_date = None
    
    # Iterate through each day
    for i, date in enumerate(strategy_data.index):
        corr = strategy_data.at[date, 'rolling_corr']
        
        # Skip if we don't have enough data yet
        if pd.isna(corr):
            continue
        
        # Check for previous day's pending entry signal
        if i > 0:
            prev_date = strategy_data.index[i-1]
            if strategy_data.at[prev_date, 'signal'] == 'Entry Signal':
                # Execute entry today
                in_position = True
                entry_date = date
                entry_price_asset = strategy_data.at[date, asset_name]
                entry_price_qqq = strategy_data.at[date, 'QQQ']
                shares_asset = asset_amount / entry_price_asset
                shares_qqq = qqq_amount / entry_price_qqq
                
                strategy_data.at[date, 'entry_exit'] = 'Entry'
                pending_entry = False
        
        # Process exit signal - but keep position open for today
        if in_position and not pending_exit and corr >= exit_corr:
            # Mark for exit, but keep position active today
            strategy_data.at[date, 'signal'] = 'Exit Signal'
            strategy_data.at[date, 'entry_exit'] = 'Exit'
            pending_exit = True
            # Don't set in_position to False yet, wait until next day
        
        # Process entry signal if not in position and no pending entry
        if not in_position and not pending_entry and corr <= entry_corr:
            # Generate entry signal today (to be executed tomorrow)
            strategy_data.at[date, 'signal'] = 'Entry Signal'
            pending_entry = True
        
        # Update position status
        strategy_data.at[date, 'in_position'] = in_position
        
        # Update share counts
        strategy_data.at[date, f'{asset_name}_shares'] = shares_asset if in_position else 0
        strategy_data.at[date, 'QQQ_shares'] = shares_qqq if in_position else 0
        
        # Calculate daily PnL if we're in position
        if in_position and i > 0:
            prev_date = strategy_data.index[i-1]
            strategy_data.at[date, f'{asset_name}_pnl_daily'] = shares_asset * (
                strategy_data.at[date, asset_name] - strategy_data.at[prev_date, asset_name]
            )
            strategy_data.at[date, 'QQQ_pnl_daily'] = shares_qqq * (
                strategy_data.at[prev_date, 'QQQ'] - strategy_data.at[date, 'QQQ']
            )
            
            # Add dividend payments if we're including them
            if include_dividends:
                # Add asset dividends if paid today
                if strategy_data.at[date, f'{asset_name}_Dividends'] > 0:
                    asset_div_payment = shares_asset * strategy_data.at[date, f'{asset_name}_Dividends']
                    strategy_data.at[date, 'dividend_payment'] += asset_div_payment
                    strategy_data.at[date, f'{asset_name}_pnl_daily'] += asset_div_payment
                
                # Subtract QQQ dividends for short position if paid today
                if strategy_data.at[date, 'QQQ_Dividends'] > 0:
                    qqq_div_payment = -shares_qqq * strategy_data.at[date, 'QQQ_Dividends']
                    strategy_data.at[date, 'dividend_payment'] += qqq_div_payment
                    strategy_data.at[date, 'QQQ_pnl_daily'] += qqq_div_payment
            
            # Calculate total daily PnL
            strategy_data.at[date, 'total_pnl_daily'] = (
                strategy_data.at[date, f'{asset_name}_pnl_daily'] + 
                strategy_data.at[date, 'QQQ_pnl_daily']
            )
        elif not in_position:
            # No PnL when not in position
            strategy_data.at[date, f'{asset_name}_pnl_daily'] = 0
            strategy_data.at[date, 'QQQ_pnl_daily'] = 0
            strategy_data.at[date, 'total_pnl_daily'] = 0
        
        # Close the position after the exit day (next day after exit signal)
        if pending_exit and i < len(strategy_data.index) - 1:
            # We'll close the position tomorrow, so in_position becomes False for the next day
            in_position = False
            pending_exit = False
    
    # Calculate position values
    for date in strategy_data.index:
        if strategy_data.at[date, 'in_position']:
            strategy_data.at[date, f'{asset_name}_value'] = asset_amount
            strategy_data.at[date, 'QQQ_value'] = qqq_amount
        else:
            strategy_data.at[date, f'{asset_name}_value'] = 0
            strategy_data.at[date, 'QQQ_value'] = 0
    
    # Add cumulative PnL columns
    strategy_data[f'{asset_name}_Cum_PnL'] = 0.0
    strategy_data['QQQ_Cum_PnL'] = 0.0
    strategy_data['Total_Cum_PnL'] = 0.0
    
    # Track cumulative PnL across multiple trades
    asset_cum_pnl = 0.0
    qqq_cum_pnl = 0.0
    total_cum_pnl = 0.0
    
    # Calculate cumulative PnLs based on daily PnLs
    for date in strategy_data.index:
        # Add today's PnL to the running total
        asset_cum_pnl += strategy_data.at[date, f'{asset_name}_pnl_daily']
        qqq_cum_pnl += strategy_data.at[date, 'QQQ_pnl_daily']
        total_cum_pnl += strategy_data.at[date, 'total_pnl_daily']
        
        # Update the cumulative columns
        strategy_data.at[date, f'{asset_name}_Cum_PnL'] = asset_cum_pnl
        strategy_data.at[date, 'QQQ_Cum_PnL'] = qqq_cum_pnl
        strategy_data.at[date, 'Total_Cum_PnL'] = total_cum_pnl
    
    # Calculate strategy metrics
    entries = strategy_data[strategy_data['entry_exit'] == 'Entry']
    exits = strategy_data[strategy_data['entry_exit'] == 'Exit']
    
    metrics = {}
    if not entries.empty:
        total_trades = len(entries)
        
        # Calculate dividend metrics
        total_dividend_pnl = strategy_data.loc[strategy_data['in_position'], 'dividend_payment'].sum()
        days_in_market = strategy_data['in_position'].sum()
        days_with_dividends = sum((strategy_data['in_position']) & 
                                 ((strategy_data[f'{asset_name}_Dividends'] > 0) | 
                                  (strategy_data['QQQ_Dividends'] > 0)))
        
        # Calculate win rate based on total PnL for each trade
        if not exits.empty:
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
        
        metrics = {
            "Total trades": total_trades,
            "Win rate": f"{win_rate:.2%}",
            "Final cumulative PnL": f"${strategy_data['Total_Cum_PnL'].iloc[-1]:.2f}",
            "Number of days in market": int(days_in_market),
            "Amount of Dividends paid": f"${total_dividend_pnl:.2f}",
            "Days with dividend payments": int(days_with_dividends)
        }
    
    # Create a slimmed-down DataFrame for inspection with dynamic column names
    columns_to_include = [
        'QQQ', asset_name, 'rolling_corr', 'signal', 'in_position', 'entry_exit',
        f'{asset_name}_shares', 'QQQ_shares',
        f'{asset_name}_Dividends', 'QQQ_Dividends', 'dividend_payment',
        f'{asset_name}_pnl_daily', 'QQQ_pnl_daily', 'total_pnl_daily',
        f'{asset_name}_Cum_PnL', 'QQQ_Cum_PnL', 'Total_Cum_PnL',
        f'{asset_name}_value', 'QQQ_value'
    ]
    
    portfolio_results = strategy_data[columns_to_include].copy()
    
    return strategy_data, portfolio_results, metrics 

# Function to implement Strategy 2
def run_strategy_2(asset_data, qqq_data, vix_data, vvix_data=None, start_date=None, end_date=None, window=30, 
                  vix_lower=20, vix_upper=30, vvix_lower=100, vvix_upper=150, corr_above=0.9, 
                  use_vix=True, use_vvix=False, asset_amount=10000, qqq_amount=10000, asset_name="Asset",
                  include_dividends=False, asset_dividends=None, qqq_dividends=None):
    """
    Implements a VIX/VVIX-based pairs trading strategy between the selected asset and QQQ.
    - Enters long asset/short QQQ position when VIX/VVIX is within specified range AND correlation is above threshold
    - Exits when VIX/VVIX or correlation moves outside specified parameters
    
    Parameters:
    -----------
    asset_data: pd.Series
        Price data for the selected asset
    qqq_data: pd.Series
        Price data for QQQ
    vix_data: pd.Series
        Price data for VIX
    vvix_data: pd.Series
        Price data for VVIX (optional)
    start_date, end_date: datetime or None
        Date range for the backtest (if None, uses all available data)
    window: int
        Rolling window in days for correlation calculation
    vix_lower: float
        Lower bound for VIX value to enter position
    vix_upper: float
        Upper bound for VIX value to enter position
    vvix_lower: float
        Lower bound for VVIX value to enter position
    vvix_upper: float
        Upper bound for VVIX value to enter position
    corr_above: float
        Correlation threshold to enter position (must be above this value)
    use_vix: bool
        Whether to use VIX data for entry/exit decisions
    use_vvix: bool
        Whether to use VVIX data for entry/exit decisions
    asset_amount: float
        Dollar amount to invest in the asset
    qqq_amount: float
        Dollar amount to short in QQQ
    asset_name: str
        Name of the selected asset for column headers
    include_dividends: bool
        Whether to include dividend payments in the calculations
    asset_dividends: pd.Series or None
        Dividend data for the selected asset
    qqq_dividends: pd.Series or None
        Dividend data for QQQ
    """
    # Merge the data
    strategy_data = pd.DataFrame({
        asset_name: asset_data,
        'QQQ': qqq_data,
    })
    
    # Add VIX if we're using it
    if use_vix and vix_data is not None:
        strategy_data['VIX'] = vix_data
        
    # Add VVIX if we're using it
    if use_vvix and vvix_data is not None:
        strategy_data['VVIX'] = vvix_data
    
    # Add dividend data if available and requested
    if include_dividends:
        # Add asset dividends
        if asset_dividends is not None:
            dividend_col_name = f'{asset_name}_Dividends'
            strategy_data = strategy_data.merge(
                asset_dividends[['Date', dividend_col_name]], 
                left_index=True, right_on='Date', 
                how='left'
            )
            strategy_data.set_index('Date', inplace=True)
            strategy_data[dividend_col_name].fillna(0, inplace=True)
        else:
            strategy_data[f'{asset_name}_Dividends'] = 0
            
        # Add QQQ dividends
        if qqq_dividends is not None:
            strategy_data = strategy_data.merge(
                qqq_dividends[['Date', 'QQQ_Dividends']], 
                left_index=True, right_on='Date', 
                how='left'
            )
            strategy_data.set_index('Date', inplace=True)
            strategy_data['QQQ_Dividends'].fillna(0, inplace=True)
        else:
            strategy_data['QQQ_Dividends'] = 0
    else:
        # Still create the columns but fill with zeros
        strategy_data[f'{asset_name}_Dividends'] = 0
        strategy_data['QQQ_Dividends'] = 0
    
    # Filter by date range if provided
    if start_date is not None and end_date is not None:
        strategy_data = strategy_data.loc[start_date:end_date]
    
    # Calculate returns
    strategy_data[f'{asset_name}_Rets'] = strategy_data[asset_name].pct_change()
    strategy_data['QQQ_Rets'] = strategy_data['QQQ'].pct_change()
    strategy_data = strategy_data.dropna()
    
    # Calculate rolling correlation
    strategy_data['rolling_corr'] = (
        strategy_data[f'{asset_name}_Rets']
          .rolling(window)
          .corr(strategy_data['QQQ_Rets'])
    )
    
    # Round the rolling correlation to 2 decimal places to avoid precision issues
    strategy_data['rolling_corr'] = strategy_data['rolling_corr'].round(2)
    
    # Initialize columns for tracking positions and PnL
    strategy_data['signal'] = None  # New column to track entry/exit signals
    strategy_data['in_position'] = False
    strategy_data[f'{asset_name}_shares'] = 0
    strategy_data['QQQ_shares'] = 0
    strategy_data[f'{asset_name}_value'] = 0.0
    strategy_data['QQQ_value'] = 0.0
    strategy_data[f'{asset_name}_pnl_daily'] = 0.0
    strategy_data['QQQ_pnl_daily'] = 0.0
    strategy_data['total_pnl_daily'] = 0.0  # New column for total daily PnL
    strategy_data['entry_exit'] = None  # To track entry and exit points
    strategy_data['dividend_payment'] = 0.0  # Track dividend payments
    
    # Entry and exit logic
    in_position = False
    pending_entry = False
    pending_exit = False
    entry_price_asset = None
    entry_price_qqq = None
    shares_asset = 0
    shares_qqq = 0
    entry_date = None
    
    # Iterate through each day
    for i, date in enumerate(strategy_data.index):
        corr = strategy_data.at[date, 'rolling_corr']
        
        # Skip if we don't have enough data yet
        if pd.isna(corr):
            continue
        
        # Check for previous day's pending entry signal
        if i > 0:
            prev_date = strategy_data.index[i-1]
            if strategy_data.at[prev_date, 'signal'] == 'Entry Signal':
                # Execute entry today
                in_position = True
                entry_date = date
                entry_price_asset = strategy_data.at[date, asset_name]
                entry_price_qqq = strategy_data.at[date, 'QQQ']
                shares_asset = asset_amount / entry_price_asset
                shares_qqq = qqq_amount / entry_price_qqq
                
                strategy_data.at[date, 'entry_exit'] = 'Entry'
                pending_entry = False
        
        # Check VIX condition if we're using VIX
        vix_condition = True
        if use_vix:
            if 'VIX' not in strategy_data.columns or pd.isna(strategy_data.at[date, 'VIX']):
                vix_condition = False
            else:
                vix = strategy_data.at[date, 'VIX']
                vix_condition = vix_lower <= vix <= vix_upper
        
        # Check VVIX condition if we're using VVIX
        vvix_condition = True
        if use_vvix:
            if 'VVIX' not in strategy_data.columns or pd.isna(strategy_data.at[date, 'VVIX']):
                vvix_condition = False
            else:
                vvix = strategy_data.at[date, 'VVIX']
                vvix_condition = vvix_lower <= vvix <= vvix_upper
        
        # Entry logic: VIX/VVIX within range AND correlation above threshold
        volatility_condition = (vix_condition if use_vix else True) and (vvix_condition if use_vvix else True)
        
        # Process exit signal - but keep position open for today
        if in_position and not pending_exit and (not volatility_condition or corr < corr_above):
            # Mark for exit, but keep position active today
            strategy_data.at[date, 'signal'] = 'Exit Signal'
            strategy_data.at[date, 'entry_exit'] = 'Exit'
            pending_exit = True
            # Don't set in_position to False yet, wait until next day
        
        # Process entry signal if not in position and no pending entry
        if not in_position and not pending_entry and volatility_condition and corr >= corr_above:
            # Generate entry signal today (to be executed tomorrow)
            strategy_data.at[date, 'signal'] = 'Entry Signal'
            pending_entry = True
        
        # Update position status
        strategy_data.at[date, 'in_position'] = in_position
        
        # Update share counts
        strategy_data.at[date, f'{asset_name}_shares'] = shares_asset if in_position else 0
        strategy_data.at[date, 'QQQ_shares'] = shares_qqq if in_position else 0
        
        # Calculate daily PnL if we're in position
        if in_position and i > 0:
            prev_date = strategy_data.index[i-1]
            strategy_data.at[date, f'{asset_name}_pnl_daily'] = shares_asset * (
                strategy_data.at[date, asset_name] - strategy_data.at[prev_date, asset_name]
            )
            strategy_data.at[date, 'QQQ_pnl_daily'] = shares_qqq * (
                strategy_data.at[prev_date, 'QQQ'] - strategy_data.at[date, 'QQQ']
            )
            
            # Add dividend payments if we're including them
            if include_dividends:
                # Add asset dividends if paid today
                if strategy_data.at[date, f'{asset_name}_Dividends'] > 0:
                    asset_div_payment = shares_asset * strategy_data.at[date, f'{asset_name}_Dividends']
                    strategy_data.at[date, 'dividend_payment'] += asset_div_payment
                    strategy_data.at[date, f'{asset_name}_pnl_daily'] += asset_div_payment
                
                # Subtract QQQ dividends for short position if paid today
                if strategy_data.at[date, 'QQQ_Dividends'] > 0:
                    qqq_div_payment = -shares_qqq * strategy_data.at[date, 'QQQ_Dividends']
                    strategy_data.at[date, 'dividend_payment'] += qqq_div_payment
                    strategy_data.at[date, 'QQQ_pnl_daily'] += qqq_div_payment
            
            # Calculate total daily PnL
            strategy_data.at[date, 'total_pnl_daily'] = (
                strategy_data.at[date, f'{asset_name}_pnl_daily'] + 
                strategy_data.at[date, 'QQQ_pnl_daily']
            )
        elif not in_position:
            # No PnL when not in position
            strategy_data.at[date, f'{asset_name}_pnl_daily'] = 0
            strategy_data.at[date, 'QQQ_pnl_daily'] = 0
            strategy_data.at[date, 'total_pnl_daily'] = 0
        
        # Close the position after the exit day (next day after exit signal)
        if pending_exit and i < len(strategy_data.index) - 1:
            # We'll close the position tomorrow, so in_position becomes False for the next day
            in_position = False
            pending_exit = False
    
    # Calculate position values
    for date in strategy_data.index:
        if strategy_data.at[date, 'in_position']:
            strategy_data.at[date, f'{asset_name}_value'] = asset_amount
            strategy_data.at[date, 'QQQ_value'] = qqq_amount
        else:
            strategy_data.at[date, f'{asset_name}_value'] = 0
            strategy_data.at[date, 'QQQ_value'] = 0
    
    # Add cumulative PnL columns
    strategy_data[f'{asset_name}_Cum_PnL'] = 0.0
    strategy_data['QQQ_Cum_PnL'] = 0.0
    strategy_data['Total_Cum_PnL'] = 0.0
    
    # Track cumulative PnL across multiple trades
    asset_cum_pnl = 0.0
    qqq_cum_pnl = 0.0
    total_cum_pnl = 0.0
    
    # Calculate cumulative PnLs based on daily PnLs
    for date in strategy_data.index:
        # Add today's PnL to the running total
        asset_cum_pnl += strategy_data.at[date, f'{asset_name}_pnl_daily']
        qqq_cum_pnl += strategy_data.at[date, 'QQQ_pnl_daily']
        total_cum_pnl += strategy_data.at[date, 'total_pnl_daily']
        
        # Update the cumulative columns
        strategy_data.at[date, f'{asset_name}_Cum_PnL'] = asset_cum_pnl
        strategy_data.at[date, 'QQQ_Cum_PnL'] = qqq_cum_pnl
        strategy_data.at[date, 'Total_Cum_PnL'] = total_cum_pnl
    
    # Calculate strategy metrics
    entries = strategy_data[strategy_data['entry_exit'] == 'Entry']
    exits = strategy_data[strategy_data['entry_exit'] == 'Exit']
    
    metrics = {}
    if not entries.empty:
        total_trades = len(entries)
        
        # Calculate dividend metrics if we're including dividends
        if include_dividends:
            total_dividend_pnl = strategy_data.loc[strategy_data['in_position'], 'dividend_payment'].sum()
            days_with_dividends = sum((strategy_data['in_position']) & 
                                     ((strategy_data[f'{asset_name}_Dividends'] > 0) | 
                                      (strategy_data['QQQ_Dividends'] > 0)))
        else:
            total_dividend_pnl = 0
            days_with_dividends = 0
        
        # Calculate days in market
        days_in_market = strategy_data['in_position'].sum()
        
        # Calculate win rate based on total PnL for each trade
        if not exits.empty:
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
        
        metrics = {
            "Total trades": total_trades,
            "Win rate": f"{win_rate:.2%}",
            "Final cumulative PnL": f"${strategy_data['Total_Cum_PnL'].iloc[-1]:.2f}",
            "Number of days in market": int(days_in_market),
            "Amount of Dividends paid": f"${total_dividend_pnl:.2f}",
            "Days with dividend payments": int(days_with_dividends)
        }
    
    # Create a slimmed-down DataFrame for inspection with dynamic column names
    columns_to_include = [
        'QQQ', asset_name, 'rolling_corr', 'signal', 'in_position', 'entry_exit',
        f'{asset_name}_shares', 'QQQ_shares'
    ]
    
    # Add VIX and VVIX to the output if they are used
    if use_vix and 'VIX' in strategy_data.columns:
        columns_to_include.insert(2, 'VIX')
    
    if use_vvix and 'VVIX' in strategy_data.columns:
        columns_to_include.insert(3 if use_vix else 2, 'VVIX')
    
    # Add the dividend columns if dividends are included
    if include_dividends:
        columns_to_include.extend([f'{asset_name}_Dividends', 'QQQ_Dividends', 'dividend_payment'])
    
    # Add PnL and cumulative PnL columns
    columns_to_include.extend([
        f'{asset_name}_pnl_daily', 'QQQ_pnl_daily', 'total_pnl_daily',
        f'{asset_name}_Cum_PnL', 'QQQ_Cum_PnL', 'Total_Cum_PnL',
        f'{asset_name}_value', 'QQQ_value'
    ])
    
    portfolio_results = strategy_data[columns_to_include].copy()
    
    return strategy_data, portfolio_results, metrics

# Function to implement Strategy 3
def run_strategy_3(asset_data, qqq_data, start_date=None, end_date=None, window=30, 
                 corr_lower=0.80, corr_upper=0.90, asset_amount=10000, qqq_amount=10000, 
                 asset_name="Asset", include_dividends=False, asset_dividends=None, qqq_dividends=None):
    """
    Implements a correlation-bound pairs trading strategy between a selected asset and QQQ.
    - Enters long asset/short QQQ position when correlation is within specified bounds (corr_lower to corr_upper, inclusive)
    - Exits when correlation moves outside the specified bounds
    
    Parameters:
    -----------
    asset_data: pd.Series
        Price data for the selected asset
    qqq_data: pd.Series
        Price data for QQQ
    start_date, end_date: datetime or None
        Date range for the backtest (if None, uses all available data)
    window: int
        Rolling window in days for correlation calculation
    corr_lower: float
        Lower bound of correlation to enter position (inclusive)
    corr_upper: float
        Upper bound of correlation to enter position (inclusive)
    asset_amount: float
        Dollar amount to invest in the asset
    qqq_amount: float
        Dollar amount to short in QQQ
    asset_name: str
        Name of the selected asset for column headers
    include_dividends: bool
        Whether to include dividend payments in the calculations
    asset_dividends: pd.Series or None
        Dividend data for the selected asset
    qqq_dividends: pd.Series or None
        Dividend data for QQQ
    """
    # Merge the data
    strategy_data = pd.DataFrame({
        asset_name: asset_data,
        'QQQ': qqq_data
    })
    
    # Add dividend data if available and requested
    if include_dividends:
        # Add asset dividends
        if asset_dividends is not None:
            dividend_col_name = f'{asset_name}_Dividends'
            strategy_data = strategy_data.merge(
                asset_dividends[['Date', dividend_col_name]], 
                left_index=True, right_on='Date', 
                how='left'
            )
            strategy_data.set_index('Date', inplace=True)
            strategy_data[dividend_col_name].fillna(0, inplace=True)
        else:
            strategy_data[f'{asset_name}_Dividends'] = 0
            
        # Add QQQ dividends
        if qqq_dividends is not None:
            strategy_data = strategy_data.merge(
                qqq_dividends[['Date', 'QQQ_Dividends']], 
                left_index=True, right_on='Date', 
                how='left'
            )
            strategy_data.set_index('Date', inplace=True)
            strategy_data['QQQ_Dividends'].fillna(0, inplace=True)
        else:
            strategy_data['QQQ_Dividends'] = 0
    else:
        # Still create the columns but fill with zeros
        strategy_data[f'{asset_name}_Dividends'] = 0
        strategy_data['QQQ_Dividends'] = 0
    
    # Filter by date range if provided
    if start_date is not None and end_date is not None:
        strategy_data = strategy_data.loc[start_date:end_date]
    
    # Calculate returns
    strategy_data[f'{asset_name}_Rets'] = strategy_data[asset_name].pct_change()
    strategy_data['QQQ_Rets'] = strategy_data['QQQ'].pct_change()
    strategy_data = strategy_data.dropna()
    
    # Calculate rolling correlation
    strategy_data['rolling_corr'] = (
        strategy_data[f'{asset_name}_Rets']
          .rolling(window)
          .corr(strategy_data['QQQ_Rets'])
    )
    
    # Round the rolling correlation to 2 decimal places to avoid precision issues
    strategy_data['rolling_corr'] = strategy_data['rolling_corr'].round(2)
    
    # Initialize columns for tracking positions and PnL
    strategy_data['signal'] = None  # Column to track entry/exit signals
    strategy_data['in_position'] = False
    strategy_data[f'{asset_name}_shares'] = 0
    strategy_data['QQQ_shares'] = 0
    strategy_data[f'{asset_name}_value'] = 0.0
    strategy_data['QQQ_value'] = 0.0
    strategy_data[f'{asset_name}_pnl_daily'] = 0.0
    strategy_data['QQQ_pnl_daily'] = 0.0
    strategy_data['total_pnl_daily'] = 0.0  # Total daily PnL
    strategy_data['entry_exit'] = None  # To track entry and exit points
    strategy_data['dividend_payment'] = 0.0  # Track dividend payments
    
    # Entry and exit logic
    in_position = False
    pending_entry = False
    pending_exit = False
    entry_price_asset = None
    entry_price_qqq = None
    shares_asset = 0
    shares_qqq = 0
    entry_date = None
    
    # Iterate through each day
    for i, date in enumerate(strategy_data.index):
        corr = strategy_data.at[date, 'rolling_corr']
        
        # Skip if we don't have enough data yet
        if pd.isna(corr):
            continue
        
        # Check for previous day's pending entry signal
        if i > 0:
            prev_date = strategy_data.index[i-1]
            if strategy_data.at[prev_date, 'signal'] == 'Entry Signal':
                # Execute entry today
                in_position = True
                entry_date = date
                entry_price_asset = strategy_data.at[date, asset_name]
                entry_price_qqq = strategy_data.at[date, 'QQQ']
                shares_asset = asset_amount / entry_price_asset
                shares_qqq = qqq_amount / entry_price_qqq
                
                strategy_data.at[date, 'entry_exit'] = 'Entry'
                pending_entry = False
        
        # Process exit signal - when correlation moves outside bounds, but keep position open for today
        if in_position and not pending_exit:
            # Correlation outside bounds - prepare to exit
            if corr < corr_lower or corr > corr_upper:
                strategy_data.at[date, 'signal'] = 'Exit Signal'
                strategy_data.at[date, 'entry_exit'] = 'Exit'
                pending_exit = True
                # Don't set in_position to False yet, wait until next day
        
        # Process entry signal if not in position and no pending entry
        if not in_position and not pending_entry:
            # Correlation inside bounds - prepare to enter
            if corr_lower <= corr <= corr_upper:
                strategy_data.at[date, 'signal'] = 'Entry Signal'
                pending_entry = True
        
        # Update position status
        strategy_data.at[date, 'in_position'] = in_position
        
        # Update share counts
        strategy_data.at[date, f'{asset_name}_shares'] = shares_asset if in_position else 0
        strategy_data.at[date, 'QQQ_shares'] = shares_qqq if in_position else 0
        
        # Calculate daily PnL if we're in position
        if in_position and i > 0:
            prev_date = strategy_data.index[i-1]
            strategy_data.at[date, f'{asset_name}_pnl_daily'] = shares_asset * (
                strategy_data.at[date, asset_name] - strategy_data.at[prev_date, asset_name]
            )
            strategy_data.at[date, 'QQQ_pnl_daily'] = shares_qqq * (
                strategy_data.at[prev_date, 'QQQ'] - strategy_data.at[date, 'QQQ']
            )
            
            # Add dividend payments if we're including them
            if include_dividends:
                # Add asset dividends if paid today
                if strategy_data.at[date, f'{asset_name}_Dividends'] > 0:
                    asset_div_payment = shares_asset * strategy_data.at[date, f'{asset_name}_Dividends']
                    strategy_data.at[date, 'dividend_payment'] += asset_div_payment
                    strategy_data.at[date, f'{asset_name}_pnl_daily'] += asset_div_payment
                
                # Subtract QQQ dividends for short position if paid today
                if strategy_data.at[date, 'QQQ_Dividends'] > 0:
                    qqq_div_payment = -shares_qqq * strategy_data.at[date, 'QQQ_Dividends']
                    strategy_data.at[date, 'dividend_payment'] += qqq_div_payment
                    strategy_data.at[date, 'QQQ_pnl_daily'] += qqq_div_payment
            
            # Calculate total daily PnL
            strategy_data.at[date, 'total_pnl_daily'] = (
                strategy_data.at[date, f'{asset_name}_pnl_daily'] + 
                strategy_data.at[date, 'QQQ_pnl_daily']
            )
        elif not in_position:
            # No PnL when not in position
            strategy_data.at[date, f'{asset_name}_pnl_daily'] = 0
            strategy_data.at[date, 'QQQ_pnl_daily'] = 0
            strategy_data.at[date, 'total_pnl_daily'] = 0
        
        # Close the position after the exit day (next day after exit signal)
        if pending_exit and i < len(strategy_data.index) - 1:
            # We'll close the position tomorrow, so in_position becomes False for the next day
            in_position = False
            pending_exit = False
    
    # Calculate position values
    for date in strategy_data.index:
        if strategy_data.at[date, 'in_position']:
            strategy_data.at[date, f'{asset_name}_value'] = asset_amount
            strategy_data.at[date, 'QQQ_value'] = qqq_amount
        else:
            strategy_data.at[date, f'{asset_name}_value'] = 0
            strategy_data.at[date, 'QQQ_value'] = 0
    
    # Add cumulative PnL columns
    strategy_data[f'{asset_name}_Cum_PnL'] = 0.0
    strategy_data['QQQ_Cum_PnL'] = 0.0
    strategy_data['Total_Cum_PnL'] = 0.0
    
    # Track cumulative PnL across multiple trades
    asset_cum_pnl = 0.0
    qqq_cum_pnl = 0.0
    total_cum_pnl = 0.0
    
    # Calculate cumulative PnLs based on daily PnLs
    for date in strategy_data.index:
        # Add today's PnL to the running total
        asset_cum_pnl += strategy_data.at[date, f'{asset_name}_pnl_daily']
        qqq_cum_pnl += strategy_data.at[date, 'QQQ_pnl_daily']
        total_cum_pnl += strategy_data.at[date, 'total_pnl_daily']
        
        # Update the cumulative columns
        strategy_data.at[date, f'{asset_name}_Cum_PnL'] = asset_cum_pnl
        strategy_data.at[date, 'QQQ_Cum_PnL'] = qqq_cum_pnl
        strategy_data.at[date, 'Total_Cum_PnL'] = total_cum_pnl
    
    # Calculate strategy metrics
    entries = strategy_data[strategy_data['entry_exit'] == 'Entry']
    exits = strategy_data[strategy_data['entry_exit'] == 'Exit']
    
    metrics = {}
    if not entries.empty:
        total_trades = len(entries)
        
        # Calculate dividend metrics
        total_dividend_pnl = strategy_data.loc[strategy_data['in_position'], 'dividend_payment'].sum()
        days_in_market = strategy_data['in_position'].sum()
        days_with_dividends = sum((strategy_data['in_position']) & 
                                 ((strategy_data[f'{asset_name}_Dividends'] > 0) | 
                                  (strategy_data['QQQ_Dividends'] > 0)))
        
        # Calculate win rate based on total PnL for each trade
        if not exits.empty:
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
        
        metrics = {
            "Total trades": total_trades,
            "Win rate": f"{win_rate:.2%}",
            "Final cumulative PnL": f"${strategy_data['Total_Cum_PnL'].iloc[-1]:.2f}",
            "Number of days in market": int(days_in_market),
            "Amount of Dividends paid": f"${total_dividend_pnl:.2f}",
            "Days with dividend payments": int(days_with_dividends)
        }
    
    # Create a slimmed-down DataFrame for inspection with dynamic column names
    columns_to_include = [
        'QQQ', asset_name, 'rolling_corr', 'signal', 'in_position', 'entry_exit',
        f'{asset_name}_shares', 'QQQ_shares',
        f'{asset_name}_Dividends', 'QQQ_Dividends', 'dividend_payment',
        f'{asset_name}_pnl_daily', 'QQQ_pnl_daily', 'total_pnl_daily',
        f'{asset_name}_Cum_PnL', 'QQQ_Cum_PnL', 'Total_Cum_PnL',
        f'{asset_name}_value', 'QQQ_value'
    ]
    
    portfolio_results = strategy_data[columns_to_include].copy()
    
    return strategy_data, portfolio_results, metrics

# Placeholder for future strategies
def run_strategy_4(asset_data, qqq_data, vix_data, vvix_data=None, start_date=None, end_date=None, window=30, 
                  vix_lower=20, vix_upper=30, vvix_lower=100, vvix_upper=150, corr_above=0.9, 
                  use_vix=True, use_vvix=False, asset_amount=10000, qqq_amount=10000, asset_name="Asset",
                  # New parameters for correlation signals
                  use_vix_asset_corr=False, vix_asset_window=30, vix_asset_corr_threshold=-0.70,
                  use_vvix_asset_corr=False, vvix_asset_window=30, vvix_asset_corr_threshold=-0.70,
                  use_vix_qqq_corr=False, vix_qqq_window=30, vix_qqq_corr_threshold=-0.70,
                  use_vvix_qqq_corr=False, vvix_qqq_window=30, vvix_qqq_corr_threshold=-0.70,
                  # New volatility value change constraint
                  use_vol_change_constraint=False, lookback_days=1,
                  include_dividends=False, asset_dividends=None, qqq_dividends=None):
    """
    Implements an advanced volatility & correlation strategy between the selected asset and QQQ.
    - Extends Strategy 2 by adding multiple correlation signals between VIX/VVIX and both the asset and QQQ
    - Enters position when ALL enabled conditions are met simultaneously (volatility levels + correlation thresholds)
    - Exits when ANY of the conditions are no longer met
    - Can stay in market if volatility index value is increasing compared to N days ago
    
    Parameters:
    -----------
    asset_data: pd.Series
        Price data for the selected asset
    qqq_data: pd.Series
        Price data for QQQ
    vix_data: pd.Series
        Price data for VIX
    vvix_data: pd.Series
        Price data for VVIX (optional)
    start_date, end_date: datetime or None
        Date range for the backtest (if None, uses all available data)
    window: int
        Rolling window in days for asset-QQQ correlation calculation
    vix_lower, vix_upper: float
        Lower and upper bounds for VIX value to enter position
    vvix_lower, vvix_upper: float
        Lower and upper bounds for VVIX value to enter position
    corr_above: float
        Asset-QQQ correlation threshold to enter position (must be above this value)
    use_vix, use_vvix: bool
        Whether to use VIX/VVIX data for entry/exit decisions based on volatility levels
        
    # New correlation parameters
    use_vix_asset_corr: bool
        Whether to check VIX-Asset correlation as an entry/exit condition
    vix_asset_window: int
        Rolling window in days for VIX-Asset correlation calculation
    vix_asset_corr_threshold: float
        Enter position when VIX-Asset correlation is below this value
    use_vvix_asset_corr: bool
        Whether to check VVIX-Asset correlation as an entry/exit condition
    vvix_asset_window: int
        Rolling window in days for VVIX-Asset correlation calculation
    vvix_asset_corr_threshold: float
        Enter position when VVIX-Asset correlation is below this value
    use_vix_qqq_corr: bool
        Whether to check VIX-QQQ correlation as an entry/exit condition
    vix_qqq_window: int
        Rolling window in days for VIX-QQQ correlation calculation
    vix_qqq_corr_threshold: float
        Enter position when VIX-QQQ correlation is below this value
    use_vvix_qqq_corr: bool
        Whether to check VVIX-QQQ correlation as an entry/exit condition
    vvix_qqq_window: int
        Rolling window in days for VVIX-QQQ correlation calculation
    vvix_qqq_corr_threshold: float
        Enter position when VVIX-QQQ correlation is below this value
    
    # New volatility value change constraint
    use_vol_change_constraint: bool
        Whether to use the volatility value change constraint to stay in market
    lookback_days: int
        Number of days to look back for comparing volatility values
        
    asset_amount: float
        Dollar amount to invest in the asset
    qqq_amount: float
        Dollar amount to short in QQQ
    asset_name: str
        Name of the selected asset for column headers
    include_dividends: bool
        Whether to include dividend payments in the calculations
    asset_dividends: pd.Series or None
        Dividend data for the selected asset
    qqq_dividends: pd.Series or None
        Dividend data for QQQ
    """
    # Merge the data
    strategy_data = pd.DataFrame({
        asset_name: asset_data,
        'QQQ': qqq_data,
    })
    
    # Add VIX if we're using it for any condition
    if (use_vix or use_vix_asset_corr or use_vix_qqq_corr) and vix_data is not None:
        strategy_data['VIX'] = vix_data
        
    # Add VVIX if we're using it for any condition
    if (use_vvix or use_vvix_asset_corr or use_vvix_qqq_corr) and vvix_data is not None:
        strategy_data['VVIX'] = vvix_data
    
    # Add dividend data if available and requested
    if include_dividends:
        # Add asset dividends
        if asset_dividends is not None:
            dividend_col_name = f'{asset_name}_Dividends'
            strategy_data = strategy_data.merge(
                asset_dividends[['Date', dividend_col_name]], 
                left_index=True, right_on='Date', 
                how='left'
            )
            strategy_data.set_index('Date', inplace=True)
            strategy_data[dividend_col_name].fillna(0, inplace=True)
        else:
            strategy_data[f'{asset_name}_Dividends'] = 0
            
        # Add QQQ dividends
        if qqq_dividends is not None:
            strategy_data = strategy_data.merge(
                qqq_dividends[['Date', 'QQQ_Dividends']], 
                left_index=True, right_on='Date', 
                how='left'
            )
            strategy_data.set_index('Date', inplace=True)
            strategy_data['QQQ_Dividends'].fillna(0, inplace=True)
        else:
            strategy_data['QQQ_Dividends'] = 0
    else:
        # Still create the columns but fill with zeros
        strategy_data[f'{asset_name}_Dividends'] = 0
        strategy_data['QQQ_Dividends'] = 0
    
    # Filter by date range if provided
    if start_date is not None and end_date is not None:
        strategy_data = strategy_data.loc[start_date:end_date]
    
    # Calculate returns for all assets
    strategy_data[f'{asset_name}_Rets'] = strategy_data[asset_name].pct_change()
    strategy_data['QQQ_Rets'] = strategy_data['QQQ'].pct_change()
    
    # Calculate returns for VIX and VVIX if they're being used
    if 'VIX' in strategy_data.columns:
        strategy_data['VIX_Rets'] = strategy_data['VIX'].pct_change()
    
    if 'VVIX' in strategy_data.columns:
        strategy_data['VVIX_Rets'] = strategy_data['VVIX'].pct_change()
    
    # Drop rows with NaN values after return calculations
    strategy_data = strategy_data.dropna()
    
    # Calculate standard asset-QQQ rolling correlation
    strategy_data['rolling_corr'] = (
        strategy_data[f'{asset_name}_Rets']
          .rolling(window)
          .corr(strategy_data['QQQ_Rets'])
    )
    
    # Calculate VIX-Asset correlation if needed
    if use_vix_asset_corr and 'VIX' in strategy_data.columns:
        strategy_data['vix_asset_corr'] = (
            strategy_data['VIX_Rets']
              .rolling(vix_asset_window)
              .corr(strategy_data[f'{asset_name}_Rets'])
        )
    
    # Calculate VVIX-Asset correlation if needed
    if use_vvix_asset_corr and 'VVIX' in strategy_data.columns:
        strategy_data['vvix_asset_corr'] = (
            strategy_data['VVIX_Rets']
              .rolling(vvix_asset_window)
              .corr(strategy_data[f'{asset_name}_Rets'])
        )
    
    # Calculate VIX-QQQ correlation if needed
    if use_vix_qqq_corr and 'VIX' in strategy_data.columns:
        strategy_data['vix_qqq_corr'] = (
            strategy_data['VIX_Rets']
              .rolling(vix_qqq_window)
              .corr(strategy_data['QQQ_Rets'])
        )
    
    # Calculate VVIX-QQQ correlation if needed
    if use_vvix_qqq_corr and 'VVIX' in strategy_data.columns:
        strategy_data['vvix_qqq_corr'] = (
            strategy_data['VVIX_Rets']
              .rolling(vvix_qqq_window)
              .corr(strategy_data['QQQ_Rets'])
        )
    
    # Add previous day values for VIX and VVIX if using volatility value change constraint
    if use_vol_change_constraint:
        if 'VIX' in strategy_data.columns:
            strategy_data[f'VIX_prev_{lookback_days}d'] = strategy_data['VIX'].shift(lookback_days)
        if 'VVIX' in strategy_data.columns:
            strategy_data[f'VVIX_prev_{lookback_days}d'] = strategy_data['VVIX'].shift(lookback_days)
    
    # Round all correlation values to 2 decimal places to avoid precision issues
    corr_columns = [col for col in strategy_data.columns if 'corr' in col]
    for col in corr_columns:
        strategy_data[col] = strategy_data[col].round(2)
    
    # Initialize columns for tracking positions and PnL
    strategy_data['signal'] = None  # New column to track entry/exit signals
    strategy_data['in_position'] = False
    strategy_data[f'{asset_name}_shares'] = 0
    strategy_data['QQQ_shares'] = 0
    strategy_data[f'{asset_name}_value'] = 0.0
    strategy_data['QQQ_value'] = 0.0
    strategy_data[f'{asset_name}_pnl_daily'] = 0.0
    strategy_data['QQQ_pnl_daily'] = 0.0
    strategy_data['total_pnl_daily'] = 0.0  # New column for total daily PnL
    strategy_data['entry_exit'] = None  # To track entry and exit points
    strategy_data['dividend_payment'] = 0.0  # Track dividend payments
    
    # Entry and exit logic
    in_position = False
    pending_entry = False
    pending_exit = False
    entry_price_asset = None
    entry_price_qqq = None
    shares_asset = 0
    shares_qqq = 0
    entry_date = None
    
    # Iterate through each day
    for i, date in enumerate(strategy_data.index):
        # Skip early days where correlations haven't been calculated yet
        if pd.isna(strategy_data.at[date, 'rolling_corr']):
            continue
        
        # Check for previous day's pending entry signal
        if i > 0:
            prev_date = strategy_data.index[i-1]
            if strategy_data.at[prev_date, 'signal'] == 'Entry Signal':
                # Execute entry today
                in_position = True
                entry_date = date
                entry_price_asset = strategy_data.at[date, asset_name]
                entry_price_qqq = strategy_data.at[date, 'QQQ']
                shares_asset = asset_amount / entry_price_asset
                shares_qqq = qqq_amount / entry_price_qqq
                
                strategy_data.at[date, 'entry_exit'] = 'Entry'
                pending_entry = False
        
        # Get current correlation values
        corr = strategy_data.at[date, 'rolling_corr']
        
        # Check VIX/VVIX level conditions if we're using them
        vix_level_condition = True
        if use_vix:
            if 'VIX' not in strategy_data.columns or pd.isna(strategy_data.at[date, 'VIX']):
                vix_level_condition = False
            else:
                vix = strategy_data.at[date, 'VIX']
                vix_level_condition = vix_lower <= vix <= vix_upper
        
        vvix_level_condition = True
        if use_vvix:
            if 'VVIX' not in strategy_data.columns or pd.isna(strategy_data.at[date, 'VVIX']):
                vvix_level_condition = False
            else:
                vvix = strategy_data.at[date, 'VVIX']
                vvix_level_condition = vvix_lower <= vvix <= vvix_upper
        
        # Check advanced correlation conditions
        vix_asset_corr_condition = True
        if use_vix_asset_corr:
            if 'vix_asset_corr' not in strategy_data.columns or pd.isna(strategy_data.at[date, 'vix_asset_corr']):
                vix_asset_corr_condition = False
            else:
                vix_asset_corr = strategy_data.at[date, 'vix_asset_corr']
                vix_asset_corr_condition = vix_asset_corr <= vix_asset_corr_threshold
        
        vvix_asset_corr_condition = True
        if use_vvix_asset_corr:
            if 'vvix_asset_corr' not in strategy_data.columns or pd.isna(strategy_data.at[date, 'vvix_asset_corr']):
                vvix_asset_corr_condition = False
            else:
                vvix_asset_corr = strategy_data.at[date, 'vvix_asset_corr']
                vvix_asset_corr_condition = vvix_asset_corr <= vvix_asset_corr_threshold
        
        vix_qqq_corr_condition = True
        if use_vix_qqq_corr:
            if 'vix_qqq_corr' not in strategy_data.columns or pd.isna(strategy_data.at[date, 'vix_qqq_corr']):
                vix_qqq_corr_condition = False
            else:
                vix_qqq_corr = strategy_data.at[date, 'vix_qqq_corr']
                vix_qqq_corr_condition = vix_qqq_corr <= vix_qqq_corr_threshold
        
        vvix_qqq_corr_condition = True
        if use_vvix_qqq_corr:
            if 'vvix_qqq_corr' not in strategy_data.columns or pd.isna(strategy_data.at[date, 'vvix_qqq_corr']):
                vvix_qqq_corr_condition = False
            else:
                vvix_qqq_corr = strategy_data.at[date, 'vvix_qqq_corr']
                vvix_qqq_corr_condition = vvix_qqq_corr <= vvix_qqq_corr_threshold
        
        # Check volatility value change constraint for staying in market
        vol_change_condition = False  # Initialize to False by default
        if use_vol_change_constraint and in_position:
            vix_change_condition = False
            vvix_change_condition = False
            
            # For VIX value change condition
            if use_vix and 'VIX' in strategy_data.columns and f'VIX_prev_{lookback_days}d' in strategy_data.columns:
                vix_curr = strategy_data.at[date, 'VIX']
                vix_prev = strategy_data.at[date, f'VIX_prev_{lookback_days}d']
                
                # Check if current <= previous (volatility decreasing or stable)
                if not pd.isna(vix_curr) and not pd.isna(vix_prev):
                    vix_change_condition = vix_curr <= vix_prev
                    
            # For VVIX value change condition
            if use_vvix and 'VVIX' in strategy_data.columns and f'VVIX_prev_{lookback_days}d' in strategy_data.columns:
                vvix_curr = strategy_data.at[date, 'VVIX']
                vvix_prev = strategy_data.at[date, f'VVIX_prev_{lookback_days}d']
                
                # Check if current <= previous (volatility decreasing or stable)
                if not pd.isna(vvix_curr) and not pd.isna(vvix_prev):
                    vvix_change_condition = vvix_curr <= vvix_prev
            
            # Set overall condition based on which volatility indices are being used
            if use_vix and use_vvix:
                # If both are used, require both to meet the condition
                vol_change_condition = vix_change_condition and vvix_change_condition
            elif use_vix:
                vol_change_condition = vix_change_condition
            elif use_vvix:
                vol_change_condition = vvix_change_condition
        
        # Base condition from Strategy 2: Asset-QQQ correlation
        asset_qqq_corr_condition = corr >= corr_above
        
        # Combine all conditions
        volatility_level_condition = (vix_level_condition if use_vix else True) and (vvix_level_condition if use_vvix else True)
        volatility_corr_condition = (vix_asset_corr_condition and vvix_asset_corr_condition and 
                                    vix_qqq_corr_condition and vvix_qqq_corr_condition)
        
        # Entry and exit conditions
        regular_conditions_met = volatility_level_condition and volatility_corr_condition and asset_qqq_corr_condition
        
        # For entry: All regular conditions must be met
        entry_conditions_met = regular_conditions_met
        
        # For exit: Consider the volatility value change constraint if enabled
        # Only exit if regular conditions fail AND vol_change_condition is false (or not using vol_change_constraint)
        if in_position and use_vol_change_constraint:
            # Exit if regular conditions not met AND volatility is decreasing
            exit_conditions_met = not regular_conditions_met and not vol_change_condition
        else:
            # If not using vol_change_constraint, just check regular conditions
            exit_conditions_met = not regular_conditions_met
        
        # Process exit signal - but keep position open for today
        if in_position and not pending_exit and exit_conditions_met:
            # Mark for exit, but keep position active today
            strategy_data.at[date, 'signal'] = 'Exit Signal'
            strategy_data.at[date, 'entry_exit'] = 'Exit'
            pending_exit = True
            # Don't set in_position to False yet, wait until next day
        
        # Process entry signal if not in position and no pending entry
        if not in_position and not pending_entry and entry_conditions_met:
            # Generate entry signal today (to be executed tomorrow)
            strategy_data.at[date, 'signal'] = 'Entry Signal'
            pending_entry = True
        
        # Update position status
        strategy_data.at[date, 'in_position'] = in_position
        
        # Update share counts
        strategy_data.at[date, f'{asset_name}_shares'] = shares_asset if in_position else 0
        strategy_data.at[date, 'QQQ_shares'] = shares_qqq if in_position else 0
        
        # Calculate daily PnL if we're in position
        if in_position and i > 0:
            prev_date = strategy_data.index[i-1]
            strategy_data.at[date, f'{asset_name}_pnl_daily'] = shares_asset * (
                strategy_data.at[date, asset_name] - strategy_data.at[prev_date, asset_name]
            )
            strategy_data.at[date, 'QQQ_pnl_daily'] = shares_qqq * (
                strategy_data.at[prev_date, 'QQQ'] - strategy_data.at[date, 'QQQ']
            )
            
            # Add dividend payments if we're including them
            if include_dividends:
                # Add asset dividends if paid today
                if strategy_data.at[date, f'{asset_name}_Dividends'] > 0:
                    asset_div_payment = shares_asset * strategy_data.at[date, f'{asset_name}_Dividends']
                    strategy_data.at[date, 'dividend_payment'] += asset_div_payment
                    strategy_data.at[date, f'{asset_name}_pnl_daily'] += asset_div_payment
                
                # Subtract QQQ dividends for short position if paid today
                if strategy_data.at[date, 'QQQ_Dividends'] > 0:
                    qqq_div_payment = -shares_qqq * strategy_data.at[date, 'QQQ_Dividends']
                    strategy_data.at[date, 'dividend_payment'] += qqq_div_payment
                    strategy_data.at[date, 'QQQ_pnl_daily'] += qqq_div_payment
            
            # Calculate total daily PnL
            strategy_data.at[date, 'total_pnl_daily'] = (
                strategy_data.at[date, f'{asset_name}_pnl_daily'] + 
                strategy_data.at[date, 'QQQ_pnl_daily']
            )
        elif not in_position:
            # No PnL when not in position
            strategy_data.at[date, f'{asset_name}_pnl_daily'] = 0
            strategy_data.at[date, 'QQQ_pnl_daily'] = 0
            strategy_data.at[date, 'total_pnl_daily'] = 0
        
        # Close the position after the exit day (next day after exit signal)
        if pending_exit and i < len(strategy_data.index) - 1:
            # We'll close the position tomorrow, so in_position becomes False for the next day
            in_position = False
            pending_exit = False
    
    # Calculate position values
    for date in strategy_data.index:
        if strategy_data.at[date, 'in_position']:
            strategy_data.at[date, f'{asset_name}_value'] = asset_amount
            strategy_data.at[date, 'QQQ_value'] = qqq_amount
        else:
            strategy_data.at[date, f'{asset_name}_value'] = 0
            strategy_data.at[date, 'QQQ_value'] = 0
    
    # Add cumulative PnL columns
    strategy_data[f'{asset_name}_Cum_PnL'] = 0.0
    strategy_data['QQQ_Cum_PnL'] = 0.0
    strategy_data['Total_Cum_PnL'] = 0.0
    
    # Track cumulative PnL across multiple trades
    asset_cum_pnl = 0.0
    qqq_cum_pnl = 0.0
    total_cum_pnl = 0.0
    
    # Calculate cumulative PnLs based on daily PnLs
    for date in strategy_data.index:
        # Add today's PnL to the running total
        asset_cum_pnl += strategy_data.at[date, f'{asset_name}_pnl_daily']
        qqq_cum_pnl += strategy_data.at[date, 'QQQ_pnl_daily']
        total_cum_pnl += strategy_data.at[date, 'total_pnl_daily']
        
        # Update the cumulative columns
        strategy_data.at[date, f'{asset_name}_Cum_PnL'] = asset_cum_pnl
        strategy_data.at[date, 'QQQ_Cum_PnL'] = qqq_cum_pnl
        strategy_data.at[date, 'Total_Cum_PnL'] = total_cum_pnl
    
    # Calculate strategy metrics
    entries = strategy_data[strategy_data['entry_exit'] == 'Entry']
    exits = strategy_data[strategy_data['entry_exit'] == 'Exit']
    
    metrics = {}
    if not entries.empty:
        total_trades = len(entries)
        
        # Calculate dividend metrics if we're including dividends
        if include_dividends:
            total_dividend_pnl = strategy_data.loc[strategy_data['in_position'], 'dividend_payment'].sum()
            days_with_dividends = sum((strategy_data['in_position']) & 
                                     ((strategy_data[f'{asset_name}_Dividends'] > 0) | 
                                      (strategy_data['QQQ_Dividends'] > 0)))
        else:
            total_dividend_pnl = 0
            days_with_dividends = 0
        
        # Calculate days in market
        days_in_market = strategy_data['in_position'].sum()
        
        # Calculate win rate based on total PnL for each trade
        if not exits.empty:
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
        
        metrics = {
            "Total trades": total_trades,
            "Win rate": f"{win_rate:.2%}",
            "Final cumulative PnL": f"${strategy_data['Total_Cum_PnL'].iloc[-1]:.2f}",
            "Number of days in market": int(days_in_market),
            "Amount of Dividends paid": f"${total_dividend_pnl:.2f}",
            "Days with dividend payments": int(days_with_dividends)
        }
    
    # Create a slimmed-down DataFrame for inspection with dynamic column names
    columns_to_include = [
        'QQQ', asset_name, 'rolling_corr', 'signal', 'in_position', 'entry_exit',
        f'{asset_name}_shares', 'QQQ_shares'
    ]
    
    # Add VIX and VVIX to the output if they are used
    if use_vix and 'VIX' in strategy_data.columns:
        columns_to_include.insert(2, 'VIX')
        if use_vol_change_constraint and f'VIX_prev_{lookback_days}d' in strategy_data.columns:
            columns_to_include.append(f'VIX_prev_{lookback_days}d')
    
    if use_vvix and 'VVIX' in strategy_data.columns:
        columns_to_include.insert(3 if use_vix else 2, 'VVIX')
        if use_vol_change_constraint and f'VVIX_prev_{lookback_days}d' in strategy_data.columns:
            columns_to_include.append(f'VVIX_prev_{lookback_days}d')
    
    # Add the correlation columns to output if they are used
    if use_vix_asset_corr and 'vix_asset_corr' in strategy_data.columns:
        columns_to_include.append('vix_asset_corr')
    
    if use_vvix_asset_corr and 'vvix_asset_corr' in strategy_data.columns:
        columns_to_include.append('vvix_asset_corr')
    
    if use_vix_qqq_corr and 'vix_qqq_corr' in strategy_data.columns:
        columns_to_include.append('vix_qqq_corr')
    
    if use_vvix_qqq_corr and 'vvix_qqq_corr' in strategy_data.columns:
        columns_to_include.append('vvix_qqq_corr')
    
    # Add the dividend columns if dividends are included
    if include_dividends:
        columns_to_include.extend([f'{asset_name}_Dividends', 'QQQ_Dividends', 'dividend_payment'])
    
    # Add PnL and cumulative PnL columns
    columns_to_include.extend([
        f'{asset_name}_pnl_daily', 'QQQ_pnl_daily', 'total_pnl_daily',
        f'{asset_name}_Cum_PnL', 'QQQ_Cum_PnL', 'Total_Cum_PnL',
        f'{asset_name}_value', 'QQQ_value'
    ])
    
    portfolio_results = strategy_data[columns_to_include].copy()
    
    return strategy_data, portfolio_results, metrics 