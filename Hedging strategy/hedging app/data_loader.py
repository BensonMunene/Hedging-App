import os
import streamlit as st
import pandas as pd

# Function to load assets
@st.cache_data
def load_assets():
    # Define the specific assets we want to load
    assets = ['YMAX', 'JEPQ', 'QQQ', 'QQQI', 'QYLD', 'YMAG', 'PBP', 'QDTY', 'QYLG', 'FEPI', 'GPIQ', 'IQQQ', 'FTQI', 'VIX', 'VVIX']
    
    # Initialize a dictionary to store data for each asset
    dfs = {}
    
    # Load each asset's data
    for asset in assets:
        try:
            # Try to load the file
            file_path = f"{asset}.csv"
            if os.path.exists(file_path):
                # Read the CSV
                df = pd.read_csv(file_path)
                
                # Keep only Date and Close columns
                if 'Date' in df.columns and 'Close' in df.columns:
                    df = df[['Date', 'Close']]
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.rename(columns={'Close': asset}, inplace=True)
                    dfs[asset] = df
            else:
                st.warning(f"{file_path} not found")
        except Exception as e:
            st.warning(f"Error loading {asset}: {e}")
    
    # If no data was loaded, return None
    if not dfs:
        return None
    
    # Merge all dataframes on Date
    merged_df = None
    for asset, df in dfs.items():
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on='Date', how='outer')
    
    # Sort by date
    if merged_df is not None:
        merged_df.sort_values('Date', inplace=True)
    
    return merged_df, dfs

# Function to load dividend data for assets
@st.cache_data
def load_dividend_data():
    # Define the specific assets we want to load dividends for
    assets = ['YMAX', 'JEPQ', 'QQQ', 'QQQI', 'QYLD', 'YMAG', 'PBP', 'QDTY', 'QYLG', 'FEPI', 'GPIQ', 'IQQQ', 'FTQI']
    
    # Initialize a dictionary to store dividend data for each asset
    dividend_dfs = {}
    
    # Load each asset's dividend data
    for asset in assets:
        try:
            # Try to load the dividend file using the pattern shown in the screenshot
            file_path = f"{asset} Dividends.csv"
            if os.path.exists(file_path):
                # Read the CSV
                df = pd.read_csv(file_path)
                
                # Process the dividend data
                if 'Date' in df.columns and 'Dividends' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.rename(columns={'Dividends': f'{asset}_Dividends'}, inplace=True)
                    dividend_dfs[asset] = df
            else:
                # Try alternative format just in case
                file_path_alt = f"{asset}_Dividends.csv"
                if os.path.exists(file_path_alt):
                    df = pd.read_csv(file_path_alt)
                    if 'Date' in df.columns and 'Dividends' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'])
                        df.rename(columns={'Dividends': f'{asset}_Dividends'}, inplace=True)
                        dividend_dfs[asset] = df
        except Exception as e:
            st.warning(f"Error loading {asset} dividend data: {e}")
    
    # Merge all dividend dataframes on Date
    merged_dividend_df = None
    for asset, df in dividend_dfs.items():
        if merged_dividend_df is None:
            merged_dividend_df = df
        else:
            merged_dividend_df = pd.merge(merged_dividend_df, df, on='Date', how='outer')
    
    # Sort by date
    if merged_dividend_df is not None:
        merged_dividend_df.sort_values('Date', inplace=True)
    
    return merged_dividend_df, dividend_dfs