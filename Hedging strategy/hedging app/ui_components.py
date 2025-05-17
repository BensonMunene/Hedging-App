import streamlit as st
import pandas as pd

# Define custom CSS styles
def load_css():
    """Load custom CSS styles"""
    st.markdown("""
    <style>
        /* Radio button styling to match screenshot */
        div.row-widget.stRadio > div {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        div.row-widget.stRadio > div > label {
            background-color: white;
            border: 1px solid #ccc;
            border-radius: 50%;
            padding: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            width: 20px;
            height: 20px;
            cursor: pointer;
        }
        
        div.row-widget.stRadio > div [data-baseweb="radio"] {
            display: none;
        }
        
        div.row-widget.stRadio > div [data-baseweb="radio"][aria-checked="true"] + label {
            background-color: #ff5c5c;
            color: white;
            border-color: #ff5c5c;
        }
        
        /* Original CSS styles */
        .main-header {
            font-size: 2.5rem;
            color: #1E88E5;
            text-align: center;
            margin-bottom: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #f0f2f6;
        }
        .section-header {
            font-size: 1.8rem;
            color: #0D47A1;
            margin-top: 1.5rem;
            margin-bottom: 0.8rem;
            padding-bottom: 0.3rem;
        }
        .description-text {
            font-size: 1rem;
            color: #424242;
            margin-bottom: 1rem;
        }
        .explanation-text {
            font-size: 1rem;
            color: #424242;
            margin: 1rem 0;
            padding: 0.8rem;
            background-color: #f8f9fa;
            border-left: 3px solid #1E88E5;
            border-radius: 0 3px 3px 0;
        }
        .select-text {
            font-weight: 500;
            color: #0D47A1;
            margin-bottom: 10px;
        }
        .stDataFrame {
            max-height: 300px;
            overflow-y: auto !important;
        }
        
        /* Other styling */
        .subsection {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            border: 1px solid #e0e0e0;
        }
        
        .correlation-value {
            font-size: 24px;
            font-weight: bold;
            color: #1f77b4;
        }
        
        .correlation-label {
            font-size: 16px;
            color: #666;
        }
        
        /* Add direct container for the analysis sections */
        .analysis-container {
            margin-top: 0.5rem;
        }
        
        /* Style for number input */
        div[data-testid="stNumberInput"] > div {
            display: flex;
            align-items: center;
        }
        
        div[data-testid="stNumberInput"] input {
            text-align: center;
            font-size: 1rem;
            padding: 0.25rem;
            width: 60px;
        }
        
        div[data-testid="stNumberInput"] button {
            background-color: #f0f2f6;
            border: 1px solid #ddd;
            cursor: pointer;
            height: 28px;
            width: 28px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        div[data-testid="stNumberInput"] button:hover {
            background-color: #e0e2e6;
        }
        
        /* Align input columns */
        .input-container {
            display: flex;
            align-items: flex-start;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        
        .input-item {
            flex: 1;
        }
        
        /* Styling for Backtest UI */
        .stRadio > div[role="radiogroup"] > label {
            margin-bottom: 8px;
            font-weight: normal;
        }
        
        /* Override radio button styling for backtesting */
        .backtest-section div.row-widget.stRadio > div [data-baseweb="radio"][aria-checked="true"] + label {
            background-color: #e55c5c; /* Slightly darker red */
            color: white;
            border-color: #e55c5c;
        }
        
        /* Style for section headers in backtesting */
        .backtest-section .stSubheader {
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 15px;
            color: #333;
        }
        
        /* Style for date inputs */
        .date-input-container {
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
        }
        
        /* Styling for date input fields */
        div[data-testid="stDateInput"] input {
            background-color: #f8f9fa;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            padding: 8px 12px;
            font-size: 0.9rem;
            width: 100%;
        }
        
        /* Style for info message in backtesting */
        .backtest-section .stAlert {
            margin-top: 20px;
            padding: 10px 15px;
            border-radius: 6px;
        }
        
        /* Change default background color of info message */
        .backtest-section .stAlert.st-ae {
            background-color: #e7f1ff;
            border-left-color: #4a86e8;
        }
        
        /* Full range message styling */
        .full-range-message {
            background-color: #f0f5ff; 
            padding: 15px; 
            border-radius: 5px; 
            margin-top: 10px;
            color: #3a6cb7;
            font-size: 0.9rem;
        }
        
        /* Add space between backtesting controls */
        .backtest-column {
            padding: 0 10px;
        }
        
        /* Style for checkbox inputs - make them more compact */
        .backtest-section .stCheckbox {
            margin-bottom: 4px;
            padding-top: 2px;
            padding-bottom: 2px;
        }
        
        /* Style the checkbox label */
        .backtest-section .stCheckbox label {
            font-size: 0.9rem;
            margin-bottom: 0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        /* Make columns in the asset selection section more compact */
        .backtest-section [data-testid="column"] {
            padding: 0 5px;
        }
        
        /* Highlight selected checkbox */
        .backtest-section .stCheckbox [data-baseweb="checkbox"][data-checked="true"] {
            background-color: #f0f5ff;
            border-radius: 4px;
        }
    </style>
    """, unsafe_allow_html=True)

def main_header(title="Multi-Leg Hedging Strategy Analysis Tool"):
    """Display the main header"""
    st.markdown(f'<div class="main-header">{title}</div>', unsafe_allow_html=True)

def section_header(title):
    """Display a section header"""
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

def description_text(text):
    """Display description text"""
    st.markdown(f'<div class="description-text">{text}</div>', unsafe_allow_html=True)

def explanation_text(text):
    """Display explanation text with special styling"""
    st.markdown(f'<div class="explanation-text">{text}</div>', unsafe_allow_html=True)

def select_text(text):
    """Display selection prompt text"""
    st.markdown(f'<div class="select-text">{text}</div>', unsafe_allow_html=True)

def subsection_start():
    """Start a styled subsection"""
    st.markdown('<div class="subsection">', unsafe_allow_html=True)

def subsection_end():
    """End a styled subsection"""
    st.markdown('</div>', unsafe_allow_html=True)

def input_container_start():
    """Start an input container for aligned inputs"""
    st.markdown('<div class="input-container">', unsafe_allow_html=True)

def input_container_end():
    """End an input container"""
    st.markdown('</div>', unsafe_allow_html=True)

def input_item_start():
    """Start an input item within the container"""
    st.markdown('<div class="input-item">', unsafe_allow_html=True)

def input_item_end():
    """End an input item"""
    st.markdown('</div>', unsafe_allow_html=True)

def backtest_section_start():
    """Start the backtest section styling"""
    st.markdown('<div class="backtest-section">', unsafe_allow_html=True)

def backtest_section_end():
    """End the backtest section styling"""
    st.markdown('</div>', unsafe_allow_html=True)

def backtest_column_start():
    """Start a backtest column with specific styling"""
    st.markdown('<div class="backtest-column">', unsafe_allow_html=True)

def backtest_column_end():
    """End a backtest column"""
    st.markdown('</div>', unsafe_allow_html=True)

def full_range_message():
    """Display a message for using full data range"""
    st.markdown("""
    <div class="full-range-message">
        Full available data range will be used for each asset
    </div>
    """, unsafe_allow_html=True)

def date_input_container_start():
    """Start a date input container"""
    st.markdown('<div class="date-input-container">', unsafe_allow_html=True)

def date_input_container_end():
    """End a date input container"""
    st.markdown('</div>', unsafe_allow_html=True)

def horizontal_rule():
    """Add a horizontal separator"""
    st.markdown("""
    <hr style="height:3px;border:none;color:#333;background-color:#f0f0f0;margin:30px 0;">
    """, unsafe_allow_html=True)

def coming_soon_placeholder():
    """Display a 'coming soon' placeholder for future features"""
    st.markdown("""
    <div style="
        display: flex;
        justify-content: center;
        align-items: center;
        height: 350px;
        background-color: #f8f9fa;
        border-radius: 10px;
        border: 1px dashed #cccccc;
        color: #666666;
        font-size: 1.5rem;
        font-style: italic;
        text-align: center;
        padding: 20px;
    ">
        <div>
            <span style="font-size: 3rem; display: block; margin-bottom: 15px;">🔜</span>
            Coming Soon
        </div>
    </div>
    """, unsafe_allow_html=True)

def asset_selector(assets, default_index=0, key=None, label="Select asset:"):
    """
    Create a horizontal radio button asset selector
    
    Parameters:
    -----------
    assets: list
        List of asset names to choose from
    default_index: int
        Index of the default selected asset
    key: str or None
        Optional key for the radio button
    label: str
        Label text for the radio button
        
    Returns:
    --------
    str
        Selected asset name
    """
    select_text(label)
    return st.radio(
        label,
        options=assets,
        index=default_index,
        horizontal=True,
        key=key,
        label_visibility="collapsed"
    )

def window_size_input(min_value=1, max_value=150, default_value=30, step=1, key=None):
    """
    Create a number input for window size selection
    
    Parameters:
    -----------
    min_value: int
        Minimum window size
    max_value: int
        Maximum window size
    default_value: int
        Default window size
    step: int
        Step size for the input
    key: str or None
        Optional key for the input
        
    Returns:
    --------
    int
        Selected window size
    """
    select_text("Select window size:")
    return st.number_input(
        "Rolling window size (trading days):",
        min_value=min_value,
        max_value=max_value,
        value=default_value,
        step=step,
        key=key,
        label_visibility="collapsed"
    ) 