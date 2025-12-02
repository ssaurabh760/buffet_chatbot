"""
AppleBee - Warren Buffett Investment Advisor
Complete Financial Analysis Dashboard + AI Chatbot
"""

import streamlit as st
import tensorflow as tf
from helper import (
    PositionalEncoding, 
    MultiHeadAttentionLayer,
    create_padding_mask,
    create_look_ahead_mask,
    scaled_dot_product_attention,
    add_pos_enc, 
    predict,
    get_tokenizer
)
from financial_analyzer import FinancialAnalyzer, get_buffett_ratio_info
import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🦉 Warren Buffett Investment Advisor",
    page_icon="🦉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown(
    """
    <style>
    .metric-pass { color: #28a745; font-weight: bold; }
    .metric-fail { color: #dc3545; font-weight: bold; }
    .metric-neutral { color: #ffc107; font-weight: bold; }
    .section-header { 
        font-size: 20px; 
        font-weight: bold; 
        color: #1f77b4;
        margin-top: 20px;
        margin-bottom: 10px;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 10px;
    }
    .chat-user {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border-left: 4px solid #2196F3;
    }
    .chat-assistant {
        background-color: #f3e5f5;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border-left: 4px solid #9c27b0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.title("📊 AppleBee Dashboard")
st.sidebar.markdown("---")
stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, MSFT, BRK.B)", value="AAPL").upper()

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📚 Warren Buffett's Investment Criteria

**Income Statement:**
- Gross Margin ≥ 40%
- SG&A Expense ≤ 30%
- R&D Expense ≤ 30%
- Depreciation ≤ 10%
- Interest Expense ≤ 15%
- Net Profit Margin ≥ 20%
- EPS Growth > 1.0

**Balance Sheet:**
- Cash > Debt
- Debt-to-Equity < 0.80
""")

# Load model with caching
@st.cache_resource
def load_model():
    """Load the trained transformer model."""
    try:
        keras_path = './models/buffett_model.keras'
        
        if os.path.exists(keras_path):
            model = tf.keras.models.load_model(
                keras_path,
                compile=False,
                custom_objects={
                    'PositionalEncoding': PositionalEncoding,
                    'MultiHeadAttentionLayer': MultiHeadAttentionLayer,
                    'create_padding_mask': create_padding_mask,
                    'create_look_ahead_mask': create_look_ahead_mask,
                    'scaled_dot_product_attention': scaled_dot_product_attention,
                    'add_pos_enc': add_pos_enc,
                }
            )
            return model
        else:
            st.error("❌ Model file not found!")
            st.stop()
            
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

# Initialize session state
if "model" not in st.session_state:
    with st.spinner("🦉 Loading Warren Buffett AI..."):
        st.session_state.model = load_model()
        st.session_state.tokenizer = get_tokenizer()

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Helper functions
def get_pass_fail_symbol(value, threshold, compare_type="greater"):
    """Returns ✓ or ✗ symbol"""
    if value is None or pd.isna(value):
        return "⚠"
    
    if compare_type == "greater":
        return "✓" if value >= threshold else "✗"
    elif compare_type == "less":
        return "✓" if value <= threshold else "✗"
    return "⚠"

def display_income_statement_metrics(metrics):
    """Display Income Statement Analysis"""
    st.markdown('<div class="section-header">📋 Income Statement Analysis</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        value = metrics.get('Gross Margin', {}).get('value')
        status = get_pass_fail_symbol(value, 40, "greater")
        st.metric("Gross Margin", f"{value:.2f}%" if value else "N/A", f"{status} Target: ≥40%")
    
    with col2:
        value = metrics.get('SGA Margin', {}).get('value')
        status = get_pass_fail_symbol(value, 30, "less")
        st.metric("SG&A Expense Margin", f"{value:.2f}%" if value else "N/A", f"{status} Target: ≤30%")
    
    with col3:
        value = metrics.get('R&D Margin', {}).get('value')
        status = get_pass_fail_symbol(value, 30, "less")
        st.metric("R&D Expense Margin", f"{value:.2f}%" if value else "N/A", f"{status} Target: ≤30%")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        value = metrics.get('Depreciation Margin', {}).get('value')
        status = get_pass_fail_symbol(value, 10, "less")
        st.metric("Depreciation Margin", f"{value:.2f}%" if value else "N/A", f"{status} Target: ≤10%")
    
    with col2:
        value = metrics.get('Interest Margin', {}).get('value')
        status = get_pass_fail_symbol(value, 15, "less")
        st.metric("Interest Expense Margin", f"{value:.2f}%" if value else "N/A", f"{status} Target: ≤15%")
    
    with col3:
        value = metrics.get('Tax Rate', {}).get('value')
        st.metric("Tax Rate", f"{value:.2f}%" if value else "N/A", "ℹ Corporate tax baseline")
    
    col1, col2 = st.columns(2)
    
    with col1:
        value = metrics.get('Net Margin', {}).get('value')
        status = get_pass_fail_symbol(value, 20, "greater")
        st.metric("Net Profit Margin", f"{value:.2f}%" if value else "N/A", f"{status} Target: ≥20%")
    
    with col2:
        value = metrics.get('EPS Growth', {}).get('value')
        status = get_pass_fail_symbol(value, 1.0, "greater")
        st.metric("EPS Growth (YoY)", f"{value:.4f}" if value else "N/A", f"{status} Target: >1.0")

def display_balance_sheet_metrics(metrics):
    """Display Balance Sheet Analysis"""
    st.markdown('<div class="section-header">🏦 Balance Sheet Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        de_ratio = metrics.get('Debt to Equity', {}).get('value')
        status = get_pass_fail_symbol(de_ratio, 0.80, "less")
        st.metric("Debt-to-Equity Ratio", f"{de_ratio:.2f}" if de_ratio else "N/A", f"{status} Target: <0.80")
    
    with col2:
        cash_vs_debt = metrics.get('Cash vs Debt', {})
        if cash_vs_debt.get('value') is not None:
            status = "✓" if cash_vs_debt.get('value') else "✗"
            st.metric("Cash vs Total Debt", f"{status}", 
                     f"Cash: ${cash_vs_debt.get('cash', 0)/1e9:.2f}B | Debt: ${cash_vs_debt.get('debt', 0)/1e9:.2f}B")
        else:
            st.metric("Cash vs Total Debt", "N/A", "Insufficient data")

def display_cashflow_metrics(metrics):
    """Display Cash Flow Analysis"""
    st.markdown('<div class="section-header">💰 Cash Flow Analysis</div>', unsafe_allow_html=True)
    
    capex = metrics.get('CapEx Margin', {}).get('value')
    st.metric("CapEx Margin", f"{capex:.2f}%" if capex else "N/A", 
             "ℹ Lower is better (less reinvestment needed)")

def display_summary_scorecard(metrics):
    """Display Overall Investment Scorecard"""
    st.markdown('<div class="section-header">🎯 Warren Buffett Scorecard</div>', unsafe_allow_html=True)
    
    criteria = {
        "Gross Margin ≥ 40%": (metrics.get('Gross Margin', {}).get('value'), 40, "greater"),
        "SG&A Expense ≤ 30%": (metrics.get('SGA Margin', {}).get('value'), 30, "less"),
        "R&D Expense ≤ 30%": (metrics.get('R&D Margin', {}).get('value'), 30, "less"),
        "Depreciation ≤ 10%": (metrics.get('Depreciation Margin', {}).get('value'), 10, "less"),
        "Interest Expense ≤ 15%": (metrics.get('Interest Margin', {}).get('value'), 15, "less"),
        "Net Profit Margin ≥ 20%": (metrics.get('Net Margin', {}).get('value'), 20, "greater"),
        "EPS Growing (> 1.0)": (metrics.get('EPS Growth', {}).get('value'), 1.0, "greater"),
        "Debt-to-Equity < 0.80": (metrics.get('Debt to Equity', {}).get('value'), 0.80, "less"),
    }
    
    scorecard_data = []
    passed = 0
    total = 0
    
    for criterion, (value, threshold, comp_type) in criteria.items():
        if value is not None:
            total += 1
            if comp_type == "greater":
                passes = value >= threshold
            else:
                passes = value <= threshold
            
            if passes:
                passed += 1
            
            status = "✓ PASS" if passes else "✗ FAIL"
            
            scorecard_data.append({
                "Criterion": criterion,
                "Value": f"{value:.2f}" if abs(value) < 10 else f"{value:.4f}",
                "Target": f"{threshold:.0f}" if abs(threshold) < 10 else f"{threshold:.2f}",
                "Status": status
            })
    
    scorecard_df = pd.DataFrame(scorecard_data)
    st.dataframe(scorecard_df, use_container_width=True, hide_index=True)
    
    if total > 0:
        st.markdown(f"**Score: {passed}/{total} criteria met ({int(passed/total*100)}%)**")

# MAIN HEADER
st.title("🦉 Warren Buffett Investment Advisor")
st.markdown("*Complete financial analysis + AI investment education*")
st.markdown("---")

# Create tabs
tab_dashboard, tab_chatbot = st.tabs(["📊 Financial Dashboard", "💬 Investment Chatbot"])

# ===== DASHBOARD TAB =====
with tab_dashboard:
    if stock_symbol:
        with st.spinner(f"🔍 Fetching data for {stock_symbol}..."):
            analyzer = FinancialAnalyzer(stock_symbol)
            metrics = analyzer.calculate_ratios()
        
        if metrics and len(metrics) > 0:
            # Company info
            info = analyzer.get_company_info()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Company", info.get('name', 'N/A'))
            with col2:
                st.metric("Sector", info.get('sector', 'N/A'))
            with col3:
                st.metric("P/E Ratio", f"{info.get('pe_ratio', 'N/A'):.2f}" if isinstance(info.get('pe_ratio'), (int, float)) else "N/A")
            with col4:
                st.metric("Current Price", f"${info.get('current_price', 'N/A'):.2f}" if isinstance(info.get('current_price'), (int, float)) else "N/A")
            
            st.markdown("---")
            
            # Display analyses
            display_income_statement_metrics(metrics)
            st.markdown("")
            display_balance_sheet_metrics(metrics)
            st.markdown("")
            display_cashflow_metrics(metrics)
            st.markdown("")
            display_summary_scorecard(metrics)
            
            # Financial statements
            st.markdown("---")
            st.markdown('<div class="section-header">📄 Financial Statements</div>', unsafe_allow_html=True)
            
            tabs_fs = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
            
            with tabs_fs[0]:
                income = analyzer.get_income_statement()
                if not income.empty:
                    st.dataframe(income.iloc[:, :5], use_container_width=True)
                else:
                    st.info("Income statement data not available")
            
            with tabs_fs[1]:
                balance = analyzer.get_balance_sheet()
                if not balance.empty:
                    st.dataframe(balance.iloc[:, :5], use_container_width=True)
                else:
                    st.info("Balance sheet data not available")
            
            with tabs_fs[2]:
                cashflow = analyzer.get_cash_flow()
                if not cashflow.empty:
                    st.dataframe(cashflow.iloc[:, :5], use_container_width=True)
                else:
                    st.info("Cash flow statement data not available")
            
            st.session_state.current_metrics = metrics
            st.session_state.current_symbol = stock_symbol
        else:
            st.error("Unable to fetch financial data for this ticker. Please try a different stock.")
    else:
        st.info("👈 Enter a stock symbol in the sidebar to analyze")

# ===== CHATBOT TAB =====
with tab_chatbot:
    st.markdown("### 💬 Ask Warren Buffett About Investing")
    st.markdown("*Ask questions about financial metrics, investment philosophy, and value investing*")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("**Chat History**")
    with col2:
        if st.button("🔄 Clear"):
            st.session_state.conversation_history = []
            st.rerun()
    
    # Display conversation
    if st.session_state.conversation_history:
        for msg in st.session_state.conversation_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user"><b>You:</b> {msg["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="chat-assistant"><b>Warren Buffett:</b> {msg["content"]}</div>',
                    unsafe_allow_html=True
                )
    else:
        st.info("💬 No conversation yet. Ask Warren Buffett something!")
    
    # Input
    st.markdown("---")
    col1, col2 = st.columns([5, 1])
    with col1:
        user_question = st.text_input(
            "Your question:",
            placeholder="e.g., What makes a good company to invest in?",
            label_visibility="collapsed"
        )
    with col2:
        send_button = st.button("📤 Send", use_container_width=True)
    
    # Process question
    if send_button and user_question.strip():
        st.session_state.conversation_history.append({
            "role": "user",
            "content": user_question
        })
        
        with st.spinner("🤔 Warren is thinking..."):
            try:
                response = predict(user_question, st.session_state.model)
                response = str(response).strip()
                
                if not response or response.lower() == "unknown":
                    response = "I appreciate your question. Could you rephrase or ask about another investment topic?"
                
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": response
                })
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Model:** TensorFlow Transformer")
with col2:
    st.markdown("**Data:** yfinance")
with col3:
    st.markdown("**Status:** ✅ Online")

st.markdown("""
---
**Disclaimer:** This app is for educational purposes only. Not financial advice. 
Always conduct your own research or consult with a financial advisor before investing.
""")