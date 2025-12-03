"""
Warren Buffett Investment Advisor Chatbot + Financial Dashboard
Complete AppleBee Application
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
import plotly.graph_objects as go

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
    .ratio-good {
        background-color: #c8e6c9;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #4caf50;
    }
    .ratio-warning {
        background-color: #fff9c4;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #fbc02d;
    }
    .ratio-bad {
        background-color: #ffcdd2;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #f44336;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load model with caching
@st.cache_resource
def load_model():
    """Load the trained transformer model."""
    try:
        keras_path = './models/model.h5'
        
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

if "current_ticker" not in st.session_state:
    st.session_state.current_ticker = None

if "current_ratios" not in st.session_state:
    st.session_state.current_ratios = None

# Header
st.markdown("# 🦉 Warren Buffett Investment Advisor")
st.markdown("*Powered by AI + Financial Analysis*")

# Navigation
tab1, tab2 = st.tabs(["💬 Investment Chatbot", "📊 Financial Dashboard"])

# ============================================================================
# TAB 1: CHATBOT
# ============================================================================
with tab1:
    st.markdown("### 💬 Ask Warren Buffett About Investing")
    
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

# ============================================================================
# TAB 2: FINANCIAL DASHBOARD
# ============================================================================
with tab2:
    st.markdown("### 📊 Financial Analysis Dashboard")
    st.markdown("*Analyze stocks using Warren Buffett's financial criteria*")
    
    # Stock input
    col1, col2 = st.columns([3, 1])
    with col1:
        ticker = st.text_input(
            "Enter stock ticker:",
            placeholder="e.g., AAPL, MSFT, BRK.B",
            label_visibility="collapsed"
        ).upper()
    with col2:
        analyze_button = st.button("🔍 Analyze", use_container_width=True)
    
    if analyze_button and ticker:
        with st.spinner(f"📊 Analyzing {ticker}..."):
            try:
                analyzer = FinancialAnalyzer(ticker)
                st.session_state.current_ticker = ticker
                st.session_state.current_ratios = analyzer.calculate_ratios()
                
                # Company info
                info = analyzer.get_company_info()
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Company", info['name'])
                with col2:
                    st.metric("Sector", info['sector'])
                with col3:
                    st.metric("P/E Ratio", info['pe_ratio'])
                with col4:
                    st.metric("Dividend Yield", info['dividend_yield'])
                
                # Financial Ratios
                st.markdown("---")
                st.markdown("## Warren Buffett Financial Ratios")
                
                if st.session_state.current_ratios:
                    ratio_info = get_buffett_ratio_info()
                    
                    for ratio_name, ratio_data in st.session_state.current_ratios.items():
                        if ratio_name == 'Current EPS':
                            continue
                        
                        value = ratio_data['value']
                        benchmark = ratio_data['benchmark']
                        rule = ratio_data['rule']
                        logic = ratio_data['logic']
                        unit = ratio_data['unit']
                        
                        # Assessment
                        if value is None:
                            assessment = "⚪ Data unavailable"
                            css_class = "ratio-warning"
                        else:
                            assessment = analyzer.get_ratio_assessment(ratio_name, value)
                            if "✅" in assessment:
                                css_class = "ratio-good"
                            elif "⚠️" in assessment:
                                css_class = "ratio-warning"
                            else:
                                css_class = "ratio-bad"
                        
                        # Display
                        with st.expander(f"📈 {ratio_name} {assessment}", expanded=False):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**Current Value:** {value:.2f}{unit}" if value else "N/A")
                                st.write(f"**Benchmark:** {benchmark}{unit if benchmark else ''}")
                                st.write(f"**Rule:** {rule}")
                            
                            with col2:
                                st.write(f"**Buffett's Logic:**")
                                st.write(logic)
                            
                            st.info(f"**Interpretation:** {ratio_info.get(ratio_name, {}).get('interpretation', 'N/A')}")
                
                # Financial Statements
                st.markdown("---")
                st.markdown("## Financial Statements")
                
                tabs_fs = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
                
                with tabs_fs[0]:
                    income = analyzer.get_income_statement()
                    if not income.empty:
                        st.dataframe(income)
                    else:
                        st.warning("Income statement data not available")
                
                with tabs_fs[1]:
                    balance = analyzer.get_balance_sheet()
                    if not balance.empty:
                        st.dataframe(balance)
                    else:
                        st.warning("Balance sheet data not available")
                
                with tabs_fs[2]:
                    cashflow = analyzer.get_cash_flow()
                    if not cashflow.empty:
                        st.dataframe(cashflow)
                    else:
                        st.warning("Cash flow statement data not available")
                
            except Exception as e:
                st.error(f"❌ Error analyzing {ticker}: {str(e)}")
                st.info("Make sure the ticker symbol is valid (e.g., AAPL, MSFT, BRK.B)")
    
    # Info sidebar
    with st.sidebar:
        st.markdown("### 📚 About Buffett Ratios")
        st.markdown("""
        These 7 key financial ratios help identify companies with:
        - **Pricing Power** (high gross margins)
        - **Efficient Operations** (low overhead)
        - **Asset-Light Models** (low depreciation)
        - **Financial Strength** (low debt)
        - **Profitability** (high net margins)
        
        **Buffett's Philosophy:**
        Look for companies that don't need to compete on price and generate strong, growing profits year after year.
        """)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Model:** TensorFlow Transformer")
with col2:
    st.markdown("**Data:** yfinance")
with col3:
    st.markdown("**Status:** ✅ Online")