import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AppleBee - Warren Buffett Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
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
</style>
""", unsafe_allow_html=True)

# Sidebar - User Input
st.sidebar.title("📊 AppleBee Dashboard")
st.sidebar.markdown("---")
stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, MSFT, TSLA)", value="AAPL").upper()

st.sidebar.markdown("---")
st.sidebar.markdown("""
### Warren Buffett's Investment Criteria

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
- No Preferred Stock
- Growing Retained Earnings

**Cash Flow:**
- CapEx Margin (lower is better)
""")

# Helper functions
def get_pass_fail_color(value, threshold, compare_type="greater"):
    """Returns HTML class for pass/fail styling"""
    if value is None or pd.isna(value):
        return "metric-neutral"
    
    if compare_type == "greater":
        return "metric-pass" if value >= threshold else "metric-fail"
    elif compare_type == "less":
        return "metric-pass" if value <= threshold else "metric-fail"
    return "metric-neutral"

def get_pass_fail_symbol(value, threshold, compare_type="greater"):
    """Returns ✓ or ✗ symbol"""
    if value is None or pd.isna(value):
        return "⚠"
    
    if compare_type == "greater":
        return "✓" if value >= threshold else "✗"
    elif compare_type == "less":
        return "✓" if value <= threshold else "✗"
    return "⚠"

def fetch_stock_data(symbol):
    """Fetch financial data from yfinance"""
    try:
        stock = yf.Ticker(symbol)
        return stock
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def calculate_metrics(stock):
    """Calculate all Warren Buffett metrics"""
    try:
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        cashflow = stock.cashflow
        
        if financials.empty or balance_sheet.empty or cashflow.empty:
            return None, None, None
        
        # Get the most recent year (first column)
        metrics = {}
        
        # ===== INCOME STATEMENT METRICS =====
        try:
            gross_profit = financials.loc['Gross Profit'].iloc[0]
            total_revenue = financials.loc['Total Revenue'].iloc[0]
            metrics['Gross Margin'] = gross_profit / total_revenue if total_revenue != 0 else 0
        except:
            metrics['Gross Margin'] = np.nan
        
        try:
            sga = financials.loc['Selling General And Administration'].iloc[0]
            metrics['SG&A Margin'] = sga / gross_profit if gross_profit != 0 else 0
        except:
            metrics['SG&A Margin'] = np.nan
        
        try:
            rnd = financials.loc['Research And Development'].iloc[0]
            metrics['R&D Margin'] = rnd / gross_profit if gross_profit != 0 else 0
        except:
            metrics['R&D Margin'] = np.nan
        
        try:
            depreciation = financials.loc['Reconciled Depreciation'].iloc[0]
            metrics['Depreciation Margin'] = depreciation / gross_profit if gross_profit != 0 else 0
        except:
            metrics['Depreciation Margin'] = np.nan
        
        try:
            interest_expense = financials.loc['Interest Expense'].iloc[0]
            operating_income = financials.loc['Operating Income'].iloc[0]
            metrics['Interest Expense Margin'] = interest_expense / operating_income if operating_income != 0 else 0
        except:
            metrics['Interest Expense Margin'] = np.nan
        
        try:
            tax_provision = financials.loc['Tax Provision'].iloc[0]
            pretax_income = financials.loc['Pretax Income'].iloc[0]
            metrics['Tax Rate'] = tax_provision / pretax_income if pretax_income != 0 else 0
        except:
            metrics['Tax Rate'] = np.nan
        
        try:
            net_income = financials.loc['Net Income'].iloc[0]
            metrics['Net Profit Margin'] = net_income / total_revenue if total_revenue != 0 else 0
        except:
            metrics['Net Profit Margin'] = np.nan
        
        try:
            # EPS Growth (compare current year to previous year)
            if len(financials.columns) > 1:
                eps_current = financials.loc['Basic EPS'].iloc[0]
                eps_previous = financials.loc['Basic EPS'].iloc[1]
                metrics['EPS Growth'] = eps_current / eps_previous if eps_previous != 0 else 0
            else:
                metrics['EPS Growth'] = np.nan
        except:
            metrics['EPS Growth'] = np.nan
        
        # ===== BALANCE SHEET METRICS =====
        try:
            cash = balance_sheet.loc['Cash And Cash Equivalents'].iloc[0]
            metrics['Cash'] = cash
        except:
            metrics['Cash'] = np.nan
        
        try:
            current_debt = balance_sheet.loc['Current Debt'].iloc[0]
            metrics['Current Debt'] = current_debt
        except:
            try:
                short_term_debt = balance_sheet.loc['Current Portion Of Long Term Debt'].iloc[0]
                metrics['Current Debt'] = short_term_debt
            except:
                metrics['Current Debt'] = np.nan
        
        try:
            total_debt = balance_sheet.loc['Total Debt'].iloc[0]
            metrics['Total Debt'] = total_debt
        except:
            metrics['Total Debt'] = np.nan
        
        try:
            total_assets = balance_sheet.loc['Total Assets'].iloc[0]
            total_liabilities = balance_sheet.loc['Total Liabilities'].iloc[0]
            stockholder_equity = balance_sheet.loc['Stockholders Equity'].iloc[0]
            
            # Adjusted Debt-to-Equity
            metrics['Debt to Equity'] = total_liabilities / stockholder_equity if stockholder_equity != 0 else 0
        except:
            metrics['Debt to Equity'] = np.nan
        
        try:
            preferred_stock = balance_sheet.loc['Preferred Stock'].iloc[0]
            metrics['Preferred Stock'] = preferred_stock
        except:
            metrics['Preferred Stock'] = np.nan
        
        try:
            retained_earnings = balance_sheet.loc['Retained Earnings'].iloc[0]
            metrics['Retained Earnings'] = retained_earnings
        except:
            metrics['Retained Earnings'] = np.nan
        
        try:
            treasury_stock = balance_sheet.loc['Treasury Stock'].iloc[0]
            metrics['Treasury Stock'] = treasury_stock
        except:
            metrics['Treasury Stock'] = np.nan
        
        # ===== CASH FLOW METRICS =====
        try:
            capex = cashflow.loc['Capital Expenditure'].iloc[0]
            net_income_ops = financials.loc['Net Income From Continuing Operations'].iloc[0]
            metrics['CapEx Margin'] = abs(capex) / net_income_ops if net_income_ops != 0 else 0
        except:
            metrics['CapEx Margin'] = np.nan
        
        try:
            operating_cash_flow = cashflow.loc['Operating Cash Flow'].iloc[0]
            metrics['Operating Cash Flow'] = operating_cash_flow
        except:
            metrics['Operating Cash Flow'] = np.nan
        
        return metrics, financials, balance_sheet
    
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        return None, None, None

def display_income_statement_metrics(metrics):
    """Display Income Statement Analysis"""
    st.markdown('<div class="section-header">📋 Income Statement Analysis</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        value = metrics.get('Gross Margin', np.nan)
        status = get_pass_fail_symbol(value, 0.40, "greater")
        st.metric(
            "Gross Margin",
            f"{value*100:.2f}%" if not pd.isna(value) else "N/A",
            f"{status} Target: ≥40%"
        )
    
    with col2:
        value = metrics.get('SG&A Margin', np.nan)
        status = get_pass_fail_symbol(value, 0.30, "less")
        st.metric(
            "SG&A Expense Margin",
            f"{value*100:.2f}%" if not pd.isna(value) else "N/A",
            f"{status} Target: ≤30%"
        )
    
    with col3:
        value = metrics.get('R&D Margin', np.nan)
        status = get_pass_fail_symbol(value, 0.30, "less")
        st.metric(
            "R&D Expense Margin",
            f"{value*100:.2f}%" if not pd.isna(value) else "N/A",
            f"{status} Target: ≤30%"
        )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        value = metrics.get('Depreciation Margin', np.nan)
        status = get_pass_fail_symbol(value, 0.10, "less")
        st.metric(
            "Depreciation Margin",
            f"{value*100:.2f}%" if not pd.isna(value) else "N/A",
            f"{status} Target: ≤10%"
        )
    
    with col2:
        value = metrics.get('Interest Expense Margin', np.nan)
        status = get_pass_fail_symbol(value, 0.15, "less")
        st.metric(
            "Interest Expense Margin",
            f"{value*100:.2f}%" if not pd.isna(value) else "N/A",
            f"{status} Target: ≤15%"
        )
    
    with col3:
        value = metrics.get('Tax Rate', np.nan)
        st.metric(
            "Tax Rate",
            f"{value*100:.2f}%" if not pd.isna(value) else "N/A",
            "ℹ Corporate tax baseline"
        )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        value = metrics.get('Net Profit Margin', np.nan)
        status = get_pass_fail_symbol(value, 0.20, "greater")
        st.metric(
            "Net Profit Margin",
            f"{value*100:.2f}%" if not pd.isna(value) else "N/A",
            f"{status} Target: ≥20%"
        )
    
    with col2:
        value = metrics.get('EPS Growth', np.nan)
        status = get_pass_fail_symbol(value, 1.0, "greater")
        st.metric(
            "EPS Growth (YoY)",
            f"{value:.4f}" if not pd.isna(value) else "N/A",
            f"{status} Target: >1.0 (Growing)"
        )

def display_balance_sheet_metrics(metrics):
    """Display Balance Sheet Analysis"""
    st.markdown('<div class="section-header">🏦 Balance Sheet Analysis</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cash = metrics.get('Cash', np.nan)
        debt = metrics.get('Total Debt', np.nan)
        if not pd.isna(cash) and not pd.isna(debt):
            status = "✓" if cash > debt else "✗"
            st.metric(
                "Cash vs Total Debt",
                f"{status}",
                f"Cash: ${cash/1e9:.2f}B | Debt: ${debt/1e9:.2f}B"
            )
        else:
            st.metric("Cash vs Total Debt", "N/A", "Insufficient data")
    
    with col2:
        value = metrics.get('Debt to Equity', np.nan)
        status = get_pass_fail_symbol(value, 0.80, "less")
        st.metric(
            "Debt-to-Equity Ratio",
            f"{value:.2f}" if not pd.isna(value) else "N/A",
            f"{status} Target: <0.80"
        )
    
    with col3:
        pref_stock = metrics.get('Preferred Stock', np.nan)
        if not pd.isna(pref_stock):
            status = "✓" if pref_stock == 0 else "✗"
            st.metric(
                "Preferred Stock",
                f"{status}",
                "No preferred stock ideal"
            )
        else:
            st.metric("Preferred Stock", "N/A", "Unknown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        retained_earnings = metrics.get('Retained Earnings', np.nan)
        st.metric(
            "Retained Earnings",
            f"${retained_earnings/1e9:.2f}B" if not pd.isna(retained_earnings) else "N/A",
            "ℹ Growing over time is positive"
        )
    
    with col2:
        treasury_stock = metrics.get('Treasury Stock', np.nan)
        if not pd.isna(treasury_stock) and treasury_stock != 0:
            st.metric(
                "Treasury Stock",
                "✓",
                "Company repurchasing stock"
            )
        else:
            st.metric(
                "Treasury Stock",
                "None",
                "ℹ Stock buybacks positive"
            )

def display_cashflow_metrics(metrics):
    """Display Cash Flow Analysis"""
    st.markdown('<div class="section-header">💰 Cash Flow Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        value = metrics.get('CapEx Margin', np.nan)
        st.metric(
            "CapEx Margin",
            f"{value*100:.2f}%" if not pd.isna(value) else "N/A",
            "ℹ Lower is better (less reinvestment needed)"
        )
    
    with col2:
        ocf = metrics.get('Operating Cash Flow', np.nan)
        st.metric(
            "Operating Cash Flow",
            f"${ocf/1e9:.2f}B" if not pd.isna(ocf) else "N/A",
            "ℹ Higher is better"
        )

def display_summary_scorecard(metrics):
    """Display Overall Investment Scorecard"""
    st.markdown('<div class="section-header">🎯 Warren Buffett Scorecard</div>', unsafe_allow_html=True)
    
    criteria = {
        "Gross Margin ≥ 40%": (metrics.get('Gross Margin', np.nan), 0.40, "greater"),
        "SG&A Expense ≤ 30%": (metrics.get('SG&A Margin', np.nan), 0.30, "less"),
        "R&D Expense ≤ 30%": (metrics.get('R&D Margin', np.nan), 0.30, "less"),
        "Depreciation ≤ 10%": (metrics.get('Depreciation Margin', np.nan), 0.10, "less"),
        "Interest Expense ≤ 15%": (metrics.get('Interest Expense Margin', np.nan), 0.15, "less"),
        "Net Profit Margin ≥ 20%": (metrics.get('Net Profit Margin', np.nan), 0.20, "greater"),
        "EPS Growing (> 1.0)": (metrics.get('EPS Growth', np.nan), 1.0, "greater"),
        "Debt-to-Equity < 0.80": (metrics.get('Debt to Equity', np.nan), 0.80, "less"),
    }
    
    scorecard_data = []
    passed = 0
    total = 0
    
    for criterion, (value, threshold, comp_type) in criteria.items():
        if not pd.isna(value):
            total += 1
            if comp_type == "greater":
                passes = value >= threshold
            else:
                passes = value <= threshold
            
            if passes:
                passed += 1
            
            status = "✓ PASS" if passes else "✗ FAIL"
            status_color = "color: green;" if passes else "color: red;"
            
            scorecard_data.append({
                "Criterion": criterion,
                "Value": f"{value*100:.2f}%" if abs(value) < 10 else f"{value:.2f}",
                "Target": f"{threshold*100:.0f}%" if abs(threshold) < 10 else f"{threshold:.2f}",
                "Status": status
            })
    
    scorecard_df = pd.DataFrame(scorecard_data)
    st.dataframe(
        scorecard_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Criterion": st.column_config.TextColumn("Criterion", width="medium"),
            "Value": st.column_config.TextColumn("Current Value", width="small"),
            "Target": st.column_config.TextColumn("Target", width="small"),
            "Status": st.column_config.TextColumn("Status", width="small"),
        }
    )
    
    if total > 0:
        st.markdown(f"**Score: {passed}/{total} criteria met ({int(passed/total*100)}%)**")

# Main content area
st.title("📈 AppleBee - Warren Buffett Stock Analyzer")
st.markdown("""
This dashboard analyzes stocks using Warren Buffett's investment criteria. 
Enter a stock symbol to see how it performs against his key financial metrics.
""")

st.markdown("---")

# Fetch and display data
if stock_symbol:
    with st.spinner(f"Fetching data for {stock_symbol}..."):
        stock = fetch_stock_data(stock_symbol)
    
    if stock is not None:
        metrics, financials, balance_sheet = calculate_metrics(stock)
        
        if metrics is not None:
            # Display company info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Stock Symbol", stock_symbol)
            with col2:
                try:
                    current_price = stock.info.get('currentPrice', 'N/A')
                    st.metric("Current Price", f"${current_price}" if current_price != 'N/A' else "N/A")
                except:
                    st.metric("Current Price", "N/A")
            with col3:
                st.metric("Last Updated", datetime.now().strftime("%Y-%m-%d %H:%M"))
            
            st.markdown("---")
            
            # Display all analyses
            display_income_statement_metrics(metrics)
            st.markdown("")
            display_balance_sheet_metrics(metrics)
            st.markdown("")
            display_cashflow_metrics(metrics)
            st.markdown("")
            display_summary_scorecard(metrics)
            
            # Add tabs for detailed financial statements
            st.markdown("---")
            tab1, tab2, tab3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
            
            with tab1:
                if financials is not None and not financials.empty:
                    st.dataframe(
                        financials.iloc[:, :5],  # Show last 5 years
                        use_container_width=True
                    )
                else:
                    st.info("Income statement data not available")
            
            with tab2:
                if balance_sheet is not None and not balance_sheet.empty:
                    st.dataframe(
                        balance_sheet.iloc[:, :5],  # Show last 5 years
                        use_container_width=True
                    )
                else:
                    st.info("Balance sheet data not available")
            
            with tab3:
                try:
                    cashflow = stock.cashflow
                    if cashflow is not None and not cashflow.empty:
                        st.dataframe(
                            cashflow.iloc[:, :5],  # Show last 5 years
                            use_container_width=True
                        )
                    else:
                        st.info("Cash flow statement data not available")
                except:
                    st.info("Cash flow statement data not available")
        
        else:
            st.error("Unable to calculate metrics. This stock may not have sufficient financial data available.")
else:
    st.info("👈 Enter a stock symbol in the sidebar to begin analysis")

st.markdown("---")
st.markdown("""
**Disclaimer:** This dashboard is for educational purposes only and does not constitute financial advice. 
Always conduct your own research or consult with a financial advisor before making investment decisions.
""")