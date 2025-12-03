"""
AppleBee - Warren Buffett Stock Analysis Dashboard
A Streamlit application for analyzing stocks using Warren Buffett's investment principles
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Try to import yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="AppleBee - Warren Buffett Stock Analyzer",
    page_icon="🐝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-gold: #D4AF37;
        --dark-green: #1B4D3E;
        --cream: #F5F5DC;
        --dark-bg: #0E1117;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #D4AF37;
        text-align: center;
        padding: 1rem;
        border-bottom: 3px solid #D4AF37;
        margin-bottom: 2rem;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #888;
        text-align: center;
        margin-top: -1.5rem;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 4px solid #D4AF37;
        margin-bottom: 1rem;
    }
    
    .metric-pass {
        border-left-color: #00C853 !important;
    }
    
    .metric-fail {
        border-left-color: #FF5252 !important;
    }
    
    .metric-neutral {
        border-left-color: #FFC107 !important;
    }
    
    /* Status badges */
    .status-pass {
        background-color: #00C853;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .status-fail {
        background-color: #FF5252;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .status-neutral {
        background-color: #FFC107;
        color: black;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #D4AF37;
        border-bottom: 2px solid #333;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Info boxes */
    .buffett-quote {
        background: linear-gradient(135deg, #1B4D3E 0%, #0d2818 100%);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #D4AF37;
        font-style: italic;
    }
    
    /* Table styling */
    .dataframe {
        font-size: 0.9rem;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Warren Buffett's ratio thresholds and explanations
BUFFETT_RATIOS = {
    "income_statement": {
        "gross_margin": {
            "name": "Gross Margin",
            "threshold": 0.40,
            "rule": "≥ 40%",
            "comparison": ">=",
            "logic": "Signals the company isn't competing on price. High margins indicate a durable competitive advantage (moat)."
        },
        "sga_margin": {
            "name": "SG&A Expense Margin",
            "threshold": 0.30,
            "rule": "≤ 30%",
            "comparison": "<=",
            "logic": "Wide-moat companies don't need to spend a lot on overhead to operate. Low SG&A relative to gross profit indicates efficiency."
        },
        "rd_margin": {
            "name": "R&D Expense Margin",
            "threshold": 0.30,
            "rule": "≤ 30%",
            "comparison": "<=",
            "logic": "R&D expenses don't always create value for shareholders. Companies overly dependent on R&D may lack sustainable advantages."
        },
        "depreciation_margin": {
            "name": "Depreciation Margin",
            "threshold": 0.10,
            "rule": "≤ 10%",
            "comparison": "<=",
            "logic": "Buffett doesn't like businesses that need to invest heavily in depreciating assets to maintain their competitive advantage."
        },
        "interest_expense_margin": {
            "name": "Interest Expense Margin",
            "threshold": 0.15,
            "rule": "≤ 15%",
            "comparison": "<=",
            "logic": "Great businesses don't need debt to finance themselves. Low interest expense indicates financial strength."
        },
        "tax_rate": {
            "name": "Effective Tax Rate",
            "threshold": 0.21,
            "rule": "~21% (Current Corporate Rate)",
            "comparison": "info",
            "logic": "Great businesses are so profitable that they are forced to pay their full tax load. Very low rates may indicate accounting tricks."
        },
        "net_margin": {
            "name": "Net Margin",
            "threshold": 0.20,
            "rule": "≥ 20%",
            "comparison": ">=",
            "logic": "Great companies convert 20% or more of their revenue into net income. This indicates exceptional business quality."
        },
        "eps_growth": {
            "name": "EPS Growth",
            "threshold": 0,
            "rule": "Positive & Growing",
            "comparison": ">",
            "logic": "Great companies increase profits every year. Consistent EPS growth indicates a sustainable competitive advantage."
        }
    },
    "balance_sheet": {
        "debt_to_equity": {
            "name": "Debt to Equity Ratio",
            "threshold": 0.80,
            "rule": "≤ 80%",
            "comparison": "<=",
            "logic": "Conservative leverage indicates financial stability. Companies with low debt can weather economic storms better."
        },
        "cash_to_debt": {
            "name": "Cash to Debt Ratio",
            "threshold": 1.0,
            "rule": "≥ 1.0x",
            "comparison": ">=",
            "logic": "Companies with more cash than debt have strong financial positions and flexibility for opportunities."
        },
        "retained_earnings_growth": {
            "name": "Retained Earnings Growth",
            "threshold": 0,
            "rule": "Positive & Growing",
            "comparison": ">",
            "logic": "Growing retained earnings indicate the company is reinvesting profits effectively and building shareholder value."
        }
    },
    "cash_flow": {
        "capex_margin": {
            "name": "CapEx to Net Income",
            "threshold": 0.50,
            "rule": "≤ 50%",
            "comparison": "<=",
            "logic": "Companies that don't need heavy capital expenditures to maintain their business generate more free cash flow for shareholders."
        }
    }
}


# Sample data for demonstration (based on actual Apple financials)
SAMPLE_DATA = {
    "AAPL": {
        "info": {
            "longName": "Apple Inc.",
            "symbol": "AAPL",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "currentPrice": 237.33,
            "marketCap": 3580000000000,
            "currency": "USD"
        },
        "income_stmt": pd.DataFrame({
            pd.Timestamp("2024-09-28"): {
                "Total Revenue": 391035000000,
                "Gross Profit": 180683000000,
                "Selling General And Administration": 26097000000,
                "Research And Development": 31370000000,
                "Reconciled Depreciation": 11445000000,
                "Operating Income": 123216000000,
                "Interest Expense": 0,
                "Pretax Income": 123485000000,
                "Tax Provision": 29749000000,
                "Net Income": 93736000000,
                "Basic EPS": 6.11
            },
            pd.Timestamp("2023-09-30"): {
                "Total Revenue": 383285000000,
                "Gross Profit": 169148000000,
                "Selling General And Administration": 24932000000,
                "Research And Development": 29915000000,
                "Reconciled Depreciation": 11519000000,
                "Operating Income": 114301000000,
                "Interest Expense": 3933000000,
                "Pretax Income": 113736000000,
                "Tax Provision": 16741000000,
                "Net Income": 96995000000,
                "Basic EPS": 6.16
            },
            pd.Timestamp("2022-09-24"): {
                "Total Revenue": 394328000000,
                "Gross Profit": 170782000000,
                "Selling General And Administration": 25094000000,
                "Research And Development": 26251000000,
                "Reconciled Depreciation": 11104000000,
                "Operating Income": 119437000000,
                "Interest Expense": 2931000000,
                "Pretax Income": 119103000000,
                "Tax Provision": 19300000000,
                "Net Income": 99803000000,
                "Basic EPS": 6.15
            },
            pd.Timestamp("2021-09-25"): {
                "Total Revenue": 365817000000,
                "Gross Profit": 152836000000,
                "Selling General And Administration": 21973000000,
                "Research And Development": 21914000000,
                "Reconciled Depreciation": 11284000000,
                "Operating Income": 108949000000,
                "Interest Expense": 2645000000,
                "Pretax Income": 109207000000,
                "Tax Provision": 14527000000,
                "Net Income": 94680000000,
                "Basic EPS": 5.67
            }
        }).T.T,  # Transpose to get correct orientation
        "balance_sheet": pd.DataFrame({
            pd.Timestamp("2024-09-28"): {
                "Total Debt": 96837000000,
                "Total Equity Gross Minority Interest": 56950000000,
                "Cash And Cash Equivalents": 29943000000,
                "Cash Cash Equivalents And Short Term Investments": 65171000000,
                "Retained Earnings": -19154000000,
                "Total Assets": 364980000000
            },
            pd.Timestamp("2023-09-30"): {
                "Total Debt": 111088000000,
                "Total Equity Gross Minority Interest": 62146000000,
                "Cash And Cash Equivalents": 29965000000,
                "Cash Cash Equivalents And Short Term Investments": 61555000000,
                "Retained Earnings": -214000000,
                "Total Assets": 352583000000
            },
            pd.Timestamp("2022-09-24"): {
                "Total Debt": 120069000000,
                "Total Equity Gross Minority Interest": 50672000000,
                "Cash And Cash Equivalents": 23646000000,
                "Cash Cash Equivalents And Short Term Investments": 48304000000,
                "Retained Earnings": -3068000000,
                "Total Assets": 352755000000
            }
        }).T.T,
        "cash_flow": pd.DataFrame({
            pd.Timestamp("2024-09-28"): {
                "Operating Cash Flow": 118254000000,
                "Capital Expenditure": -9447000000,
                "Net Income": 93736000000
            },
            pd.Timestamp("2023-09-30"): {
                "Operating Cash Flow": 110543000000,
                "Capital Expenditure": -10959000000,
                "Net Income": 96995000000
            },
            pd.Timestamp("2022-09-24"): {
                "Operating Cash Flow": 122151000000,
                "Capital Expenditure": -10708000000,
                "Net Income": 99803000000
            }
        }).T.T
    },
    "MSFT": {
        "info": {
            "longName": "Microsoft Corporation",
            "symbol": "MSFT",
            "sector": "Technology",
            "industry": "Software—Infrastructure",
            "currentPrice": 423.46,
            "marketCap": 3150000000000,
            "currency": "USD"
        },
        "income_stmt": pd.DataFrame({
            pd.Timestamp("2024-06-30"): {
                "Total Revenue": 245122000000,
                "Gross Profit": 171006000000,
                "Selling General And Administration": 28054000000,
                "Research And Development": 29510000000,
                "Reconciled Depreciation": 22287000000,
                "Operating Income": 109433000000,
                "Interest Expense": 2935000000,
                "Pretax Income": 110654000000,
                "Tax Provision": 22033000000,
                "Net Income": 88136000000,
                "Basic EPS": 11.86
            },
            pd.Timestamp("2023-06-30"): {
                "Total Revenue": 211915000000,
                "Gross Profit": 146052000000,
                "Selling General And Administration": 24506000000,
                "Research And Development": 27195000000,
                "Reconciled Depreciation": 13861000000,
                "Operating Income": 88523000000,
                "Interest Expense": 1968000000,
                "Pretax Income": 89694000000,
                "Tax Provision": 16950000000,
                "Net Income": 72361000000,
                "Basic EPS": 9.72
            }
        }).T.T,
        "balance_sheet": pd.DataFrame({
            pd.Timestamp("2024-06-30"): {
                "Total Debt": 72394000000,
                "Total Equity Gross Minority Interest": 268477000000,
                "Cash And Cash Equivalents": 18315000000,
                "Cash Cash Equivalents And Short Term Investments": 75530000000,
                "Retained Earnings": 173144000000,
                "Total Assets": 512163000000
            },
            pd.Timestamp("2023-06-30"): {
                "Total Debt": 59965000000,
                "Total Equity Gross Minority Interest": 206223000000,
                "Cash And Cash Equivalents": 34704000000,
                "Cash Cash Equivalents And Short Term Investments": 111262000000,
                "Retained Earnings": 118848000000,
                "Total Assets": 411976000000
            }
        }).T.T,
        "cash_flow": pd.DataFrame({
            pd.Timestamp("2024-06-30"): {
                "Operating Cash Flow": 118548000000,
                "Capital Expenditure": -44477000000,
                "Net Income": 88136000000
            },
            pd.Timestamp("2023-06-30"): {
                "Operating Cash Flow": 87582000000,
                "Capital Expenditure": -28107000000,
                "Net Income": 72361000000
            }
        }).T.T
    },
    "BRK-B": {
        "info": {
            "longName": "Berkshire Hathaway Inc.",
            "symbol": "BRK-B",
            "sector": "Financial Services",
            "industry": "Insurance—Diversified",
            "currentPrice": 472.83,
            "marketCap": 1030000000000,
            "currency": "USD"
        },
        "income_stmt": pd.DataFrame({
            pd.Timestamp("2024-09-30"): {
                "Total Revenue": 371900000000,
                "Gross Profit": 93000000000,
                "Selling General And Administration": 11200000000,
                "Research And Development": 0,
                "Reconciled Depreciation": 12800000000,
                "Operating Income": 42800000000,
                "Interest Expense": 2100000000,
                "Pretax Income": 89600000000,
                "Tax Provision": 11700000000,
                "Net Income": 89497000000,
                "Basic EPS": 41.32
            },
            pd.Timestamp("2023-09-30"): {
                "Total Revenue": 364482000000,
                "Gross Profit": 88500000000,
                "Selling General And Administration": 10800000000,
                "Research And Development": 0,
                "Reconciled Depreciation": 12200000000,
                "Operating Income": 37353000000,
                "Interest Expense": 1800000000,
                "Pretax Income": 46527000000,
                "Tax Provision": 10500000000,
                "Net Income": 96223000000,
                "Basic EPS": 44.02
            }
        }).T.T,
        "balance_sheet": pd.DataFrame({
            pd.Timestamp("2024-09-30"): {
                "Total Debt": 125800000000,
                "Total Equity Gross Minority Interest": 625100000000,
                "Cash And Cash Equivalents": 325200000000,
                "Cash Cash Equivalents And Short Term Investments": 325200000000,
                "Retained Earnings": 612400000000,
                "Total Assets": 1146000000000
            },
            pd.Timestamp("2023-09-30"): {
                "Total Debt": 127100000000,
                "Total Equity Gross Minority Interest": 561400000000,
                "Cash And Cash Equivalents": 157200000000,
                "Cash Cash Equivalents And Short Term Investments": 157200000000,
                "Retained Earnings": 548900000000,
                "Total Assets": 1005000000000
            }
        }).T.T,
        "cash_flow": pd.DataFrame({
            pd.Timestamp("2024-09-30"): {
                "Operating Cash Flow": 49800000000,
                "Capital Expenditure": -19200000000,
                "Net Income": 89497000000
            },
            pd.Timestamp("2023-09-30"): {
                "Operating Cash Flow": 43200000000,
                "Capital Expenditure": -16800000000,
                "Net Income": 96223000000
            }
        }).T.T
    }
}


def generate_sample_history(days=730):
    """Generate sample stock price history"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Generate random walk for price
    returns = np.random.normal(0.0005, 0.02, days)
    price = 150 * np.cumprod(1 + returns)
    
    # Generate OHLC data
    data = {
        'Open': price * (1 + np.random.uniform(-0.01, 0.01, days)),
        'High': price * (1 + np.random.uniform(0, 0.02, days)),
        'Low': price * (1 - np.random.uniform(0, 0.02, days)),
        'Close': price,
        'Volume': np.random.randint(50000000, 150000000, days)
    }
    
    return pd.DataFrame(data, index=dates)


def get_stock_data(symbol: str) -> dict:
    """Fetch comprehensive stock data using yfinance or sample data"""
    
    # First, try to use sample data if available
    symbol_upper = symbol.upper()
    if symbol_upper in SAMPLE_DATA:
        sample = SAMPLE_DATA[symbol_upper]
        return {
            "info": sample["info"],
            "income_stmt": sample["income_stmt"],
            "balance_sheet": sample["balance_sheet"],
            "cash_flow": sample["cash_flow"],
            "history": generate_sample_history(),
            "success": True,
            "is_sample": True
        }
    
    # Try yfinance if available
    if YFINANCE_AVAILABLE:
        try:
            stock = yf.Ticker(symbol)
            
            # Get financial statements
            income_stmt = stock.income_stmt
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow
            
            # Get basic info
            info = stock.info
            
            # Get historical data for price chart
            history = stock.history(period="2y")
            
            # Check if we got valid data
            if income_stmt is not None and not income_stmt.empty:
                return {
                    "info": info,
                    "income_stmt": income_stmt,
                    "balance_sheet": balance_sheet,
                    "cash_flow": cash_flow,
                    "history": history,
                    "success": True,
                    "is_sample": False
                }
        except Exception as e:
            pass
    
    # Return error if no data available
    return {
        "success": False, 
        "error": f"No data available for {symbol}. Try AAPL, MSFT, or BRK-B for sample data."
    }


def safe_get(df: pd.DataFrame, keys: list, column_idx: int = 0):
    """Safely get a value from a DataFrame with multiple possible key names"""
    if df is None or df.empty:
        return None
    
    for key in keys:
        if key in df.index:
            try:
                value = df.loc[key].iloc[column_idx]
                if pd.notna(value):
                    return float(value)
            except:
                continue
    return None


def calculate_buffett_ratios(data: dict) -> dict:
    """Calculate all Warren Buffett ratios from financial data"""
    income_stmt = data.get("income_stmt")
    balance_sheet = data.get("balance_sheet")
    cash_flow = data.get("cash_flow")
    
    ratios = {
        "income_statement": {},
        "balance_sheet": {},
        "cash_flow": {}
    }
    
    # ===== INCOME STATEMENT RATIOS =====
    if income_stmt is not None and not income_stmt.empty:
        # Get key values
        revenue = safe_get(income_stmt, ["Total Revenue", "Revenue"])
        gross_profit = safe_get(income_stmt, ["Gross Profit"])
        sga = safe_get(income_stmt, ["Selling General And Administration", "SG&A", "Selling And Marketing Expense"])
        rd = safe_get(income_stmt, ["Research And Development", "R&D"])
        depreciation = safe_get(income_stmt, ["Reconciled Depreciation", "Depreciation", "Depreciation And Amortization"])
        operating_income = safe_get(income_stmt, ["Operating Income", "EBIT"])
        interest_expense = safe_get(income_stmt, ["Interest Expense", "Interest Expense Non Operating"])
        pretax_income = safe_get(income_stmt, ["Pretax Income", "Income Before Tax"])
        tax_provision = safe_get(income_stmt, ["Tax Provision", "Income Tax Expense"])
        net_income = safe_get(income_stmt, ["Net Income", "Net Income Common Stockholders"])
        
        # Basic EPS for current and previous year
        eps_current = safe_get(income_stmt, ["Basic EPS", "Diluted EPS"], 0)
        eps_previous = safe_get(income_stmt, ["Basic EPS", "Diluted EPS"], 1)
        
        # Calculate ratios
        # 1. Gross Margin
        if revenue and gross_profit:
            ratios["income_statement"]["gross_margin"] = gross_profit / revenue
        
        # 2. SG&A Margin
        if gross_profit and sga:
            ratios["income_statement"]["sga_margin"] = sga / gross_profit
        elif gross_profit:
            ratios["income_statement"]["sga_margin"] = 0  # No SG&A reported
        
        # 3. R&D Margin
        if gross_profit and rd:
            ratios["income_statement"]["rd_margin"] = rd / gross_profit
        elif gross_profit:
            ratios["income_statement"]["rd_margin"] = 0  # No R&D reported
        
        # 4. Depreciation Margin
        if gross_profit and depreciation:
            ratios["income_statement"]["depreciation_margin"] = depreciation / gross_profit
        
        # 5. Interest Expense Margin
        if operating_income and interest_expense:
            ratios["income_statement"]["interest_expense_margin"] = abs(interest_expense) / operating_income
        elif operating_income:
            ratios["income_statement"]["interest_expense_margin"] = 0  # No interest expense
        
        # 6. Effective Tax Rate
        if pretax_income and tax_provision and pretax_income > 0:
            ratios["income_statement"]["tax_rate"] = tax_provision / pretax_income
        
        # 7. Net Margin
        if revenue and net_income:
            ratios["income_statement"]["net_margin"] = net_income / revenue
        
        # 8. EPS Growth
        if eps_current and eps_previous and eps_previous != 0:
            ratios["income_statement"]["eps_growth"] = (eps_current - eps_previous) / abs(eps_previous)
    
    # ===== BALANCE SHEET RATIOS =====
    if balance_sheet is not None and not balance_sheet.empty:
        # Get key values
        total_debt = safe_get(balance_sheet, ["Total Debt", "Long Term Debt", "Total Liabilities Net Minority Interest"])
        total_equity = safe_get(balance_sheet, ["Total Equity Gross Minority Interest", "Stockholders Equity", "Total Stockholder Equity"])
        cash = safe_get(balance_sheet, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash"])
        retained_earnings_current = safe_get(balance_sheet, ["Retained Earnings"], 0)
        retained_earnings_previous = safe_get(balance_sheet, ["Retained Earnings"], 1)
        
        # Calculate ratios
        # 1. Debt to Equity
        if total_equity and total_debt and total_equity != 0:
            ratios["balance_sheet"]["debt_to_equity"] = total_debt / total_equity
        
        # 2. Cash to Debt
        if cash and total_debt and total_debt != 0:
            ratios["balance_sheet"]["cash_to_debt"] = cash / total_debt
        elif cash and (not total_debt or total_debt == 0):
            ratios["balance_sheet"]["cash_to_debt"] = float('inf')  # No debt is great!
        
        # 3. Retained Earnings Growth
        if retained_earnings_current and retained_earnings_previous and retained_earnings_previous != 0:
            ratios["balance_sheet"]["retained_earnings_growth"] = (retained_earnings_current - retained_earnings_previous) / abs(retained_earnings_previous)
    
    # ===== CASH FLOW RATIOS =====
    if cash_flow is not None and not cash_flow.empty:
        # Get key values
        capex = safe_get(cash_flow, ["Capital Expenditure", "Capital Expenditures"])
        operating_cash_flow = safe_get(cash_flow, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"])
        net_income_cf = safe_get(cash_flow, ["Net Income", "Net Income From Continuing Operations"])
        
        # Calculate CapEx Margin
        if capex and net_income_cf and net_income_cf != 0:
            ratios["cash_flow"]["capex_margin"] = abs(capex) / net_income_cf
    
    return ratios


def evaluate_ratio(value: float, threshold: float, comparison: str) -> str:
    """Evaluate if a ratio passes Buffett's criteria"""
    if value is None:
        return "neutral"
    
    if comparison == ">=":
        return "pass" if value >= threshold else "fail"
    elif comparison == "<=":
        return "pass" if value <= threshold else "fail"
    elif comparison == ">":
        return "pass" if value > threshold else "fail"
    elif comparison == "<":
        return "pass" if value < threshold else "fail"
    else:  # info
        return "neutral"


def format_percentage(value: float) -> str:
    """Format a decimal as percentage"""
    if value is None:
        return "N/A"
    if value == float('inf'):
        return "∞ (No Debt)"
    return f"{value * 100:.2f}%"


def format_ratio(value: float) -> str:
    """Format a ratio value"""
    if value is None:
        return "N/A"
    if value == float('inf'):
        return "∞ (No Debt)"
    return f"{value:.2f}x"


def display_ratio_card(name: str, value: float, rule: str, logic: str, comparison: str, threshold: float, is_percentage: bool = True):
    """Display a single ratio as a styled card"""
    status = evaluate_ratio(value, threshold, comparison)
    
    if is_percentage:
        formatted_value = format_percentage(value)
    else:
        formatted_value = format_ratio(value)
    
    status_class = f"status-{status}"
    metric_class = f"metric-{status}"
    
    status_text = "✓ PASS" if status == "pass" else ("✗ FAIL" if status == "fail" else "ℹ INFO")
    
    st.markdown(f"""
    <div class="metric-card {metric_class}">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <h4 style="margin: 0; color: #D4AF37;">{name}</h4>
            <span class="{status_class}">{status_text}</span>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.75rem;">
            <span style="font-size: 2rem; font-weight: bold; color: white;">{formatted_value}</span>
            <span style="color: #888;">Buffett's Rule: {rule}</span>
        </div>
        <p style="color: #aaa; font-size: 0.9rem; margin: 0;">{logic}</p>
    </div>
    """, unsafe_allow_html=True)


def create_gauge_chart(value: float, threshold: float, title: str, comparison: str):
    """Create a gauge chart for a ratio"""
    if value is None:
        return None
    
    # Determine color based on pass/fail
    status = evaluate_ratio(value, threshold, comparison)
    color = "#00C853" if status == "pass" else "#FF5252"
    
    # Create gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={'suffix': '%', 'font': {'size': 24}},
        title={'text': title, 'font': {'size': 14}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'steps': [
                {'range': [0, threshold * 100], 'color': 'rgba(0, 200, 83, 0.3)' if comparison == "<=" else 'rgba(255, 82, 82, 0.3)'},
                {'range': [threshold * 100, 100], 'color': 'rgba(255, 82, 82, 0.3)' if comparison == "<=" else 'rgba(0, 200, 83, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "#D4AF37", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}
    )
    
    return fig


def calculate_buffett_score(ratios: dict) -> tuple:
    """Calculate overall Buffett score"""
    total = 0
    passed = 0
    
    for category, metrics in BUFFETT_RATIOS.items():
        for ratio_key, ratio_info in metrics.items():
            if ratio_info["comparison"] != "info":
                total += 1
                if category in ratios and ratio_key in ratios[category]:
                    value = ratios[category][ratio_key]
                    if evaluate_ratio(value, ratio_info["threshold"], ratio_info["comparison"]) == "pass":
                        passed += 1
    
    return passed, total


def display_financial_statement(df: pd.DataFrame, title: str):
    """Display a financial statement DataFrame"""
    if df is None or df.empty:
        st.warning(f"No {title} data available")
        return
    
    # Format the DataFrame
    df_display = df.copy()
    
    # Convert column names to readable dates
    if isinstance(df_display.columns[0], (pd.Timestamp, datetime)):
        df_display.columns = [col.strftime('%Y-%m-%d') if hasattr(col, 'strftime') else str(col) for col in df_display.columns]
    
    # Format numbers
    def format_value(x):
        if pd.notna(x) and isinstance(x, (int, float)):
            if abs(x) >= 1e9:
                return f"${x/1e9:.2f}B"
            elif abs(x) >= 1e6:
                return f"${x/1e6:.2f}M"
            else:
                return f"${x:,.0f}"
        return x
    
    df_display = df_display.map(format_value)
    
    st.dataframe(df_display, use_container_width=True, height=400)


# ===== MAIN APPLICATION =====
def main():
    # Header
    st.markdown('<h1 class="main-header">🐝 AppleBee</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Warren Buffett Stock Analysis Dashboard</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/51/Warren_Buffett_KU_Visit.jpg/220px-Warren_Buffett_KU_Visit.jpg", width=150)
        st.markdown("### 📊 Stock Selection")
        
        symbol = st.text_input(
            "Enter Stock Symbol",
            value="AAPL",
            help="Enter a valid stock ticker symbol (e.g., AAPL, MSFT, GOOGL)"
        ).upper()
        
        analyze_button = st.button("🔍 Analyze Stock", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📖 About")
        st.markdown("""
        **AppleBee** analyzes stocks using Warren Buffett's investment principles.
        
        The dashboard evaluates:
        - 📈 Income Statement Ratios
        - 📊 Balance Sheet Ratios  
        - 💰 Cash Flow Ratios
        
        All metrics are compared against Buffett's proven thresholds.
        """)
        
        st.markdown("---")
        st.markdown("### 💡 Buffett's Wisdom")
        st.markdown("""
        > *"Rule No. 1: Never lose money. Rule No. 2: Never forget Rule No. 1."*
        
        > *"It's far better to buy a wonderful company at a fair price than a fair company at a wonderful price."*
        """)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "💬 AI Chatbot", "📚 Learn"])
    
    # Initialize session state
    if 'stock_data' not in st.session_state:
        st.session_state.stock_data = None
    if 'current_symbol' not in st.session_state:
        st.session_state.current_symbol = None
    if 'quick_symbol' not in st.session_state:
        st.session_state.quick_symbol = None
    
    # Handle quick symbol selection
    if st.session_state.quick_symbol:
        symbol = st.session_state.quick_symbol
        st.session_state.quick_symbol = None  # Reset
    
    # Fetch data if button clicked or symbol changed
    if analyze_button or (symbol and symbol != st.session_state.current_symbol):
        with st.spinner(f"Fetching data for {symbol}..."):
            data = get_stock_data(symbol)
            if data["success"]:
                st.session_state.stock_data = data
                st.session_state.current_symbol = symbol
            else:
                st.error(f"Error fetching data: {data.get('error', 'Unknown error')}")
                st.session_state.stock_data = None
    
    # ===== TAB 1: DASHBOARD =====
    with tab1:
        if st.session_state.stock_data:
            data = st.session_state.stock_data
            info = data.get("info", {})
            
            # Company header
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                company_name = info.get("longName", symbol)
                st.markdown(f"## {company_name} ({symbol})")
                sector = info.get("sector", "N/A")
                industry = info.get("industry", "N/A")
                st.markdown(f"**Sector:** {sector} | **Industry:** {industry}")
                
                # Show sample data indicator
                if data.get("is_sample", False):
                    st.info("📊 **Sample Data Mode**: Using pre-loaded financial data for demonstration. For live data, run the app locally with `streamlit run app.py`.")
            
            with col2:
                current_price = info.get("currentPrice", info.get("regularMarketPrice", "N/A"))
                if isinstance(current_price, (int, float)):
                    st.metric("Current Price", f"${current_price:.2f}")
                else:
                    st.metric("Current Price", "N/A")
            
            with col3:
                market_cap = info.get("marketCap", 0)
                if market_cap >= 1e12:
                    st.metric("Market Cap", f"${market_cap/1e12:.2f}T")
                elif market_cap >= 1e9:
                    st.metric("Market Cap", f"${market_cap/1e9:.2f}B")
                else:
                    st.metric("Market Cap", f"${market_cap/1e6:.2f}M")
            
            st.markdown("---")
            
            # Calculate ratios
            ratios = calculate_buffett_ratios(data)
            passed, total = calculate_buffett_score(ratios)
            
            # Buffett Score Summary
            st.markdown('<h3 class="section-header">🎯 Warren Buffett Score</h3>', unsafe_allow_html=True)
            
            score_col1, score_col2, score_col3 = st.columns([1, 2, 1])
            
            with score_col2:
                score_pct = (passed / total * 100) if total > 0 else 0
                score_color = "#00C853" if score_pct >= 70 else ("#FFC107" if score_pct >= 50 else "#FF5252")
                
                # Create score gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score_pct,
                    number={'suffix': '%', 'font': {'size': 48, 'color': 'white'}},
                    title={'text': f"Buffett Criteria Passed: {passed}/{total}", 'font': {'size': 18, 'color': '#888'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#888'},
                        'bar': {'color': score_color},
                        'bgcolor': 'rgba(255,255,255,0.1)',
                        'borderwidth': 2,
                        'bordercolor': '#333',
                        'steps': [
                            {'range': [0, 50], 'color': 'rgba(255, 82, 82, 0.2)'},
                            {'range': [50, 70], 'color': 'rgba(255, 193, 7, 0.2)'},
                            {'range': [70, 100], 'color': 'rgba(0, 200, 83, 0.2)'}
                        ],
                    }
                ))
                fig.update_layout(
                    height=300,
                    margin=dict(l=30, r=30, t=50, b=30),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'color': 'white'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            if score_pct >= 70:
                st.success(f"🌟 **Strong Buffett Candidate!** {company_name} passes {passed} out of {total} Warren Buffett criteria. This stock shows characteristics of a quality business with durable competitive advantages.")
            elif score_pct >= 50:
                st.warning(f"⚠️ **Mixed Results.** {company_name} passes {passed} out of {total} criteria. Some aspects align with Buffett's principles, but there are areas of concern.")
            else:
                st.error(f"❌ **Does Not Meet Criteria.** {company_name} only passes {passed} out of {total} criteria. This stock may not align well with Buffett's investment philosophy.")
            
            st.markdown("---")
            
            # ===== INCOME STATEMENT SECTION =====
            st.markdown('<h3 class="section-header">📈 Income Statement Analysis</h3>', unsafe_allow_html=True)
            
            income_ratios = ratios.get("income_statement", {})
            
            # Display income statement ratios in a grid
            col1, col2 = st.columns(2)
            
            with col1:
                for i, (key, info) in enumerate(BUFFETT_RATIOS["income_statement"].items()):
                    if i % 2 == 0:
                        value = income_ratios.get(key)
                        display_ratio_card(
                            info["name"],
                            value,
                            info["rule"],
                            info["logic"],
                            info["comparison"],
                            info["threshold"]
                        )
            
            with col2:
                for i, (key, info) in enumerate(BUFFETT_RATIOS["income_statement"].items()):
                    if i % 2 == 1:
                        value = income_ratios.get(key)
                        display_ratio_card(
                            info["name"],
                            value,
                            info["rule"],
                            info["logic"],
                            info["comparison"],
                            info["threshold"]
                        )
            
            # Show raw income statement
            with st.expander("📄 View Full Income Statement"):
                display_financial_statement(data.get("income_stmt"), "Income Statement")
            
            st.markdown("---")
            
            # ===== BALANCE SHEET SECTION =====
            st.markdown('<h3 class="section-header">📊 Balance Sheet Analysis</h3>', unsafe_allow_html=True)
            
            balance_ratios = ratios.get("balance_sheet", {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                key = "debt_to_equity"
                info = BUFFETT_RATIOS["balance_sheet"][key]
                value = balance_ratios.get(key)
                display_ratio_card(
                    info["name"],
                    value,
                    info["rule"],
                    info["logic"],
                    info["comparison"],
                    info["threshold"],
                    is_percentage=False
                )
            
            with col2:
                key = "cash_to_debt"
                info = BUFFETT_RATIOS["balance_sheet"][key]
                value = balance_ratios.get(key)
                display_ratio_card(
                    info["name"],
                    value,
                    info["rule"],
                    info["logic"],
                    info["comparison"],
                    info["threshold"],
                    is_percentage=False
                )
            
            with col3:
                key = "retained_earnings_growth"
                info = BUFFETT_RATIOS["balance_sheet"][key]
                value = balance_ratios.get(key)
                display_ratio_card(
                    info["name"],
                    value,
                    info["rule"],
                    info["logic"],
                    info["comparison"],
                    info["threshold"]
                )
            
            # Show raw balance sheet
            with st.expander("📄 View Full Balance Sheet"):
                display_financial_statement(data.get("balance_sheet"), "Balance Sheet")
            
            st.markdown("---")
            
            # ===== CASH FLOW SECTION =====
            st.markdown('<h3 class="section-header">💰 Cash Flow Analysis</h3>', unsafe_allow_html=True)
            
            cash_ratios = ratios.get("cash_flow", {})
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col2:
                key = "capex_margin"
                info = BUFFETT_RATIOS["cash_flow"][key]
                value = cash_ratios.get(key)
                display_ratio_card(
                    info["name"],
                    value,
                    info["rule"],
                    info["logic"],
                    info["comparison"],
                    info["threshold"]
                )
            
            # Show raw cash flow statement
            with st.expander("📄 View Full Cash Flow Statement"):
                display_financial_statement(data.get("cash_flow"), "Cash Flow Statement")
            
            st.markdown("---")
            
            # ===== STOCK PRICE CHART =====
            st.markdown('<h3 class="section-header">📈 Stock Price History</h3>', unsafe_allow_html=True)
            
            history = data.get("history")
            if history is not None and not history.empty:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=history.index,
                    open=history['Open'],
                    high=history['High'],
                    low=history['Low'],
                    close=history['Close'],
                    name='Price'
                ))
                fig.update_layout(
                    title=f'{symbol} Stock Price (2 Years)',
                    yaxis_title='Price ($)',
                    xaxis_title='Date',
                    template='plotly_dark',
                    height=500,
                    xaxis_rangeslider_visible=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
        else:
            # Welcome screen
            st.markdown("""
            <div style="text-align: center; padding: 3rem;">
                <h2>Welcome to AppleBee! 🐝</h2>
                <p style="font-size: 1.2rem; color: #888;">
                    Enter a stock symbol in the sidebar and click "Analyze Stock" to begin your Warren Buffett-style analysis.
                </p>
                <br>
                <p style="color: #D4AF37;">
                    <strong>Sample Data Available:</strong><br>
                    AAPL (Apple) | MSFT (Microsoft) | BRK-B (Berkshire Hathaway)
                </p>
                <br>
                <p style="font-size: 0.9rem; color: #666;">
                    💡 For live data on any stock, run this app locally with: <code>streamlit run app.py</code>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Quick start cards
            st.markdown("### 🚀 Quick Start")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h4 style="color: #D4AF37;">📈 Apple (AAPL)</h4>
                    <p style="color: #888;">The world's most valuable company. See why Buffett loves it!</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Analyze AAPL", key="quick_aapl"):
                    st.session_state.quick_symbol = "AAPL"
                    st.rerun()
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h4 style="color: #D4AF37;">💻 Microsoft (MSFT)</h4>
                    <p style="color: #888;">Tech giant with strong moats. Excellent financials!</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Analyze MSFT", key="quick_msft"):
                    st.session_state.quick_symbol = "MSFT"
                    st.rerun()
            
            with col3:
                st.markdown("""
                <div class="metric-card">
                    <h4 style="color: #D4AF37;">🏛️ Berkshire (BRK-B)</h4>
                    <p style="color: #888;">Buffett's own company. The gold standard!</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Analyze BRK-B", key="quick_brkb"):
                    st.session_state.quick_symbol = "BRK-B"
                    st.rerun()
    
    # ===== TAB 2: AI CHATBOT =====
    with tab2:
        st.markdown('<h3 class="section-header">💬 Warren Buffett AI Investment Advisor</h3>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="buffett-quote">
            <p>"The most important investment you can make is in yourself."</p>
            <p style="text-align: right; color: #D4AF37;">— Warren Buffett</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("🚧 **Coming Soon!** The AI Chatbot feature is under development. This chatbot will be trained on Warren Buffett's investment principles and will help you understand financial ratios and stock analysis.")
        
        # Placeholder chat interface
        st.markdown("### Ask the Buffett Bot")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about Warren Buffett's investment principles..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Placeholder response
            with st.chat_message("assistant"):
                response = "🚧 The AI Chatbot is currently being developed. Soon, I'll be able to answer your questions about Warren Buffett's investment strategies, explain financial ratios, and provide insights based on the stock analysis dashboard!"
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    # ===== TAB 3: LEARN =====
    with tab3:
        st.markdown('<h3 class="section-header">📚 Understanding Buffett\'s Investment Criteria</h3>', unsafe_allow_html=True)
        
        st.markdown("""
        Warren Buffett, known as the "Oracle of Omaha," has developed a systematic approach to evaluating businesses.
        His investment philosophy focuses on finding companies with **durable competitive advantages** (moats) at **fair prices**.
        """)
        
        # Income Statement Section
        st.markdown("### 📈 Income Statement Ratios")
        
        for key, info in BUFFETT_RATIOS["income_statement"].items():
            with st.expander(f"**{info['name']}** — {info['rule']}"):
                st.markdown(f"""
                **What it measures:** {info['logic']}
                
                **Buffett's Threshold:** {info['rule']}
                
                **Why it matters:** This ratio helps identify companies that have genuine competitive advantages
                rather than those that are simply competing on price or spending excessively to maintain their market position.
                """)
        
        # Balance Sheet Section
        st.markdown("### 📊 Balance Sheet Ratios")
        
        for key, info in BUFFETT_RATIOS["balance_sheet"].items():
            with st.expander(f"**{info['name']}** — {info['rule']}"):
                st.markdown(f"""
                **What it measures:** {info['logic']}
                
                **Buffett's Threshold:** {info['rule']}
                
                **Why it matters:** A strong balance sheet indicates financial stability and provides
                the company with flexibility to weather economic downturns and capitalize on opportunities.
                """)
        
        # Cash Flow Section
        st.markdown("### 💰 Cash Flow Ratios")
        
        for key, info in BUFFETT_RATIOS["cash_flow"].items():
            with st.expander(f"**{info['name']}** — {info['rule']}"):
                st.markdown(f"""
                **What it measures:** {info['logic']}
                
                **Buffett's Threshold:** {info['rule']}
                
                **Why it matters:** Free cash flow is what's left after maintaining the business—
                companies that generate abundant free cash flow can reinvest in growth, pay dividends,
                or buy back shares.
                """)
        
        st.markdown("---")
        
        st.markdown("""
        ### 🎓 Key Takeaways
        
        1. **Focus on Quality:** Buffett looks for businesses with high margins, low debt, and consistent earnings growth.
        
        2. **Think Like a Business Owner:** Don't just look at stock prices—understand the underlying business.
        
        3. **Patience is Key:** Great businesses are rare. Wait for the right opportunity at the right price.
        
        4. **Margin of Safety:** Buy at prices that provide a buffer against errors in analysis.
        
        5. **Long-term Perspective:** Buffett's favorite holding period is "forever."
        """)


if __name__ == "__main__":
    main()