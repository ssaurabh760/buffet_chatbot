"""
Warren Buffett Financial Analyzer
Calculates key financial ratios for stock analysis
"""

import yfinance as yf
import pandas as pd
import streamlit as st
from typing import Dict, Tuple, Optional

class FinancialAnalyzer:
    """Analyze stocks using Warren Buffett's financial ratio criteria."""
    
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(self.ticker)
        self.info = self.stock.info
        
    def get_income_statement(self) -> pd.DataFrame:
        """Get most recent income statement."""
        try:
            income_stmt = self.stock.quarterly_financials
            if income_stmt.empty:
                return pd.DataFrame()
            return income_stmt.iloc[:, 0]  # Most recent quarter
        except:
            return pd.DataFrame()
    
    def get_balance_sheet(self) -> pd.DataFrame:
        """Get most recent balance sheet."""
        try:
            balance = self.stock.quarterly_balance_sheet
            if balance.empty:
                return pd.DataFrame()
            return balance.iloc[:, 0]  # Most recent quarter
        except:
            return pd.DataFrame()
    
    def get_cash_flow(self) -> pd.DataFrame:
        """Get most recent cash flow statement."""
        try:
            cash_flow = self.stock.quarterly_cashflow
            if cash_flow.empty:
                return pd.DataFrame()
            return cash_flow.iloc[:, 0]  # Most recent quarter
        except:
            return pd.DataFrame()
    
    def calculate_ratios(self) -> Dict:
        """Calculate all 8 Warren Buffett financial ratios."""
        ratios = {}
        
        try:
            income = self.get_income_statement()
            
            # Get necessary values
            revenue = income.get('Total Revenue', 0)
            gross_profit = income.get('Gross Profit', 0)
            sga_expense = income.get('Operating Expense', 0)
            rd_expense = income.get('Research Development', 0)
            depreciation = income.get('Depreciation', 0)
            operating_income = income.get('Operating Income', 0)
            interest_expense = income.get('Interest Expense', 0)
            taxes = income.get('Tax Provision', 0)
            net_income = income.get('Net Income', 0)
            
            # Safe division helper
            def safe_div(num, denom, default=None):
                if denom == 0 or denom is None:
                    return default
                return (num / denom) * 100
            
            # 1. Gross Margin
            ratios['Gross Margin'] = {
                'value': safe_div(gross_profit, revenue),
                'benchmark': 40,
                'rule': '>= 40%',
                'logic': "Signals the company isn't competing on price.",
                'unit': '%'
            }
            
            # 2. SG&A Expense Margin
            ratios['SGA Margin'] = {
                'value': safe_div(sga_expense, gross_profit),
                'benchmark': 30,
                'rule': '<= 30%',
                'logic': "Wide-moat companies don't need high overhead.",
                'unit': '%'
            }
            
            # 3. R&D Expense Margin
            ratios['R&D Margin'] = {
                'value': safe_div(rd_expense, gross_profit),
                'benchmark': 30,
                'rule': '<= 30%',
                'logic': "R&D doesn't always create shareholder value.",
                'unit': '%'
            }
            
            # 4. Depreciation Margin
            ratios['Depreciation Margin'] = {
                'value': safe_div(depreciation, gross_profit),
                'benchmark': 10,
                'rule': '<= 10%',
                'logic': "Buffett avoids asset-heavy businesses.",
                'unit': '%'
            }
            
            # 5. Interest Expense Margin
            if operating_income > 0:
                int_margin = (interest_expense / operating_income) * 100
            else:
                int_margin = None
                
            ratios['Interest Margin'] = {
                'value': int_margin,
                'benchmark': 15,
                'rule': '<= 15%',
                'logic': "Great businesses don't need debt.",
                'unit': '%'
            }
            
            # 6. Tax Rate
            if revenue > 0:
                try:
                    pre_tax_income = operating_income - interest_expense
                    if pre_tax_income > 0:
                        tax_rate = (taxes / pre_tax_income) * 100
                    else:
                        tax_rate = None
                except:
                    tax_rate = None
            else:
                tax_rate = None
                
            ratios['Tax Rate'] = {
                'value': tax_rate,
                'benchmark': 21,  # Current US corporate tax rate
                'rule': '≈ 21%',
                'logic': "Profitable companies pay their full tax load.",
                'unit': '%'
            }
            
            # 7. Net Profit Margin
            ratios['Net Margin'] = {
                'value': safe_div(net_income, revenue),
                'benchmark': 20,
                'rule': '>= 20%',
                'logic': "Great companies convert 20%+ revenue to profit.",
                'unit': '%'
            }
            
            # 8. EPS Growth (using available data)
            try:
                eps = self.info.get('trailingEps', None)
                ratios['Current EPS'] = {
                    'value': eps,
                    'benchmark': None,
                    'rule': 'Positive & Growing',
                    'logic': "Great companies increase profits yearly.",
                    'unit': '$'
                }
            except:
                ratios['Current EPS'] = {
                    'value': None,
                    'benchmark': None,
                    'rule': 'Positive & Growing',
                    'logic': "Great companies increase profits yearly.",
                    'unit': '$'
                }
            
            return ratios
            
        except Exception as e:
            st.error(f"Error calculating ratios: {str(e)}")
            return {}
    
    def get_ratio_assessment(self, ratio_name: str, value: Optional[float]) -> str:
        """Get assessment of whether ratio meets Buffett criteria."""
        if value is None:
            return "⚪ Data unavailable"
        
        ratio_rules = {
            'Gross Margin': (lambda v: v >= 40, "Excellent" if v >= 40 else "Needs improvement"),
            'SGA Margin': (lambda v: v <= 30, "Efficient" if v <= 30 else "Inefficient"),
            'R&D Margin': (lambda v: v <= 30, "Reasonable" if v <= 30 else "High"),
            'Depreciation Margin': (lambda v: v <= 10, "Asset-light" if v <= 10 else "Asset-heavy"),
            'Interest Margin': (lambda v: v <= 15, "Low debt" if v <= 15 else "High debt"),
            'Tax Rate': (lambda v: 15 <= v <= 25, "Normal" if 15 <= v <= 25 else "Abnormal"),
            'Net Margin': (lambda v: v >= 20, "Excellent" if v >= 20 else "Needs improvement"),
        }
        
        if ratio_name in ratio_rules:
            check, label = ratio_rules[ratio_name]
            if check(value):
                return f"✅ {label}"
            else:
                return f"⚠️ {label}"
        
        return "❓ Unknown"
    
    def get_company_info(self) -> Dict:
        """Get basic company information."""
        return {
            'name': self.info.get('longName', 'N/A'),
            'sector': self.info.get('sector', 'N/A'),
            'industry': self.info.get('industry', 'N/A'),
            'market_cap': self.info.get('marketCap', 'N/A'),
            'pe_ratio': self.info.get('trailingPE', 'N/A'),
            'dividend_yield': self.info.get('dividendYield', 'N/A'),
        }


def get_buffett_ratio_info() -> Dict:
    """Get detailed information about each Buffett ratio."""
    return {
        'Gross Margin': {
            'description': 'Gross Profit / Revenue',
            'benchmark': 40,
            'rule': '>= 40%',
            'logic': "Buffett looks for companies with high gross margins because it shows they have pricing power and aren't competing on price.",
            'interpretation': 'Higher is better. Shows competitive advantage.'
        },
        'SGA Margin': {
            'description': 'SG&A Expense / Gross Profit',
            'benchmark': 30,
            'rule': '<= 30%',
            'logic': 'Wide-moat companies operate efficiently without massive overhead. Low SG&A relative to gross profit indicates operational excellence.',
            'interpretation': 'Lower is better. Efficient operations.'
        },
        'R&D Margin': {
            'description': 'R&D Expense / Gross Profit',
            'benchmark': 30,
            'rule': '<= 30%',
            'logic': 'Buffett prefers businesses that don\'t require constant R&D to stay competitive. Excessive R&D suggests no moat.',
            'interpretation': 'Lower is better. Stable business model.'
        },
        'Depreciation Margin': {
            'description': 'Depreciation / Gross Profit',
            'benchmark': 10,
            'rule': '<= 10%',
            'logic': 'Buffett avoids capital-intensive businesses. Low depreciation means the company doesn\'t need to constantly reinvest.',
            'interpretation': 'Lower is better. Asset-light model.'
        },
        'Interest Margin': {
            'description': 'Interest Expense / Operating Income',
            'benchmark': 15,
            'rule': '<= 15%',
            'logic': 'Great businesses generate strong cash flow and don\'t need debt. Low interest expense shows financial strength.',
            'interpretation': 'Lower is better. Low debt burden.'
        },
        'Tax Rate': {
            'description': 'Tax Provision / Pre-Tax Income',
            'benchmark': 21,
            'rule': '≈ 21% (current US rate)',
            'logic': 'Highly profitable companies pay their full tax load. A lower tax rate might indicate tax deferral issues.',
            'interpretation': 'Should be close to statutory rate. Indicates profitability.'
        },
        'Net Margin': {
            'description': 'Net Income / Revenue',
            'benchmark': 20,
            'rule': '>= 20%',
            'logic': 'Net margin is the ultimate measure. Buffett seeks companies that convert 20%+ of revenue to profit.',
            'interpretation': 'Higher is better. Great profitability.'
        },
    }