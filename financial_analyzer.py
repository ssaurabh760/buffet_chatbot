"""
Warren Buffett Financial Analyzer
Calculates key financial ratios for stock analysis
"""

import yfinance as yf
import pandas as pd
import streamlit as st
from typing import Dict, Optional
import numpy as np
import warnings

warnings.filterwarnings('ignore')

class FinancialAnalyzer:
    """Analyze stocks using Warren Buffett's financial ratio criteria."""
    
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        
        try:
            self.stock = yf.Ticker(self.ticker)
            self.info = self.stock.info
        except Exception as e:
            st.error(f"Could not fetch data for {ticker}: {str(e)}")
            self.stock = None
            self.info = {}
        
    def get_income_statement(self) -> pd.DataFrame:
        """Get most recent income statement."""
        try:
            if self.stock is None:
                return pd.DataFrame()
            
            financials = self.stock.financials
            if financials.empty:
                return pd.DataFrame()
            
            return financials
        except Exception as e:
            return pd.DataFrame()
    
    def get_balance_sheet(self) -> pd.DataFrame:
        """Get most recent balance sheet."""
        try:
            if self.stock is None:
                return pd.DataFrame()
            
            balance_sheet = self.stock.balance_sheet
            if balance_sheet.empty:
                return pd.DataFrame()
            
            return balance_sheet
        except Exception as e:
            return pd.DataFrame()
    
    def get_cash_flow(self) -> pd.DataFrame:
        """Get most recent cash flow statement."""
        try:
            if self.stock is None:
                return pd.DataFrame()
            
            cashflow = self.stock.cashflow
            if cashflow.empty:
                return pd.DataFrame()
            
            return cashflow
        except Exception as e:
            return pd.DataFrame()
    
    def calculate_ratios(self) -> Dict:
        """Calculate all 8 Warren Buffett financial ratios."""
        ratios = {}
        
        if self.stock is None:
            st.error("Could not fetch stock data")
            return {}
        
        try:
            financials = self.get_income_statement()
            balance_sheet = self.get_balance_sheet()
            cashflow = self.get_cash_flow()
            
            if financials.empty or balance_sheet.empty or cashflow.empty:
                st.error("Insufficient financial data available for this ticker")
                return {}
            
            # INCOME STATEMENT METRICS
            try:
                gross_profit = financials.loc['Gross Profit'].iloc[0]
                total_revenue = financials.loc['Total Revenue'].iloc[0]
                ratios['Gross Margin'] = {
                    'value': (gross_profit / total_revenue) * 100 if total_revenue != 0 else None,
                    'benchmark': 40,
                    'rule': '>= 40%',
                    'logic': "Signals the company isn't competing on price.",
                    'unit': '%'
                }
            except:
                ratios['Gross Margin'] = {
                    'value': None,
                    'benchmark': 40,
                    'rule': '>= 40%',
                    'logic': "Signals the company isn't competing on price.",
                    'unit': '%'
                }
            
            try:
                sga = financials.loc['Selling General And Administration'].iloc[0]
                ratios['SGA Margin'] = {
                    'value': (sga / gross_profit) * 100 if gross_profit != 0 else None,
                    'benchmark': 30,
                    'rule': '<= 30%',
                    'logic': "Wide-moat companies don't need high overhead.",
                    'unit': '%'
                }
            except:
                ratios['SGA Margin'] = {
                    'value': None,
                    'benchmark': 30,
                    'rule': '<= 30%',
                    'logic': "Wide-moat companies don't need high overhead.",
                    'unit': '%'
                }
            
            try:
                rnd = financials.loc['Research And Development'].iloc[0]
                ratios['R&D Margin'] = {
                    'value': (rnd / gross_profit) * 100 if gross_profit != 0 else None,
                    'benchmark': 30,
                    'rule': '<= 30%',
                    'logic': "R&D doesn't always create shareholder value.",
                    'unit': '%'
                }
            except:
                ratios['R&D Margin'] = {
                    'value': None,
                    'benchmark': 30,
                    'rule': '<= 30%',
                    'logic': "R&D doesn't always create shareholder value.",
                    'unit': '%'
                }
            
            try:
                depreciation = financials.loc['Reconciled Depreciation'].iloc[0]
                ratios['Depreciation Margin'] = {
                    'value': (depreciation / gross_profit) * 100 if gross_profit != 0 else None,
                    'benchmark': 10,
                    'rule': '<= 10%',
                    'logic': "Buffett avoids asset-heavy businesses.",
                    'unit': '%'
                }
            except:
                ratios['Depreciation Margin'] = {
                    'value': None,
                    'benchmark': 10,
                    'rule': '<= 10%',
                    'logic': "Buffett avoids asset-heavy businesses.",
                    'unit': '%'
                }
            
            try:
                interest_expense = financials.loc['Interest Expense'].iloc[0]
                operating_income = financials.loc['Operating Income'].iloc[0]
                ratios['Interest Margin'] = {
                    'value': (interest_expense / operating_income) * 100 if operating_income != 0 else None,
                    'benchmark': 15,
                    'rule': '<= 15%',
                    'logic': "Great businesses don't need debt.",
                    'unit': '%'
                }
            except:
                ratios['Interest Margin'] = {
                    'value': None,
                    'benchmark': 15,
                    'rule': '<= 15%',
                    'logic': "Great businesses don't need debt.",
                    'unit': '%'
                }
            
            try:
                tax_provision = financials.loc['Tax Provision'].iloc[0]
                pretax_income = financials.loc['Pretax Income'].iloc[0]
                ratios['Tax Rate'] = {
                    'value': (tax_provision / pretax_income) * 100 if pretax_income != 0 else None,
                    'benchmark': 21,
                    'rule': '≈ 21%',
                    'logic': "Profitable companies pay their full tax load.",
                    'unit': '%'
                }
            except:
                ratios['Tax Rate'] = {
                    'value': None,
                    'benchmark': 21,
                    'rule': '≈ 21%',
                    'logic': "Profitable companies pay their full tax load.",
                    'unit': '%'
                }
            
            try:
                net_income = financials.loc['Net Income'].iloc[0]
                ratios['Net Margin'] = {
                    'value': (net_income / total_revenue) * 100 if total_revenue != 0 else None,
                    'benchmark': 20,
                    'rule': '>= 20%',
                    'logic': "Great companies convert 20%+ revenue to profit.",
                    'unit': '%'
                }
            except:
                ratios['Net Margin'] = {
                    'value': None,
                    'benchmark': 20,
                    'rule': '>= 20%',
                    'logic': "Great companies convert 20%+ revenue to profit.",
                    'unit': '%'
                }
            
            try:
                if len(financials.columns) > 1:
                    eps_current = financials.loc['Basic EPS'].iloc[0]
                    eps_previous = financials.loc['Basic EPS'].iloc[1]
                    ratios['EPS Growth'] = {
                        'value': eps_current / eps_previous if eps_previous != 0 else None,
                        'benchmark': 1.0,
                        'rule': '> 1.0 (Growing)',
                        'logic': "Great companies increase profits yearly.",
                        'unit': ''
                    }
                else:
                    ratios['EPS Growth'] = {
                        'value': None,
                        'benchmark': 1.0,
                        'rule': '> 1.0 (Growing)',
                        'logic': "Great companies increase profits yearly.",
                        'unit': ''
                    }
            except:
                ratios['EPS Growth'] = {
                    'value': None,
                    'benchmark': 1.0,
                    'rule': '> 1.0 (Growing)',
                    'logic': "Great companies increase profits yearly.",
                    'unit': ''
                }
            
            # BALANCE SHEET METRICS
            try:
                cash = balance_sheet.loc['Cash And Cash Equivalents'].iloc[0]
                total_debt = balance_sheet.loc['Total Debt'].iloc[0]
                ratios['Cash vs Debt'] = {
                    'value': cash > total_debt,
                    'cash': cash,
                    'debt': total_debt,
                    'benchmark': True,
                    'rule': 'Cash > Debt',
                    'logic': "Strong financial position.",
                    'unit': ''
                }
            except:
                ratios['Cash vs Debt'] = {
                    'value': None,
                    'benchmark': True,
                    'rule': 'Cash > Debt',
                    'logic': "Strong financial position.",
                    'unit': ''
                }
            
            try:
                total_assets = balance_sheet.loc['Total Assets'].iloc[0]
                total_liabilities = balance_sheet.loc['Total Liabilities'].iloc[0]
                stockholder_equity = balance_sheet.loc['Stockholders Equity'].iloc[0]
                ratios['Debt to Equity'] = {
                    'value': (total_liabilities / stockholder_equity) if stockholder_equity != 0 else None,
                    'benchmark': 0.80,
                    'rule': '< 0.80',
                    'logic': "Lower debt burden.",
                    'unit': ''
                }
            except:
                ratios['Debt to Equity'] = {
                    'value': None,
                    'benchmark': 0.80,
                    'rule': '< 0.80',
                    'logic': "Lower debt burden.",
                    'unit': ''
                }
            
            try:
                capex = cashflow.loc['Capital Expenditure'].iloc[0]
                net_income_ops = financials.loc['Net Income From Continuing Operations'].iloc[0]
                ratios['CapEx Margin'] = {
                    'value': (abs(capex) / net_income_ops) * 100 if net_income_ops != 0 else None,
                    'benchmark': None,
                    'rule': 'Lower is better',
                    'logic': "Less reinvestment needed.",
                    'unit': '%'
                }
            except:
                ratios['CapEx Margin'] = {
                    'value': None,
                    'benchmark': None,
                    'rule': 'Lower is better',
                    'logic': "Less reinvestment needed.",
                    'unit': '%'
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
            'Debt to Equity': (lambda v: v <= 0.80, "Healthy" if v <= 0.80 else "High"),
        }
        
        if ratio_name in ratio_rules:
            check, label = ratio_rules[ratio_name]
            try:
                if check(value):
                    return f"✅ {label}"
                else:
                    return f"⚠️ {label}"
            except:
                return "❓ Error"
        
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
            'current_price': self.info.get('currentPrice', 'N/A'),
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
        'Debt to Equity': {
            'description': 'Total Liabilities / Stockholders Equity',
            'benchmark': 0.80,
            'rule': '< 0.80',
            'logic': 'Lower debt burden means less financial risk and stronger financial position.',
            'interpretation': 'Lower is better. Healthy balance sheet.'
        },
    }