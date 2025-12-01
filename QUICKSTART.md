# AppleBee Dashboard - Quick Start

## Installation
1. pip install -r requirements.txt
2. streamlit run warren_buffett_dashboard.py

## Usage
1. Type stock symbol (AAPL, MSFT, TSLA)
2. View the analysis
3. Check the scorecard

## Try These Stocks
- AAPL (Apple) - Usually scores 6-7/8
- MSFT (Microsoft) - Usually scores 7-8/8  
- TSLA (Tesla) - Usually scores 3-5/8
- KO (Coca-Cola) - Usually scores 5-6/8

## Understanding Scores
- 7-8/8: Excellent (Buffett-quality)
- 5-6/8: Good (solid fundamentals)
- 3-4/8: Fair (mixed signals)
- <3/8: Weak (doesn't fit criteria)

## Metrics Quick Reference
- **Gross Margin**: Should be ≥ 40%
- **SG&A Expense**: Should be ≤ 30%
- **Net Profit Margin**: Should be ≥ 20%
- **Debt-to-Equity**: Should be < 0.80
- **EPS Growth**: Should be > 1.0

## Common Issues
- "ModuleNotFoundError": Run pip install -r requirements.txt
- "Stock symbol not found": Check spelling, try AAPL
- "Metric shows N/A": Normal if company doesn't report that data