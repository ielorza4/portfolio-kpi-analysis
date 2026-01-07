# Portfolio KPI Analysis (Python)

Python-based portfolio performance and risk analysis framework using
daily account balances and benchmark ETFs.

The script adjusts for external cash flows and evaluates portfolio
performance relative to major equity indices, producing an investor-ready
PDF report.

## Overview
- Uses daily account balances (Ameritrade → Schwab migration)
- Adjusts returns for deposits and withdrawals
- Benchmarks performance against major ETFs using market data
- Generates visual diagnostics and a consolidated PDF report

## Metrics
- Time-Weighted Return (YTD)
- Annualized Volatility
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Rolling Volatility
- Benchmark-relative performance (SPY, QQQ, ACWI, IWM, RSP, IAU)

## Data
- Portfolio data derived from daily account balances
- Benchmark prices downloaded via `yfinance`

## Files
- `kpi_analysis.py` — main analysis script
- `balances_2024.csv` — account balance history (anonymized)
- `balance_history.csv` — account balance history (anonymized)
- `output/KPI_Report_YTD_2025.pdf` — example output

## How to Run
```bash
pip install pandas numpy matplotlib yfinance pandas-market-calendars
python kpi_analysis.py
