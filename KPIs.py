#!/usr/bin/env python3
"""
Created on Tue May 16 16:17:38 2023
# -*- coding: utf-8 -*-

@author: nacho
"""
#%% Import necessary libraries

import pandas as pd
import numpy as np
import datetime as dt
from pandas_market_calendars import get_calendar
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import spyder
print(spyder.__version__)

#%% Read and clean the csv from TD Ameritrade Account, the entire balance history

balance_history = pd.read_csv('2024_04_30.csv')    
balance_history = balance_history.copy()

# Remove last 3 rows, irrelevant
balance_history = balance_history.iloc[:-3]

# Convert the "Account value" column to a numeric data type, and rename it
balance_history['Account value'] = pd.to_numeric(balance_history['Account value'].str.replace(',', ''))
balance_history = balance_history.rename(columns={'Account value': 'balance'})

# Set the index to the "Date" column, and to datetime
balance_history['Date'] = pd.to_datetime(balance_history['Date'])
balance_history.set_index('Date', inplace=True)

#%% Import schwab data

# Load the data, skipping the first row and specifying no header initially
schwab_data = pd.read_csv('balance_history.csv', header=None, skiprows=1)

# Rename columns explicitly
schwab_data.columns = ['date', 'balance']

# Reverse the DataFrame order
schwab_data = schwab_data.iloc[::-1].reset_index(drop=True)

# Split the 'date' column to remove "$" and "," then convert it to float
schwab_data['balance'] = schwab_data['balance'].replace('[\$,]', '', regex=True).astype(float)

# Set the index to the "Date" column, and to datetime
schwab_data['date'] = pd.to_datetime(schwab_data['date'])
schwab_data.set_index('date', inplace=True)

#%% Add 12 days of missing data from migration (taking avg increase per day to end in starting balance)

missing_schwab_data = {
    "date": [
        "05/01/2024", "05/02/2024", "05/03/2024", "05/05/2024",
        "05/06/2024", "05/07/2024", "05/08/2024", "05/09/2024", "05/12/2024",
    ],
    "balance": [
        "$53,696.99", "$53,796.99", "$53,896.99", "$53,996.99",
        "$53,896.99", "$53,796.99", "$53,996.99", "$54,100.99", "$54,296.99",
    ]
}

missing_schwab_df = pd.DataFrame(missing_schwab_data)
missing_schwab_df['date'] = pd.to_datetime(missing_schwab_df['date'], format='%m/%d/%Y')
missing_schwab_df['balance'] = pd.to_numeric(missing_schwab_df['balance'].str.replace('[\$,]', '', regex=True))
missing_schwab_df.set_index('date', inplace=True)

# Append missing data to schwab data
schwab_data = pd.concat([missing_schwab_df, schwab_data, ])

# Filter out weekends
schwab_data = schwab_data[schwab_data.index.dayofweek < 5]

schwab_data.tail(20)
schwab_data.info()

#%% Merge and add cash flows

schwab_balance_history = pd.DataFrame(schwab_data)
#schwab_balance_history['Date'] = pd.to_datetime(schwab_balance_history['Date'], format='%m/%d/%Y')
#schwab_balance_history['Total Value'] = pd.to_numeric(schwab_balance_history['Total Value'].str.replace('[\$,]', '', regex=True))
#schwab_balance_history.set_index('Date', inplace=True)

# Append Schwab data to the original balance history
combined_balance_history = pd.concat([balance_history, schwab_balance_history])

# Sort the combined DataFrame by date
combined_balance_history = combined_balance_history.sort_index()

balance_history = combined_balance_history

# Create dictionary of cash flows, then add them to the df
cash_flows = {'2020-10-21': 9748.00, '2020-11-04': 6002.00, '2021-01-11': 6948.00, 
 '2021-03-31': 4000.00, '2021-12-03': 4000.00, '2022-03-04': 1990.00, 
 '2022-04-29': 1960.00, '2022-12-02': -4025.00, '2023-04-19': -2025.00,'2023-06-12': -2050.00,
 '2023-12-20': 9950.00, '2024-04-30': 24940.00, '2024-10-10': 12950.00, '2025-04-04': 25120.00,
 '2025-07-03': -4504.09}

for date, amount in cash_flows.items():
    balance_history.loc[date, 'cash_flow'] = amount
    
# Fill null CF values, then check create column for net sum of CFs  
balance_history['cash_flow'].fillna(0, inplace=True)
balance_history['net_cash_flow'] = balance_history['cash_flow'].cumsum()

balance_history.info()

balance_history.tail(25)

#%% Select CUTOFF

balance_history = balance_history.iloc[:-2]

# Specify the cutoff date
cutoff_date = '2025-12-31'
# Find the integer location of the cutoff date in the index
cutoff_loc = balance_history.index.get_loc(cutoff_date)
# Slice the DataFrame up to (and including) the cutoff date
balance_history = balance_history.iloc[:cutoff_loc+1]
balance_history

#%% 2025 YTD

# Filter balance for 2023 only, then add daily returns column
balance_history_2025 = balance_history.loc[(balance_history.index.year == 2025), :]
balance_history_2025['daily_return'] = (balance_history_2025['balance'].diff() - balance_history_2025['cash_flow']) / balance_history_2025['balance'].shift(1)

# Time weighted return, TWR
balance_history_2025['twr'] = (1 + balance_history_2025['daily_return']).cumprod() - 1
twr_2025 = balance_history_2025['twr'][-1]

# Sharpe ratio YTD 
daily_std_2025 = balance_history_2025['daily_return'].std()
annualized_vol_2025 = daily_std_2025 * np.sqrt(252)

# Get the number of trading days YTD
start_date_25 = dt.datetime.strptime('31/12/2024', '%d/%m/%Y')
today = pd.Timestamp.today().date()

# Get NYSE calendar
trading_calendar = get_calendar('NYSE')
trading_days_2025 = trading_calendar.valid_days(start_date_25, today)
trading_days_2025 = len(trading_days_2025)
trading_days_2025

# 3month Tbill at 01.04.2025 = 5.44 , https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics?data=billrates
rf_2025 = 0.0544
twr_2025_annualized = (1 + twr_2025) ** (252/trading_days_2025) - 1
twr_2025_annualized

# Sharpe

sr_ytd = (twr_2025_annualized-rf_2025)/annualized_vol_2025
sr_ytd

# Sortino

neg_return_ytd = np.where(balance_history_2025["daily_return"]>0,0,balance_history_2025["daily_return"])
neg_vol_ytd = np.sqrt((pd.Series(neg_return_ytd[neg_return_ytd != 0]) ** 2).mean() * 252)
sortino_ytd = (twr_2025_annualized-rf_2025)/neg_vol_ytd
sortino_ytd

# Max Drawdown

balance_history_2025["cum_roll_max"] = balance_history_2025["twr"].cummax()
balance_history_2025["drawdown"] = balance_history_2025["cum_roll_max"] - balance_history_2025["twr"]
max_dd_ytd = balance_history_2025["drawdown"].max()
max_dd_ytd
    
# Calmar Ratio

calmar_ytd = twr_2025_annualized/max_dd_ytd
calmar_ytd

#%% Create list to save plots, PDF collector

figures_to_pdf = []

#%% PLOTS

# Daily returns and daily returns histogram

# Create a figure and 2 subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
# Get 'daily_return' data
daily_return = balance_history_2025['daily_return']

# Plot 'daily_return' on the first subplot (ax1)
ax1.plot(balance_history_2025.index, balance_history_2025['daily_return'], color='blue')
ax1.set_ylabel('Daily Return')
ax1.set_title('Daily Returns')

# Plot 'daily_return' as a histogram on the second subplot (ax2)
# Define bin edges with a separation of 0.25%
bin_width = 0.002
bin_edges = np.arange(-0.04, 0.07, bin_width)
ax2.hist(daily_return, bins=bin_edges, color='blue', edgecolor='black')
ax2.set_ylabel('Frequency')
ax2.set_title('Daily Return Histogram')

# Adjust layout
plt.tight_layout()

# Append to PDF list
figures_to_pdf.append(fig)

# Show the plot
plt.show()

#%% BENCHMARKS

import yfinance as yf
import pandas as pd

tickers = ["SPY", "QQQ", "IWM", "RSP", "ACWI", "IAU"]
ohlcv_data = {}

for ticker in tickers:
    # Download historical data with auto_adjust=False to try and obtain "Adj Close"
    ticker_obj = yf.Ticker(ticker)
    df = ticker_obj.history(start="2024-12-31", end="2026-01-01", interval='1d', auto_adjust=False)
    
    # Flatten MultiIndex columns if they exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # If "Adj Close" is missing, create it by copying the "Close" column
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]
    
    ohlcv_data[ticker] = df

# Optional: print columns for one ticker to verify
for ticker, data in ohlcv_data.items():
    print(f"{ticker} columns: {data.columns.tolist()}")

#%% KPI functions

def TWR(DF): 
    df = DF.copy()
    df["return"] = df["Adj Close"].pct_change()
    df["cum_return"] = (1 + df["return"]).cumprod() -1
    twr = (df["cum_return"][-1]) 
    return twr

def CAGR(DF): 
    df = DF.copy()
    df["return"] = df["Adj Close"].pct_change()
    df["cum_return"] = (1 + df["return"]).cumprod()
    n = len(df)/252
    cagr = (df["cum_return"][-1]) ** (1/n) - 1
    return cagr
    
def volatility(DF):
    "function to calculate annualized volatility of a trading strategy"
    df = DF.copy()
    df["daily_ret"] = DF["Adj Close"].pct_change()
    vol = df["daily_ret"].std() * np.sqrt(252)
    return vol

def sharpe(DF, rf):
    "function to calculate Sharpe Ratio of a trading strategy"
    df = DF.copy()
    return (CAGR(df) - rf)/volatility(df)

def sortino(DF, rf):
    "function to calculate Sortino Ratio of a trading strategy"
    df = DF.copy()
    df["return"] = df["Adj Close"].pct_change()
    neg_return = np.where(df["return"]>0,0,df["return"])
    #below you will see three ways to calculate the denominator (neg_vol), some people use the
    #standard deviation of negative returns while others use a downward deviation approach,
    #you can use either. However, downward deviation approach is more widely used
    neg_vol = np.sqrt((pd.Series(neg_return[neg_return != 0]) ** 2).mean() * 252)
    #neg_vol = pd.Series(neg_return[neg_return != 0]).std() * np.sqrt(252)
    return (CAGR(df) - rf)/neg_vol

# -- Sortino only takes into account the STD of negative asset returns (bad days)
# Will increase sharpe ratio, if Sharpe negative, sortino will be more negative
# if Sharpe positive, sortino will be more positive

def max_dd(DF):
    "function to calculate max drawdown"
    df = DF.copy()
    df["return"] = df["Adj Close"].pct_change()
    df["cum_return"] = (1+df["return"]).cumprod()
    df["cum_roll_max"] = df["cum_return"].cummax()
    df["drawdown"] = df["cum_roll_max"] - df["cum_return"]
    return (df["drawdown"]/df["cum_roll_max"]).max()
    
def calmar(DF):
    "function to calculate calmar ratio"
    df = DF.copy()
    return CAGR(df)/max_dd(df)

#%%% Benchmarking  KPIs

for ticker in ohlcv_data:
    print("TWR_YTD for {} = {}".format(ticker, TWR(ohlcv_data[ticker])))
    print("CAGR_YTD for {} = {}".format(ticker, CAGR(ohlcv_data[ticker])))
    print("vol for {} = {}".format(ticker,volatility(ohlcv_data[ticker])))
    print("Sharpe of {} = {}".format(ticker,sharpe(ohlcv_data[ticker],rf_2025)))
    print("Sortino of {} = {}".format(ticker,sortino(ohlcv_data[ticker],rf_2025)))
    print("Max drawdown of {} = {}".format(ticker,max_dd(ohlcv_data[ticker])))
    print("Calmar ratio of {} = {}".format(ticker,calmar(ohlcv_data[ticker])))
    
#%%% Summary of KPIs

# Define empty lists to store the KPIs
tickers = []
twr = []
vol = []
sharpe_ratio = []
sortino_ratio = []
max_drawdown = []
calmar_ratio = []

# Calculate the KPIs for each ticker
for ticker in ohlcv_data:
    tickers.append(ticker)
    twr.append(TWR(ohlcv_data[ticker]))
    vol.append(volatility(ohlcv_data[ticker]))
    sharpe_ratio.append(sharpe(ohlcv_data[ticker], rf_2025))
    sortino_ratio.append(sortino(ohlcv_data[ticker], rf_2025))
    max_drawdown.append(max_dd(ohlcv_data[ticker]))
    calmar_ratio.append(calmar(ohlcv_data[ticker]))

# Create a DataFrame with the metrics
kpis_df = pd.DataFrame({
    'Ticker': tickers,
    'TWR_YTD': twr,
    'Volatility': vol,
    'Sharpe_Ratio': sharpe_ratio,
    'Sortino_Ratio': sortino_ratio,
    'max_drawdown': max_drawdown,
    'Calmar_Ratio': calmar_ratio})

# Add your Nacho portfolio KPIs
kpis_nacho = {
    'Ticker': 'Fund',
    'TWR_YTD': twr_2025,
    'Volatility': annualized_vol_2025,
    'Sharpe_Ratio': sr_ytd,
    'Sortino_Ratio': sortino_ytd,
    'max_drawdown': max_dd_ytd,
    'Calmar_Ratio': calmar_ytd
}

kpis_nacho_df = pd.DataFrame([kpis_nacho])

# Combine everything
kpis_all = pd.concat([kpis_df, kpis_nacho_df], ignore_index=True)

# Sort by Sharpe Ratio
sorted_kpis = kpis_all.sort_values(by='Sharpe_Ratio', ascending=False)

# >>> Display full table (Option 2: temporary display settings)
with pd.option_context('display.max_columns', None,
                       'display.max_rows', None,
                       'display.width', None,
                       'display.max_colwidth', None):
    print(sorted_kpis)

#%% Plot KPIs TABLE

plot_df = sorted_kpis.copy()

# Drop Calmar column
plot_df = plot_df.drop(columns=["Calmar_Ratio"])

# Format percentage columns
pct_cols = ["TWR_YTD", "Volatility", "max_drawdown"]
plot_df[pct_cols] = plot_df[pct_cols] * 100

# Round for display
plot_df = plot_df.round({
    "TWR_YTD": 2,
    "Volatility": 2,
    "Sharpe_Ratio": 3,
    "Sortino_Ratio": 3,
    "max_drawdown": 2
})

fig, ax = plt.subplots(figsize=(10, 3))
ax.axis("off")

table = ax.table(
    cellText=plot_df.values,
    colLabels=plot_df.columns,
    loc="center",
    cellLoc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.4)

ax.set_title("KPI Summary (YTD 2025)", fontsize=12, pad=10)

# Disclosure (Excel-style footnote)
disclosure_text = (
    "SPY=SP500, QQQ=Nasdaq100, IWM=Russell2000, "
    "RSP=SP500EqualWeight, ACWI=WorldIndex, IAU=Gold"
)

fig.text(
    0.01, -0.05, disclosure_text,
    fontsize=9, ha="left", va="top"
)

plt.tight_layout()

# Append to PDF list
figures_to_pdf.append(fig)

plt.show()

#%% Plot benchmark performances

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

def TWR_plot_YTD(tickers, ohlcv_data, fondo=None):
    fig, ax = plt.subplots(figsize=(15, 15))

    # List to store the tickers and their final returns
    performance_data = []

    for ticker in tickers:
        # Retrieve the DataFrame for the current ticker from the dictionary
        ticker_data = ohlcv_data[ticker]
        # Calculate the daily returns
        ticker_data["return"] = ticker_data["Adj Close"].pct_change()
        # Calculate the cumulative returns
        ticker_data["cum_return"] = (1 + ticker_data["return"]).cumprod() - 1
        # Get the final cumulative return percentage
        final_return = ticker_data["cum_return"].iloc[-1] * 100
        # Store the ticker and its final return
        performance_data.append((ticker, final_return, ticker_data.index, ticker_data["cum_return"]))
    
    # Add the fondo performance if provided
    if fondo is not None:
        fondo_data = fondo.copy()
        final_return_fondo = fondo_data['twr'].iloc[-1] * 100
        performance_data.append(("Fund", final_return_fondo, fondo_data.index, fondo_data['twr']))
    
    # Sort the performance data by final return in descending order
    performance_data.sort(key=lambda x: x[1], reverse=True)

    # Plot the cumulative returns in sorted order
    for ticker, final_return, index, cum_return in performance_data:
        ax.plot(index, cum_return, label=f"{ticker} ({final_return:.2f}%)")

    ax.set_title('Performance')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.grid(True)
    ax.legend()

    # Set the date format for x-axis tick labels, to show only month name
    date_format = mdates.DateFormatter("%b")
    ax.xaxis.set_major_formatter(date_format)

    # Ensure that the date labels fit nicely
    fig.autofmt_xdate()
    
    # Add the disclosure below the x-axis, aligned with the origin of the graph
    disclosure_text = ("SPY=SP500, QQQ=Nasdaq100, IWM=Russell2000, "
                       "RSP=SP500EqualWeight, ACWI=WorldIndex, IAU=Gold")
    ax.annotate(disclosure_text, xy=(0, -0.18), xycoords='axes fraction', fontsize=10,
                verticalalignment='top', horizontalalignment='left')
    
    # Adjust layout to minimize padding
    plt.subplots_adjust(bottom=0.5)
    # Append to PDF list
    figures_to_pdf.append(fig)
    plt.show()

# Example call with correct fondo DataFrame
TWR_plot_YTD(tickers, ohlcv_data, fondo=balance_history_2025)

#%%% Plot vs. SPY

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

def TWR_plot_YTD(ohlcv_data, fondo=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # List to store the tickers and their final returns
    performance_data = []

    # Plot SPY performance
    spy_data = ohlcv_data["SPY"]
    spy_data["return"] = spy_data["Adj Close"].pct_change()
    spy_data["cum_return"] = (1 + spy_data["return"]).cumprod() - 1
    final_return_spy = spy_data["cum_return"].iloc[-1] * 100
    performance_data.append(("SPY", final_return_spy, spy_data.index, spy_data["cum_return"]))
    
    # Add your own performance if provided
    if fondo is not None:
        fondo_data = fondo.copy()  # Assuming fondo is already provided as a DataFrame
        final_return_fondo = fondo_data['twr'].iloc[-1] * 100
        performance_data.append(("Fund", final_return_fondo, fondo_data.index, fondo_data['twr']))
    
    # Sort the performance data by final return in descending order
    performance_data.sort(key=lambda x: x[1], reverse=True)

    # Plot the cumulative returns in sorted order
    for ticker, final_return, index, cum_return in performance_data:
        ax.plot(index, cum_return, label=f"{ticker} ({final_return:.2f}%)")

    ax.set_title('Performance')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.grid(True)
    ax.legend()
    
    # Append to PDF list
    figures_to_pdf.append(fig)
    plt.show()

# Example call with correct fondo DataFrame
TWR_plot_YTD(ohlcv_data, fondo=balance_history_2025)

#%% PLOTS: Rolling Volatility

window = 30

# Your portfolio
volatility_roll = balance_history_2025['daily_return'].rolling(window=window).std() * np.sqrt(252)

# SPY rolling volatility
spy_data = ohlcv_data['SPY']
spy_data['daily_return'] = spy_data['Adj Close'].pct_change()
volatility_roll_spy = spy_data['daily_return'].rolling(window=window).std() * np.sqrt(252)

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 6))
volatility_roll.plot(ax=ax, color='blue', linewidth=2, label='Fund')
volatility_roll_spy.plot(ax=ax, color='red', linestyle='--', linewidth=2, label='SPY')

ax.set_title('30-day Rolling Volatility')
ax.set_xlabel('Date')
ax.set_ylabel('Volatility')
ax.legend()
plt.tight_layout()
plt.show()

# Append to PDF list
figures_to_pdf.append(fig)

#%% EXPORT ALL PLOTS TO PDF

pdf_path = "KPIs_PDF.pdf"

with PdfPages(pdf_path) as pdf:
    for fig in figures_to_pdf:
        pdf.savefig(fig, bbox_inches="tight")
        
print(f"Saved PDF with {len(figures_to_pdf)} figures -> {pdf_path}")
