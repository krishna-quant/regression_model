# SPY Global Indices Linear Regression Strategy

Predict next-day SPY open-to-open return using a multiple linear regression on global index signals (US, Europe, Asia), then trade a simple long/short signal and evaluate performance (Sharpe, max drawdown).

## Why this project
- Recreates a core quant workflow: data → features → OLS model → backtest → risk metrics.
- Clean, reproducible code with a toggle for **live data** vs **snapshot CSV**.

## Features
- Pulls historical prices from Yahoo Finance (`yfinance`)
- Builds predictors:
  - US (SPY, S&P 500, Nasdaq, Dow): 1-day open-to-open return (lagged)
  - Europe (DAX, CAC40): 1-day open-to-open return (lagged)
  - Asia (Nikkei, HSI, AORD): prior day **(Close – Open) / Open**
- Fits **OLS** with `statsmodels`
- Generates a sign‐based long/short strategy
- Computes **Sharpe Ratio** and **Max Drawdown**
- Plots cumulative wealth (with optional 5-day smoothing)

## Project Structure
- regression_model.py # main script
- requirements.txt # dependencies
- README.md # project description
- .gitignore # ignore cache/data/etc.
- indicepanel.csv # created automatically when live=True
