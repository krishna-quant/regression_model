import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Step 1: Download & Prepare Global Data

def download_global_data(start="2015-01-01", end=None, live=True):
    tickers = {
        "SPY": "SPY",
        "S&P500": "^GSPC",
        "Nasdaq": "^IXIC",
        "Dow": "^DJI",
        "DAX": "^GDAXI",
        "CAC40": "^FCHI",
        "Nikkei": "^N225",
        "HSI": "^HSI",
        "AORD": "^AORD"
    }

    if live:
        end = pd.Timestamp.today().strftime('%Y-%m-%d')  # use today's date if live

        data = {}
        for name, ticker in tickers.items():
            df = yf.download(ticker, start=start, end=end)[['Open', 'Close']]
            data[name] = df

        df_final = pd.DataFrame(index=data['SPY'].index)

        # Response: SPY next-day open change
        df_final['SPY_Change'] = (data['SPY']['Open'].shift(-1) - data['SPY']['Open']) / data['SPY']['Open']

        # US markets (1-day lag)
        for name in ['SPY', 'S&P500', 'Nasdaq', 'Dow']:
            df_final[f'{name}_Lag1'] = (data[name]['Open'] - data[name]['Open'].shift(1)) / data[name]['Open'].shift(1)

        # European markets (1-day lag)
        for name in ['DAX', 'CAC40']:
            df_final[f'{name}_Lag1'] = (data[name]['Open'] - data[name]['Open'].shift(1)) / data[name]['Open'].shift(1)

        # Asian markets (Open-Close previous day)
        for name in ['Nikkei', 'HSI', 'AORD']:
            df_final[f'{name}_OC'] = (data[name]['Close'] - data[name]['Open']) / data[name]['Open']

        df_final = df_final.dropna()

        # Save CSV snapshot
        df_final.to_csv("indicepanel.csv")

    else:
        # load from saved file
        df_final = pd.read_csv("indicepanel.csv", index_col=0, parse_dates=True)

    # Build X, y
    X = df_final.drop('SPY_Change', axis=1)
    y = df_final['SPY_Change']

    return X, y

# Step 2: Train/Test Split

def split_train_test(X, y, train_size=1000, test_size=1000):
    X_train = X.iloc[-(train_size + test_size):-test_size]
    X_test = X.iloc[-test_size:]
    y_train = y.iloc[-(train_size + test_size):-test_size]
    y_test = y.iloc[-test_size:]
    return X_train, X_test, y_train, y_test

# Step 3: Fit OLS

def fit_ols(X, y):
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const)
    results = model.fit()
    return results

# Step 4: Generate Trading Signals

def generate_signals(model, X):
    X_const = sm.add_constant(X)
    pred = model.predict(X_const)
    position = np.where(pred > 0, 1, -1)  # Long if positive, short if negative
    return pred, position

# Step 5: Compute Profit and Wealth

def compute_profit(position, y):
    profit = position * y
    wealth = (profit + 1).cumprod()  # cumulative wealth
    return profit, wealth

# Step 6: Compute Sharpe Ratio

def compute_sharpe(profit):
    daily_return = profit
    return (daily_return.mean() / daily_return.std()) * np.sqrt(252)

# Step 7: Maximum Drawdown

def max_drawdown(wealth):
    peak = wealth.cummax()
    drawdown = (wealth - peak) / peak
    return drawdown.min()

# Step 8: Plot Wealth

def plot_wealth(wealth_train, wealth_test):
    plt.figure(figsize=(12,6))
    plt.plot(wealth_train.index, wealth_train, label="Train Wealth")
    plt.plot(wealth_test.index, wealth_test, label="Test Wealth")

    # Smoothed (5-day moving average) versions
    plt.plot(wealth_train.index, wealth_train.rolling(window=5).mean(), label="Train Wealth (5-day MA)", linestyle="--")       
    plt.plot(wealth_test.index, wealth_test.rolling(window=5).mean(), label="Test Wealth (5-day MA)", linestyle="--")
    plt.title("Cumulative Wealth of Signal-Based Strategy")
    plt.xlabel("Date")
    plt.ylabel("Wealth")
    plt.legend()
    plt.show()


    # Main Execution

def main():
    X, y = download_global_data()
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    results = fit_ols(X_train, y_train)
    print(results.summary())
    
    _, position_train = generate_signals(results, X_train)
    _, position_test = generate_signals(results, X_test)
    
    profit_train, wealth_train = compute_profit(position_train, y_train)
    profit_test, wealth_test = compute_profit(position_test, y_test)
    
    print(f"Sharpe Ratio Train: {compute_sharpe(profit_train):.2f}")
    print(f"Sharpe Ratio Test: {compute_sharpe(profit_test):.2f}")
    
    print(f"Max Drawdown Train: {max_drawdown(wealth_train):.2%}")
    print(f"Max Drawdown Test: {max_drawdown(wealth_test):.2%}")
    
    plot_wealth(wealth_train, wealth_test)

if __name__ == "__main__":
    main()