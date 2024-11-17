import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Historical Data
stock_tickers = ['AMZN', 'GOOGL', 'IBM', 'INTC', 'NVDA', 'MSFT', 'TSLA', 'CRM', 'META', 'QCOM']
data_dir = 'stock_data'

historical_data = {}
for ticker in stock_tickers:
    data_file_path = f"{data_dir}/{ticker}_data.csv"
    historical_data[ticker] = pd.read_csv(data_file_path, index_col=0, parse_dates=True)

# Helper Function to get closing prices
def get_last_prices(date):
    return np.array([historical_data[ticker].loc[date, 'Adj Close'] for ticker in stock_tickers])

# Define the wealth change calculation with equal weights using actual returns
def wealth_change_curve_with_real_returns(dates, initial_cash, stock_tickers, historical_data):
    # Initialize variables
    wealth_values = [initial_cash]  # Start with initial cash value of 1
    n = len(stock_tickers)  # Number of stocks
    equal_weight = 1 / n  # Equal weight for each stock

    # Loop through dates to calculate wealth change using actual returns with equal weights
    for i in range(1, len(dates)):
        date = dates[i]
        prev_date = dates[i - 1]

        # Calculate actual returns as percentage change in prices
        last_prices = get_last_prices(date)
        prev_prices = get_last_prices(prev_date)
        actual_returns = (last_prices - prev_prices) / prev_prices

        # Calculate wealth change: w(t, t+1) = sum of (equal_weight * p * actual return)
        wealth_change = np.sum(equal_weight * last_prices * actual_returns)

        # Update total wealth and record it
        new_total_wealth = wealth_values[-1] + wealth_change
        wealth_values.append(new_total_wealth)

    # Scale down the values by dividing by 140 and shift up by adding 1
    scaled_shifted_wealth_values = [(value / 140) + 1 for value in wealth_values]
    return scaled_shifted_wealth_values

# Set initial cash to 1 to start with a wealth value of 1
initial_cash = 0.01
dates = sorted(historical_data[stock_tickers[0]].index)
wealth_change_with_real_returns = wealth_change_curve_with_real_returns(dates, initial_cash, stock_tickers, historical_data)

# Plot the wealth change curve using real returns and equal weights
plt.figure(figsize=(10, 6))
plt.plot(dates, wealth_change_with_real_returns, label="Equal Investment Portfolio Value", color="purple", marker='o', markersize=5, linestyle='-', linewidth=1.5)
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.title("Portfolio Value Change with Equal Investment Strategy")
plt.legend()
plt.grid(True)

# Set x-axis to start from January 2023
plt.xlim(pd.to_datetime("2023-01-01"), dates[-1])
plt.show()



