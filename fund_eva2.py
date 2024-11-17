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
    wealth_values = [initial_cash]  # Start with initial cash value of 1
    n = len(stock_tickers)  # Number of stocks
    equal_weight = 1 / n  # Equal weight for each stock

    for i in range(1, len(dates)):
        date = dates[i]
        prev_date = dates[i - 1]

        last_prices = get_last_prices(date)
        prev_prices = get_last_prices(prev_date)
        actual_returns = (last_prices - prev_prices) / prev_prices

        wealth_change = np.sum(equal_weight * actual_returns) * wealth_values[-1]
        new_total_wealth = wealth_values[-1] + wealth_change
        wealth_values.append(new_total_wealth)

    return wealth_values

# Set initial cash to 1 to start with a wealth value of 1
initial_cash = 1
dates = sorted(historical_data[stock_tickers[0]].index)
wealth_values = wealth_change_curve_with_real_returns(dates, initial_cash, stock_tickers, historical_data)

# Calculate performance metrics
# Annualized return rate
total_return = (wealth_values[-1] / wealth_values[0]) - 1
annualized_return = (1 + total_return) ** (12 / len(dates)) - 1  # Monthly compounding assumption

# Standard deviation of monthly returns
monthly_returns = [wealth_values[i] / wealth_values[i - 1] - 1 for i in range(1, len(wealth_values))]
std_dev = np.std(monthly_returns) * np.sqrt(12)  # Annualized standard deviation

# Sharpe ratio (Assume a risk-free rate of 0 for simplicity)
sharpe_ratio = annualized_return / std_dev

# Maximum drawdown
wealth_array = np.array(wealth_values)
drawdown = (wealth_array - np.maximum.accumulate(wealth_array)) / np.maximum.accumulate(wealth_array)
max_drawdown = drawdown.min()

# Display the performance metrics
print("Performance Metrics:")
print(f"Return Rate: {annualized_return * 100:.2f}%")
print(f"Standard Deviation: {std_dev * 100:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Maximum Drawdown: {max_drawdown * 100:.2f}%")

# Plot the wealth change curve using real returns and equal weights
plt.figure(figsize=(10, 6))
plt.plot(dates, wealth_values, label="Equal Investment Portfolio Value", color="purple", marker='o', markersize=5, linestyle='-', linewidth=1.5)
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.title("Portfolio Value Change with Equal Investment Strategy")
plt.legend()
plt.grid(True)

# Set x-axis to start from January 2023
plt.xlim(pd.to_datetime("2023-01-01"), dates[-1])
plt.show()

