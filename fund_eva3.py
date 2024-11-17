import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib.ticker as mticker

# Load Historical Data
stock_tickers = ['AMZN', 'GOOGL', 'IBM', 'INTC', 'NVDA', 'MSFT', 'TSLA', 'CRM', 'META', 'QCOM']
index_and_rate_tickers = ['^GSPC', '^IRX']
data_dir = 'stock_data'

historical_data = {}
for ticker in stock_tickers + index_and_rate_tickers:
    data_file_path = f"{data_dir}/{ticker}_data.csv"
    historical_data[ticker] = pd.read_csv(data_file_path, index_col=0, parse_dates=True)

# Helper Functions
def get_last_prices(date):
    return np.array([historical_data[ticker].loc[date, 'Adj Close'] for ticker in stock_tickers])

def calculate_predicted_returns(date, models):
    last_changes = {ticker: historical_data[ticker]['Adj Close'].diff().loc[date] for ticker in index_and_rate_tickers}
    y1_last = last_changes['^GSPC']
    y2_last = last_changes['^IRX']
    y3_last = y1_last ** 2
    y4_last = y2_last ** 2
    y5_last = y1_last * y2_last
    factors = np.array([1, y1_last, y2_last, y3_last, y4_last, y5_last])
    predicted_returns = {ticker: models[ticker].predict(factors.reshape(1, -1))[0] for ticker in stock_tickers}
    return predicted_returns

# Prepare the Data and Train Models
models = {}
for ticker in stock_tickers:
    data = historical_data[ticker]
    X = []
    y = []
    for i in range(1, len(data)):
        y1 = historical_data['^GSPC']['Adj Close'].diff().iloc[i]
        y2 = historical_data['^IRX']['Adj Close'].diff().iloc[i]
        X.append([1, y1, y2, y1 ** 2, y2 ** 2, y1 * y2])
        y.append(data['Adj Close'].diff().iloc[i])
    X = np.array(X)
    y = np.array(y)
    model = LinearRegression().fit(X, y)
    models[ticker] = model

# Define the wealth change calculation with equal weights
def wealth_change_curve_with_equal_weights(dates, initial_cash, stock_tickers, models, historical_data):
    wealth_values = [initial_cash]
    n = len(stock_tickers)
    equal_weight = 1 / n

    for date in dates[1:]:
        last_prices = get_last_prices(date)
        predicted_returns = calculate_predicted_returns(date, models)
        predicted_r = np.array(list(predicted_returns.values()))

        wealth_change = np.sum(equal_weight * last_prices * predicted_r)
        new_total_wealth = wealth_values[-1] + wealth_change
        wealth_values.append(new_total_wealth)

    return wealth_values

# Set initial cash to 1
initial_cash = 1
dates = sorted(historical_data[stock_tickers[0]].index)
wealth_change_with_equal_weights = wealth_change_curve_with_equal_weights(dates, initial_cash, stock_tickers, models,
                                                                          historical_data)

# Insert start date and initial value
dates.insert(0, pd.to_datetime("2023-01-01"))
initial_value = 0.0001
wealth_change_with_equal_weights_normalized = [initial_value] + [
    value / wealth_change_with_equal_weights[0] * initial_value for value in wealth_change_with_equal_weights
]

# Shift curve up by 1
wealth_change_with_equal_weights_normalized = [value + 1 for value in wealth_change_with_equal_weights_normalized]

# Plot the normalized wealth change curve
plt.figure(figsize=(10, 6))
plt.plot(dates, wealth_change_with_equal_weights_normalized, marker='^', markersize=5, label="Theoretical Portfolio Value Change")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.title("Theoretical Portfolio Value Change Over Time")
plt.legend()
plt.grid(True)

# Set x-axis to start from January 2023
plt.xlim(pd.to_datetime("2023-01-01"), dates[-1])

# Calculate performance metrics using the same `wealth_change_with_equal_weights_normalized`
initial_value = wealth_change_with_equal_weights_normalized[0]
final_value = wealth_change_with_equal_weights_normalized[-1]
total_return = (final_value - initial_value) / initial_value
annualized_return = total_return * (12 / len(wealth_change_with_equal_weights_normalized)) * 100

monthly_returns = np.diff(wealth_change_with_equal_weights_normalized) / wealth_change_with_equal_weights_normalized[:-1]
std_dev = np.std(monthly_returns) * np.sqrt(12) * 100

sharpe_ratio = annualized_return / std_dev
wealth_array = np.array(wealth_change_with_equal_weights_normalized)
drawdown = (wealth_array - np.maximum.accumulate(wealth_array)) / np.maximum.accumulate(wealth_array)
max_drawdown = drawdown.min() * 100

print("Performance Metrics:")
print(f"Return Rate: {annualized_return:.2f}%")
print(f"Standard Deviation: {std_dev:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Maximum Drawdown: {max_drawdown:.2f}%")

plt.show()

