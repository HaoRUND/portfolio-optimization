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

# First Plot: Equal Investment Portfolio Value
def wealth_change_curve_with_real_returns(dates, initial_cash, stock_tickers, historical_data):
    wealth_values = [initial_cash]
    n = len(stock_tickers)
    equal_weight = 1 / n

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

initial_cash = 1
dates = sorted(historical_data[stock_tickers[0]].index)
wealth_values = wealth_change_curve_with_real_returns(dates, initial_cash, stock_tickers, historical_data)

# Second Plot: Adjusted Portfolio Value
def optimize_portfolio(date, models, last_prices):
    predicted_returns = calculate_predicted_returns(date, models)
    mu = np.array(list(predicted_returns.values()))
    Sigma = np.cov([historical_data[ticker]['Adj Close'].pct_change().dropna() for ticker in stock_tickers])

    n = len(stock_tickers)
    w = cp.Variable(n)
    gamma = cp.Parameter(nonneg=True)

    portfolio_return = mu.T @ w
    portfolio_risk = cp.quad_form(w, Sigma)
    objective = cp.Maximize(portfolio_return - gamma * portfolio_risk)
    constraints = [cp.sum(w) == 1, w >= 0]

    prob = cp.Problem(objective, constraints)
    gamma.value = 1
    prob.solve()

    return w.value

portfolio_values = [initial_cash]
current_cash = initial_cash
current_portfolio = np.zeros(len(stock_tickers))

for date in dates[1:]:
    last_prices = get_last_prices(date)
    optimal_weights = optimize_portfolio(date, models, last_prices)
    current_portfolio_value = np.sum(current_portfolio * last_prices)
    current_cash += current_portfolio_value

    current_portfolio = (current_cash * optimal_weights) / last_prices
    current_cash = 0
    portfolio_values.append(np.sum(current_portfolio * last_prices))

# Third Plot: Theoretical Portfolio Value Change
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

wealth_change_with_equal_weights = wealth_change_curve_with_equal_weights(dates, initial_cash, stock_tickers, models, historical_data)

# Insert start date and initial values for each series to ensure consistent lengths
dates.insert(0, pd.to_datetime("2023-01-01"))
wealth_values.insert(0, initial_cash)
portfolio_values.insert(0, initial_cash)
initial_value = 0.0001
wealth_change_with_equal_weights_normalized = [initial_value] + [
    value / wealth_change_with_equal_weights[0] * initial_value for value in wealth_change_with_equal_weights
]
wealth_change_with_equal_weights_normalized = [value + 1 for value in wealth_change_with_equal_weights_normalized]

# Plot all three curves in the same figure without any modification
plt.figure(figsize=(12, 8))
plt.plot(dates, wealth_change_with_equal_weights_normalized, marker='^', markersize=5, label="Theoretical Portfolio Value Change", linestyle='-', linewidth=1.5)
plt.plot(dates, portfolio_values, label='Actual Portfolio Value', marker='o', color='blue', markersize=5)
plt.plot(dates, wealth_values, label="Equal Investment Portfolio Value", color="purple", marker='o', markersize=5, linestyle='-', linewidth=1.5)
# Set labels and title
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.title("Comparison of Portfolio Value Changes Over Time")
plt.legend()
plt.grid(True)
plt.xlim(pd.to_datetime("2023-01-01"), dates[-1])

plt.show()
