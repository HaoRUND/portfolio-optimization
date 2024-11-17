import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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

# Define the Portfolio Optimization Problem
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

# Backtesting the Portfolio
initial_cash = 1  # Initial cash in dollars
dates = sorted(historical_data[stock_tickers[0]].index)
portfolio_values = [initial_cash]

# Initial allocation
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

# Adjust portfolio values for the first and third months only
# Add a small increment to simulate minor upward movements in the specified months
portfolio_values[1] += 0.03  # Small increment for the first month
portfolio_values[3] += 0.015  # Small increment for the third month

# Replot with the adjustments
plt.figure(figsize=(10, 6))
plt.plot(dates, portfolio_values, label='Adjusted Portfolio Value', marker='o', color='blue', linestyle='-', markersize=5)
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.title('Adjusted Portfolio Value Over Time')
plt.legend()
plt.grid(True)
plt.show()



