import pickle
import os

# 定义存储模型系数的字典
models_coefficients = {}

# 指定模型文件夹路径
models_dir = 'trained_models'

# 遍历模型文件夹中的所有文件
for filename in os.listdir(models_dir):
    if filename.endswith('.pkl'):  # 确保文件是以.pkl结尾的模型文件
        # 构建模型文件的完整路径
        model_file_path = os.path.join(models_dir, filename)
        # 加载模型
        with open(model_file_path, 'rb') as model_file:
            model = pickle.load(model_file)
        # 提取模型系数
        coefficients = model.params
        # 提取截距项
        intercept = coefficients['const']
        # 提取自变量系数
        variables_coefficients = coefficients.drop('const')
        # 将系数存储到字典中
        models_coefficients[filename[:-10]] = (intercept, variables_coefficients)

# 打印模型系数
#for ticker, (intercept, variables_coefficients) in models_coefficients.items():
 #   print(f"{ticker}:")
  #  print(f"Intercept: {intercept}")
   # print("Variable Coefficients:")
    #print(variables_coefficients)
    #print()

import pandas as pd
import os

# 定义股票的列表，不包括指数和利率
stock_tickers = ['AMZN', 'GOOGL', 'IBM', 'INTC', 'NVDA', 'MSFT', 'TSLA', 'CRM', 'META', 'QCOM']

# 指数和利率的
index_and_rate_tickers = ['^GSPC', '^IRX']

# 加载数据
data_dir = 'stock_data'

# 存储最后一次变化量的字典
last_changes = {}

# 遍历每个股票和指数/利率
for ticker in stock_tickers + index_and_rate_tickers:
    try:
        # 加载数据
        data_file_path = os.path.join(data_dir, f"{ticker}_data.csv")
        data = pd.read_csv(data_file_path, index_col=0, parse_dates=True)

        # 计算变化量
        change = data['Adj Close'].diff()

        # 获取最后一次的变化量
        last_change = change.iloc[-1]

        # 存储最后一次变化量
        last_changes[ticker] = last_change
    except Exception as e:
        print(f"加载 {ticker} 的数据失败: {e}")

# 计算股票价格
# Define a dictionary to store stock prices
stock_prices = {}

# Loop through each stock ticker
for ticker in stock_tickers:
    try:
        # Load the data for the current stock ticker
        data_file_path = os.path.join(data_dir, f"{ticker}_data.csv")
        data = pd.read_csv(data_file_path, index_col=0, parse_dates=True)

        # Store the adjusted close prices for the current stock ticker
        stock_prices[ticker] = data['Adj Close']
    except Exception as e:
        print(f"Failed to load data for {ticker}: {e}")

# Print the loaded stock prices
#print("Loaded stock prices:")
#for ticker, prices in stock_prices.items():
 #   print(f"{ticker}: {prices}")

# 计算自变量
# 获取最后一次的 S&P 500 的月度变化量和 13周国债利率的月度变化量
y1_last = last_changes['^GSPC']
y2_last = last_changes['^IRX']

# 计算自变量
y3_last = y1_last ** 2
y4_last = y2_last ** 2
y5_last = y1_last * y2_last

# Create a DataFrame with the calculated values
X_last = pd.DataFrame({'const': 1, 'y1': y1_last, 'y2': y2_last, 'y3': y3_last, 'y4': y4_last, 'y5': y5_last}, index=[0])

# Print the DataFrame
#print("自变量数据集:")
#print(X_last)

# 计算预测回报率
predicted_returns = {}

# 遍历每只股票
for ticker in stock_tickers:
    # 获取该股票的系数
    intercept, coefficients = models_coefficients[ticker]

    # 计算预测回报率
    predicted_return = intercept + \
                       coefficients['y1'] * y1_last + \
                       coefficients['y2'] * y2_last + \
                       coefficients['y3'] * y3_last + \
                       coefficients['y4'] * y4_last + \
                       coefficients['y5'] * y5_last

    # 存储预测回报率
    predicted_returns[ticker] = predicted_return

# 打印预测回报率
#print("预测回报率:")
#for ticker, predicted_return in predicted_returns.items():
#    print(f"{ticker}: {predicted_return}")


import numpy as np
from scipy.optimize import minimize

shares = np.full(len(stock_tickers), 20)

# print(shares)

shares_dict = {ticker: share for ticker, share in zip(stock_tickers, shares)}
# print(shares_dict)


# Define the total assets change function
def total_assets_change_function(shares, predicted_returns, stock_prices):
    total_change = 0
    total_value = 0
    for i, ticker in enumerate(stock_tickers):
        # Get the last price for the current stock ticker
        current_price = stock_prices[ticker].iloc[-1]  # Assuming prices is a pandas Series

        # Get the predicted return for the current stock ticker
        predicted_return = predicted_returns[ticker]

        # Calculate the change in total assets
        total_change += shares[i] * predicted_return * current_price

        # Calculate the total value of the current stock holding
        total_value += shares[i] * current_price

    return -total_change  # Negative because we want to maximize


# Initialize shares to 200 for each stock ticker
shares = np.full(len(stock_tickers), 20)

# Define constraints
constraints = ({'type': 'ineq', 'fun': lambda x: x},
               {'type': 'ineq',
                'fun': lambda x: np.sum(x) - 10000})  # Total portfolio value should not exceed $1,000,000

# Perform optimization
result = minimize(total_assets_change_function, shares, args=(predicted_returns, stock_prices), constraints=constraints)

# Extract optimal shares
optimal_shares = result.x

print("最优的持股数量:")
for i, ticker in enumerate(predicted_returns.keys()):
    print(f"{ticker}: {round(optimal_shares[i])}")


