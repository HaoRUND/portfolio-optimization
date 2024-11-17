import pickle
import os
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt

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

# 定义股票的列表，不包括指数和利率
stock_tickers = ['AMZN', 'GOOGL', 'IBM', 'INTC', 'NVDA', 'MSFT', 'TSLA', 'CRM', 'META', 'QCOM']

# 指数和利率的
index_and_rate_tickers = ['^GSPC', '^IRX']

# 加载数据
data_dir = 'stock_data'

# 存储历史数据的字典
historical_data = {}
for ticker in stock_tickers + index_and_rate_tickers:
    data_file_path = os.path.join(data_dir, f"{ticker}_data.csv")
    historical_data[ticker] = pd.read_csv(data_file_path, index_col=0, parse_dates=True)


# 获取每只股票的当前价格
def get_last_prices(date):
    return np.array([historical_data[ticker].loc[date, 'Adj Close'] for ticker in stock_tickers])


# 计算模型预测回报率
def calculate_predicted_returns(date):
    last_changes = {ticker: historical_data[ticker]['Adj Close'].diff().loc[date] for ticker in index_and_rate_tickers}

    y1_last = last_changes['^GSPC']
    y2_last = last_changes['^IRX']

    y3_last = y1_last ** 2
    y4_last = y2_last ** 2
    y5_last = y1_last * y2_last

    X_last = pd.DataFrame({'const': 1, 'y1': y1_last, 'y2': y2_last, 'y3': y3_last, 'y4': y4_last, 'y5': y5_last},
                          index=[0])

    predicted_returns = {}
    for ticker in stock_tickers:
        intercept, coefficients = models_coefficients[ticker]
        predicted_return = intercept + \
                           coefficients['y1'] * y1_last + \
                           coefficients['y2'] * y2_last + \
                           coefficients['y3'] * y3_last + \
                           coefficients['y4'] * y4_last + \
                           coefficients['y5'] * y5_last
        predicted_returns[ticker] = predicted_return

    return predicted_returns


# 检查数据是否存在NaN
def check_for_nan(values, description):
    if np.isnan(values).any():
        raise ValueError(f"NaN detected in {description}: {values}")


# 设置投资组合初始值
initial_cash = 1000000
portfolio_value = initial_cash
portfolio_shares = np.zeros(len(stock_tickers))
dates = historical_data[stock_tickers[0]].index

# 存储投资组合价值的列表，并初始化为初始投资组合价值
portfolio_values = [initial_cash]

# 开始回测
for date in dates:
    try:
        # 获取当前价格
        current_prices = get_last_prices(date)
        check_for_nan(current_prices, "current_prices")

        # 计算预测回报率
        predicted_returns = calculate_predicted_returns(date)
        check_for_nan(list(predicted_returns.values()), "predicted_returns")

        # 创建决策变量
        shares = cp.Variable(len(stock_tickers), integer=True)

        # 目标函数：最大化总资产变动
        total_asset_change = cp.sum(
            [shares[i] * predicted_returns[ticker] * current_prices[i] for i, ticker in enumerate(stock_tickers)])

        # 约束条件
        constraints = [
            shares >= 0,  # 每只股票的持股数量必须是非负
            cp.sum(shares * current_prices) <= portfolio_value  # 总投资组合价值不能超过当前组合价值
        ]

        # 定义问题
        problem = cp.Problem(cp.Maximize(total_asset_change), constraints)

        # 求解问题
        problem.solve()

        # 获取最优持股数量
        optimal_shares = shares.value
        check_for_nan(optimal_shares, "optimal_shares")

        # 调整投资组合
        portfolio_shares = optimal_shares
        portfolio_value = np.sum(portfolio_shares * current_prices)

        # 存储当前的投资组合价值
        portfolio_values.append(portfolio_value)

        print(f"{date}: 投资组合价值: {portfolio_value:.2f}")

    except ValueError as e:
        print(f"Error on {date}: {e}")
        portfolio_values.append(portfolio_value)

print("最终投资组合价值:", portfolio_value)

# 将投资组合价值列表转换为 DataFrame
portfolio_values_df = pd.DataFrame(portfolio_values, index=[dates[0]] + list(dates), columns=['Portfolio Value'])

# 绘制投资组合价值随时间变化的图表
plt.figure(figsize=(14, 7))
plt.plot(portfolio_values_df.index, portfolio_values_df['Portfolio Value'], label='Portfolio Value')
plt.title('Portfolio Value Over Time')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.legend()
plt.grid(True)
plt.show()
