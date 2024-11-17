import yfinance as yf
import pandas as pd
import statsmodels.api as sm

# 定义股票的列表，不包括指数和利率
stock_tickers = ['AMZN', 'GOOGL', 'IBM', 'INTC', 'NVDA', 'MSFT', 'TSLA', 'CRM', 'META', 'QCOM']
# 定义指数和利率的名称
index_and_rate_tickers = ['^GSPC', '^IRX']

# 尝试下载数据
try:
    data = yf.download(stock_tickers + index_and_rate_tickers, start="2023-01-01", end="2023-12-31", interval="1mo")
except Exception as e:
    print(f"下载数据失败: {e}")
monthly_returns = data['Adj Close'].pct_change()

# 处理数据，填充缺失值
data.fillna(method='ffill', inplace=True)  # 向前填充
data.fillna(method='bfill', inplace=True)  # 向后填充

# 计算特征值
change = data['Adj Close'].diff()
y1 = change['^GSPC']  # S&P 500的月度变化量
y2 = change['^IRX']  # 13周国债利率的月度变化量
y3 = y1 ** 2
y4 = y2 ** 2
y5 = y1 * y2
# 准备自变量数据集
X = pd.DataFrame({'const': 1, 'y1': y1, 'y2': y2, 'y3': y3, 'y4': y4, 'y5': y5}).dropna()


import pickle
import os

# 创建一个新文件夹来保存模型
models_dir = 'trained_models'
os.makedirs(models_dir, exist_ok=True)

# 循环遍历每只股票，构建和拟合模型
model_summaries = {}
for ticker in stock_tickers:
    y = monthly_returns[ticker].dropna()  # 获取该股票的月度回报率
    X_aligned = X[X.index.isin(y.index)]  # 确保自变量和因变量在相同的时间点上对齐
    # 检查对齐后的X和y的长度是否一致
    if len(X_aligned) != len(y):
        print(f"数据长度不匹配: {ticker}")
        continue
    # 添加常数项用于截距
    X_aligned = sm.add_constant(X_aligned)
    # 构建和拟合模型
    model = sm.OLS(y, X_aligned).fit()
    # 保存模型摘要
    model_summaries[ticker] = model.summary()
    # Save the Model
    model_file_path = os.path.join(models_dir, f"{ticker}_model.pkl")
    with open(model_file_path, 'wb') as model_file:
        pickle.dump(model, model_file)



#提取回归系数
coefficients = {}
for ticker in stock_tickers:
    y = monthly_returns[ticker].dropna()  # 获取该股票的月度回报率
    X_aligned = X[X.index.isin(y.index)]  # 确保自变量和因变量在相同的时间点上对齐
    model = sm.OLS(y, X_aligned).fit()
    # 提取并保存回归系数
    coefficients[ticker] = model.params

import numpy as np
import cvxpy as cp

# 假设 n 是股票的数量
n = len(stock_tickers)  # 此处设置为您的股票数量

# 初始化决策变量
w = cp.Variable(n)

# 创建一个 n x n 的零矩阵 A
A = np.zeros((n, n))
# 假设 b 是一个长度为 n 的向量
b = np.random.randn(n)

# 常数 c
c = 0

initial_prices = data['Adj Close'].iloc[0]



# 累加每只股票的贡献
# 假设 n 是股票的数量
n = len(stock_tickers)  # 此处设置为您的股票数量

# 初始化决策变量
w = cp.Variable(n)

# 创建一个 n x n 的零矩阵 A
A = np.zeros((n, n))
# 假设 b 是一个长度为 n 的向量
b = np.random.randn(n)

# 常数 c
c = 0

initial_prices = data['Adj Close'].iloc[0]



# 累加每只股票的贡献
for ticker in stock_tickers:
    if ticker in coefficients:
        coef = coefficients[ticker]
        x_i = initial_prices[ticker]  # 这里是这个月第一天的价格
        A[0, 0] += coef['y3'] * x_i # y1^2 的系数
        A[1, 1] += coef['y4'] * x_i # y2^2 的系数
        A[0, 1] += coef['y5'] / 2  *x_i  # y1*y2 的系数需要除以2，因为会被计算两次
        A[1, 0] += coef['y5'] / 2  *x_i
        b[0] += coef['y1'] *x_i   # y1 的系数
        b[1] += coef['y2'] *x_i   # y2 的系数
        c += coef['const'] *x_i   # 常数项
# 确保 A 是半正定
A = A.T @ A

import cvxpy as cp

# 定义二次型目标函数
try:
    objective = cp.Minimize(cp.quad_form(w, A) + b.T @ w + c)
except Exception as e:
    print(f"Error: {e}")

# 定义约束条件
constraints = [cp.sum(w) == 1, w >= -1,  # 权重下界
               w <= 1]   # 权重上界]  # 权重和为1

# 创建和求解问题
prob = cp.Problem(objective, constraints)
prob.solve()


print("Optimized Weights:", w.value)

import cvxpy as cp

# 定义二次型目标函数
try:
    objective = cp.Minimize(cp.quad_form(w, A) + b.T @ w + c)
except Exception as e:
    print(f"Error: {e}")

# 定义约束条件
constraints = [cp.sum(w) == 1, w >= -1, w <= 1]

# 创建和求解问题
prob = cp.Problem(objective, constraints)
prob.solve()


print("Optimized Weights:", w.value)

