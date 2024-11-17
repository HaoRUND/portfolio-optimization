import yfinance as yf
import os

# 定义股票的列表，不包括指数和利率
stock_tickers = ['AMZN', 'GOOGL', 'IBM', 'INTC', 'NVDA', 'MSFT', 'TSLA', 'CRM', 'META', 'QCOM']
# 定义指数和利率的名称
index_and_rate_tickers = ['^GSPC', '^IRX']

# 创建一个新文件夹来保存股票数据
data_dir = 'stock_data'
os.makedirs(data_dir, exist_ok=True)

# 下载数据并保存到文件夹
for ticker in stock_tickers + index_and_rate_tickers:
    try:
        # 下载数据
        data = yf.download(ticker, start="2023-01-01", end="2023-12-31", interval="1mo")
        # 将数据保存到文件夹
        data_file_path = os.path.join(data_dir, f"{ticker}_data.csv")
        data.to_csv(data_file_path)
        print(f"已下载并保存 {ticker} 的数据到 {data_file_path}")
    except Exception as e:
        print(f"下载 {ticker} 的数据失败: {e}")
