import pandas as pd
import numpy as np
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

# Map stock tickers to s0, s1, ..., s9
ticker_to_label = {ticker: f's{i}' for i, ticker in enumerate(stock_tickers)}


# Function to Plot Stock Price Changes with Triangle Markers
def plot_stock_changes(start_date, end_date):
    plt.figure(figsize=(14, 8))

    for ticker in stock_tickers:
        data = historical_data[ticker]

        # Ensure start_date and end_date are within the actual data range
        actual_start_date = data.index[data.index >= start_date][0]
        actual_end_date = data.index[data.index <= end_date][-1]

        filtered_data = data.loc[actual_start_date:actual_end_date]

        # Plot with triangle markers
        plt.plot(filtered_data.index, filtered_data['Adj Close'],
                 label=ticker_to_label[ticker], marker='v')  # 'v' for triangle down marker

    plt.title(f'Stock Price Changes from {actual_start_date.date()} to {actual_end_date.date()}')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.legend()
    plt.grid(True)
    plt.show()


# Define the date range for which to plot the stock changes
start_date = '2023-01-01'
end_date = '2023-12-31'

# Plot the stock price changes
plot_stock_changes(start_date, end_date)
