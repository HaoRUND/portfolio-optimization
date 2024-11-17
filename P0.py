import numpy as np
import pandas as pd
import pickle
import os

# Path to the directory containing the trained models
model_dir = 'trained_models'

# List of stock tickers
stock_tickers = ['AMZN', 'GOOGL', 'IBM', 'INTC', 'NVDA', 'MSFT', 'TSLA', 'CRM', 'META', 'QCOM']
index_and_rate_tickers = ['^GSPC', '^IRX']

# Load the historical data
data_dir = 'stock_data'
historical_data = {}
for ticker in stock_tickers + index_and_rate_tickers:
    data_file_path = f"{data_dir}/{ticker}_data.csv"
    historical_data[ticker] = pd.read_csv(data_file_path, index_col=0, parse_dates=True)

# Load the models and extract coefficients
A_values = []
for ticker in stock_tickers:
    model_file_path = os.path.join(model_dir, f'{ticker}_model.pkl')
    with open(model_file_path, 'rb') as model_file:
        model = pickle.load(model_file)
        A_values.append(model.params)

# Convert to numpy array (matrix)
A_matrix = np.array(A_values)


# Calculate P(0), Q, c
def calculate_initial_matrices(date, A_matrix):
    last_prices = np.array([historical_data[ticker].loc[date, 'Adj Close'] for ticker in stock_tickers])

    # P(0) is a diagonal matrix with last prices
    P_0 = np.diag(last_prices)

    # Q matrix
    Q = P_0 @ A_matrix @ A_matrix.T @ P_0

    # Calculate the factors at t=0
    y1 = historical_data['^GSPC']['Adj Close'][0]
    y2 = historical_data['^IRX']['Adj Close'][0]

    # Debugging information
    print(f"y1: {y1}, y2: {y2}")

    if pd.isna(y1) or pd.isna(y2):
        raise ValueError("One of the factors is NaN. Check the input data for missing values.")

    factors = np.array([1, y1, y2, y1 ** 2, y2 ** 2, y1 * y2])

    # Print Delta x(0)
    print(f"Delta x(0): {factors}")

    # c vector
    c = P_0 @ A_matrix @ factors

    return P_0, Q, c


# Example date
date = '2023-01-01'
P_0, Q, c = calculate_initial_matrices(date, A_matrix)

# Print the matrices
print("Matrix P(0):")
print(P_0)
print("\nMatrix Q:")
print(Q)
print("\nVector c:")
print(c)
