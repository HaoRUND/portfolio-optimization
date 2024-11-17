import pickle
import numpy as np
import os

# Path to the directory containing the trained models
model_dir = 'trained_models'

# List of stock tickers
stock_tickers = ['AMZN', 'GOOGL', 'IBM', 'INTC', 'NVDA', 'MSFT', 'TSLA', 'CRM', 'META', 'QCOM']

# Initialize a list to store the coefficients
A_values = []

# Load each model and extract the coefficients
for ticker in stock_tickers:
    model_file_path = os.path.join(model_dir, f'{ticker}_model.pkl')
    with open(model_file_path, 'rb') as model_file:
        model = pickle.load(model_file)
        A_values.append(model.params)

# Convert the list to a numpy array (matrix)
A_matrix = np.array(A_values)

# Print the parameter matrix A
print("Parameter matrix A:")
print(A_matrix)
