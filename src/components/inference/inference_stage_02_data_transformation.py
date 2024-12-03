import numpy as np
import pandas as pd
from scipy.stats import boxcox
import joblib

def box_cox_with_min_max_scaling(input_prices, lambda_value):
    """
    Applies Box-Cox transformation and Min-Max scaling to the input prices, 
    and returns the data in a format suitable for LSTM model inference.

    :param input_prices: List or array of prices (30 days sequence).
    :param lambda_value: The lambda value used for Box-Cox transformation.
    :return: Scaled prices reshaped for LSTM model (1, 30, 1).
    """
    # Step 1: Apply Box-Cox transformation using the provided lambda value
    input_prices = np.array(input_prices)

    transformed_prices = boxcox(input_prices, lambda_value)
    
    # Step 2: Load the MinMaxScaler
    scaler = joblib.load('artifacts/data_transformation/min_max_scaler.pkl')
    
    # Convert transformed prices to a DataFrame with the same column names as during training
    df_prices = pd.DataFrame(transformed_prices, columns=['pettah_average'])

    # Step 3: Apply Min-Max scaling
    scaled_prices = scaler.transform(df_prices)
    
    # Step 4: Reshape the scaled prices to (1, sequence_length, num_features) for LSTM
    input_sequence = np.reshape(scaled_prices, (1, scaled_prices.shape[0], 1))  # Shape: (1, 30, 1)

    return input_sequence

