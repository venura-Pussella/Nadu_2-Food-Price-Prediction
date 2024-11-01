import numpy as np
import pandas as pd
from scipy.stats import boxcox
import joblib

def box_cox_with_min_max_scaling(input_prices, lambda_value =-0.1207758043220706 ):
    """
    Applies Box-Cox transformation and Min-Max scaling to the input prices, 
    and returns the data in a format suitable for LSTM model inference.

    :param input_prices: List or array of prices (30 days sequence).
    :param lambda_value: The lambda value used for Box-Cox transformation.
    :return: Scaled prices reshaped for LSTM model (1, 30, 1).
    """
    # Step 1: Apply Box-Cox transformation using the provided lambda value
    input_prices = np.array(input_prices)
    transformed_prices = boxcox(input_prices, lmbda=lambda_value)
    
    # Step 2: Load the MinMaxScaler
    scaler = joblib.load('artifacts/model_trainer/min_max_scaler/min_max_scaler.pkl')
    
    # Convert transformed prices to a DataFrame with the same column names as during training
    df_prices = pd.DataFrame(transformed_prices, columns=['pettah_average'])

    # Step 3: Apply Min-Max scaling
    scaled_prices = scaler.transform(df_prices)
    
    # Step 4: Reshape the scaled prices to (1, sequence_length, num_features) for LSTM
    input_sequence = np.reshape(scaled_prices, (1, scaled_prices.shape[0], 1))  # Shape: (1, 30, 1)

    return input_sequence

# if __name__ == '__main__':
#     # Example input: Prices for 30 days
#     input_prices = [
#     100, 110, 120, 115, 130, 
#     140, 135, 150, 160, 155, 
#     170, 165, 180, 175, 190, 
#     200, 205, 210, 220, 225, 
#     230, 240, 250, 260, 255, 
#     270, 280, 290, 295, 300, 
#     310
# ]

#     # Lambda value obtained from training phase
#     lambda_value = -0.1207758043220706

#     # Call the function to get the input sequence for LSTM
#     input_sequence = min_max_scale_inference_with_boxcox(input_prices, lambda_value)

#     # Print the reshaped input sequence
#     print("Reshaped Input Sequence for LSTM:")
#     print(input_sequence)
