import numpy as np
import pandas as pd
import joblib

def inverse_transform_output(predicted_prices, lambda_value):
    """
    Inverse transforms the predicted prices from the model to get the original scale.
    
    :param predicted_prices: Array of predicted prices (scaled).
    :param lambda_value: Lambda value used during the Box-Cox transformation.
    :return: Inverse transformed prices to original scale.
    """
    # Step 1: Inverse Min-Max Scaling

    # Load the MinMaxScaler
    scaler = joblib.load('artifacts/model_trainer/min_max_scaler/min_max_scaler.pkl')
    
    # Convert predicted prices to a DataFrame
    df_predicted = pd.DataFrame(predicted_prices, columns=['pettah_average'])

    # Inverse transform the scaled prices back to the original scale
    original_prices = scaler.inverse_transform(df_predicted)

    # Step 2: Inverse Box-Cox Transformation
    # Apply inverse Box-Cox transformation
    if lambda_value == 0:
        inverse_transformed_prices = np.exp(original_prices)  # Special case for lambda = 0
    else:
        inverse_transformed_prices = np.power(original_prices * lambda_value + 1, 1 / lambda_value)

    return inverse_transformed_prices.flatten()  # Return as 1D array

# Example usage
# if __name__ == '__main__':
#     # Example output from the model (predicted prices)
#     predicted_scaled_prices = np.array([[0.5], [0.6], [0.7], [0.8], [0.9]])  # Example scaled output

#     # Lambda value used in Box-Cox transformation
#     lambda_value = -0.1207758043220706

#     # Get the inverse transformed prices
#     real_predicted_prices = inverse_transform_output(predicted_scaled_prices, lambda_value)
    
#     # Print the real predicted prices
#     print("Real Predicted Prices:")
#     print(real_predicted_prices)
