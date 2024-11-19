import numpy as np
import pandas as pd
from datetime import timedelta
import joblib

def inverse_transform_output(predicted_prices, lambda_value):
    """
    Inverse transforms the predicted prices from the model to get the original scale.
    
    :param predicted_prices: Array of predicted prices (scaled).
    :param lambda_value: Lambda value used during the Box-Cox transformation.
    :return: Inverse transformed prices to original scale.
    """
    # Load the MinMaxScaler
    scaler = joblib.load('artifacts/data_transformation/min_max_scaler.pkl')
    
    # Convert predicted prices to a DataFrame
    df_predicted = pd.DataFrame(predicted_prices[0], columns=['pettah_average'])

    print(df_predicted)

    # Inverse transform the scaled prices back to the original scale
    original_prices = scaler.inverse_transform(df_predicted)

    # Apply inverse Box-Cox transformation
    if lambda_value == 0:
        inverse_transformed_prices = np.exp(original_prices)  # Special case for lambda = 0
    else:
        inverse_transformed_prices = np.power(original_prices * lambda_value + 1, 1 / lambda_value)

    return inverse_transformed_prices.flatten()  # Return as 1D array

def create_prediction_dataframe(predicted_prices, last_date):

    # Reshape to (1, 5) if necessary
    if predicted_prices.ndim == 1:
        predicted_prices = predicted_prices.reshape(1, -1)

    # Ensure the shape is (1, 5) as expected
    if predicted_prices.shape != (1, 5):
        raise ValueError("Unexpected output from prediction pipeline. Expected a (1, 5) shape, but got something else.")

    # Initialize lists to store predictions and dates
    predicted_values = []
    predicted_dates = []

    # Extract each predicted value and corresponding date
    for j in range(5):
        predicted_price_value = predicted_prices[0, j]
        next_date = last_date + timedelta(days=j + 1)
        predicted_values.append(predicted_price_value)
        predicted_dates.append(next_date)

    # Create a DataFrame with predicted prices and dates
    df = pd.DataFrame({
        'date': predicted_dates,
        'predicted_value': predicted_values
    })

    return df