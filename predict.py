import pandas as pd
from datetime import timedelta
from src.connectors.cosmos import cosmos_url, cosmos_key, database_name, container_name, query_30_day
from src.pipeline.inference.prediction import prediction_pipeline
from src.pipeline.inference.inference_stage_data_ingestion_latest_30days import fetch_cosmosdb_data_to_dataframe_latest30days, process_cosmosdb_dataframe_latest30days
from src.configuration.configuration import load_configuration, get_data_transformation_config
from src.utils.common import read_lambda_value

def predict_future_prices():

    # Fetch data from Cosmos DB
    df = fetch_cosmosdb_data_to_dataframe_latest30days(cosmos_url, cosmos_key, database_name, container_name, query_30_day)

    df = process_cosmosdb_dataframe_latest30days(df)

    # Sort the DataFrame by date (if not already sorted)
    df = df.sort_values(by='date', ascending=True)
    
    # Get the initial sequence (last 30 prices)
    current_sequence = df['pettah_average'].values[-30:].tolist()
    
    # Get the last date in the data to generate future dates
    last_date = pd.to_datetime(df['date'].values[-1])

    # Load the configuration
    config, _ = load_configuration()

    # Retrieve the data ingestion configuration from the loaded config
    data_transformation_config = get_data_transformation_config(config)

    # Read the lambda value
    lambda_value = read_lambda_value(data_transformation_config)
    
    # Make a single 5-day prediction using the current sequence
    predicted_prices = prediction_pipeline(current_sequence, lambda_value = lambda_value)
    
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

if __name__ == '__main__':

    # Predict future values for 5 days
    predicted_df = predict_future_prices()

    print(predicted_df)
