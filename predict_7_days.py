import pandas as pd
from datetime import timedelta
from prediction import prediction_pipeline
from src.connectors.cosmos import cosmos_url, cosmos_key, database_name, container_name, query_30_day
from src.pipeline.inference.inference_stage_data_ingestion_latest_30days import fetch_cosmosdb_data_to_dataframe_latest30days
from src.pipeline.inference.inference_stage_data_ingestion_latest_30days import process_cosmosdb_dataframe_latest30days

def predict_future_prices(lambda_value=-0.1207758043220706, days_to_predict=7):

    # Fetch data from Cosmos DB
    df = fetch_cosmosdb_data_to_dataframe_latest30days(cosmos_url, cosmos_key, database_name, container_name, query_30_day)
    
    df = process_cosmosdb_dataframe_latest30days(df)

    # Sort the DataFrame by date (if not already sorted)
    df = df.sort_values(by='date', ascending=True)
    
    # Get the initial sequence (last 30 prices)
    current_sequence = df['pettah_average'].values[-30:].tolist()  # Start with last 30 prices

    # Get the last date in the data to generate future dates
    last_date = pd.to_datetime(df['date'].values[-1])

    # Initialize lists to store predictions and dates
    predicted_values = []
    predicted_dates = []

    for i in range(days_to_predict):
        # Make a prediction for the next day using the current sequence
        predicted_price = prediction_pipeline(current_sequence, lambda_value)

        # Append the predicted price and the date
        next_date = last_date + timedelta(days=i + 1)  # Increment the date
        predicted_values.append(predicted_price[0])  # Extract scalar value from prediction
        predicted_dates.append(next_date)

        # Update the sequence: drop the first value, add the predicted value
        current_sequence = current_sequence[1:] + [predicted_price[0]]

    # Create a DataFrame with predicted prices and dates
    prediction_df = pd.DataFrame({
        'date': predicted_dates,
        'predicted_value': predicted_values
    })

    return prediction_df

if __name__ == '__main__':

    # Predict future values for 7 days
    predicted_df = predict_future_prices(lambda_value=-0.1207758043220706, days_to_predict=7)

    # Display the results
    print(predicted_df)
