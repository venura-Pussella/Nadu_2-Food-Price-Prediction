from src.connectors.cosmos import cosmos_url, cosmos_key, database_name, container_name, query_30_day
from src.pipeline.inference.prediction import prediction_pipeline
from src.pipeline.inference.inference_stage_data_ingestion_latest_30days import fetch_cosmosdb_data_to_dataframe_latest30days, process_cosmosdb_dataframe_latest30days
from src.pipeline.inference.inference_stage_output_preprocessing import create_prediction_dataframe
from src.configuration.configuration import load_configuration, get_data_transformation_config
from src.utils.common import read_lambda_value

def predict_future_prices():

    # Fetch data from Cosmos DB
    df = fetch_cosmosdb_data_to_dataframe_latest30days(cosmos_url, cosmos_key, database_name, container_name, query_30_day)

    df, current_sequence , last_date = process_cosmosdb_dataframe_latest30days(df)

    # Load the configuration
    config, _ = load_configuration()

    # Retrieve the data ingestion configuration from the loaded config
    data_transformation_config = get_data_transformation_config(config)

    # Read the lambda value
    lambda_value = read_lambda_value(data_transformation_config)
    
    # Make a single 5-day prediction using the current sequence
    predicted_prices = prediction_pipeline(current_sequence, lambda_value = lambda_value)
    
    # Create a DataFrame with predicted prices and dates
    predicted_df = create_prediction_dataframe(predicted_prices, last_date)

    return predicted_df

if __name__ == '__main__':

    # Predict future values for 5 days
    predicted_df = predict_future_prices()

    print(predicted_df)
