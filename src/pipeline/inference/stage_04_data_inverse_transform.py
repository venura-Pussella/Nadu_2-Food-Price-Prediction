from src.components.inference.inference_stage_03_inverse_transformation import inverse_transform_output , create_prediction_dataframe
from src.utils.common import read_lambda_value
from src.configuration.configuration import load_configuration, get_data_transformation_config

def data_inverse_transform(prediction, last_date):

    # Load the configuration
    config, _ = load_configuration()

    # Retrieve data ingestion configuration from the loaded config
    data_transformation_config = get_data_transformation_config(config)

    # Read the lambda value
    lambda_value = read_lambda_value(data_transformation_config)

    # inverse transform the prediction to see real values
    predicted_prices = inverse_transform_output(prediction, lambda_value)
    
    # Create a DataFrame with predicted prices and dates
    predicted_df = create_prediction_dataframe(predicted_prices, last_date)

    return predicted_df
