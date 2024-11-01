from src import logger
import pandas as pd
import numpy as np
import tensorflow as tf
from src.pipeline.inference.inference_stage_preprocessing import box_cox_with_min_max_scaling
from src.pipeline.inference.inference_stage_output_preprocessing import inverse_transform_output
from src.connectors.cosmos import cosmos_url, cosmos_key, database_name, container_name, query
from src.pipeline.inference.inference_stage_data_ingestion import fetch_cosmosdb_data_to_dataframe
from src.pipeline.inference.inference_stage_data_ingestion import process_cosmosdb_dataframe
from src.pipeline.inference.inference_stage_sequence_creation import create_sequences


def prediction_pipeline(input_prices, lambda_value = -0.1207758043220706):
    
    try:
        # input the sequences toBox_cox transform and Min_Max_Scaling 
        transformed_prices = box_cox_with_min_max_scaling(input_prices, lambda_value)

        # Load your trained LSTM model
        model = tf.keras.models.load_model('artifacts/model_trainer/model/best_model.keras')

        # make prediction
        prediction = model.predict(transformed_prices)

        # Inverse transform the prediction to see real values
        true_price = inverse_transform_output(prediction, lambda_value)

        return true_price
    
    except Exception as e:
        logger.error(f"An error occurred during inference pipeline: {e}")

def run_prediction_pipeline(sequence_length=30, lambda_value=-0.1207758043220706):

    # fetch data from the cosmos db
    df = fetch_cosmosdb_data_to_dataframe(cosmos_url, cosmos_key, database_name, container_name, query)

    # process the fethed data from the cosmos db to create a dataframe
    processed_df = process_cosmosdb_dataframe(df)

    # create sequences from the dataframe
    sequences, real_values, dates = create_sequences(processed_df, sequence_length)
    
    predictions = []
    
    # Step 2: Iterate over each sequence and predict the next value
    for sequence in sequences:
        predicted_value = prediction_pipeline(sequence, lambda_value)
        if predicted_value is not None:
            predictions.append(predicted_value[0])  # Get the first element from the 1D array
        else:
            predictions.append(np.nan)  # Handle missing predictions
    
    # Step 3: Create a DataFrame for comparison
    comparison_df = pd.DataFrame({
        'date': dates,
        'predicted_value': np.array(predictions).flatten(),
        'real_value': real_values,
    })
    
    return comparison_df

if __name__ == "__main__":

    # Example usage (assuming df contains the 'date' and 'pettah_average' columns)
    comparison_df = run_prediction_pipeline()

    # Display the comparison DataFrame
    print(comparison_df.head())