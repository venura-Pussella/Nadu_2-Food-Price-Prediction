import os
import pandas as pd

def save_predictions_to_csv(predicted_df, config):
    """
    Save new predictions to a CSV file. If the file exists, append the predictions.
    Ensure the 'date' column is formatted as a date string before saving.

    Args:
        predicted_df (pd.DataFrame): DataFrame containing new predictions.
        csv_file_path (str): Path to the CSV file where predictions will be saved.
    """
    try:
        # Check if the CSV file exists
        if os.path.exists(config.data_path):
            # Load existing data
            existing_df = pd.read_csv(config.data_path)

            # Append new predictions to the existing DataFrame
            combined_df = pd.concat([existing_df, predicted_df], ignore_index=True)
        else:
            # If the file doesn't exist, use the new predictions DataFrame
            combined_df = predicted_df

        # Ensure the 'date' column is formatted as a date string
        combined_df['date'] = pd.to_datetime(combined_df['date']).dt.strftime('%Y-%m-%d')

        # Save the combined DataFrame back to the CSV
        combined_df.to_csv(config.data_path, index=False)
        print(f"Data successfully saved to {config.data_path}")
    except Exception as e:
        print(f"An error occurred while saving predictions: {e}")